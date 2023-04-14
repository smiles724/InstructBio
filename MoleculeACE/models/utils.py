"""
Note: torch.autograd.set_detect_anomaly(True) makes the calculation very slow
"""
import os
import logging
import random
import numpy as np
from typing import List
import sklearn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from MoleculeACE.benchmark.const import CONFIG_PATH_SMILES
from MoleculeACE.benchmark.utils import get_config, calc_rmse

smiles_encoding = get_config(CONFIG_PATH_SMILES)
_MODEL_DICT = {}
get_model = lambda cfg: _MODEL_DICT[cfg]


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls

    return decorator


@register_model('GNN')
class GNN:
    def __init__(self):
        self.val_losses, self.val_roc = [], []
        self.model, self.critic, self.loss_fn, self.criterion, self.teacher_model = None, None, None, nn.BCELoss(), None
        self.optimizer, self.lr_scheduler = None, None
        self.device, self.save_path = None, None

    def train(self, x_train: List[Data], y_train: List[float], x_val, y_val, mean=0.0, mad=1.0, epochs=100, batch_size=64, patience=None, save_path=None, classification=False,
              x_test=None, y_test=None, log=None):
        wait = 0
        test_score = None
        t = tqdm(range(epochs))
        y_val = torch.tensor(y_val)
        train_loader = graphs_to_loader(x_train, y_train, batch_size=batch_size + 1 if len(x_train) % batch_size == 1 else batch_size)
        val_loader = DataLoader(x_val, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)
        if x_test is not None:
            y_test = torch.tensor(y_test)
            test_loader = DataLoader(x_test, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)

        for e in t:
            if patience is not None and wait >= patience:
                self.model.load_state_dict(torch.load(save_path))
                break
            else:
                loss = self._one_epoch(train_loader, mean, mad) / len(x_train)
                val_pred = self.predict(val_loader, mean=mean, mad=mad)
                if classification:
                    val_score = self.classification_test(val_pred, y_val.tolist(), self.model.num_class)
                    self.val_roc.append(val_score)
                else:
                    val_score = self.loss_fn(squeeze_if_needed(val_pred), y_val)
                    self.val_losses.append(val_score)

                if x_test is not None:
                    test_pred = self.predict(test_loader, mean=mean, mad=mad)
                    if classification:
                        test_score = self.classification_test(test_pred, y_test.tolist(), self.model.num_class)
                    else:
                        test_score = self.loss_fn(squeeze_if_needed(test_pred), y_test)

                if log is not None:
                    log.logger.info(f"Epoch {e + 1} | Train {loss:.3f} | Val {val_score:.3f} | Test: {test_score:.3f}")
                else:
                    t.set_description(f"Epoch {e + 1} | Train {loss:.3f} | Val {val_score:.3f} |  Best_Val {max(self.val_roc) if classification else min(self.val_losses):.3f} "
                                      f"| Lr: {self.optimizer.param_groups[0]['lr']}")

                if self.lr_scheduler is not None: self.lr_scheduler.step()

                if (classification and val_score >= max(self.val_roc)) or (not classification and val_score <= min(self.val_losses)):
                    torch.save([self.model.state_dict(), val_score], save_path)
                else:
                    wait += 1

    def _one_epoch(self, train_loader, mean, mad):
        self.model.train()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            batch.to(self.device)
            y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
            target = (batch.y - mean) / mad
            if self.model.num_class > 1:
                target = target.reshape(-1, self.model.num_class)
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(target))
            if loss.dim() > 1:
                valids = (target != 0.5)
                loss = (loss * valids).sum() / valids.sum()
            if not loss > 0:
                print(idx, y_hat[0], batch.y[0], target[0], loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * len(batch.y)
        return total_loss

    def InstructLearning(self, x_train, y_train, x_val, y_val, x_unlabeled, x_test, y_test, mean=0.0, mad=1.0, epochs=100, critic_epochs=50, patience=5, batch_size=64,
                         instruct_lr=1e-6, clip_grad=False, loss_path=None, update_label_freq=10, weight_factor=0.1, classification=False, soft_label=True, lambda_=0.0):
        for g in self.optimizer.param_groups:  # reset learning rate
            g['lr'] = instruct_lr

        test_score = -1.0
        t = tqdm(range(epochs))
        y_val = torch.tensor(y_val)
        mask = [1] * len(x_train) + [0] * len(x_unlabeled)
        unlabeled_loader = DataLoader(x_unlabeled, batch_size=batch_size + 1 if len(x_unlabeled) % batch_size == 1 else batch_size, shuffle=False)
        val_loader = DataLoader(x_val, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)
        test_loader = DataLoader(x_test, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.6)

        for epoch in t:
            ############################################################
            ### get pseudo-labels
            ############################################################
            if epoch % update_label_freq == 0:
                if self.teacher_model is not None and epoch == 0:
                    print('Predicting pseudo-labels by teacher model...')
                    predict_labels = self.predict(unlabeled_loader, model=self.teacher_model, mean=mean, mad=mad)
                elif self.teacher_model is None:
                    predict_labels = self.predict(unlabeled_loader, mean=mean, mad=mad)

                if self.teacher_model is None or (self.teacher_model is not None and epoch == 0):
                    if classification and soft_label:
                        predict_labels = torch.sigmoid(predict_labels)
                    elif classification:
                        predict_labels = (predict_labels > 0).float()
                    train_loader = graphs_to_loader_mask(x_train + x_unlabeled, y_train + predict_labels.tolist(), mask, batch_size=batch_size, shuffle=True)

            ############################################################
            ###  pretrain critic
            ############################################################
            if epoch == 0:
                print('Start pretraining the critic.')
                self.model.eval()
                self.critic.train()
                t_critic = tqdm(range(critic_epochs))
                best_running_loss_B, cnt = np.inf, 0
                y_hat_list, loss_A_list = [], []
                for epoch_critic in t_critic:
                    running_loss_B = 0.0
                    for idx, batch in enumerate(train_loader):
                        batch.to(self.device)
                        if epoch_critic == 0:
                            with torch.no_grad():
                                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).detach().clone()
                                if classification:
                                    y_hat = torch.sigmoid(y_hat)  # normalize between 0 and 1
                                    loss_A = F.binary_cross_entropy(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y.reshape(-1, self.model.num_class)),
                                                                    reduction='none').detach().clone()
                                else:
                                    loss_A = F.mse_loss(squeeze_if_needed(y_hat), squeeze_if_needed((batch.y - mean) / mad), reduction='none').detach().clone()
                            y_hat_list.append(y_hat)
                            loss_A_list.append(loss_A)
                        else:
                            y_hat = y_hat_list[idx]
                            loss_A = loss_A_list[idx]
                        if loss_A.mean() < 0:
                            print(idx, y_hat.max(), y_hat.min(), torch.max(batch.y), torch.min(batch.y), loss_A.mean())
                            raise ValueError
                        p = self.critic(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, y_hat, loss_A)
                        mask_target = batch.mask
                        if self.model.num_class > 1:
                            mask_target = mask_target.unsqueeze(dim=1).repeat(1, self.model.num_class)
                        loss_B = self.criterion(p, mask_target.unsqueeze(dim=-1).float())

                        self.optimizer.zero_grad()
                        loss_B.backward()
                        self.optimizer.step()
                        running_loss_B += float(loss_B) * len(y_hat)
                    t_critic.set_description(f'Critic Epoch: {epoch_critic + 1} | Loss_B: {running_loss_B / len(mask):.3f} | Lr: {self.optimizer.param_groups[0]["lr"]:.3f}')
                    lr_scheduler.step(running_loss_B)
                    if running_loss_B < best_running_loss_B:
                        best_running_loss_B = running_loss_B
                        cnt = 0
                    else:
                        cnt += 1
                        if cnt >= patience:
                            break  # early stop
                del y_hat_list, loss_A_list
                torch.cuda.empty_cache()
                for g in self.optimizer.param_groups:  # reset learning rate
                    g['lr'] = instruct_lr

            ############################################################
            ###  start semi-supervised training
            ############################################################
            self.model.train()
            self.critic.train()
            total_loss, total_loss_B, total_loss_C = 0.0, 0.0, 0.0
            for idx, batch in enumerate(train_loader):
                batch.to(self.device)
                y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                if classification:
                    y_hat = torch.sigmoid(y_hat)
                    loss_A = F.binary_cross_entropy(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y.reshape(-1, self.model.num_class)), reduction='none')
                else:
                    loss_A = F.mse_loss(squeeze_if_needed(y_hat), squeeze_if_needed((batch.y - mean) / mad), reduction='none')
                if loss_A.mean() < 0:
                    print(idx, y_hat[0], batch.y[0], loss_A.mean())
                    raise ValueError

                ############################################################
                ### compute loss for critic
                ############################################################
                p = self.critic(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, y_hat.detach().clone(), loss_A.clone().detach())
                soft_mask = batch.mask.clone().float()
                soft_mask[soft_mask > 0.9] = 0.9
                soft_mask[soft_mask < 0.1] = 0.1
                if self.model.num_class > 1:
                    soft_mask = soft_mask.unsqueeze(dim=1).repeat(1, self.model.num_class)
                loss_B = self.criterion(p, soft_mask.unsqueeze(dim=-1))

                ############################################################
                ### compute loss for model
                ############################################################
                p1 = p.detach().clone().squeeze()
                sample_weights = torch.zeros_like(p1, dtype=torch.float, device=self.device)
                num_label = (batch.mask > 0.5).sum()
                num_unlabel = (batch.mask < 0.5).sum()
                sample_weights[batch.mask > 0.5] = (1 + lambda_ * p1[batch.mask > 0.5]) / num_label
                sample_weights[batch.mask < 0.5] = weight_factor * (2 * p1[batch.mask < 0.5] - 1) / num_unlabel
                loss_C = (sample_weights * loss_A).sum() / sample_weights.sum()

                self.optimizer.zero_grad()
                loss = loss_B + loss_C
                if clip_grad:  # clip gradient to avoid the gradient explosion
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * len(batch.y)
                total_loss_B += float(loss_B) * len(batch.y)
                total_loss_C += float(loss_C) * len(batch.y)

            ############################################################
            ### evaluation on validation and test sets
            ############################################################
            val_pred = self.predict(val_loader, mean=mean, mad=mad)
            if classification:
                val_score = self.classification_test(val_pred, y_val.tolist(), self.model.num_class)
                self.val_roc.append(val_score)
            else:
                val_score = self.loss_fn(squeeze_if_needed(val_pred), y_val)
                self.val_losses.append(val_score)
            if (classification and val_score >= max(self.val_roc)) or (not classification and val_score <= min(self.val_losses)):
                torch.save(self.model.state_dict(), self.save_path)
                y_hat = self.predict(test_loader, mean=mean, mad=mad)
                if classification:
                    test_score = self.classification_test(y_hat, y_test, self.model.num_class)
                else:
                    test_score = calc_rmse(y_test, y_hat)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            t.set_description(f"Epoch {epoch + 1} | Loss: {total_loss / len(mask):.1f} | Loss_B: {total_loss_B / len(mask):.1f} | Loss_C: {total_loss_C / len(mask):.1f}"
                              f" | Val {val_score:.3f} |  Best_Val {max(self.val_roc) if classification else min(self.val_losses):.3f} | Test: {test_score:.3f} |"
                              f" Lr: {self.optimizer.param_groups[0]['lr']:.3f}")
        torch.save(self.val_losses, loss_path + '.pt')  # save loss weight
        plot_loss(self.val_losses, loss_path + '.pdf')  # plot the loss change

    def classification_test(self, preds, targets, num_class, ):
        if num_class > 1:
            rocauc_list = []
            targets = np.array(targets)
            for i in range(num_class):
                if 0 in targets[:, i] and 1 in targets[:, i]:  # AUC is only defined when there are two classes.
                    valids = (targets[:, i] != 0.5)
                    rocauc_list.append(sklearn.metrics.roc_auc_score(targets[valids, i], torch.sigmoid(preds[valids, i]).cpu().numpy()))
            test_score = sum(rocauc_list) / len(rocauc_list)
        else:
            test_score = sklearn.metrics.roc_auc_score(targets, torch.sigmoid(preds).cpu().numpy())
        return test_score

    def predict(self, loader, mean=0.0, mad=1.0, model=None):
        if model is None:
            model = self.model
        y_pred = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                y_hat = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).detach() * mad + mean
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)
        return torch.tensor(y_pred)


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'crit': logging.CRITICAL}

    def __init__(self, path, filename, level='info'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = logging.getLogger(path + filename)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

        th = logging.FileHandler(path + filename, encoding='utf-8')
        self.logger.addHandler(th)


def graphs_to_loader(x: List[Data], y: List[float], batch_size=64, shuffle: bool = False):
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def graphs_to_loader_mask(x: List[Data], y: List[float], masks: List[int], batch_size=64, shuffle: bool = False):
    for graph, label, mask in zip(x, y, masks):
        graph.y = torch.tensor(label)
        graph.mask = torch.tensor(mask)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle, drop_last=True)  # prevent BatchNorm error


def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor


def plot_loss(losses, save_path="loss.pdf"):
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.savefig(save_path)
