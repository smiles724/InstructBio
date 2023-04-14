import os
import argparse
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

import torch
from torch_geometric.loader import DataLoader
from MoleculeACE import get_model, Data, Descriptors, calc_rmse, RANDOM_SEED, set_seed, datasets, moleculenet_reg, moleculenet_cls, UnlabeledData, Logger


def cross_validate(algorithm, data, n_folds, save_path=None, **hyper):
    x_train, y_train, x_test, y_test, mean, mad = data.x_train, data.y_train, data.x_test, data.y_test, data.mean, data.mad
    test_loader = DataLoader(x_test, batch_size=args.batch + 1 if len(x_test) % args.batch == 1 else args.batch, shuffle=False)

    ss = StratifiedKFold(n_splits=n_folds, shuffle=True)
    cutoff = np.median(y_train)
    labels = [0 if i < cutoff else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    unlabeled_data = UnlabeledData(args.unlabeled_path, args.unlabeled_size)
    if unlabeled_data.x is None:
        unlabeled_data(Descriptors.GRAPH)
    x_unlabeled = unlabeled_data.x

    scores_before, scores = [], []
    log.logger.info(f'{"=" * 35} {args.data} {"=" * 35}\nModel: {args.model}, Aggr: {args.aggr}, Normalize: {bool(args.normalize)}, '
                    f'Train/Val/Test/Unlabeled: {int(len(y_train) * (1 - 1 / n_folds))}/{int(len(y_train) / n_folds)}/{len(y_test)}/{len(x_unlabeled)},'
                    f' Epochs: {args.instruct_epochs}, Batch: {args.batch}, Seed: {RANDOM_SEED}, Frequence: {args.update_label_freq}')
    for i_split, split in enumerate(splits):
        x_tr_fold = [x_train[i] for i in split['train_idx']] if type(x_train) is list else x_train[split['train_idx']]
        x_val_fold = [x_train[i] for i in split['val_idx']] if type(x_train) is list else x_train[split['val_idx']]
        y_tr_fold = [y_train[i] for i in split['train_idx']] if type(y_train) is list else y_train[split['train_idx']]
        y_val_fold = [y_train[i] for i in split['val_idx']] if type(y_train) is list else y_train[split['val_idx']]

        hyper['save_path'] = save_path
        model = algorithm(**hyper)
        print(f'{"=" * 30} Start Training [Fold {i_split + 1}/{n_folds}] {"=" * 33}')
        base_path = f"results/{args.model}_{args.data}_base.pt"
        if not os.path.exists(base_path):
            print('Start pretraining the model.')
            model.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, mean=mean, mad=mad, epochs=args.epochs, batch_size=args.batch, save_path=base_path)
        else:
            print(f'Loading pretrained weight from {base_path}.')
        weight, val_last = torch.load(base_path)
        model.model.load_state_dict(weight)
        model.val_losses = [val_last]

        y_hat = model.predict(test_loader, mean=mean, mad=mad)
        score_before = calc_rmse(y_test, y_hat)
        print(f"Before instruct learning -- rmse: {score_before:.3f}")

        score = model.InstructLearning(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, x_unlabeled, x_test, y_test, mean, mad, epochs=args.instruct_epochs, batch_size=args.batch,
                                       instruct_lr=args.instruct_lr, loss_path=args.loss_path, update_label_freq=args.update_label_freq, weight_factor=args.weight_factor,
                                       critic_epochs=args.critic_epochs, clip_grad=args.clip_grad, lambda_=args.lambda_)
        scores_before.append(score_before)
        scores.append(score)
        if args.debug: break
        del model
        torch.cuda.empty_cache()
    return sum(scores_before) / len(scores_before), sum(scores) / len(scores)  # return the rmse for all folds


def molecule_train(algorithm, data, save_path=None, **hyper):
    x_train, y_train, x_val, y_val, x_test, y_test = data.x_train, data.y_train, data.x_val, data.y_val, data.x_test, data.y_test
    if args.data in moleculenet_cls:
        mean, mad = 0.0, 1.0
        classification = True
    else:
        mean, mad = data.mean, data.mad
        classification = False
    log.logger.info(f'{"=" * 35} {args.data} TASKS {"=" * 35}\nModel: {args.model}, Aggr: {args.aggr}, Normalize: {bool(args.normalize)}, '
                    f'Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}\nEpochs: {args.instruct_epochs}, Batch: {args.batch}, Seed: {RANDOM_SEED},'
                    f' Weight factor: {args.weight_factor}, Frequency: {args.update_label_freq}, CLS: {classification}\n{"=" * 30} Start Training {"=" * 33}')
    hyper['save_path'] = save_path
    model = algorithm(**hyper)
    base_path = f"results/{args.model}_{args.data}_base.pt"
    if not os.path.exists(base_path):
        print('Start pretraining the model.')
        model.train(x_train, y_train, x_val, y_val, mean=mean, mad=mad, epochs=args.epochs, batch_size=args.batch, save_path=base_path, classification=classification)
    else:
        print(f'Loading pretrained weight from {base_path}.')
    weight, val_last = torch.load(base_path)
    model.model.load_state_dict(weight)
    model.val_losses = [val_last]

    test_loader = DataLoader(x_test, batch_size=args.batch + 1 if len(x_test) % args.batch == 1 else args.batch, shuffle=False)
    y_hat = model.predict(test_loader, mean=mean, mad=mad)
    if args.data in moleculenet_cls:
        if model.model.num_class > 1:
            rocauc_list = []
            y_test = np.array(y_test)
            for i in range(model.model.num_class):
                if 0 in y_test[:, i] and 1 in y_test[:, i]:  # AUC is only defined when there are two classes.
                    valids = (y_test[:, i] != 0.5)
                    rocauc_list.append(sklearn.metrics.roc_auc_score(y_test[valids, i], torch.sigmoid(y_hat[valids, i]).cpu().numpy()))
            score_before = sum(rocauc_list) / len(rocauc_list)
        else:
            score_before = sklearn.metrics.roc_auc_score(data.y_test, torch.sigmoid(y_hat).cpu().numpy())   # sigmoid to 0 - 1
        print(f"Before instruct learning -- roc-auc: {score_before:.3f}")
    else:
        score_before = calc_rmse(y_test, y_hat)
        print(f"Before instruct learning -- rmse: {score_before:.3f}")

    unlabeled_data = UnlabeledData(args.unlabeled_path, args.unlabeled_size)
    if unlabeled_data.x is None:
        unlabeled_data(Descriptors.GRAPH)
    x_unlabeled = unlabeled_data.x
    print(f'Unlabeled Size: {len(x_unlabeled)}')
    score = model.InstructLearning(x_train, y_train, x_val, y_val, x_unlabeled, x_test, y_test, mean, mad, epochs=args.instruct_epochs, batch_size=args.batch,
                                   instruct_lr=args.instruct_lr, loss_path=args.loss_path, update_label_freq=args.update_label_freq, weight_factor=args.weight_factor,
                                   critic_epochs=args.critic_epochs, clip_grad=args.clip_grad, classification=classification, lambda_=args.lambda_)
    return score_before, score


def main():
    moleculenet_tasks = {'clintox': 2, 'sider': 27, 'tox21': 12, 'toxcast': 617}
    set_seed(RANDOM_SEED)
    data = Data(args.data)
    data(Descriptors.GRAPH)  # we use GNNs
    if args.data not in moleculenet_cls and args.normalize:
        data.compute_mean_mad()
    save_path = f'results/{args.model}_{args.data}_best.pt'
    args.loss_path = f'results/{args.model}_{args.data}_loss'
    if args.data in ['tox21', 'toxcast']:  # nan value
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.data in moleculenet_cls:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()
    hyper = {'dropout': 0.4, 'edge_hidden': 256, 'fc_hidden': 512, 'lr': args.lr, 'num_layers': 3, 'n_fc_layers': 1, 'node_hidden': 128, 'transformer_hidden': 128,
             'aggr': args.aggr, 'loss_fn': loss_fn}
    if args.data in moleculenet_tasks.keys():
        hyper['num_class'] = moleculenet_tasks[args.data]
    hyper = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyper.items()}  # Convert numpy types to regular python type
    if args.data not in datasets:
        score_before, score = molecule_train(algorithm, data, save_path=save_path, **hyper)
        return score_before, score
    else:
        score_before, score = cross_validate(algorithm, data, n_folds=args.n_folds, early_stopping=10, seed=RANDOM_SEED, save_path=save_path, **hyper)
        return sum(score_before) / len(score_before), sum(score) / len(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='DATA', type=str, default='CHEMBL4203_Ki', choices=datasets + moleculenet_reg + moleculenet_cls)
    parser.add_argument('--normalize', type=int, default=1, help='scale the label between 0 and 1')
    parser.add_argument('--model', metavar='MODEL', type=str, default='GIN', choices=['MPNN', 'GAT', 'GCN', 'GIN'])
    parser.add_argument('--aggr', type=str, default='transformer', choices=['pool', 'transformer'], help='aggregation function to get graph-level features')
    parser.add_argument('--n_folds', metavar='FOLD', type=int, default=5, help='cross validation')  # do not use too few folds, since the dataset is small
    parser.add_argument('--epochs', metavar='N', type=int, default=100, help='training epochs, default=50')
    parser.add_argument('--batch', metavar='SIZE', type=int, default=64, help='batch size')
    parser.add_argument('--lr', metavar='RATE', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--clip_grad', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--instruct_epochs', metavar='N', type=int, default=400)
    parser.add_argument('--critic_epochs', metavar='N', type=int, default=10, help='epochs to pretrain the critic')
    parser.add_argument('--instruct_lr', metavar='RATE', default=1e-5, type=float, help='learning rate for reinforcement learning')
    parser.add_argument('--unlabeled_path', metavar='PRE', type=str, default='./250k_zinc')
    parser.add_argument('--unlabeled_size', type=int, default=-1, help='how many samples used for the entire instruct learning, -1 means not limitation.')
    parser.add_argument('--update_label_freq', type=int, default=5, help='interval epochs to re-label the data')
    parser.add_argument('--lambda_', type=float, default=0.0, help='control the signal for labeled data')
    parser.add_argument('--weight_factor', type=float, default=0.1, help='loss weight for different datasets, between 0 and 1')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    algorithm = get_model(args.model)

    log = Logger('./', f'training_{args.model}_{args.data}.log')
    try:
        metric_before, metric, = main()
        log.logger.info(f'{args.data}: {metric_before} --> {metric}. ')
    except KeyboardInterrupt:
        log.logger.info(f'Stop training for {args.data}')
