import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from torch_geometric.loader import DataLoader
from MoleculeACE import get_model, Data, Descriptors, calc_rmse, RANDOM_SEED, set_seed, datasets, UnlabeledData, Logger


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
                    f'Train/Val/Test/Unlabeled: {int(len(y_train) * (1 - 1 / n_folds))}/{int(len(y_train) / n_folds)}/{len(y_test)}/{len(x_unlabeled)}\n'
                    f'Epochs: {args.instruct_epochs}, Batch: {args.batch}, Seed: {RANDOM_SEED}, Frequence: {args.update_label_freq}')
    for i_split, split in enumerate(splits):
        x_tr_fold = [x_train[i] for i in split['train_idx']] if type(x_train) is list else x_train[split['train_idx']]
        x_val_fold = [x_train[i] for i in split['val_idx']] if type(x_train) is list else x_train[split['val_idx']]
        y_tr_fold = [y_train[i] for i in split['train_idx']] if type(y_train) is list else y_train[split['train_idx']]
        y_val_fold = [y_train[i] for i in split['val_idx']] if type(y_train) is list else y_train[split['val_idx']]

        hyper['save_path'] = save_path
        model = algorithm(**hyper)

        if args.teacher_model:
            if 'GIN' in args.teacher_model:  # TODO: record hyper-parameters of model training
                from MoleculeACE.models.gin import GINVirtual_node
                model.teacher_model = GINVirtual_node(aggr='transformer')
            teacher_weight = torch.load(f'results/{args.teacher_model}_{args.data}_best.pt')
            model.teacher_model.load_state_dict(teacher_weight)
            model.teacher_model.cuda()
            print(f'Loading teacher model {args.teacher_model}_{args.data}_best.pt successfully.')

        print(f'{"=" * 30} Start Training [Fold {i_split + 1}/{n_folds}] {"=" * 33}')
        base_path = f"results/{args.model}_{args.data}_base.pt"
        if not os.path.exists(base_path) or (args.teacher_model and args.teacher_stage == 'supervised'):
            if args.teacher_model:
                print('Start training the model with a teacher model.')
                base_path = f"results/{args.model}_{args.data}_student.pt"
                unlabeled_loader = DataLoader(x_unlabeled, batch_size=args.batch + 1 if len(x_unlabeled) % args.batch == 1 else args.batch, shuffle=False)
                predict_labels = model.predict(unlabeled_loader, model=model.teacher_model, mean=mean, mad=mad)

                model.train(x_tr_fold + x_unlabeled, y_tr_fold + predict_labels.tolist(), x_val_fold, y_val_fold, x_test=x_test, y_test=y_test, mean=mean, mad=mad,
                            epochs=args.epochs, batch_size=args.batch, save_path=base_path, log=log)
                exit()
            else:
                print('Start pretraining the model.')
                model.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, x_test=x_test, y_test=y_test, mean=mean, mad=mad, epochs=args.epochs,
                            batch_size=args.batch, save_path=base_path, log=log)
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
                                       critic_epochs=args.critic_epochs, clip_grad=args.clip_grad)
        scores_before.append(score_before)
        scores.append(score)
        if args.debug: break
        del model
        torch.cuda.empty_cache()
    return sum(scores_before) / len(scores_before), sum(scores) / len(scores)  # return the rmse for all folds


def main():
    set_seed(RANDOM_SEED)
    data = Data(args.data)
    data(Descriptors.GRAPH)  # we use GNNs
    if args.normalize:
        data.compute_mean_mad()
    save_path = f'results/{args.model}_{args.data}_best.pt'
    args.loss_path = f'results/{args.model}_{args.data}_loss'
    loss_fn = torch.nn.MSELoss()
    hyper = {'dropout': 0.4, 'edge_hidden': 256, 'fc_hidden': 512, 'lr': args.lr, 'num_layers': 3, 'n_fc_layers': 1, 'node_hidden': 128, 'transformer_hidden': 128,
             'aggr': args.aggr, 'loss_fn': loss_fn}
    hyper = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyper.items()}
    score_before, score = cross_validate(algorithm, data, n_folds=args.n_folds, early_stopping=10, seed=RANDOM_SEED, save_path=save_path, **hyper)
    return sum(score_before) / len(score_before), sum(score) / len(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='DATA', type=str, default='CHEMBL2147_Ki', choices=datasets)
    parser.add_argument('--normalize', type=int, default=1, help='scale the label between 0 and 1')
    parser.add_argument('--model', metavar='MODEL', type=str, default='', choices=['MPNN', 'GAT', 'GCN', 'GIN'])
    parser.add_argument('--aggr', type=str, default='transformer', choices=['pool', 'transformer'], help='aggregation function to get graph-level features')
    parser.add_argument('--teacher_model', type=str, default='', help='type of teacher model', choices=['GAT', 'GCN', 'GIN'])
    parser.add_argument('--teacher_stage', type=str, default='supervised', help='which stage to use the teacher model', choices=['supervised', 'semi-supervised'])

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
    parser.add_argument('--weight_factor', type=float, default=0.5, help='loss weight for different datasets, between 0 and 1')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    algorithm = get_model(args.model)

    log = Logger('./', f's_{args.model}{"_t_" + args.teacher_model if args.teacher_model else ""}_{args.data}.log')
    try:
        metric_before, metric, = main()
        print(f'{args.data}: {metric_before} --> {metric}. ')
    except KeyboardInterrupt:
        print(f'Stop training for {args.data}')
