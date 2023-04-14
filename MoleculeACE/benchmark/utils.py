import json
import os
import pickle
import random
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from yaml import load, Loader, dump

from MoleculeACE.benchmark import moleculenet_reg, moleculenet_cls
from MoleculeACE.benchmark.const import Descriptors, DATA_PATH, CONFIG_PATH_SMILES, CONFIG_PATH


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    Args:
        smiles: smiles sequence
        include_chirality: Default=False

    Return:
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def split(dataset, frac_train=0.8, frac_valid=0.1):
    """
    Args:
        dataset(pandas.DataFrame): the dataset to split. Make sure each element in
            the dataset has key "smiles" which will be used to calculate the
            scaffold.
        frac_train(float): the fraction of data to be used for the train split.
        frac_valid(float): the fraction of data to be used for the valid split.
    """
    RDLogger.DisableLog('rdApp.*')  # turn off warning from rdkit
    N = len(dataset)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i in range(N):
        try:
            scaffold = generate_scaffold(dataset['smiles'][i], include_chirality=True)
        except ValueError:  # not chemically reasonable smiles
            continue
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    return train_idx, valid_idx, test_idx


class UnlabeledData:
    def __init__(self, file: str, size_limit: int):
        if os.path.exists(f"{file}.pt"):
            self.x = torch.load(f"{file}.pt")[:size_limit]
        else:
            df = pd.read_csv(f"{file}.csv")
            self.smiles = df['smiles']
            if size_limit > 0: self.smiles = self.smiles[:size_limit]  # control the number of unlabeled data

            from MoleculeACE.benchmark.featurization import Featurizer  # do not move
            self.featurizer = Featurizer()
            self.x = None
            self.save_path = f"{file}.pt"

    def featurize_data(self, descriptor: Descriptors, **kwargs):
        self.x = self.featurizer(descriptor, smiles=self.smiles, **kwargs)

    def __call__(self, descriptor: Descriptors, **kwargs):
        self.featurize_data(descriptor, **kwargs)
        torch.save(self.x, self.save_path)


class Data:
    def __init__(self, file: Union[str, pd.DataFrame]):
        # Either load a .csv file or use a provided dataframe
        if type(file) is str:
            df = pd.read_csv(os.path.join(DATA_PATH, f"{file}.csv"))
        else:
            df = file

        if file == 'freesolv':
            df = df.rename(columns={'expt': 'y'})
        elif file == 'esol':
            df = df.rename(columns={'measured log solubility in mols per litre': 'y'})
        elif file == 'lipo':
            df = df.rename(columns={'exp': 'y'})
        elif file == 'bbbp':
            df = df.rename(columns={'p_np': 'y'})
        elif file == 'bace':
            df = df.rename(columns={'Class': 'y', 'mol': 'smiles'})
        elif file == 'tox21':
            df = df.drop(columns=['mol_id'])
            df = df.fillna(0.5)  # convert nan to 0.5
            cols = df.columns.tolist()
            df = df[cols[-1:] + cols[:-1]]  # reorder the columns
        elif file == 'toxcast':
            df = df.fillna(0.5)

        if 'y' in df:
            df['y'] = df['y'].astype('float')

        self.smiles_val, self.y_val = None, None
        if file in moleculenet_reg + moleculenet_cls:
            _split = [''] * len(df)
            train_idx, valid_idx, test_idx = split(df)
            for i in train_idx:
                _split[i] = 'train'
            for i in valid_idx:
                _split[i] = 'val'
            for i in test_idx:
                _split[i] = 'test'
            df['split'] = _split
            self.smiles_val = df[df['split'] == 'val']['smiles'].tolist()
            if 'y' in df:
                self.y_val = df[df['split'] == 'val']['y'].tolist()
            else:
                y_val = df[df['split'] == 'val']
                self.y_val = [y_val.iloc[i, 1:-1].astype('float').tolist() for i in range(len(y_val))]
            assert len(self.y_val) != 0

        self.smiles_train = df[df['split'] == 'train']['smiles'].tolist()
        self.smiles_test = df[df['split'] == 'test']['smiles'].tolist()

        if 'y' in df:
            self.y_train = df[df['split'] == 'train']['y'].tolist()
            self.y_test = df[df['split'] == 'test']['y'].tolist()
        else:
            y_train = df[df['split'] == 'train']
            y_test = df[df['split'] == 'test']
            self.y_train = [y_train.iloc[i, 1:-1].astype('float').tolist() for i in range(len(y_train))]
            self.y_test = [y_test.iloc[i, 1:-1].astype('float').tolist() for i in range(len(y_test))]

        from MoleculeACE.benchmark.featurization import Featurizer  # do not move

        self.featurizer = Featurizer()
        self.x_train, self.x_test, self.x_val = None, None, None
        self.mean, self.mad = 0.0, 1.0
        self.augmented = 0

    def featurize_data(self, descriptor: Descriptors, **kwargs):
        self.x_train = self.featurizer(descriptor, smiles=self.smiles_train, **kwargs)
        self.x_test = self.featurizer(descriptor, smiles=self.smiles_test, **kwargs)

        self.y_train = [self.y_train[i] for i, g in enumerate(self.x_train) if g is not None]
        self.x_train = [g for g in self.x_train if g is not None]
        self.y_test = [self.y_test[i] for i, g in enumerate(self.x_test) if g is not None]
        self.x_test = [g for g in self.x_test if g is not None]
        if self.smiles_val is not None:
            self.x_val = self.featurizer(descriptor, smiles=self.smiles_val, **kwargs)
            self.y_val = [self.y_val[i] for i, g in enumerate(self.x_val) if g is not None]
            self.x_val = [g for g in self.x_val if g is not None]

    def shuffle(self):
        """ Shuffle training data """
        c = list(zip(self.smiles_train, self.y_train))  # Shuffle all lists together
        random.shuffle(c)
        self.smiles_train, self.y_train = zip(*c)
        self.smiles_train, self.y_train = list(self.smiles_train), list(self.y_train)

    def compute_mean_mad(self):
        """ Normalize the label to prevent training instability. """
        mean = sum(self.y_train) / len(self.y_train)
        ma = [abs(i - self.mean) for i in self.y_train]
        mad = sum(ma) / len(ma)
        self.mean, self.mad = mean, mad

    def __call__(self, descriptor: Descriptors, **kwargs):
        self.featurize_data(descriptor, **kwargs)

    def __repr__(self):
        return f"{len(self.y_train)} train/{len(self.y_test)} test"


def load_model(filename: str):
    """ Load a algorithm """
    if filename.endswith('.h5'):
        raise ValueError('The algorithm weight is Tensorflow-based')
    else:
        with open(filename, 'rb') as handle:
            model = pickle.load(handle)
    return model


def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config


def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    args = {k: v.item() if isinstance(v, np.generic) else v for k, v in args.items()}
    with open(filename, 'w') as file:
        dump(args, file)


def get_benchmark_config(dataset: str, algorithm, descriptor):
    from MoleculeACE.models import GCN, MPNN, GAT, GIN
    list_of_algos = [GCN, MPNN, GAT, GIN]

    # Descriptor
    if type(descriptor) is not str:
        if not descriptor in [i for i in Descriptors]:
            raise ValueError(f"Chosen descriptor is not supported, please pick from: {[i.__str__() for i in Descriptors]}")
        descriptor = descriptor.name
    else:
        if not descriptor in [i.name for i in Descriptors]:
            raise ValueError(f"Chosen descriptor is not supported, please pick from: {[i.name for i in Descriptors]}")

    # Algorithm
    if type(algorithm) is not str:
        if algorithm not in list_of_algos:
            raise ValueError(f"Chosen algorithm is not supported, please pick from: {list_of_algos}")
        algorithm = algorithm.__name__
    else:
        if algorithm not in [i.__name__ for i in list_of_algos]:
            raise ValueError(f"Chosen algorithm is not supported, please pick from: {[i.__name__ for i in list_of_algos]}")

    combinations = {'ECFP': ['RF', 'SVM', 'GBM', 'KNN', 'MLP'], 'MACCS': ['RF', 'SVM', 'GBM', 'KNN'], 'PHYSCHEM': ['RF', 'SVM', 'GBM', 'KNN'], 'WHIM': ['RF', 'SVM', 'GBM', 'KNN'],
                    'GRAPH': ['GIN', 'GCN', 'MPNN', 'GAT', 'XGIN'], 'TOKENS': ['Transformer'], 'SMILES': ['CNN']}

    if algorithm not in combinations[descriptor]:
        raise ValueError(f'Given combination of descriptor and algorithm is not supported. Pick from: {combinations}')

    config_path = os.path.join(CONFIG_PATH, 'benchmark', dataset, f"{algorithm}_{descriptor}.yml")
    hyperparameters = get_config(config_path)

    return hyperparameters


def calc_rmse(true, pred):
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


smiles_encoding = get_config(CONFIG_PATH_SMILES)
