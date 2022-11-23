import numpy as np
import pandas as pd
import os
import csv
from glob import glob


def load_kfold(path_to_kfold):
    """Load all csv file (fold.csv)

    Args:
        path_to_kfold (str): path to folder containing csv files

    Returns:
        list: list of csv directory
    """
    lst_kfolds = []
    for f in glob(path_to_kfold + '/*.csv'):
        if f.find('fold') != -1:
            lst_kfolds.append(f.replace('\\', '/'))
    return lst_kfolds


def cross_validate(lst_kfolds):
    """Cross validation validation

    Args:
        lst_kfolds (list): list of csv directory (executed by load_kfold() function)

    Yields:
        tuple of dataframe: train dataframe and test dataframe
    """
    n_folds = len(lst_kfolds)
    for idx in range(n_folds):
        test_fold = n_folds - idx - 1
        test_file = lst_kfolds[test_fold]
        train_files = lst_kfolds[:test_fold] + lst_kfolds[test_fold + 1:]
        train_fold = []
        for train_file in train_files:
            with open(train_file, 'r') as file_fold:
                train_ = pd.read_csv(file_fold, header=None)
                train_fold.append(train_)
                del train_
        train_data = pd.concat(train_fold)
        # train_data = pd.concat((pd.read_csv(train_file, header=None) for train_file in train_files))
        test_data = pd.read_csv(test_file, header=None)
        yield train_data, test_data
