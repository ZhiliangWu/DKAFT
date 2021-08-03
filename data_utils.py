#
# DKAFT
#
# Copyright (c) Siemens AG, 2021
# Authors:
# Zhiliang Wu <zhiliang.wu@siemens.com>
# License-Identifier: MIT

import gzip
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from logging_conf import logger


def do_train_valid_test_split(dataframe, id_col='uid', test_ratio=0.1,
                              cross=False):
    """

    Args:
        dataframe (pd.DataFrame): The DataFrame with an column of sample ids.
        id_col (str): The column name of the id.
        test_ratio (int): The ratio of the test set
        cross (bool): Whether do five-fold cross-validation split.

    Returns:
        pd.DataFrame: A split df with multiple columns indicating the label
            of each sample (train, valid, test).

    Example:
        >>> fn = './DL_lung_0.5.csv'
        >>> df = pd.read_csv(fn, index_col=0)
        >>> # sanity check
        >>> # df = pd.DataFrame(data=np.arange(100), columns=['uid'])
        >>> df_split = do_train_valid_test_split(df)
        >>> df_split.to_csv(f'{fn[:-4]}_idx_split.csv')
    """

    data_df = dataframe.reset_index()
    num_samples = len(data_df)
    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio,
                                           random_state=42)

    df_split = data_df.loc[:, [id_col]]
    if cross:
        kf = KFold(n_splits=5, shuffle=False)

        for i, (t, v) in enumerate(kf.split(train_idx)):
            train = train_idx[t]
            valid = train_idx[v]
            fold_name = f'fold_{i}'
            df_split[fold_name] = 0
            df_split.loc[train, fold_name] = 1
            df_split.loc[valid, fold_name] = 2
            df_split.loc[test_idx, fold_name] = 3
    else:
        for i in range(5):
            train, valid = train_test_split(train_idx, test_size=test_ratio,
                                            random_state=i)
            fold_name = f'fold_{i}'
            df_split[fold_name] = 0
            df_split.loc[train, fold_name] = 1
            df_split.loc[valid, fold_name] = 2
            df_split.loc[test_idx, fold_name] = 3

    return df_split


def get_tensor_loaders_in_file(data_fp, split_df, train_batch_size=64,
                               valid_batch_size=128, n_fold=0, cate=False,
                               fix=False):
    """Gets dataloaders for training / inference on mimic datasets.

    Args:
        data_fp (str): The file path of the csv file.
        split_df (pd.DataFrame): The DataFrame for the train-valid-test splits.
        train_batch_size (int): The batch size used for training.
        valid_batch_size (int): The batch size used for validation and testing.
        n_fold (int): The index of the training and validation set, from 1 to 5.
        cate (bool): Whether load the category information as targets.
        fix (bool): Whether do the shuffle during the training/inference.

    Returns:
        (DataLoader, DataLoader, DataLoader, int, int): Dataloaders for
            training, training/training_evaluation, validation and testing.

    """

    with gzip.open(data_fp, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    if cate:
        # category is loaded as target
        static, temp_tensor, _, target = data
    else:
        static, temp_tensor, target = data

    n_feature_sta = static.shape[-1]
    n_feature_temp = temp_tensor.shape[-1]

    dataset = TensorDataset(torch.tensor(static, dtype=torch.float),
                            torch.tensor(temp_tensor, dtype=torch.float),
                            torch.tensor(target, dtype=torch.float))

    train_idx = split_df[split_df[f'fold_{n_fold}'] == 1].index.to_list()
    valid_idx = split_df[split_df[f'fold_{n_fold}'] == 2].index.to_list()
    test_idx = split_df[split_df[f'fold_{n_fold}'] == 3].index.to_list()

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    logger.info(f'Size of the training: {len(train_idx)}.')
    logger.info(f'Size of the validation: {len(valid_idx)}.')
    logger.info(f'Size of the testing: {len(test_idx)}.')

    train_loader = DataLoader(dataset, batch_size=train_batch_size,
                              sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(dataset, batch_size=valid_batch_size,
                              sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=valid_batch_size,
                             sampler=test_sampler, num_workers=4)
    if fix:
        subset_train = Subset(dataset, indices=train_idx)
        train_loader = DataLoader(subset_train, batch_size=train_batch_size,
                                  shuffle=False)
        subset_test = Subset(dataset, indices=test_idx)
        test_loader = DataLoader(subset_test, batch_size=valid_batch_size,
                                 shuffle=False)

    return train_loader, valid_loader, test_loader, n_feature_sta, \
           n_feature_temp


def prepare_batch_tensor(batch, device, non_blocking, new_shape=None):
    """Prepare the batch data for training/inference, move data to GPU, reshape
    the target if necessary.

    Args:
        batch (torch.Tensor): A batch of data.
        device (torch.device or str): Device to load the backbone and data.
        non_blocking (bool): Whether tries to convert asynchronously with
            respect to the host if possible.
            https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to
        new_shape (tuple): The new shape of the target variable, sometimes
            necessary for certain API calls.

    Returns:
        (torch.Tensor, torch.Tensor)

    """

    sta, temp, y = batch
    sta = sta.to(device, dtype=torch.float, non_blocking=non_blocking)
    temp = temp.to(device, dtype=torch.float, non_blocking=non_blocking)
    y = y.to(device, dtype=torch.float, non_blocking=non_blocking)

    if new_shape:
        y = y.view(*new_shape)

    return (sta, temp), y


def get_tensor_loaders_in_file_pfs(data_fp, split_df, train_batch_size=64,
                                   valid_batch_size=128, n_fold=0,
                                   cate=False, fix=False):
    """Gets dataloaders for training / inference on PFS datasets.

    Args:
        data_fp (str): The file path of the csv file.
        split_df (pd.DataFrame): The DataFrame for the train-valid-test splits.
        train_batch_size (int): The batch size used for training.
        valid_batch_size (int): The batch size used for validation and testing.
        n_fold (int): The index of the training and validation set, from 1 to 5.
        cate (bool): Whether load the category information as targets.
        fix (bool): Whether do the shuffle during the training/inference.

    Returns:
        (DataLoader, DataLoader, DataLoader, int, int): Dataloaders for
            training, validation and testing, the number of hidden static
            representations and sequential representations.

    """

    data = torch.load(data_fp)

    if cate:
        X_padded, lengths, static_data, target, label = data
        dataset = TensorDataset(static_data, X_padded, lengths, target, label)

    else:
        X_padded, lengths, static_data, target = data
        dataset = TensorDataset(static_data, X_padded, lengths, target)

    n_feature_sta = static_data.size()[-1]
    n_feature_temp = X_padded.size()[-1]

    train_idx = split_df[split_df[f'fold_{n_fold}'] == 1].index.to_list()
    valid_idx = split_df[split_df[f'fold_{n_fold}'] == 2].index.to_list()
    test_idx = split_df[split_df[f'fold_{n_fold}'] == 3].index.to_list()

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    logger.info(f'Size of the training: {len(train_idx)}.')
    logger.info(f'Size of the validation: {len(valid_idx)}.')
    logger.info(f'Size of the testing: {len(test_idx)}.')

    train_loader = DataLoader(dataset, batch_size=train_batch_size,
                              sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(dataset, batch_size=valid_batch_size,
                              sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=valid_batch_size,
                             sampler=test_sampler, num_workers=4)

    if fix:
        subset_train = Subset(dataset, indices=train_idx)
        train_loader = DataLoader(subset_train, batch_size=train_batch_size,
                                  shuffle=False)
        subset_test = Subset(dataset, indices=test_idx)
        test_loader = DataLoader(subset_test, batch_size=valid_batch_size,
                                 shuffle=False)

    return train_loader, valid_loader, test_loader, n_feature_sta, \
           n_feature_temp


def prepare_batch_tensor_pfs(batch, device, non_blocking, new_shape=(-1, 1),
                             cate=False):
    """Prepare the batch data for training/inference, move data to GPU, reshape
    the target if necessary.

    Args:
        batch (torch.Tensor): A batch of data.
        device (torch.device or str): Device to load the backbone and data.
        non_blocking (bool): Whether tries to convert asynchronously with
            respect to the host if possible.
            https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to
        new_shape (tuple): The new shape of the target variable, sometimes
            necessary for certain API calls.
        cate (bool): Whether load the category information as targets.

    Returns:
        (torch.Tensor, torch.Tensor)

    """
    if cate:
        static_data, X_padded, lengths, _, label = batch
        y = label.to(device, dtype=torch.int64, non_blocking=non_blocking)

    else:
        static_data, X_padded, lengths, target = batch
        y = target.to(device, dtype=torch.float, non_blocking=non_blocking)

    sta = static_data.to(device, dtype=torch.float, non_blocking=non_blocking)
    temp = X_padded.to(device, dtype=torch.float, non_blocking=non_blocking)
    lgths = lengths.to(device, dtype=torch.float, non_blocking=non_blocking)

    if new_shape:
        y = y.view(*new_shape)

    return (sta, temp, lgths), y


if __name__ == '__main__':
    pass
