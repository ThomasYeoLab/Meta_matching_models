#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Tong He and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init

import time
import itertools
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV


def sum_of_mul(A, B):
    '''sum of multiplication (inner product) of two array

    Args:
        A (ndarray): first array for computaion
        A (ndarray): second array for computaion

    Returns:
        ndarray: sum of multiplication
    '''
    return np.einsum('ij,ij->i', A, B)

def covariance_rowwise(A, B):
    '''rowwise covariance computation

    Args:
        A (ndarray): first array for covariance computaion
        B (ndarray): second array for covariance computaion

    Returns:
        ndarray: rowwise covariance between two array
    '''
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(0, keepdims=True)
    B_mB = B - B.mean(0, keepdims=True)

    N = A.shape[0]
    if B_mB.ndim == 1:
        B_mB = np.expand_dims(B_mB, -1)
    a_nsample = A_mA.shape[1]
    b_nsample = B_mB.shape[1]
    rnt = np.zeros((a_nsample, b_nsample))
    comb = np.array(list(itertools.product(range(a_nsample),
                                           range(b_nsample))))

    n_comb = len(comb)
    chunk = 100000
    if n_comb > chunk:
        start_time = time.time()
        cov = np.empty(n_comb)
        for i in range(chunk, n_comb, chunk):
            cov[i - chunk:i] = sum_of_mul(A_mA[:, comb[i - chunk:i, 0]].T,
                                          B_mB[:, comb[i - chunk:i, 1]].T)

            # print(i, time.time() - start_time)
        cov[i:] = sum_of_mul(A_mA[:, comb[i:, 0]].T, B_mB[:, comb[i:, 1]].T)
    else:
        cov = sum_of_mul(A_mA[:, comb[:, 0]].T, B_mB[:, comb[:, 1]].T)
    rnt[comb[:, 0], comb[:, 1]] = cov
    return np.squeeze(rnt) / (N - 1)

def demean_norm(val):
    '''de-mean and normalize data

    Args:
        val (ndarray): value to be de-meaned and normalized

    Returns:
        ndarray: de-meaned and normalized data
    '''
    mu = np.nanmean(val, axis=1, keepdims=True)
    val = val - mu
    return normalize(val, axis=1, norm='l2')


def stacking(y_pred_k, y_pred_test, y_k):
    '''perform stacking

    Args:
        y_pred_k (ndarray): predicted for K subjects with base model trained
            on training meta-set
        y_pred_test (ndarray): predicted for remaining test subjects with base
            model trained on training meta-set
        y_k (ndarray): original test data on k subjects
    
    Returns:
        ndarray: predicted value on remaining test subjects with stacking
    '''
    parameters = {'alpha': [5, 10, 15, 20]}
    krr = KernelRidge()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = GridSearchCV(krr, parameters, cv=cv)
    clf.fit(demean_norm(y_pred_k), y_k)
    return clf.predict(demean_norm(y_pred_test)), clf.predict(
        demean_norm(y_pred_k))


def torch_nanmean(x, mask):
    '''Calculate mean and omit NAN 

    Args:
        x (torch.tensor): input data
        mask (torch.tensor, optional): mask indicated NAN

    Returns:
        torch.Tensor: mean value (omit NAN)
    '''
    num = torch.where(mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(mask, torch.full_like(x, 0), x).sum()
    return value / num


def msenanloss(input, target, mask=None):
    '''Calculate MSE (mean absolute error) and omit NAN 

    Args:
        input (torch.tensor): predicted value
        target (torch.tensor): original value
        mask (torch.tensor, optional): mask indicated NAN

    Returns:
        torch.Tensor: MSE loss (omit NAN)
    '''
    ret = (input - target)**2
    if mask is None:
        mask = torch.isnan(ret)
    return torch_nanmean(ret, mask)


class multi_task_dataset(torch.utils.data.Dataset):
    """PyTorch dataset class

    Attributes:
        x (torch.tensor): tensor for x data
        y (torch.tensor): tensor for y data
    """
    def __init__(self, x, y, for_finetune=False):
        """initialization of PyTorch dataset class

        Args:
            x (ndarray): x data
            y (ndarray): y data
            for_finetune (bool, optional): whether the network is used for
                finetune
        """
        self.x = torch.from_numpy(x).float()
        if for_finetune:
            self.y = torch.from_numpy(y).float().view(-1, 1)
        else:
            self.y = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def __len__(self):
        return int(self.x.shape[0])


class dnn_5l(nn.Module):
    '''3 layer DNN model

    Attributes:
        fc1 (torch.nn.Sequential): First layer of DNN
        fc2 (torch.nn.Sequential): Second layer of DNN
        fc3 (torch.nn.Sequential): Third layer of DNN
        fc4 (torch.nn.Sequential): Fourth layer of DNN
        fc5 (torch.nn.Sequential): Fifth layer of DNN
    '''
    def __init__(self,
                 input_size,
                 n_l1,
                 n_l2,
                 n_l3,
                 n_l4,
                 dropout,
                 output_size=1):
        """initialization of 3 layer DNN

        Args:
            input_size (int): dimension of input data
            n_l1 (int): number of node in first layer
            n_l2 (int): number of node in second layer
            n_l3 (int): number of node in third layer
            n_l4 (int): number of node in fourth layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn_5l, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l2, n_l3),
            nn.ReLU(),
            nn.BatchNorm1d(n_l3),
        )
        self.fc4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l3, n_l4),
            nn.ReLU(),
            nn.BatchNorm1d(n_l4),
        )
        self.fc5 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l4, output_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


class dnn_4l(nn.Module):
    '''4 layer DNN model

    Attributes:
        fc1 (torch.nn.Sequential): First layer of DNN
        fc2 (torch.nn.Sequential): Second layer of DNN
        fc3 (torch.nn.Sequential): Third layer of DNN
        fc4 (torch.nn.Sequential): Fourth layer of DNN
    '''
    def __init__(self,
                 input_size,
                 n_l1,
                 n_l2,
                 n_l3,
                 dropout,
                 output_size=1):
        """initialization of 3 layer DNN

        Args:
            input_size (int): dimension of input data
            n_l1 (int): number of node in first layer
            n_l2 (int): number of node in second layer
            n_l3 (int): number of node in third layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn_4l, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l2, n_l3),
            nn.ReLU(),
            nn.BatchNorm1d(n_l3),
        )
        self.fc4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l3, output_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class dnn_3l(nn.Module):
    '''3 layer DNN model

    Attributes:
        fc1 (torch.nn.Sequential): First layer of DNN
        fc2 (torch.nn.Sequential): Second layer of DNN
        fc3 (torch.nn.Sequential): Third layer of DNN
    '''
    def __init__(self,
                 input_size,
                 n_l1,
                 n_l2,
                 dropout,
                 output_size=1):
        """initialization of 3 layer DNN

        Args:
            input_size (int): dimension of input data
            n_l1 (int): number of node in first layer
            n_l2 (int): number of node in second layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn_3l, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l2, output_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class dnn_2l(nn.Module):
    '''2 layer DNN model

    Attributes:
        fc1 (torch.nn.Sequential): First layer of DNN
        fc2 (torch.nn.Sequential): Second layer of DNN
    '''
    def __init__(self, input_size, n_l1, dropout, output_size=1):
        """initialization of 2 layer DNN

        Args:
            input_size (int): dimension of input data
            n_l1 (int): number of node in first layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn_2l, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l1, output_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def main():
    pass


if __name__ == '__main__':
    main()
