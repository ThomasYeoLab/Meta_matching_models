#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Pansheng Chen, Tong He and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import time
import pickle
import hashlib
import itertools

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV


def check_models_v20(model_path):
    """
    Check whether model files are correct by calculating the MD5 checksum.

    Args:
        model_path (str): The path to the model files.

    """
    # reference MD5 checksum values
    MD5_ref = {"meta_matching_v2.0_model.pkl_torch": "588ef6159617a6a9eda793ef3f812032",
               "UKBB_rr_models.sav": "a1a07906bee1702a9b33e2afc0ff022d",
               "ABCD_rr_models_multilayer.sav": "a8775a11a750aa7a50eaab2098ceb877",
               "ABCD_rr_models_base.sav": "c88f8d9e8b0d9eefc7c1817f246b73b3",
               "GSP_rr_models_base.sav": "4e62f1cfc5ee0df74b1df13e164137d1",
               "GSP_rr_models_multilayer.sav": "51c03f6e235f2fd1e72c605aa15f3603",
               "HBN_rr_models_base.sav": "c117dd8325fbfd4c223eb3529bd8f0ec",
               "HBN_rr_models_multilayer.sav": "5825fc86cc44e55eae7e309991b7f2ff",
               "eNKI_rr_models_base.sav": "257c67670797e0c5015bc94a4ea1d279",
               "eNKI_rr_models_multilayer.sav": "f3db2f85ac6134f8829387a5056f1358"
    }
    for model_file in MD5_ref:
        file_path = os.path.join(model_path, model_file)
        assert os.path.isfile(file_path), file_path + " doesn't exist."

        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Define the chunk size (8KB)
            chunk_size = 8192
            while True:
                # Read a chunk of data from the file
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # Update the MD5 hash object with the chunk
                md5_hash.update(chunk)
        assert md5_hash.hexdigest() == MD5_ref[model_file], \
            model_file + " is not update-to-date, please download the latest version."
            

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


def stacking(y_pred_k, y_pred_test, y_k, alpha=[0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15, 20]):
    '''perform stacking

    Args:
        y_pred_k (ndarray): predicted for K subjects with base model trained
            on training meta-set
        y_pred_test (ndarray): predicted for remaining test subjects with base
            model trained on training meta-set
        y_k (ndarray): original test data on k subjects
        alpha (list): Regularization strength range for KRR
    
    Returns:
        ndarray: predicted value on remaining test subjects with stacking
    '''
    parameters = {'alpha': alpha}
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


def multilayer_metamatching_infer(x, y, model_path, dataset_names):
    '''Predict using multilayer meta-matching models
    
    Args:
        x (ndarray): input FC data
        y (ndarray): target phenotype label
        model_path (str): multilayer meta-matching models' path
        dataset_names (dict): names of extra-large, large, medium source datasets 
    
    Returns:
        ndarray: prediction on x from multilayer metamatching models
        list: names of phenotypes be predicted
    '''
    # Load DNN, which takes FC as input and outputs 67 phenotypes prediction trained on 67 UK Biobank phenotypes
    gpu = 0 # modify the gpu number here if you want to use different gpu. 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_model_weight = os.path.join(model_path, 'meta_matching_v2.0_model.pkl_torch') 
    net = torch.load(path_model_weight, map_location=device)
    net.to(device)
    net.train(False)

    # Prepare data for DNN
    n_subj = x.shape[0]
    batch_size = 16
    y_dummy = np.zeros(y.shape)
    dset = multi_task_dataset(x, y_dummy, True)
    dataLoader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Phenotypic prediction from extra-large source dataset
    dataset_XL = dataset_names['extra-large']
    n_phe_dict = {}
    models = pickle.load(open(os.path.join(model_path, dataset_XL + '_rr_models.sav'), 'rb'))
    n_phe_dict[dataset_XL] = len(models.keys())

    y_pred_dnn_XL = np.zeros((0, n_phe_dict[dataset_XL]))
    for (x_batch, _) in dataLoader:
        x_batch= x_batch.to(device)
        outputs = net(x_batch)
        y_pred_dnn_XL = np.concatenate((y_pred_dnn_XL, outputs.data.cpu().numpy()), axis=0)

    y_pred_rr_XL = np.zeros((x.shape[0], n_phe_dict[dataset_XL]))
    for phe_idx, phe_name in enumerate(models):
        y_pred_rr_XL[:, phe_idx] = models[phe_name].predict(x)
        
    y_pred_rr_1layer = {}
    y_pred_rr_2layer = {}
    
    # Phenotypic prediction from large source dataset
    dataset_L = dataset_names['large']
    models_1layer = pickle.load(open(os.path.join(model_path, dataset_L + '_rr_models_base.sav'), 'rb'))
    models_2layer = pickle.load(open(os.path.join(model_path, dataset_L + '_rr_models_multilayer.sav'), 'rb'))
    n_phe_dict[dataset_L] = len(models_1layer.keys())
    y_pred_rr_1layer[dataset_L] = np.zeros((n_subj, n_phe_dict[dataset_L]))
    y_pred_rr_2layer[dataset_L] = np.zeros((n_subj, n_phe_dict[dataset_L]))

    RR_names = [] # record phenotype names from RR model
    for phe_idx, phe_name in enumerate(models_1layer):
        y_pred_rr_1layer[dataset_L][:, phe_idx] = models_1layer[phe_name].predict(x)
        RR_names.append(phe_name + '_ABCD')

    for phe_idx, phe_name in enumerate(models_2layer):
        x_stacking = np.concatenate((y_pred_dnn_XL, y_pred_rr_XL), axis = 1)
        y_pred_rr_2layer[dataset_L][:, phe_idx] = models_2layer[phe_name].predict(x_stacking)

    # Phenotypic prediction from medium source dataset
    for dataset_M in dataset_names['medium']:
        models_1layer = pickle.load(open(os.path.join(model_path, dataset_M + '_rr_models_base.sav'), 'rb'))
        models_2layer = pickle.load(open(os.path.join(model_path, dataset_M + '_rr_models_multilayer.sav'), 'rb'))
        n_phe =n_phe_dict[dataset_M] = len(models_1layer.keys())
        y_pred_rr_1layer[dataset_M] = np.zeros((n_subj, n_phe))
        y_pred_rr_2layer[dataset_M] = np.zeros((n_subj, n_phe))
        
        for phe_idx, phe_name in enumerate(models_1layer):
            y_pred_rr_1layer[dataset_M][:, phe_idx] = models_1layer[phe_name].predict(x)
            RR_names.append(phe_name + '_' +dataset_M)
            
        for phe_idx, phe_name in enumerate(models_2layer):
            x_stacking = np.concatenate((y_pred_dnn_XL, y_pred_rr_XL, y_pred_rr_1layer[dataset_L]), axis = 1)
            y_pred_rr_2layer[dataset_M][:, phe_idx] = models_2layer[phe_name].predict(x_stacking)

    y_pred = np.concatenate([y_pred_dnn_XL] + [y_pred_rr_XL] + 
                            list(y_pred_rr_1layer.values()) +
                            list(y_pred_rr_2layer.values()), axis = 1)

    y_names = []  # record predicted phenotype names
    data_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "data")
    phe_UKBB_txt = os.path.join(data_dir, "UKBB_phe_list.txt")
    with open(phe_UKBB_txt, 'r') as f:
        UKBB_names = [line.strip() for line in f.readlines()]
    y_names.extend([name + '_UKBB' + '_DNN' for name in UKBB_names])
    y_names.extend([name + '_UKBB' + '_KRR' for name in UKBB_names])
    y_names.extend([name + '_1layer' for name in RR_names])
    y_names.extend([name + '_2layer' for name in RR_names])

    return y_pred, y_names

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


class dnn(nn.Module):
    '''DNN model (2 - 5 layers)
    '''

    def __init__(self, input_size, n_layer, n_l1, n_l2, n_l3, n_l4, dropout, output_size=1):
        """initialization of 3 layer DNN

        Args:
            input_size (int): dimension of input data
            n_layer (int): number of layers
            n_l1 (int): number of node in first layer
            n_l2 (int): number of node in second layer
            n_l3 (int): number of node in third layer
            n_l4 (int): number of node in fourth layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn, self).__init__()

        self.layers = nn.ModuleList()
        n_hidden = [n_l1, n_l2, n_l3, n_l4]
        self.layers.append(nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        ))

        for i in range(n_layer):
            self.layers.append(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(n_hidden[i], n_hidden[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(n_hidden[i + 1]),
            ))
        self.layers.append(nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_hidden[n_layer], output_size),
        ))

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
        for layer in self.layers:
            x = layer(x)
        return x


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
