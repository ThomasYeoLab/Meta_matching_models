#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Naren Wulan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import torch
import time
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
import itertools
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from cbig.CBIG_model_pytorch import *

if torch.cuda.is_available():
    data_type = torch.cuda.FloatTensor
else:
    data_type = torch.FloatTensor

def mics_z_norm(train_y, valid_y, test_y=None):
    '''z normalize y of training, validation and test set based on training set

    Args:
        train_y (ndarray): training y data
        valid_y (ndarray): validation y data
        test_y (ndarray, optional): testing y data

    Returns:
        Tuple: contains z-normed y data and std of training y data
    '''

    # subtract mean of y of training set
    t_mu = np.nanmean(train_y, axis=0, keepdims=True)
    train_y = train_y - t_mu
    valid_y = valid_y - t_mu
    if test_y:
        test_y = test_y - t_mu

    # divide std of y of training set
    t_sigma = np.nanstd(train_y, axis=0)
    if train_y.ndim == 2:
        t_sigma_d = t_sigma[np.newaxis, :]
    else:
        t_sigma_d = t_sigma
        if t_sigma == 0:
            print('t_sigma is 0, pass divide std')
            return [train_y, valid_y, test_y, t_sigma]
    train_y = train_y / t_sigma_d
    valid_y = valid_y / t_sigma_d
    if test_y:
        test_y = test_y / t_sigma_d

    return [train_y, valid_y, test_y, t_sigma]

def crop_center(data, out_sp):
    '''Crop 3D volumetric input size

    Args:
        data (ndarray): input data (182, 218, 182)
        out_sp (tuple): output size (160, 192, 160)

    Returns:
        data_crop (ndarray): cropped data
    '''

    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

class vol_dataset(torch.utils.data.Dataset):
    """PyTorch dataset class for volumetric data

    Attributes:
        x (ndarray): volumetric data
        y (ndarray): phenotype data
        icv (ndarray): intracerebroventricular (ICV)
    """

    def __init__(self, x, y, icv=None):
        """initialization of PyTorch dataset class
        """
        self.sublist = x
        self.y = torch.from_numpy(y).float()
        self.cutoff_size = (160, 192, 160)
        self.icv = icv

        if icv is not None:
            self.icv = torch.from_numpy(icv).float()[:, None, None, None]

    def __getitem__(self, idx):
        nii_data = self.sublist[idx]
        # Preprocessing
        nii_data = crop_center(nii_data, self.cutoff_size)
        nii_data[nii_data < 0] = 0
        x = torch.unsqueeze(
            torch.from_numpy(nii_data / nii_data.mean()).float(), 0)

        y = self.y[idx]

        if self.icv is not None:
            icv = self.icv[idx]
            return x, y, icv
        else:
            return x, y

    def __len__(self):
        return int(self.sublist.shape[0])

def znorm_icv(icv):
    """z normlize icv for target dataset using icv statistics from UK Biobank

    Attributes:
        icv (ndarray): intracerebroventricular (ICV)
    """

    icv = icv - 0.75488614
    icv = icv / 0.06941227

    return icv


def metamatching_infer(x, icv, y, model_dir):
    '''Predict using multilayer meta-matching models

   Args:
       x (ndarray): input T1 data
       icv (ndarray): input icv data
       y (ndarray): target phenotype label
       model_dir (str): multilayer meta-matching models' path

   Returns:
       ndarray: prediction on x from mmetamatching models
    '''

    # set gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    n_phe = 67 # number of phenotypes in UK Biobank dataset
    dset_test = vol_dataset(x, y, icv=icv)
    batch_size = 4
    testLoader = DataLoader(dset_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=batch_size)

    # load trained model
    opt_index = 98
    weight_path = os.path.join(
        model_dir,
        'CBIG_ukbb_dnn_run_0_epoch_' + str(opt_index) + '.pkl_torch')
    print(weight_path)

    net = torch.load(weight_path)  # map_location=torch.device('cpu')
    net.to(device)
    net.train(False)

    record_pred = np.zeros((0, n_phe))  # prediction value
    tes_res_record = np.zeros((1, 1, x.shape[0], n_phe))
    for (x, y, icv) in testLoader:
        x, y, icv = x.to(device), y.to(device), icv.to(device)
        outputs = net(x, icv)
        record_pred = np.concatenate((record_pred, outputs.data.cpu().numpy()),
                                     axis=0)

        del outputs, x, y
        torch.cuda.empty_cache()
    tes_res_record[0, 0, :, :] = np.squeeze(record_pred)

    return np.squeeze(tes_res_record)

def stacking(y_pred_k, y_pred_test, y_k):
    '''perform KRR for meta-matching stacking

    Args:
        y_pred_k (ndarray): input data in meta-test set for training
        y_pred_test (ndarray): input data in meta-test set for testing
        y_k (ndarray) : output data in meta-test set for training
        args (argparse.ArgumentParser) : args that could be used by
          other function

    Returns:
        Tuple: prediction for testing and training data in meta-test set

    '''

    parameters = {
        'alpha': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    }

    krr = KernelRidge()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = GridSearchCV(krr, parameters, cv=cv)
    clf.fit(y_pred_k, y_k)

    return clf.predict(y_pred_test), clf.predict(y_pred_k)

def load_3D_input(sublist):
    '''load volumetric T1 data
       Args:
           sublist (list): list of participants
       Returns:
           data_arr (ndarray): T1 data of all participants in list
    '''

    cutoff_size = (160, 192, 160)

    data_list = []
    for idx in range(len(sublist)):
        nii_data = sublist[idx]

        nii_data = crop_center(nii_data, cutoff_size)
        nii_data[nii_data < 0] = 0
        x = (nii_data / nii_data.mean()).flatten()
        data_list.append(x)

    data_arr = np.array(data_list)
    return data_arr

def sum_of_mul(A, B):
    '''sum of multiplication of two array over axis=1
    Args:
        A (ndarray): first array for calculation
        B (ndarray): second array for calculation
    Returns:
        ndarray: sum of multiplication calculated
    '''

    return np.einsum('ij,ij->i', A, B)

def covariance_rowwise(A, B):
    '''compute rowwise covariance
    Args:
       A (ndarray): first array for covariance calculation, n_subject x
          n_features
       B (ndarray): second array for covariance calculation, n_subject x 1
    Returns:
       ndarray: covariance calculated
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

            print(i, time.time() - start_time)
        cov[i:] = sum_of_mul(A_mA[:, comb[i:, 0]].T, B_mB[:, comb[i:, 1]].T)
    else:
        cov = sum_of_mul(A_mA[:, comb[:, 0]].T, B_mB[:, comb[:, 1]].T)
    rnt[comb[:, 0], comb[:, 1]] = cov
    return np.squeeze(rnt) / (N - 1)