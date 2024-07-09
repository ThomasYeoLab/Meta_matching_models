#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Naren Wulan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
import itertools
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.kernel_ridge import KernelRidge

if torch.cuda.is_available():
    data_type = torch.cuda.FloatTensor
else:
    data_type = torch.FloatTensor


class SFCN(nn.Module):
    ''' Simple Fully Convolutional Network

    Attributes:
        channel_number (list): channel number of each convolution layer
        output_dim (int): output dimensionality of SFCN
        dropout (float): dropout rate
        feature_extractor (torch.nn.Sequential): feature extractior of SFCN
        classifier (torch.nn.Sequential): classifier of SFCN
    '''

    def __init__(self,
                 channel_number=[32, 64, 128, 256, 256, 64],
                 output_dim=33,
                 dropout=0.1):
        super(SFCN, self).__init__()

        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel,
                                    out_channel,
                                    maxpool=True,
                                    kernel_size=3,
                                    padding=1))
            else:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel,
                                    out_channel,
                                    maxpool=False,
                                    kernel_size=1,
                                    padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(dropout))
        i = n_layer
        in_channel = channel_number[-1] + 1
        out_channel = output_dim
        self.classifier.add_module(
            'conv_%d' % i,
            nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel,
                   out_channel,
                   maxpool=True,
                   kernel_size=3,
                   padding=0,
                   maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel,
                          out_channel,
                          padding=padding,
                          kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel,
                          out_channel,
                          padding=padding,
                          kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel), nn.ReLU())
        return layer

    def forward(self, x, icv):

        x = self.feature_extractor(x)
        x = self.classifier.average_pool(x)
        x = torch.cat((x, icv), 1)
        x = self.classifier.dropout(x)
        x = self.classifier.conv_6(x)

        return x.reshape(x.shape[0], -1)
