#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Apple, Inc. All Rights Reserved 
#
# @Time    : 2022/3/29 17:28
# @Author  : SeptKing
# @Email   : WJH0923@mail.dlut.edu.cn
# @File    : data_QX.py
# @Software: PyCharm
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset, \
    DataLoader  ##对于数据处理，最为简单的⽅式就是将数据组织成为⼀个。但许多训练需要⽤到mini-batch，直 接组织成Tensor不便于我们操作。pytorch为我们提供了Dataset和Dataloader两个类来方便的构建。
import torch
import pickle
import datetime

class data_loader(Dataset):
    def __init__(self, df_feature_left, df_feature_right, df_ytrue, t=None):
        assert len(df_feature_left) == len(df_ytrue)
        assert len(df_feature_left) == len(df_feature_right)

        self.df_feature_left = df_feature_left
        self.df_feature_right = df_feature_right
        self.df_ytrue = df_ytrue

        self.T = t
        # self.df_feature_left = torch.tensor(
        #     self.df_feature_left, dtype=torch.float32)
        # self.df_feature_right = torch.tensor(
        #     self.df_feature_right, dtype=torch.float32)
        # self.df_ytrue = torch.tensor(
        #     self.df_ytrue, dtype=torch.float32)

    def __getitem__(self, index):

        sample_left, sample_right, y_truth = self.df_feature_left[index], self.df_feature_right[index], self.df_ytrue[index]
        if self.T:
            return self.T(sample_left), self.T(sample_right), y_truth
        else:
            return sample_left, sample_right,  y_truth

    def __len__(self):
        return len(self.df_feature_left)


def create_dataset(data_file, station, start_date, end_date, mean=None, std=None):
    df = pd.read_pickle(data_file)
    data = df[station]
    feat, ytruth = data[0], data[1]
    referece_start_time = datetime.datetime(2011, 1, 1, 0, 0)
    referece_end_time = datetime.datetime(2020, 12, 31, 23, 0)
    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    start_hours = ((pd.to_datetime(start_date) - referece_start_time).seconds) / (60 * 60)
    start_days = ((pd.to_datetime(start_date) - referece_start_time).days) * 24
    index_start = int(start_days + start_hours)
    end_hours = ((pd.to_datetime(end_date) - referece_start_time).seconds) / (60 * 60)
    end_days = ((pd.to_datetime(end_date) - referece_start_time).days) * 24
    index_end = int(end_days + end_hours)
    feat = feat[index_start: index_end + 1]
    ytruth = ytruth[index_start: index_end + 1]

    return feat, ytruth


def get_dataset_statistic(df, station, start_date, end_date):
    data = df[station]
    feat, ytruth = data[0], data[1]
    referece_start_time = datetime.datetime(2011, 1, 1, 0, 0)
    referece_end_time = datetime.datetime(2020, 12, 31, 23, 0)
    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    start_hours = ((pd.to_datetime(start_date) - referece_start_time).seconds) / (60 * 60)
    start_days = ((pd.to_datetime(start_date) - referece_start_time).days) * 24
    index_start = int(start_days + start_hours)
    end_hours = ((pd.to_datetime(end_date) - referece_start_time).seconds) / (60 * 60)
    end_days = ((pd.to_datetime(end_date) - referece_start_time).days) * 24
    index_end = int(end_days + end_hours)
    feat = feat[index_start: index_end + 1]
    ytruth = ytruth[index_start: index_end + 1]
    mu_train = np.mean(feat, axis=0)
    sigma_train = np.std(feat, axis=0)

    return mu_train, sigma_train


##对生成的训练数据进行处理
def create_train_dataset(feat_train, y_train, batch_size, shuffle=False, mean=None, std=None):
    feat_train_left = feat_train[0]
    feat_train_right = feat_train[1]
    y_train = y_train
    feat_train_left = torch.tensor(feat_train_left,dtype=torch.float32)
    feat_train_right = torch.tensor(feat_train_right,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32)
    dataset = data_loader(
        feat_train_left, feat_train_right, y_train)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader


def get_weather_data_statistic(data_file, station, start_time, end_time):
    df = pd.read_pickle(data_file)
    mean_train, std_train = get_dataset_statistic(df, station, start_time, end_time)

    return mean_train, std_train
