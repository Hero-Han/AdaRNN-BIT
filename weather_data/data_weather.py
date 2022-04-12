# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import datetime

class data_loader(Dataset):##这里是传入一个dataset函数来对数据集进行处理
    def __init__(self, df_feature, df_label_reg, t=None):

        assert len(df_feature) ==len(df_label_reg)
        self.df_feature = df_feature
        self.df_label_reg = df_label_reg

        self.T = t
        self.df_feature = torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label_reg = torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        ##获取第index的数据和标签
        sample, label_reg = self.df_feature[index], self.df_label_reg[index]
        if self.T:
            return self.T(sample), label_reg

        else:
            return sample, label_reg

    def __len__(self):
        return len(self.df_feature)

def create_dataset(df, station, start_date, end_date, mean=None, std=None):
    data = df[station]
    feat, label_reg = data[0], data[1]##该部分要看数据的维度，正常的reg应该是第2

    referece_start_time = datetime.datetime(2011, 1, 1, 0, 0, 0)##这一部分要根据现有数据进行调整
    referece_end_time = datetime.datetime(2020, 12, 31, 23, 0, 0)


    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    #assert (pd.to_datetime(end_date) - referece_end_time).days <= 0 ##这个有bug
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    start = (pd.to_datetime(start_date) - referece_start_time).total_seconds()
    end = (pd.to_datetime(end_date) - referece_start_time).total_seconds()
    index_start = start / 3600
    index_start = int(index_start)
    index_end= end / 3600
    index_end = int(index_end)
    feat = feat[index_start: index_end + 1]
    label_reg = label_reg[index_start: index_end + 1]

    return data_loader(feat, label_reg)

def create_dataset_shallow(df, station, start_date, end_date, mean=None, std=None):
    data=df[station]
    feat, label_reg =data[0], data[1] ##这个在处理数据的时候需要进行维度调整
    referece_start_time=datetime.datetime(2011, 1, 1, 0, 0, 0)
    referece_end_time=datetime.datetime(2020, 12, 31, 23, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    start=(pd.to_datetime(start_date) - referece_start_time).total_seconds()
    index_start = int(start) / 3600
    index_start = int(index_start)
    end = (pd.to_datetime(end_date) - referece_start_time).total_seconds()
    print(end)
    index_end= int(end) / 3600
    index_end = int(index_end)
    feat=feat[index_start: index_end + 1]
    #label=label[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]

    # ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    # feat=feat.reshape(-1, feat.shape[2])
    # feat=(feat - mean) / std
    # feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    return feat,  label_reg

def get_dataset_statistic(df, station, start_date, end_date):
    ##获取的都是feature的特征，这一部分跟label没啥关系，主要是获取feature的均值和方差
    data=df[station]
    feat=data[0]
    label_reg = data[1]# 调整数据
    referece_start_time=datetime.datetime(2011, 1, 1, 0, 0, 0)
    referece_end_time=datetime.datetime(2020, 12, 31, 23, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    start=(pd.to_datetime(start_date) - referece_start_time).total_seconds()
    index_start = int(start) / 3600
    index_start = int(index_start)
    end=(pd.to_datetime(end_date) - referece_start_time).total_seconds()
    index_end = int(end) / 3600
    index_end = int(index_end)
    feat=feat[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]##源代码象征这target的label
    #feat=feat.reshape(-1, feat.shape[2])
    mu_train=np.mean(feat, axis=0)
    sigma_train=np.std(feat, axis=0)

    return mu_train, sigma_train


def get_weather_data(data_file, station, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    df=pd.read_pickle(data_file)

    dataset=create_dataset(df, station, start_time,
                             end_time, mean=mean, std=std)
    train_loader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_weather_data_shallow(data_file, station, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    df=pd.read_pickle(data_file)

    feat, label_reg =create_dataset_shallow(df, station, start_time,
                             end_time, mean=mean, std=std)

    return feat, label_reg


def get_weather_data_statistic(data_file, station, start_time, end_time):
    df=pd.read_pickle(data_file)
    mean_train, std_train =get_dataset_statistic(
        df, station, start_time, end_time)
    return mean_train, std_train








