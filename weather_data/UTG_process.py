#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Apple, Inc. All Rights Reserved 
#
# @Time    : 2022/4/12 10:25
# @Author  : SeptKing
# @Email   : WJH0923@mail.dlut.edu.cn
# @File    : UTG_process.py
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
import weather_data.data_QX as data_QX
import datetime
from loss.nloss_transfer import TransferLoss
import torch
import math
from weather_data.data_vlsm import pad_all_cases
from weather_data import UTG_process
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
random.seed(SEED)


def get_split_time(num_domain=2, mode='pre_process', data_file=None, station=None, dis_type='mmd'):
    spilt_time = {
        '2': [('2011-1-1 0:0', '2015-11-30 23:0'), ('2015-12-2 0:0', '2020-3-15 23:0')]
    }
    if mode == 'pre_process':
        return spilt_time[str(num_domain)]
    if mode == 'tdc':
        return TDC(num_domain, data_file, station, dis_type=dis_type)
    else:
        print("error in mode")


def TDC(num_domain, data_file, station, dis_type='mmd'):
    start_time = datetime.datetime.strptime(
        '2011-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(
        '2020-3-15 23:00:00', '%Y-%m-%d %H:%M:%S')
    num_day = (end_time - start_time).days  ##一共有多少天
    split_N = 10  ##分为10个点
    data = pd.read_pickle(data_file)[station]
    feat = data[0]  ##选取训练集数据
    feat = feat.reshape(-1,24,feat.shape[1])
    feat = feat[0:num_day]
    feat = torch.tensor(feat, dtype=torch.float32)
    feat = feat.reshape(-1, feat.shape[2])  ##对数据按照小时维度进行展开(num_day*24,6)
    print(feat.size())

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0

    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_day)
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day)
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_day)
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_day)
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        # dis_temp += criterion_transder.compute(torch.as_tensor(torch.from_numpy(feat_part1), dtype=torch.float32), torch.as_tensor(torch.from_numpy(feat_part2), dtype=torch.float32))
                        dis_temp += criterion_transder.compute(feat_part1,feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))  ##计算最大的can——index
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]), hours=0)
            else:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]) + 1,
                                                                 hours=0)
            sel_end_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i]), hours=23)
            sel_start_time = datetime.datetime.strftime(sel_start_time, '%Y-%m-%d %H:%M')
            sel_end_time = datetime.datetime.strftime(sel_end_time, '%Y-%m-%d %H:%M')
            res.append((sel_start_time, sel_end_time))
        print(res)
        return res
    else:
        print("error in number of domain")

##训练数据生成部分

def train_val_test_generate(feat, ytrue, model_params):
    model_params = {
        'dim_in': 17,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
    }
    train_x, train_y, len_x_samples, len_before_x_samples = pad_all_cases(
        feat, ytrue, model_params, model_params['min_before'], model_params['max_before'],
        model_params['min_after'], model_params['max_after'],
        model_params['output_length'])
    ##扩充维度
    train_y = np.expand_dims(train_y, axis=2)

    return train_x, train_y, len_x_samples, len_before_x_samples


def train_test_split_SSIM(train_x, train_y, x_len, x_before_len, model_params, SEED):
    model_params = {
        'dim_in': 17,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
    }
    index_list = []
    for index, (x_s, y_s, len_s, len_before_s) in enumerate(zip(train_x, train_y, x_len, x_before_len)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    x = np.delete(train_x, index_list, axis=0)
    y = np.delete(train_y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)

    # print('x:{}'.format(x.shape))
    # print('y:{}'.format(y.shape))

    return x, y, x_len, x_before_len


def train_qld_single_station(x, y):
    train_sampling_params = {
        'dim_in': 17,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
    }
    feat_train, y_train, feat_train_len, feat_train_before_len = train_val_test_generate(x, y, train_sampling_params)
    # feat_train, y_train, feat_train_len, feat_train_before_len = train_test_split_SSIM(
    #     x_sample, y_sample, sample_len, sample_before_len,train_sampling_params, SEED)

    # print('x_train:{}'.format(feat_train.shape))
    # print('y_train:{}'.format(y_train.shape))

    feat_train = np.split(feat_train, [10, 16], axis=1)
    feat_train.pop(1)

    return feat_train, y_train


##从后向前进行修改
def load_weather_data_multi_domain(file_path, batch_size=6, station='Jintang', number_domain=2, mode='pre_process',
                                   dis_type='mmd'):
    # mode: 'auto', 'pre_process'
    data_file = os.path.join(file_path, "UTG_weather.pkl")

    ##选取训练数据，计算均值和方差
    mean_train, std_train = data_QX.get_weather_data_statistic(data_file, station=station,
                                                               start_time='2011-1-1 0:0',
                                                               end_time='2020-3-15 23:0')
    ##对训练数据进行分割，形成分割list
    split_time_list = get_split_time(number_domain, mode=mode, data_file=data_file, station=station, dis_type=dis_type)
    train_list = []  ##如果分成两段就有两个train——loader
    ##对于每个分割列表应用vlstm形成(n,leq,dim)
    for i in range(len(split_time_list)):
        time_temp = split_time_list[i]

        feat, ytrue = data_QX.create_dataset(data_file, station=station, start_date=time_temp[0],
                                             end_date=time_temp[1], mean=None, std=None)

        ##feat是一个list,主要有两个元素，第一个元素是array(770,10,16),第二个元素也是，然后将array转换成
        feat_train, y_train = train_qld_single_station(feat, ytrue)

        # print(feat_train)
        # print(y_train)
        train_loader = data_QX.create_train_dataset(
            feat_train, y_train, batch_size=batch_size, mean=mean_train, std=std_train)


        train_list.append(train_loader)

    ##对于验证集和测试集的数据准备
    feat_v, ytrue_v = data_QX.create_dataset(
        data_file, station=station, start_date='2020-6-1 0:0', end_date='2020-10-31 23:0', mean=None, std=None)
    feat_valid, y_valid = train_qld_single_station(feat_v, ytrue_v)
    valid_vld_loader = data_QX.create_train_dataset(
        feat_valid, y_valid, batch_size=batch_size, mean=mean_train, std=std_train)

    ##测试集数据调整
    feat_te, ytrue_te = data_QX.create_dataset(
        data_file, station=station, start_date='2019-6-1 0:0', end_date='2020-5-31 23:0', mean=None, std=None)
    feat_test, y_test = train_qld_single_station(feat_te, ytrue_te)
    test_loader = data_QX.create_train_dataset(
        feat_test, y_test, batch_size=batch_size, mean=mean_train, std=std_train)
    return train_list, valid_vld_loader, test_loader


if __name__ == '__main__':
    file_path = r'/Volumes/王九和/科研/农业大数据相关/实验/实验程序/玉皇/AdaRNN-BIT/weather_data'
    train_list, valid_vld_loader, test_loder = load_weather_data_multi_domain(file_path, batch_size=2,
                                                                              station='Jintang', number_domain=5,
                                                                              mode='tdc', dis_type='mmd')