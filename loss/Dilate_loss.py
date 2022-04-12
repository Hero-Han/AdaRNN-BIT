#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Apple, Inc. All Rights Reserved 
#
# @Time    : 2022/3/30 21:55
# @Author  : SeptKing
# @Email   : WJH0923@mail.dlut.edu.cn
# @File    : Dilate_loss.py
# @Software: PyCharm
import torch
from . import soft_dtw
from . import path_soft_dtw


def dilate_loss(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)（24，6，1）
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    ##24，6，6
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output+1).view(N_output, 1)).to(device)  ##是个对称矩阵
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal