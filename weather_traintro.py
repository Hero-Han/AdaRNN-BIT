#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Apple, Inc. All Rights Reserved 
#
# @Time    : 2022/3/28 21:14
# @Author  : SeptKing
# @Email   : WJH0923@mail.dlut.edu.cn
# @File    : weather_train.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.optim as optim
from tslearn.metrics import dtw, dtw_path
import os
import argparse
import datetime
import numpy as np
from loss import Dilate_loss
from tqdm import tqdm
from untils import utils
from model_Sept.Dual_Adarnn import Dual_Adarnn, Cross_Attention, Decoder, Share_Encoder
import weather_data.TE_process as data_process
import matplotlib.pyplot as plt
from untils.support import *
from d2l import torch as d2l
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pprint(*text):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow() +
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *text, flush=True)
    if args.log_file is None:
        return
    with open(args.log_file, 'a') as f:
        print(time, *text, flush=True, file=f)

def get_model(name='DualAdarnn'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    share_encoder = Share_Encoder( n_input=args.d_feat, n_hiddens = n_hiddens,dec_layers=args.dec_layers, dropout=args.dropout,model_type=args.model_name,
                          len_seq=args.len_seq, trans_loss=args.loss_type)
    cross_attention = Cross_Attention(n_hiddens=n_hiddens)
    decoder = Decoder(output_dim=args.class_num, n_hiddens=n_hiddens,dec_layers=args.dec_layers, dropout=args.dropout, attention=cross_attention)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Dual_Adarnn(share_encoder=share_encoder, decoder=decoder,output_dim=args.class_num, len_seq=args.len_seq, device=device).to(device)

def train_DualRNN(args, model_king, optimizer, train_loader_list, epoch, teacher_forcing_ratio, alpha, beta, dist_old_before = None, dist_old_after = None, weight_mat_before=None, weight_mat_after=None):
    model_king.train()
    criterion = nn.MSELoss()  ##代价函数
    criterion_1 = nn.L1Loss()  ##代价函数
    loss_all = []
    loss_1_all = []
    dist_mat_before = torch.zeros(args.num_layers, args.len_seq).to(device)  ##[1，10列]
    dist_mat_after = torch.zeros(args.num_layers, args.len_seq).to(device)
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)

    for data_all in tqdm(zip(*train_loader_list), total=len_loader):##按照batch加载数据，data_all表示的是分段的时间序列
        optimizer.zero_grad()
        list_feat_left = []
        list_feat_right = []
        list_ytrue = []  ##用的是0，1编码

        for data in data_all:##data——all也是个list
            fea_left, fea_right, yture = data[0].to(device).float(
            ), data[1].to(device).float(), data[2].to(device)
            list_feat_left.append(fea_left)  ##两个tensor
            list_feat_right.append(fea_right)
            list_ytrue.append(yture)
        flag = False
    ##test时间11：21

        index = get_index(len(data_all)-1)

        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat_left[s1].shape[0] != list_feat_right[s2].shape[0]:
                flag = True
                break
        if flag:
            continue
        # out_weight_list_before = []
        total_loss = torch.zeros(1).to(device)

        for i in range(len(index)):
            feature_s_left = list_feat_left[index[i][0]]
            feature_t_left = list_feat_left[index[i][1]]
            feature_s_right = list_feat_right[index[i][0]]
            feature_t_right = list_feat_right[index[i][0]]
            ytrue_s = list_ytrue[index[i][1]]
            ytrue_t = list_ytrue[index[i][1]]
            feature_all_left = torch.cat((feature_s_left,feature_t_left), 0)
            feature_all_right = torch.cat((feature_s_right,feature_t_right),0)
            feature_all_left = feature_all_left.permute(1,0,2)##10，2，16
            feature_all_right = feature_all_right.permute(1,0,2)## 10，2，16
            ytrue_all = torch.cat((ytrue_s,ytrue_t),0)
            ytrue_all = ytrue_all.permute(1,0,2)##6，2，1
            # print(feature_all_left.size())
            # print(feature_all_right.size())

            if epoch < args.pre_epoch:
                pred_s, pred_t,out_weight_list_before,out_weight_list_after,decoder_att,loss_transfer = model_king.for_custom_pre(
                    feature_all_left,feature_all_right,ytrue_all,teacher_forcing_ratio )


            else:
                pred_s, pred_t,decoder_att, loss_transfer, dist_before,dist_after,weight_mat_before,weight_mat_after = model_king.for_Boosting(
                    feature_all_left,feature_all_right,ytrue_all,teacher_forcing_ratio ,weight_mat_before ,weight_mat_after)
                dist_mat_before = dist_mat_before + dist_before
                dist_mat_after = dist_mat_after + dist_after

            loss_s = criterion(pred_s, ytrue_s)
            loss_t = criterion(pred_t, ytrue_t)
            loss_l1 = criterion_1(pred_s, ytrue_s)
            Loss_s, loss_shape_s, loss_temporal_s = Dilate_loss.dilate_loss(
                ytrue_s,pred_s, alpha,beta, device)  ##在这里第一个是真实，第二个是预测
            Loss_t, loss_shape_t, loss_temporal_t = Dilate_loss.dilate_loss(
                ytrue_t,pred_t, alpha,beta, device)  ##在这里第一个是真实，第二个是预测

            total_loss = total_loss + Loss_s +Loss_t+ args.dw * loss_transfer
        # print("@", out_weight_list_before)
        loss_all.append(
            [total_loss.item(), (Loss_s+Loss_t).item(),loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model_king.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean(axis=0)
    if epoch >=args.pre_epoch:
        if epoch > args.pre_epoch:
            weight_mat_before, weight_mat_after = model_king.update_weight_Boosting(
                    weight_mat_before,dist_old_before,dist_mat_before,weight_mat_after,dist_old_after,dist_mat_after)

        return loss, loss_l1, weight_mat_before, weight_mat_after, dist_mat_before, dist_mat_after

    else:
        weight_mat_before = transform_type(out_weight_list_before)
        weight_mat_after = transform_type(out_weight_list_after)
        return loss, loss_l1, weight_mat_before, weight_mat_after, None, None


def train_epoch_transfer_Boosting(model, optimizer, train_loader_list, epoch, teacher_forcing_ratio,alpha,beta, dist_old_before = None, dist_old_after = None, weight_mat_before=None, weight_mat_after=None):
    model.train()
    criterion = nn.MSELoss()  ##代价函数
    criterion_1 = nn.L1Loss()  ##代价函数
    loss_all = []
    loss_1_all = []
    dist_mat_before = torch.zeros(args.num_layers, args.len_seq).to(device)  ##[两行，24列]
    dist_mat_after = torch.zeros(args.num_layers, args.len_seq).to(device)
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)
    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        optimizer.zero_grad()
        list_feat_left = []
        list_feat_right = []
        list_ytrue = []  ##用的是0，1编码

        for data in data_all:
            fea_left, fea_right, yture = data[0].to(device).float(
            ), data[1].to(device).float(), data[2].to(device)
            list_feat_left.append(fea_left)  ##两个tensor
            list_feat_right.append(fea_right)
            list_ytrue.append(yture)
        flag = False
        index = get_index(len(data_all)-1)

        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat_left[s1].shape[0] != list_feat_right[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1).to(device)
        for i in range(len(index)):
            feature_s_left = list_feat_left[index[i][0]]
            feature_t_left = list_feat_left[index[i][1]]
            feature_s_right = list_feat_right[index[i][0]]
            feature_t_right = list_feat_right[index[i][0]]
            ytrue_s = list_ytrue[index[i][1]]
            ytrue_t = list_ytrue[index[i][1]]
            feature_all_left = torch.cat((feature_s_left, feature_t_left), 0)
            feature_all_right = torch.cat((feature_s_right, feature_t_right), 0)
            feature_all_left = feature_all_left.permute(1, 0, 2)
            feature_all_right = feature_all_right.permute(1, 0, 2)
            ytrue_all = torch.cat((ytrue_s, ytrue_t), 0)
            ytrue_all = ytrue_all.permute(1, 0, 2)

            pred_s, pred_t, decoder_att, loss_transfer, dist_before, \
            dist_after, weight_mat_before, weight_mat_after = model.for_Boosting(
                feature_all_left, feature_all_right, ytrue_all, teacher_forcing_ratio,
                weight_mat_before, weight_mat_after)
            dist_mat_before = dist_mat_before + dist_before
            dist_mat_after = dist_mat_after + dist_after

            loss_s = criterion(pred_s, ytrue_s)
            loss_t = criterion(pred_t, ytrue_t)
            loss_l1 = criterion_1(pred_s, ytrue_s)
            Loss_s, loss_shape_s, loss_temporal_s = Dilate_loss.dilate_loss(
                pred_s, ytrue_s, alpha,beta, device)  ##在这里第一个是真实，第二个是预测
            Loss_t, loss_shape_t, loss_temporal_t = Dilate_loss.dilate_loss(
                pred_t, ytrue_t,alpha,beta, device)  ##在这里第一个是真实，第二个是预测

            total_loss = total_loss + Loss_s + Loss_t + args.dw * loss_transfer
        loss_all.append([total_loss.item(), (Loss_s + Loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean(axis=0)
    if epoch > 0:
        weight_mat_before, weight_mat_after = model.update_weight_Boosting(
                weight_mat_before, dist_old_before, dist_mat_before, weight_mat_after, dist_old_after, dist_mat_after)

    return loss, loss_l1, weight_mat_before, weight_mat_after, dist_mat_before, dist_mat_after


def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index


def train_epoch_transfer(args, model, optimizer, train_loader_list,alpha,beta,teacher_forcing_ratio):
    model.train()
    criterion = nn.MSELoss()  ##代价函数
    criterion_1 = nn.L1Loss()  ##代价函数
    loss_all = []
    loss_1_all = []
    dist_mat_before = torch.zeros(args.num_layers, args.len_seq).to(device)  ##[两行，24列]
    dist_mat_after = torch.zeros(args.num_layers, args.len_seq).to(device)
    len_loader = np.inf
    for loader in train_loader_list:
        if len(loader) < len_loader:
            len_loader = len(loader)
    for data_all in tqdm(zip(*train_loader_list), total=len_loader):
        optimizer.zero_grad()
        list_feat_left = []
        list_feat_right = []
        list_ytrue = []  ##用的是0，1编码

        for data in data_all:
            fea_left, fea_right, yture = data[0].to(device).float(
            ), data[1].to(device).float(), data[2].to(device)
            list_feat_left.append(fea_left)  ##两个tensor
            list_feat_right.append(fea_right)
            list_ytrue.append(yture)
            print(fea_left)
        flag = False
        index = get_index(len(data_all - 1))

        for temp_index in index:
            s1 = temp_index[0]
            s2 = temp_index[1]
            if list_feat_left[s1].shape[0] != list_feat_right[s2].shape[0]:
                flag = True
                break
        if flag:
            continue

        total_loss = torch.zeros(1).to(device)
        for i in range(len(index)):
            feature_s_left = list_feat_left[index[i][0]]
            feature_t_left = list_feat_left[index[i][1]]
            feature_s_right = list_feat_right[index[i][0]]
            feature_t_right = list_feat_right[index[i][0]]
            ytrue_s = list_ytrue[index[i][1]]
            ytrue_t = list_ytrue[index[i][1]]
            feature_all_left = torch.cat((feature_s_left, feature_t_left), 0)
            feature_all_right = torch.cat((feature_s_right, feature_t_right), 0)
            feature_all_left = feature_all_left.permute(1, 0, 2)
            feature_all_right = feature_all_right.permute(1, 0, 2)
            ytrue_all = torch.cat((ytrue_s, ytrue_t), 0)
            ytrue_all = ytrue_all.permute(1, 0, 2)

            pred_s, pred_t, out_weight_list_before, out_weight_list_after,decoder_att, loss_transfer = model.for_pre_train(
                feature_all_left, feature_all_right, ytrue_all, teacher_forcing_ratio)
            loss_s = criterion(pred_s, ytrue_s)
            loss_t = criterion(pred_t, ytrue_t)
            loss_l1 = criterion_1(pred_s, ytrue_s)
            Loss_s, loss_shape_s, loss_temporal_s = Dilate_loss.dilate_loss(
                pred_s, ytrue_s, alpha, beta, device)  ##在这里第一个是真实，第二个是预测
            Loss_t, loss_shape_t, loss_temporal_t = Dilate_loss.dilate_loss(
                pred_t, ytrue_t, alpha,beta, device)  ##在这里第一个是真实，第二个是预测

            total_loss = total_loss + Loss_s + Loss_t + args.dw * loss_transfer
        loss_all.append([total_loss.item(), (Loss_s + Loss_t).item(), loss_transfer.item()])
        loss_1_all.append(loss_l1.item())
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean(axis=0)
    return loss, loss_l1, out_weight_list_before, out_weight_list_after


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_epoch(model, test_loader,teacher_forcing_ratio, batch, N_output,prefix='Test'):
    model.eval()

    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    total_loss_dtw = 0
    total_loss_tdi = 0
    loss_dtw = 0
    loss_tdi = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    for fea_left,fea_right, y_truth in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        fea_left, fea_right, y_truth = fea_left.to(device).float(),fea_right.to(device).float(),y_truth.to(device).float()
        with torch.no_grad():
            pred, atten = model.predict_ts(fea_left, fea_right,y_truth,teacher_forcing_ratio)
        pred_t = pred.permute(1, 0, 2)
        ran = pred_t.shape[1]
        ##按照每一个批次求loss
        for k in range(ran):
            target = y_truth.permute(1, 0, 2)
            target_m = target[:, k, :].view(-1).detach().cpu().numpy()
            pred_y = pred.permute(1, 0, 2)
            pred_m = pred_y[:, k, :].view(-1).detach().cpu().numpy()
            # plot_result(pred_m, target_m)
            fea_left_tensor = fea_left.permute(1, 0, 2)
            fea_lt = fea_left_tensor[:, k, :]
            fea_right_tensor = fea_right.permute(1, 0, 2)
            fea_rt = fea_right_tensor[:, k, :].detach().cpu().numpy()
            att_k = atten[:, k, :].detach().cpu().numpy()
            # show_attention(fea_lt, fea_rt, pred_m, att_k)
            # plt.show()
            loss_dtw += dtw(target_m, pred_m)
            path, sim = dtw_path(target_m, pred_m)
            Dist = 0
            for i, j in path:
                Dist += (i - j) * (i - j)
            loss_tdi += Dist / (N_output * N_output)
        loss_dtw = loss_dtw / ran
        loss_tdi = loss_tdi / ran
        loss = criterion(pred, y_truth)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, y_truth)

        total_loss_tdi += loss_tdi
        total_loss_dtw += loss_dtw
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()

    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = total_loss_r / len(test_loader)
    loss_tdis = total_loss_tdi / len(test_loader)
    loss_dtws = total_loss_dtw / len(test_loader)
    return loss, loss_1, loss_r, loss_dtws, loss_tdis


def test_epoch_inference(model, test_loader, batch, N_output, teacher_forcing_ratio, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    total_loss_dtw = 0
    total_loss_tdi = 0
    loss_dtw = 0
    loss_tdi = 0
    correct = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    i = 0
    for fea_left, fea_right, y_truth in tqdm(test_loader, desc=prefix, total=len(test_loader)):
        fea_left, fea_right, y_truth = fea_left.to(device).float(), fea_right.to(device).float(), y_truth.to(
            device).float()
        with torch.no_grad():
            pred,atten = model.predict_ts(fea_left, fea_right, y_truth,teacher_forcing_ratio )
        pred_t = pred.permute(1, 0, 2)
        ran = pred_t.shape[1]
        ##按照每一个批次求loss
        for k in range(ran):
            target = y_truth.permute(1, 0, 2)
            target_m = target[:, k, :].view(-1).detach().cpu().numpy()
            pred_y = pred.permute(1, 0, 2)
            pred_m = pred_y[:, k, :].view(-1).detach().cpu().numpy()
            # plot_result(pred_m, target_m)
            fea_left_tensor = fea_left.permute(1, 0, 2)
            fea_lt = fea_left_tensor[:, k, :]
            fea_right_tensor = fea_right.permute(1, 0, 2)
            fea_rt = fea_right_tensor[:, k, :].detach().cpu().numpy()
            att_k = atten[:, k, :].detach().cpu().numpy()
            # show_attention(fea_lt, fea_rt, pred_m, att_k)
            # plt.show()
            loss_dtw += dtw(target_m, pred_m)
            path, sim = dtw_path(target_m, pred_m)
            Dist = 0
            for h, j in path:
                Dist += (h - j) * (h - j)
            loss_tdi += Dist / (N_output * N_output)
        loss_dtw = loss_dtw / ran
        loss_tdi = loss_tdi / ran
        loss = criterion(pred, y_truth)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, y_truth)
        total_loss_tdi += loss_tdi
        total_loss_dtw += loss_dtw
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()

        if y_truth.shape[0] == batch:
            if i == 0:
                y_list = y_truth.to(device).numpy()
                predict_list = pred.to(device).numpy()
            else:
                y_list = np.hstack((y_list, y_truth.cpu().numpy()))
                predict_list = np.hstack((predict_list, pred.cpu().numpy()))

        i = i + 1
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = total_loss_r / len(test_loader)
    loss_tdis = total_loss_tdi / len(test_loader)
    loss_dtws = total_loss_dtw / len(test_loader)
    return loss, loss_1, loss_r,y_list, predict_list,loss_tdis,loss_dtws


def inference(model, data_loader, batch, N_output, teacher_forcing_ratio ):
    loss, loss_1, loss_r, y_list, predict_list, loss_tdis,loss_dtws = test_epoch_inference(
        model, data_loader, batch, N_output, teacher_forcing_ratio, prefix='Inference')
    return loss, loss_1, loss_r, y_list, predict_list,loss_tdis,loss_dtws


def inference_all(output_path, model, model_path, loaders,batch, N_output, teacher_forcing_ratio):
    pprint('inference...')
    loss_list = []
    loss_l1_list = []
    loss_r_list = []
    loss_dtw_list = []
    loss_tdi_list = []
    model.load_state_dict(torch.load(model_path))
    i = 0
    list_name = ['train', 'valid', 'test']
    for loader in loaders:
        loss, loss_1, loss_r, label_list, predict_list,loss_tdis,loss_dtws = inference(
            model, loader,batch, N_output, teacher_forcing_ratio)
        loss_list.append(loss)
        loss_l1_list.append(loss_1)
        loss_r_list.append(loss_r)
        loss_dtw_list.append(loss_dtws)
        loss_tdi_list.append(loss_tdis)
        i = i + 1
    return loss_list, loss_l1_list, loss_r_list, loss_dtw_list, loss_tdi_list


def transform_type(init_weight):
    weight = torch.ones(args.num_layers, args.len_seq).to(device)##1行10列
    for i in range(args.num_layers):
        for j in range(args.len_seq):
            weight[i, j] = init_weight[i][j].item()
    return weight

def main_transfer(args):
    print(args)

    output_path = args.outdir + '_' + args.station + '_' + args.model_name + '_weather_' + \
                  args.loss_type + '_' + str(args.pre_epoch) + \
                  '_' + str(args.dw) + '_' + str(args.alpha) + '_'+ str(args.beta) + '_' + str(args.lr)
    save_model_name = args.model_name + '_' + args.loss_type + \
                      '_' + str(args.dw) + '_' + str(args.alpha) + '_'+ str(args.beta) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)
    pprint('create loaders...')

    train_loader_list, valid_loader, test_loader = data_process.load_weather_data_multi_domain(
        args.data_path, args.batch_size, args.station, args.num_domain, args.data_mode)  ##进行了数据的分割，也就是数据的载入过程

    args.log_file = os.path.join(output_path, 'run.log')
    pprint('create model_king...')
    ##模型部分

    model = get_model(args.model_name)
    num_model = count_parameters(model)
    print('#model_king params:', num_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  ##优化器的设置采用adam的优化器
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 200])
    best_score = np.inf
    best_epoch, stop_round = 0, 0

    weight_mat_before, weight_mat_after, dist_mat_before, dist_mat_after = None, None, None, None
    teacher_forcing_ratio=args.teacher_forcing_ratio
    alpha = args.alpha
    beta = args.beta
    batch = args.batch_size
    N_output = args.output_size

    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)
        pprint('training...')
        if args.model_name in ['Boosting']:
            loss, loss1, weight_mat_before, weight_mat_after, dist_mat_before, dist_mat_after = train_epoch_transfer_Boosting(
                model, optimizer, train_loader_list, epoch, teacher_forcing_ratio,alpha,beta,dist_mat_before, dist_mat_after , weight_mat_before, weight_mat_after)
        elif args.model_name in ['DualAdarnn']:
            loss, loss1, weight_mat_before, weight_mat_after, dist_mat_before, dist_mat_after = train_DualRNN(
                args, model, optimizer, train_loader_list, epoch,teacher_forcing_ratio, alpha, beta, dist_mat_before, dist_mat_after, weight_mat_before, weight_mat_after)
        else:
            print("error in model_name!")
        pprint(loss, loss1)

        pprint('evaluating...')
        train_loss, train_loss_l1, train_loss_r,train_loss_dtw, train_loss_tdi = test_epoch(
            model, train_loader_list[0],teacher_forcing_ratio,batch,N_output, prefix='Train')
        val_loss, val_loss_l1, val_loss_r, val_loss_dtw, val_loss_tdi = test_epoch(
            model, valid_loader,teacher_forcing_ratio,batch,N_output, prefix='Valid')
        test_loss, test_loss_l1, test_loss_r, test_loss_tdw, test_loss_tdi = test_epoch(
            model, test_loader, teacher_forcing_ratio, batch,N_output, prefix='Test')

        pprint('valid_l1 %.6f, test_l1 %.6f' %
               (val_loss_l1, test_loss_l1))
        pprint('valid_dtw %.6f, test_dtw %.6f' %
               (val_loss_dtw, test_loss_tdw))

        if val_loss < best_score:
            best_score = val_loss
            stop_round = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_path, save_model_name))
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best val score:', best_score, '@', best_epoch)

    loaders = train_loader_list[0], valid_loader, test_loader
    loss_list, loss_l1_list, loss_r_list,loss_dtw_list, loss_tdi_list = inference_all(output_path, model, os.path.join(
        output_path, save_model_name), loaders,batch,N_output, teacher_forcing_ratio)
    pprint('MSE: train %.6f, valid %.6f, test %.6f' %
           (loss_list[0], loss_list[1], loss_list[2]))
    pprint('L1:  train %.6f, valid %.6f, test %.6f' %
           (loss_l1_list[0], loss_l1_list[1], loss_l1_list[2]))
    pprint('RMSE: train %.6f, valid %.6f, test %.6f' %
           (loss_r_list[0], loss_r_list[1], loss_r_list[2]))
    pprint('DTW: train %.6f, valid %.6f, test %.6f' %
           (loss_dtw_list[0],loss_dtw_list[1], loss_dtw_list[2]))
    pprint('TDI: train %.6f, valid %.6f, test %.6f' %
           (loss_tdi_list[0], loss_tdi_list[1], loss_tdi_list[2]))
    pprint('Finished.')


def get_args():
    parser = argparse.ArgumentParser()

    # model_king
    ##share部分
    parser.add_argument('--model_name', default='DualAdarnn')
    parser.add_argument('--d_feat', type=int, default=17)  ##特征数
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--dec_layers', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=20)  # 30, 40, 50

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=60)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=24)  ##batch_size是分批
    parser.add_argument('--dw', type=float, default=0.0005)  # 0.05, 1.0, 5.0, 0.05
    parser.add_argument('--loss_type', type=str, default='cosine')
    parser.add_argument('--station', type=str, default='Jintang')
    parser.add_argument('--data_mode', type=str,default='tdc')
    parser.add_argument('--num_domain', type=int, default=3)
    parser.add_argument('--len_seq', type=int, default=10)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--output_size', type=int, default=6)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_path', default=r'/Volumes/王九和/科研/农业大数据相关/实验/实验程序/AdaRNN-BIT/weather_data')
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main_transfer(args)















