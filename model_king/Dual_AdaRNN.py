#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Apple, Inc. All Rights Reserved 
#
# @Time    : 2022/3/25 08:54
# @Author  : SeptKing
# @Email   : WJH0923@mail.dlut.edu.cn
# @File    : Dual_AdaRNN.py
# @Software: PyCharm
import torch
import torch.nn as nn
import random
from loss.nloss_transfer import TransferLoss
import torch.nn.functional as F
import numpy as np
from untils.support import numpy_to_tvar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dual_AdaRNN(nn.Module):
    def __init__(self, share_encoder, decoder,output_dim,len_seq, device):
        super(Dual_AdaRNN, self).__init__()
        self.share_encoder = share_encoder
        self.decoder = decoder
        self.device = device
        self.output_dim = output_dim
        self.len_seq = len_seq
        self.device = device

    def for_custom_pre(self, src_left, src_right, trg, teacher_forcing_ratio, len_win=0):

        batch_size = src_left.shape[1]  ##应该就是正常的batch_size
        max_len = trg.shape[0]  ##需要预测的长度6
        outputs = torch.zeros(max_len, batch_size,
                              self.output_dim).to(device)  ##(6,2,1)

        ##注意力状态
        decoder_att = torch.zeros(max_len, batch_size,
                                  self.len_seq*2).to(device)  ##可以是(6,2,10)
        ##双向输入且具有分布学习的编码层
        encoder_outputs_left, encoder_outputs_right, out_weight_list_before, out_weight_list_after, hidden, hidden_decoder, loss_transfer = self.share_encoder.forward_pre_train(
            src_left, src_right, len_win)  ##（12，2，128）,hidden(2,2,64)##10，2，16，10，2，16

        ##选取最后一个时间节点的y
        output = src_left[-1, :, 2]  ##选取的是最后一个时间节点的第一个元素


        for t in range(0, max_len):
            output, hidden, attn_weight = self.decoder(output, hidden,hidden_decoder,
                                                       encoder_outputs_left,
                                                       encoder_outputs_right)  ##返回的output是[ys,yt],hidden是(1,2,64),a是(2,64)
            decoder_att[t] = attn_weight.squeeze()
            outputs[t] = output.unsqueeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t].view(-1) if teacher_force else output)  ##下一步的输入是output还是目标output

        output_all = outputs.permute(1, 0, 2)  ##2,6,1
        output_s = output_all[0:output_all.size(0) // 2]  ##1,6,1
        output_t = output_all[output_all.size(0) // 2:]
        #output_s = output_s.permute(1, 0, 2)
        #output_t = output_t.permute(1, 0, 2)

        return output_s, output_t, out_weight_list_before, out_weight_list_after, decoder_att, loss_transfer

    def for_Boosting(self, src_left, src_right, trg, teacher_forcing_ratio=0.5, weight_mat_before=None, weight_mat_after=None):
        batch_size = src_left.shape[1]  ##应该就是正常的batch_size
        max_len = trg.shape[0]  ##需要预测的长度6
        outputs = torch.zeros(max_len, batch_size,
                              self.output_dim).to(device)  ##(6,2,1)

        ##注意力状态
        decoder_att = torch.zeros(max_len, batch_size,
                                  self.len_seq*2).to(device)  ##可以是(6,2,12)
        ##双向输入且具有分布学习的编码层
        encoder_outputs_left, encoder_outputs_right, weight_before, weight_after, hidden, hidden_decoder, loss_transfer, dist_mat_before, dist_mat_after = self.share_encoder.forward_boosting(
            src_left, src_right, weight_mat_before, weight_mat_after)  ##（12，2，128）,hidden(2,2,64)
        ##选取最后一个时间节点的y
        output = src_left[-1, :, 2]  ##选取的是最后一个时间节点的第一个元素

        for t in range(0, max_len):
            output, hidden, attn_weight = self.decoder(output, hidden,hidden_decoder,
                                                       encoder_outputs_left,
                                                       encoder_outputs_right)  ##返回的output是[ys,yt],hidden是(1,2,64),a是(2,64)
            decoder_att[t] = attn_weight.squeeze()
            outputs[t] = output.unsqueeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t].view(-1) if teacher_force else output)  ##下一步的输入是output还是目标output

        output_all = outputs.permute(1, 0, 2)  ##2,6,1
        output_s = output_all[0:output_all.size(0) // 2]  ##1,6,1
        output_t = output_all[output_all.size(0) // 2:]
        #output_s = output_s.permute(1, 0, 2)
        #output_t = output_t.permute(1, 0, 2)

        return output_s, output_t, decoder_att, loss_transfer, dist_mat_before, dist_mat_after, weight_before, weight_after
    def update_weight_Boosting(self, weight_mat_before, dist_old_before, dist_new_before, weight_mat_after,
                               dist_old_after, dist_new_after):
        epsilon = 1e-5
        ##左端
        dist_old_before = dist_old_before.detach()
        dist_new_before = dist_new_before.detach()
        ind = dist_new_before > dist_old_before + epsilon
        weight_mat_before[ind] = weight_mat_before[ind] * \
                                 (1 + torch.sigmoid(dist_new_before[ind] - dist_old_before[ind]))

        weight_norm_b = torch.norm(weight_mat_before, dim=1, p=1)
        weight_mat_before = weight_mat_before / weight_norm_b.t().unsqueeze(1).repeat(1, self.len_seq)
        ##右端
        dist_old_after = dist_old_after.detach()
        dist_new_after = dist_new_after.detach()
        ind = dist_new_after > dist_old_after + epsilon
        weight_mat_after[ind] = weight_mat_after[ind] * \
                                (1 + torch.sigmoid(dist_new_after[ind] - dist_old_after[ind]))

        weight_norm_a = torch.norm(weight_mat_after, dim=1, p=1)
        weight_mat_after = weight_mat_after / weight_norm_a.t().unsqueeze(1).repeat(1, self.len_seq)

        return weight_mat_before, weight_mat_after

    def predict_ts(self, src_left, src_right, trg, teacher_forcing_ratio=0.5):
        tst_left = src_left.permute(1,0,2)
        tst_right = src_right.permute(1,0,2)
        tst_trg = trg.permute(1,0,2)
        batch_size = tst_left.shape[1]  ##应该就是正常的batch_size
        max_len = tst_trg.shape[0]  ##需要预测的长度6
        outputs = torch.zeros(max_len, batch_size,
                              self.output_dim).to(self.device)  ##(6,2,1)
        decoder_att = torch.zeros(max_len, batch_size,
                                  self.len_seq*2).to(self.device)  ##可以是(6,2,12)
        encoder_outputs_left, encoder_outputs_right, hidden,hidden_decoder = self.share_encoder.predict(
            tst_left, tst_right)
        output = tst_left[-1, :, 2]  ##选取的是最后一个时间节点的第一个元素
        for t in range(0, max_len):
            output, hidden, attn_weight = self.decoder(output, hidden,hidden_decoder,
                                                       encoder_outputs_left,
                                                       encoder_outputs_right)  ##返回的output是[ys,yt],hidden是(1,2,64),a是(2,64)
            decoder_att[t] = attn_weight.squeeze()
            outputs[t] = output.unsqueeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            output = (tst_trg[t].view(-1) if teacher_force else output)  ##下一步的输入是output还是目标output

        output_all = outputs.permute(1, 0, 2)  ##1,6,1
        #output_all = output_all.permute(1, 0, 2)
        #output_all = output_all.squeeze()
        return output_all, decoder_att


class Share_Encoder(nn.Module):
    def __init__(self, n_input=128, n_hiddens=[64, 64], dec_layers=1,dropout=0.1, len_seq=10, model_type='Dual_AdaRNN',
                 trans_loss='mmd'):
        super(Share_Encoder, self).__init__()
        self.n_input = n_input
        self.num_layers = len(n_hiddens)  ##2
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq  ##序列长度
        self.dropout = dropout
        self.hiddens = n_hiddens
        self.enc_hid_dim = n_hiddens[0]
        self.dec_hid_dim = n_hiddens[0]
        self.dec_layers = dec_layers
        in_size_l = self.n_input  ##输入要的维度
        in_size_r = self.n_input

        features_left = nn.ModuleList()
        for hidden in n_hiddens:
            rnn_l = nn.GRU(input_size=in_size_l,
                         num_layers=1,
                         hidden_size=hidden,
                         bidirectional=True,
                         )
            features_left.append(rnn_l)
            in_size_l = hidden*2
        self.features_left = nn.Sequential(*features_left)

        features_right = nn.ModuleList()
        for hidden in n_hiddens:
            rnn_r = nn.GRU(input_size=in_size_r,
                         num_layers=1,
                         hidden_size=hidden,
                         bidirectional=True,
                         )
            features_right.append(rnn_r)
            in_size_r = hidden * 2
        self.features_right = nn.Sequential(*features_right)

        self.output_linear_left = nn.Linear(self.enc_hid_dim * 2,
                                            self.dec_hid_dim)  ##128,64
        self.output_linear_right = nn.Linear(self.enc_hid_dim * 2,
                                             self.dec_hid_dim)  ##128,64
        self.dropout = nn.Dropout(self.dropout)

        if self.model_type == 'DualAdaRNN':
            gate = nn.ModuleList()
            for i in range(len(n_hiddens)):
                gate_weight = nn.Linear(
                    len_seq * self.hiddens[i] * 4, len_seq)
                gate.append(gate_weight)
            self.gate = gate

            bnlst = nn.ModuleList()
            for i in range(len(n_hiddens)):
                bnlst.append(nn.BatchNorm1d(len_seq))

            self.bn_lst = bnlst
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05)
            self.gate[i].bias.data.fill_(0.0)

    ##假设batch = 1的话，input_before表示的是[24,2,16]这里12表示的是一个batch的seq

    def forward_pre_train(self, input_before, input_after, len_win=0):

        ##左侧输入以及迁移权重计算
        outputs_before, hidden_before, out_lis_before, out_weight_list_before = self.gru_features_left(input_before)

        ##通过cat将hidden中前向和后向的部分拼接在一起(2,128)2表示的是batch,然后通过线性变换形成(2,64),最后通过tanh激活函数进行调整为(-1,1)之间
        hidden_before = torch.tanh(
            self.output_linear_left(
                torch.cat((hidden_before[-2, :, :], hidden_before[-1, :, :]),
                          dim=1)))

        outputs_after, hidden_after, out_lis_after, out_weight_list_after = self.gru_features_right(input_after)
        hidden_after = torch.tanh(
            self.output_linear_right(
                torch.cat((hidden_after[-2, :, :], hidden_after[-1, :, :]),
                          dim=1)))

        ##我们只使用前向输入的hidden来初始化gru
        hidden_decoder_l = hidden_before.repeat(self.dec_layers, 1, 1)  ##(1,2,64)
        hidden_decoder_r = hidden_after.repeat(self.dec_layers, 1, 1)
        hidden_decoder = torch.cat((hidden_decoder_l,hidden_decoder_r),dim=0)

        ##计算迁移损失
        out_lis_before_s, out_lis_before_t = self.get_features(out_lis_before)
        loss_transfer_before = torch.zeros((1,)).to(device)
        for i in range(len(out_lis_before_s)):  ##长度为2
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_lis_before_s[i].shape[2])  ##输入维度为64
            h_start = 0
            for j in range(h_start, self.len_seq, 1):  ##从1到12每次增加1
                i_start = j - len_win if j - len_win >= 0 else 0  ##0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1  ##0
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list_before[i][j] if self.model_type == 'DualAdaRNN' else 1 / (
                            self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer_before = loss_transfer_before + weight * criterion_transder.compute(
                        out_lis_before_s[i][:, j, :], out_lis_before_t[i][:, k, :])
        ##计算右端输入transfter
        out_lis_after_s, out_lis_after_t = self.get_features(out_lis_after)
        loss_transfer_after = torch.zeros((1,)).to(device)
        for i in range(len(out_lis_after_s)):  ##长度为2
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_lis_after_s[i].shape[2])  ##输入维度为64
            h_start = 0
            for j in range(h_start, self.len_seq, 1):  ##从1到24每次增加1
                i_start = j - len_win if j - len_win >= 0 else 0  ##0
                i_end = j + len_win if j + len_win < self.len_seq else self.len_seq - 1  ##0
                for k in range(i_start, i_end + 1):
                    weight = out_weight_list_after[i][j] if self.model_type == 'DualAdaRNN' else 1 / (
                            self.len_seq - h_start) * (2 * len_win + 1)
                    loss_transfer_after = loss_transfer_after + weight * criterion_transder.compute(
                        out_lis_before_s[i][:, j, :], out_lis_before_t[i][:, k, :])
        loss_transfer = loss_transfer_before + loss_transfer_after

        return outputs_before, outputs_after, out_weight_list_before, out_weight_list_after, hidden_decoder_l,hidden_decoder, loss_transfer

    def predict(self, input_before, input_after):
        outputs_before, hidden_before, out_lis_before, out_weight_list_before = self.gru_features_left(input_before,predict=True)
        hidden_before = torch.tanh(
            self.output_linear_left(
                torch.cat((hidden_before[-2, :, :], hidden_before[-1, :, :]),
                          dim=1)))
        outputs_after, hidden_after, out_lis_after, out_weight_list_after = self.gru_features_right(input_after,predict=True)

        hidden_decoder_l = hidden_before.repeat(self.dec_layers, 1, 1)  ##(1,2,64)
        hidden_decoder_r = hidden_after.repeat(self.dec_layers, 1, 1)
        hidden_decoder = torch.cat((hidden_decoder_l, hidden_decoder_r), dim=0)

        return outputs_before, outputs_after, hidden_decoder_l, hidden_decoder

    def forward_boosting(self, input_before, input_after, weight_mat_before=None, weight_mat_after=None):
        ##左侧输入以及迁移权重计算
        ##(10,4,128),(2,4,64),[(4,10,128),(4,10,128)],[(10),(10)]
        outputs_before, hidden_before, out_lis_before, out_weight_list_before = self.gru_features_left(input_before)

        ##通过cat将hidden中前向和后向的部分拼接在一起(2,128)2表示的是batch,然后通过线性变换形成(2,64),最后通过tanh激活函数进行调整为(-1,1)之间
        hidden_before = torch.tanh(
            self.output_linear_left(
                torch.cat((hidden_before[-2, :, :], hidden_before[-1, :, :]),
                          dim=1)))##(4,64)

        outputs_after, hidden_after, out_lis_after, out_weight_list_after = self.gru_features_right(input_after)
        hidden_after = torch.tanh(
            self.output_linear_right(
                torch.cat((hidden_after[-2, :, :], hidden_after[-1, :, :]),
                          dim=1)))

        ##我们只使用前向输入的hidden来初始化gru
        hidden_decoder_l = hidden_before.repeat(self.dec_layers, 1, 1)  ##(1,2,64)
        hidden_decoder_r = hidden_after.repeat(self.dec_layers, 1, 1)
        hidden_decoder = torch.cat((hidden_decoder_l, hidden_decoder_r), dim=0)

        ##计算迁移损失、以及更新权重
        ##左侧
        ##[(2,10,128),(2,10,128)],weight_before[2,10]不是分之一而是用的weight权重
        out_lis_before_s, out_lis_before_t = self.get_features(out_lis_before)
        loss_transfer_before = torch.zeros((1,)).to(device)
        if weight_mat_before is None:
            weight_before = (1.0 / self.len_seq *
                             torch.ones(self.num_layers, self.len_seq)).to(device)
        else:
            weight_before = weight_mat_before

        dist_mat_before = torch.zeros(self.num_layers, self.len_seq).to(device)
        for i in range(len(out_lis_before_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_lis_before_s[i].shape[2])
            for j in range(self.len_seq):
                loss_trans_before = criterion_transder.compute(
                    out_lis_before_s[i][:, j, :], out_lis_before_t[i][:, j, :])
                loss_transfer_before = loss_transfer_before + weight_before[i, j] * loss_trans_before
                dist_mat_before[i, j] = loss_trans_before

        ##计算右侧
        out_lis_after_s, out_lis_after_t = self.get_features(out_lis_after)
        loss_transfer_after = torch.zeros((1,)).to(device)
        if weight_mat_after is None:
            weight_after = (1.0 / self.len_seq *
                            torch.ones(self.num_layers, self.len_seq)).to(device)
        else:
            weight_after = weight_mat_after

        dist_mat_after = torch.zeros(self.num_layers, self.len_seq).to(device)
        for i in range(len(out_lis_after_s)):
            criterion_transder = TransferLoss(
                loss_type=self.trans_loss, input_dim=out_lis_after_s[i].shape[2])
            for j in range(self.len_seq):
                loss_trans_after = criterion_transder.compute(
                    out_lis_after_s[i][:, j, :], out_lis_after_t[i][:, j, :])
                loss_transfer_after = loss_transfer_after + weight_after[i, j] * loss_trans_after
                dist_mat_after[i, j] = loss_trans_after
        loss_transfer = loss_transfer_before + loss_transfer_after
        return outputs_before, outputs_after, weight_before, weight_after, hidden_decoder_l,hidden_decoder, loss_transfer, dist_mat_before, dist_mat_after

    def gru_features_left(self, x, predict=False):
        x_input = x
        out = None
        hidden = None
        out_lis = []

        out_weight_list = [] if (self.model_type == 'DualAdaRNN') else None

        for i in range(self.num_layers):
            ##这里面输出是(10,2,128),hidden的构成是(2,2,64)
            out, hidden = self.features_left[i](x_input.float())
            x_input = out
            out_l = out.permute(1, 0, 2)  ##将输出恢复到(batch_size,leq,dim)上
            out_lis.append(out_l)

            if self.model_type == 'DualAdaRNN' and predict == False:
                ##将输出调整回(batch,seq,dim)为计算transformloss作准备(2,12,128)
                x_input_m = x_input.permute(1, 0, 2)
                out_gate = self.process_gate_weight(x_input_m, i)
                out_weight_list.append(out_gate)


        return out, hidden, out_lis, out_weight_list

    def gru_features_right(self, x, predict=False):
        x_input = x
        out = None
        hidden = None
        out_lis = []
        out_weight_list = [] if (self.model_type == 'DualAdaRNN') else None
        for i in range(self.num_layers):
            out, hidden = self.features_right[i](x_input.float())
            x_input = out
            out_l = out.permute(1, 0, 2)  ##将输出恢复到(batch_size,leq,dim)上
            out_lis.append(out_l)

            if self.model_type == 'DualAdaRNN' and predict == False:
                x_input_m = x_input.permute(1, 0, 2)
                out_gate = self.process_gate_weight(x_input_m, i)
                out_weight_list.append(out_gate)

        return out, hidden, out_lis, out_weight_list

    ##计算迁移权重

    def process_gate_weight(self, out, index):
        x_s = out[0: int(out.shape[0] // 2)]
        x_t = out[out.shape[0] // 2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)  ##这部分的维度是1，10，128*2
        x_all = x_all.view(x_all.shape[0], -1)  ##维度已经调整（1，10*128*2）
        weight = self.gate[index](x_all.float())
        weight = self.bn_lst[index](weight)
        weight = torch.sigmoid(weight)  ##将双向gru输出的结果通过线性层转化为(1,10),然后通过bn_lst进行归一化然后通过sigmoid进行进一步处理
        weight = torch.mean(weight, dim=0)  # 对权重在batch_size层面上求均值
        res = self.softmax(weight).squeeze()  ##通过softmax求的权重和为1得到12个小时的不同维度
        return res

    def get_features(self, output_list):
        fea_lis_src, fea_list_tar = [], []
        for fea in output_list:
            fea_lis_src.append(fea[0:fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])

        return fea_lis_src, fea_list_tar


class Cross_Attention(nn.Module):
    def __init__(self, n_hiddens):
        super(Cross_Attention, self).__init__()

        self.enc_hid_dim = n_hiddens[0]  # 50
        self.dec_hid_dim = n_hiddens[0]  # 50

        self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim,
                              self.dec_hid_dim)  ##(64*3,64)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))  ##随机创建64个变量

    ##hidden的dim为(1,2,64),输入是(24,2,128)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]  ##2
        src_len = encoder_outputs.shape[0]  ##20

        # only pick up last layer hidden from decoder
        hidden = torch.unbind(hidden, dim=0)[0]  ##只取最近层的hidden信息,是按照维度做切片,(2,64)

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  ##2,24,64

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  ##2,24,128

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs),
                                dim=2)))  ##主要操作是将hidden的一维向量重复24次作为上一阶段的状态s，然后与h粘贴生成（(2,24,64*3)通过全联接生成(2,24,64)

        energy = energy.permute(0, 2, 1)  ##选取20行的各个特征维度进行重构形成(2,64,24)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)  ##生成（b,n,d）(2,24)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, n_hiddens, dec_layers, dropout, attention):  ##1,60,60,1
        super(Decoder, self).__init__()

        self.enc_hid_dim = n_hiddens[0]
        self.dec_hid_dim = n_hiddens[0]
        self.output_dim = output_dim
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.attention = attention
        self.input_dec = nn.Linear(self.output_dim, self.dec_hid_dim)
        self.gru = nn.GRU(input_size=self.enc_hid_dim * 2 + self.dec_hid_dim,
                          hidden_size=self.dec_hid_dim,
                          num_layers=self.dec_layers)
        self.out = nn.Linear(
            self.enc_hid_dim * 2 + self.dec_hid_dim + self.dec_hid_dim,
            self.output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def get_features(self, output_list):
        fea_list_src, fea_list_tar = [], []
        for fea in output_list:
            fea_list_src.append(fea[0: fea.size(0) // 2])
            fea_list_tar.append(fea[fea.size(0) // 2:])

    ##输入一组y

    def forward(self, input, hidden,hidden_decoder,encoder_outputs_left, encoder_outputs_right):
        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)  ##(1,2,1)按照当前的batch有两个y被提取出来
        embedded = self.dropout(torch.tanh(self.input_dec(input)))  ##(1,2,64)
        ##将前向的和后向的在0维进行拼接
        encoder_outputs = torch.cat(
            (encoder_outputs_left, encoder_outputs_right), dim=0)  ##(20,1,128)
        a = self.attention(hidden_decoder, encoder_outputs)  ##生成权重a(2，24)输入的hidden是(1,2,64)
        a = a.unsqueeze(1)  ##(2,1,20)，（4，1，20）
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  ##(2,20,128)，还原一个batch20h

        weighted = torch.bmm(a, encoder_outputs)  ##(2,1,128)这个就是ct
        weighted = weighted.permute(1, 0, 2)  ##(1,2,128)
        gru_input = torch.cat((embedded, weighted), dim=2)  ##(1,2,64*3)将y和ct链接
        output, hidden = self.gru(gru_input, hidden)  ##OK这里面就是将yt-1，ct，st-1投入的gru里

        input_dec = embedded.squeeze(0)  ##(2,64)
        output = output.squeeze(0)  ##(2,64)
        weighted_f = weighted.squeeze(0)  ##(2,128)


        output = self.out(torch.cat((output, weighted_f, input_dec), dim=1))  ##(2,1)

        ##返回的output是[ys,yt],hidden是(1,2,64),a是(2,20)
        return output.squeeze(1), hidden, a.squeeze(1)
