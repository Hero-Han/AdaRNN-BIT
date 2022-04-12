import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReverseLayerF(Function):

    @staticmethod  ##声明下述为静态函数
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)


    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha #对tensor 取相反数
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self. dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


def adv(source, target, input_dim=256, hidden_dim=512):
    domain_loss = nn.BCELoss() ##求二元交叉熵
    adv_net = Discriminator(input_dim, hidden_dim).to(device)
    domain_src = torch.ones(len(source)).to(device)
    domain_tar = torch.ones(len(target)).to(device)
    domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(domain_tar.shape[0], 1)
    reverse_src = ReverseLayerF.apply(source, 1) #这一块不太懂呢
    reverse_tar = ReverseLayerF.apply(target, 1)
    pred_src = adv_net(reverse_src)
    pred_tar = adv_net(reverse_tar)
    loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(pred_tar, domain_tar)
    loss = loss_s, loss_t
    return loss







