import torch
import math
from opt import args
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Compact_learning(object):
    def __init__(self, k=32, stage_num=9, beta=3, lamd=1, momentum=0.9):
        self.k = k
        self.lamd = lamd
        self.stage_num = stage_num
        self.beta = beta
        self.momentum = momentum
        self.mu = torch.Tensor(1, self.k)
        self.mu.normal_(0, math.sqrt(2. / self.k))
        self.mu = self.mu / (1e-6 + self.mu.norm(dim=0, keepdim=True))

    def __call__(self, embds):
        b, n = embds.size()
        mu = self.mu.repeat(b, 1).cuda(embds.device) # mu=lambda z=y 
        # mu = mu / (mu.norm(dim=0, keepdim=True))
        # _embds = embds
        embd_nor = embds / (embds.norm(dim=0, keepdim=True))
        _embds = embds / (embds.norm(dim=0, keepdim=True))

        with torch.no_grad():
            for i in range(self.stage_num):
                _embds_t = _embds.permute(1, 0)  # n * b
                z = torch.mm(_embds_t, mu)  # n * k
                z = z / self.lamd
                z = F.softmax(z, dim=1)
                z = z / (1e-6 + z.sum(dim=0, keepdim=True))
                mu = torch.mm(_embds, z)  # b * k
                mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))
        z_t = z.permute(1, 0)  # k * n
        _embds = torch.mm(mu, z_t)  # b * n 

        # for i in range(self.stage_num):
        #     _embds_t = _embds.permute(1, 0)  # n * b
        #     z = torch.mm(_embds_t, mu)  # n * k
        #     z = z / self.lamd
        #     z = F.softmax(z, dim=1)
        #     z = z / (1e-6 + z.sum(dim=0, keepdim=True))
        #     mu = torch.mm(_embds, z)  # b * k
        #     mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))
        # z_t = z.permute(1, 0)  # k * n
        # _embds = torch.mm(mu, z_t)  # b * n 

        # for i in range(self.stage_num):
        #     _embds_t = _embds.permute(1, 0)  # n * b
        #     z = torch.mm(_embds_t, mu)  # n * k
        #     z = z / self.lamd
        #     z = F.softmax(z, dim=1)
        #     z = z / (1e-6 + z.sum(dim=0, keepdim=True))
        #     mu = torch.mm(_embds, z)  # b * k
        #     mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))
        # _embds = mu

        # if if_train:
        #     mu = mu.cpu()
        #     self.mu = self.momentum * self.mu + (1 - self.momentum) * mu.mean(dim=0, keepdim=True)

        # return self.beta * _embds + embds
        return self.beta * _embds + embd_nor
    


class compact_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, k, stage_num, beta, cluster_num):
        super(compact_net, self).__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(input_dim, hidden_dim)
        self.compact = Compact_learning(k, stage_num, beta)

        if act == "ident":
            self.activate = lambda x: x
        elif act == "sigmoid":
            self.activate = nn.Sigmoid()
        elif act == "relu":
            self.activate = nn.ReLU()
        elif act == "leakyrelu":
            self.activate = nn.LeakyReLU(0.2, inplace=True)

        self.reset()
        
    def reset(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()

    def forward(self, X_l, X_g, A):
        Z1 = self.activate(self.mlp1(X_l))
        Z2 = self.activate(self.mlp2(X_g))
        all_embedding = torch.cat((Z1, Z2), dim=0)
        all_embedding = self.compact(all_embedding)

        Z1, Z2 = torch.split(all_embedding, all_embedding.size(0) // 2)
        Z1, Z2 = F.normalize(Z1, dim=1, p=2), F.normalize(Z2, dim=1, p=2)
        return Z1, Z2