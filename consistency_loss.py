import torch
from torch import nn
import math
import random
import torch.nn.functional as F


class SampleSimilarities(nn.Module):
    def __init__(self, feats_dim, queueSize, T, args):
        super(SampleSimilarities, self).__init__()
        self.args = args
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q, update=True):
        N, _ = q.shape
        S = 128
        index = torch.LongTensor(random.sample(range(N), S)).to(self.args.device)
        B = torch.index_select(q, 0, index)
        # B = q
        batchSize = B.shape[0]
        # batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), B.transpose(1, 0))
        out = out.transpose(0, 1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        if update:
            # update memory bank
            with torch.no_grad():
                out_ids = torch.arange(batchSize).to(self.args.device)
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, B)
                self.index = (self.index + batchSize) % self.queueSize
        return out



class Consist_net(nn.Module):
    def __init__(self, args):
        super(Consist_net, self).__init__()
        local_feats_dim = args.dims
        global_feats_dim = args.dims
        # local_feats_dim = args.k
        # global_feats_dim = args.k

        queue_size = args.consistency_memory_size
        T = args.consistency_t
        self.l2norm = Normalize(2).to(args.device)
        self.criterion = KLD().to(args.device)
        self.local_sample_similarities = SampleSimilarities(local_feats_dim, queue_size, T, args).to(args.device)
        self.global_sample_similarities = SampleSimilarities(global_feats_dim, queue_size, T, args).to(args.device)

    def forward(self, local_feats, global_feats):
        # local_feats = self.l2norm(local_feats)
        # global_feats = self.l2norm(global_feats)

        similarities_local = self.local_sample_similarities(local_feats)
        similarities_global = self.global_sample_similarities(global_feats)

        loss = self.criterion(similarities_local, similarities_global)
        # mse_loss = F.mse_loss(similarities_local, similarities_global)
        return loss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class KLD(nn.Module):
    def forward(self, targets, inputs):
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')
