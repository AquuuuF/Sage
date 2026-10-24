import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import EstimatorCV
from torch.distributions import normal
# from torch.distributions import MultivariateNormal

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, target):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.9999):
        super().__init__()
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class GeneralizedReweightLoss(nn.Module):
    def __init__(self, cls_num_list, exp_scale=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, target):
        logit_adjusted = logit + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)



class OursLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1., Sage_gamma=0.05, Sage_mu=0.5):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)# size (num_cls, )
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

        num_cls = cls_num_list.size(0)
        cls_sum = cls_num_list.unsqueeze(0).repeat(num_cls, 1)
        cls_sum = ((1. - torch.eye(num_cls).cuda()) * cls_sum).sum(1)

        self.num_cls = cls_num_list
        self.Sage_gamma = Sage_gamma
        self.Sage_mu = Sage_mu
        self.s = torch.log(self.Sage_mu * (cls_num_list.sum()/num_cls/cls_num_list.min()) * (cls_num_list**self.Sage_gamma))


    def map_range(self, x, d, c):
        # mapping values from [a, b] to [d, c]
        a, b = x.min(), x.max()
        return d + (x - a) * (c - d) / (b - a)

    def forward(self, logit, target, features=None):

        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0) + self.s.unsqueeze(0)

        return F.cross_entropy(logit_adjusted, target)

