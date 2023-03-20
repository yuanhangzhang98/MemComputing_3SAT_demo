# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:52:05 2021

@author: Yuanhang Zhang
"""


import numpy as np
import torch
import torch.nn as nn
from operators import OR
from dataset import SAT_dataset

class SAT(nn.Module):
    def __init__(self, batch, n, r, param):
        super(SAT, self).__init__()
        self.batch = batch
        self.n = n
        self.r = r
        self.param = param.copy()
        self.max_xl = 1e4 * self.n
        self.lr = self.param['lr']
        
        self.dataset = SAT_dataset(batch, n, r)
        clause_idx, clause_sign = self.dataset.generate_instances()
        # clause_idx, clause_sign = self.dataset.import_data()
        self.OR = OR(clause_idx, clause_sign)
            
        self.v = nn.Parameter(2 * torch.rand(batch, self.n) - 1)
        self.v.grad = torch.zeros_like(self.v)
        
        self.OR.init_memory(self.v)
        
    @torch.no_grad()
    def backward(self, param):
        param = [param['alpha'], param['beta'], param['gamma'], \
                     param['delta'], param['epsilon'], param['zeta']]
        self.C = self.OR.calc_grad(self.v, param)
        
        is_solved = (self.C<0.5).all(dim=1)

        # max_dv = torch.max(torch.abs(self.v.grad) + 1e-6, dim=1)[0]
        # for param in self.parameters():
        #     param.grad.data /= max_dv.view((-1, ) + (1, )*(len(param.shape)-1))
        return is_solved

    @torch.no_grad()
    def add_noise(self, noise_strength):
        self.OR.xl.data += noise_strength * torch.randn_like(self.OR.xl)
        self.OR.xs.data += noise_strength * torch.randn_like(self.OR.xs)
    
    def clamp(self, max_xl):
        self.v.data.clamp_(-1, 1)
    
    @torch.no_grad()
    def clip_weights(self):
        self.clamp(self.max_xl)
        self.OR.clamp(self.max_xl)
