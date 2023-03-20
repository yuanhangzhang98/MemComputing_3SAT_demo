# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:43:57 2021

Bidirectional logic operations used for building DMMs
Each "logic variable" is a continuous variable between [-1, 1]

@author: Yuanhang Zhang
"""

import numpy as np
import torch
import torch.nn as nn


class OR(nn.Module):
    def __init__(self, input_idx, input_sign):
        super(OR, self).__init__()
        self.shape_in = input_idx.shape
        self.n_sat = self.shape_in[-1]
        self.input_idx = input_idx
        self.input_sign = input_sign
        # [alpha, beta, gamma, delta, epsilon, zeta]
        
    @torch.no_grad()
    def init_memory(self, v):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), \
                  self.input_idx]
        input = input * self.input_sign
        C = torch.max(input, dim=-1)[0]
        C = (1 - C) / 2
        
        self.xl = nn.Parameter(torch.ones(self.shape_in[:-1]))
        self.xs = nn.Parameter(C)
        self.xl.grad = torch.zeros_like(self.xl)
        self.xs.grad = torch.zeros_like(self.xs)
    
    @torch.no_grad()
    def calc_grad(self, v, param):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), \
                  self.input_idx]
        input = input * self.input_sign
        input = input.reshape(-1, self.n_sat)
        batch = input.shape[0]
        
        xl = self.xl.reshape(-1, 1)
        xs = self.xs.reshape(-1, 1)
        [alpha, beta, gamma, delta, epsilon, zeta] = param

        v_top, v_top_idx = torch.topk(input, 2, dim=-1)
        v_top = (1 - v_top) / 2
        
        C = v_top[:, 0]
        G = C.unsqueeze(-1).repeat(1, self.n_sat)
        G[torch.arange(batch), v_top_idx[:, 0]] = v_top[:, 1]
        G *= (xl*xs)
        
        R = torch.zeros(batch, self.n_sat)
        R[torch.arange(batch), v_top_idx[:, 0]] = C
        R *= ((1+zeta*xl)*(1-xs))
        
        dv = -(G+R).reshape(self.shape_in) * self.input_sign
        dxl = (-alpha * (C-delta)).reshape(self.xl.shape)
        dxs = (-beta * (xs.squeeze()+epsilon) * (C-gamma)).reshape(self.xs.shape)
        
        v.grad.data.scatter_add_(1, self.input_idx.reshape(batch0, -1), dv.reshape(batch0, -1))
        self.xl.grad.data += dxl
        self.xs.grad.data += dxs
        
        return C.reshape(batch0, -1)
    
    @torch.no_grad()
    def calc_C(self, v):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), \
                  self.input_idx]
        input = input * self.input_sign        
        C = torch.max(input, dim=-1)[0]
        C = (1 - C) / 2
        return C
    
    @torch.no_grad()
    def clamp(self, max_xl):
        self.xl.data.clamp_(1, max_xl)
        self.xs.data.clamp_(0, 1)
        