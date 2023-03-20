# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:16:09 2021

@author: Yuanhang Zhang
"""

import numpy as np
import torch

class Optimizer():
    def __init__(self, dmm, param, max_step=int(1e6)):
        self.dmm = dmm
        self.batch = dmm.batch
        self.n = dmm.n
        self.param = param.copy()
        
        # param_fixed = {
        #     'epsilon': 1e-3,
        #     'lr': 1
        #     }
        # self.param.update(param_fixed)
        
        self.lr = self.param['lr']
        self.optimizer = torch.optim.SGD(self.dmm.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.dmm.parameters(), lr=self.lr)
        self.max_step = max_step
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, \
        #                     mode='min', factor=0.5, patience=500, verbose=True)
        
    @torch.no_grad()
    def solve(self):
        is_solved = torch.zeros(self.batch, dtype=torch.bool)
        is_solved_last = is_solved.clone()
        solved_step = self.max_step * torch.ones(self.batch)
        
        for step in range(self.max_step):
            self.optimizer.zero_grad()
            is_solved_i = self.dmm.backward(self.param)
            self.optimizer.step()
            self.dmm.add_noise(np.sqrt(0.12))
            self.dmm.clip_weights()

            is_solved = is_solved | is_solved_i
            solved_step[is_solved^is_solved_last] = step
            is_solved_last = is_solved.clone()
            n_solved = torch.sum(is_solved).detach().cpu().numpy()
            print('n:', self.n, 'step:', step, '  unsolved:', self.batch - n_solved)
            # self.scheduler.step((self.batch - n_solved) / self.batch)
            # if n_solved == self.batch:
            if n_solved == self.batch:
                break

        return solved_step
