# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:48:02 2021

@author: Yuanhang Zhang
"""

import os
import time
import numpy as np
import torch
from model import SAT
from optimizer import Optimizer

torch.set_default_tensor_type(torch.cuda.FloatTensor\
                              if torch.cuda.is_available()\
                                  else torch.FloatTensor)

try:
    os.mkdir('results/')
except FileExistsError:
    pass


# We are solving 1000 3SAT instances with 500 variables each, at clause-to-variable ratio 7
batch = 1000
n = 500
r = 7

# Parameters used in the digital MemComputing machine.
param = {
    'alpha': 5,
    'beta': 20,
    'gamma': 0.25,
    'delta': 0.05,
    'epsilon': 1e-3,
    'zeta': 1e-1,
    'lr': 0.1
}

# Initialize the model
dmm = SAT(batch, n, r, param)
optim = Optimizer(dmm, param, max_step=int(1e6))

# Solve the instances and collect statistics
solved_step = optim.solve()

# Plot the histogram of the time steps
import matplotlib.pyplot as plt
params = {
    "figure.dpi": 80,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    'axes.titlesize': 24,
}
plt.rcParams.update(params)
plt.style.use("seaborn-deep")

plt.hist(solved_step.detach().cpu().numpy(), bins=50, density=True)
plt.xlabel('Time Steps')
plt.ylabel('Probability Density')
plt.savefig('results/histogram.png', bbox_inches='tight')
plt.close()
