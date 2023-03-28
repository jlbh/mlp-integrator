#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:49:51 2023

@author: johannes
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#%%

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.hidden1 = nn.Linear(4, 4)
        self.hidden2 = nn.Linear(4, 3)
        self.output = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x
    
#%%

net = RegressionNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()

x = torch.randn(1, 4)
y_true = torch.randn(1, 2).expand(100, -1)

for i in range(100):
    y_pred = net(x)
    loss = criterion(y_pred, y_true[i:i+1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    x = y_pred


