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

class TrainingFunc():
    def __init__(self):
        self.amp = np.random.uniform(0.2, 1.8)
        self.sqrt_amp = np.sqrt(self.amp)
        self.offsets = np.random.uniform(-2, 2, 2)
        
        
    def pos(self, t): return np.sin(self.sqrt_amp * t + self.offsets[1]) + self.offsets[0]
    def vel(self, t): return self.sqrt_amp * np.cos(self.sqrt_amp * t + self.offsets[1])
    def acc(self, t): return -self.amp * np.sin(self.sqrt_amp * t + self.offsets[1])

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x
    
class Oscillator:
    def __init__(self, a, h):
        self.a = a
        self.sqrta = np.sqrt(a)
        self.ini = np.array([0, self.sqrta, 0])

    def acc(self, pos):
        return self.a * np.sin(pos)

    def vel(self, t):
        return self.sqrta * np.cos(self.sqrta * t)

    def pos(self, t):
        return np.sin(self.sqrta * t)
    
#%%

num_training_funcs = 100
epochs = 100
h = 0.01

X_train = []
Y_train = []

for i in range(num_training_funcs):
    training_func = TrainingFunc()
    X_train.append([training_func.pos(0), training_func.vel(0), training_func.acc(0)])

    for j in range(epochs):
        Y_train.append([training_func.pos((j + 1) * h), training_func.vel((j + 1) * h)])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
    
#%%

net = RegressionNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()

for i in range(num_training_funcs):
    x = torch.tensor(X_train[i], dtype=torch.float32)
    for j in range(epochs):
        pos_pred = net(x[:-1])
        vel_pred = net(x[1:])
        loss = criterion(pos_pred, Y_train[i*j][0]) 
        #+ criterion(vel_pred, Y_train[i*j][1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = torch.cat((pos_pred, vel_pred), 0)

#%%

a = 1

oscillator = Oscillator(a, h)

exact_sol = [oscillator.ini[0]]
model_sol = [oscillator.ini[0]]

exact_sol.append(oscillator.pos(h + oscillator.ini[1]))

model_pos = net(torch.tensor(oscillator.ini[:-1], dtype=torch.float32))
model_vel = net(torch.tensor(oscillator.ini[1:], dtype=torch.float32))

x = torch.cat((model_pos, model_vel, torch.tensor(oscillator.acc(model_pos), dtype=torch.float32)), 0)

model_sol.append(x[0])

for i in range(2, 400):
    pos_pred = net(x[:-1])
    vel_pred = net(x[1:])
    x = torch.cat((pos_pred, vel_pred, torch.tensor(oscillator.acc(pos_pred), dtype=torch.float32)), 0)

    exact_sol.append(oscillator.pos(h * i + oscillator.ini[1]))
    model_sol.append(x[0])

#%%

plt.plot(exact_sol, label='exact')
plt.plot(model_sol, label='model')
plt.legend()
plt.show()
