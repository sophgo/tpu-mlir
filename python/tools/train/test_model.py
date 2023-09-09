import os
import torch
import pdb
import gc
import time
import argparse
import copy
import numpy as np
import logging
import torch.nn as nn

# class test_model(nn.Module):
#     def __init__(self):
#         super(test_model, self).__init__()
#         self.name = 'test_model'
#         self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
#         self.pool1 = nn.MaxPool2d(3, 2, 1)
#         self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
#         self.linear = nn.Linear(401408,10, bias=True)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool1(x)
#         identity = x
#         x = self.conv2(x)
#         identity = self.conv3(identity)
#         x += identity
#         #x = torch.cat((x, identity))
#         x = x.view(-1)
#         x = self.linear(x)
#         return x

class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        self.name = 'test_model'
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        # self.norm = nn.BatchNorm2d(64, eps=0, momentum=0.5, affine=True, track_running_stats=True)
        # self.norm = nn.LayerNorm([64,8,8], elementwise_affine=True)
        # self.pool1 = nn.MaxPool2d(3, 2, 1)
        # self.relu = nn.ReLU()
        self.linear = nn.Linear(1024,10, bias=False)

    # def forward(self, x, y):
    #     x = torch.bmm(x, y)
    def forward(self, x):
        x = self.conv1(x)
        # x = 2*x
        # x = self.norm(x)
        # x = self.relu(x)
        # x = self.pool1(x)
        # x = x.view(-1)
        # x = self.linear(x)
        # x = x[0,0,0,0] #aten.select Op not support
        # x = x.view(-1)[0] #aten.select Op not support
        x = x.sum()
        return x
