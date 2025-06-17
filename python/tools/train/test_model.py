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


class test_model0(nn.Module):

    def __init__(self, for_train=False):
        super(test_model0, self).__init__()
        self.name = 'test_model0'
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.for_train = for_train

    # def forward(self, x): #残差分支
    #     x = self.relu(x)
    #     x = self.conv1(x)
    #     # x = self.conv2(x)
    #     if self.for_train:
    #         x = x.sum()
    #     return x

    def forward(self, x):  #残差分支
        x = self.conv1(x)
        x = self.maxpool(x)
        identity = x
        x = self.conv2(x)
        x = self.conv3(x)
        x += identity
        if self.for_train:
            x = x.sum()
        return x


class test_model1(nn.Module):

    def __init__(self, for_train=False):
        super(test_model1, self).__init__()
        self.name = 'test_model1'
        self.conv1 = nn.Conv2d(3, 128, 7, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.for_train = for_train

    def forward(self, x):  #单分支
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.for_train:
            x = x.sum()
        return x


class test_model2(nn.Module):

    def __init__(self, for_train=False):
        super(test_model2, self).__init__()
        self.name = 'test_model2'
        self.conv1 = nn.Conv2d(3, 128, 7, 2, 3, bias=False)
        # self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm = nn.BatchNorm2d(128, eps=0, momentum=0.5, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)
        self.relu = nn.ReLU()
        self.for_train = for_train

    def forward(self, x):  #残差分支
        x = self.conv1(x)
        x = self.norm(x)
        identity = x
        x = self.conv2(x)
        x += identity
        # if self.for_train:
        #     x = x.sum()
        return x


class test_model3(nn.Module):

    def __init__(self, for_train=False):
        super(test_model3, self).__init__()
        self.name = 'test_model3'
        self.conv1 = nn.Conv2d(3, 128, 7, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.for_train = for_train

    def forward(self, x):  #带2个conv分支
        x = self.conv1(x)
        identity = x
        x = self.conv2(x)
        x = self.relu(x)
        identity = self.conv3(identity)
        x += identity
        # if self.for_train:
        #     x = x.sum()
        return x


class test_model4(nn.Module):

    def __init__(self, for_train=False):
        super(test_model4, self).__init__()
        self.name = 'test_model4'
        self.conv1 = nn.Conv2d(3, 32, 7, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sig = nn.Sigmoid()
        self.for_train = for_train

    def forward(self, x):  #带2个conv分支+加1个残差分支
        x = self.conv1(x)
        identity = x
        # x = self.relu(x) #todo 加上这个relu会出错
        x = self.conv2(x)
        x = self.silu(x)
        x = self.sig(x)
        x = self.silu(x)
        identity = self.conv3(identity)
        identity = self.silu(identity)
        identity += x
        return identity

    # def forward(self, x): #带2个conv分支+加1个残差分支
    #     x = self.conv1(x)
    #     identity = x
    #     identity2 = x
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     x = self.silu(x)
    #     x = self.sig(x)
    #     x = self.silu(x)
    #     identity = self.conv3(identity)
    #     identity = self.silu(identity)
    #     y = torch.cat([x, identity, identity2], dim=1)
    #     if self.for_train:
    #         y = y.sum()
    #     return y

    # def forward(self, x): #带2个conv分支+加1个残差分支
    #     x = self.conv1(x)
    #     identity = x
    #     identity2 = x
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     identity = self.conv3(identity)
    #     y = torch.cat([x, identity], dim=1)
    #     # y = torch.cat([x, identity, identity2], dim=1) #fail
    #     if self.for_train:
    #         y = y.sum()
    #     return y


class test_model5(nn.Module):

    def __init__(self, for_train=False):
        super(test_model5, self).__init__()
        self.name = 'test_model5'
        self.conv1 = nn.Conv2d(3, 32, 7, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.for_train = for_train

    def forward(self, x):  #带2个conv分支+加1个残差分支
        x = self.conv1(x)
        identity = x
        identity2 = x
        x = self.relu(x)
        x = self.relu(x)
        x = self.relu(x)
        x = self.relu(x)
        x = self.conv2(x)
        identity = self.relu(identity)
        identity = self.relu(identity)
        identity = self.relu(identity)
        identity = self.conv3(identity)
        y = torch.cat([x, identity, identity2], dim=1)
        if self.for_train:
            y = y.sum()
        return y


class test_model6(nn.Module):

    def __init__(self, for_train=False):
        super(test_model6, self).__init__()
        self.name = 'test_model6'
        self.conv1 = nn.Conv2d(3, 128, 7, 2, 3, bias=False)
        self.norm = nn.BatchNorm2d(128, eps=0, momentum=0.5, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, 10, bias=True)
        self.relu = nn.ReLU()
        self.for_train = for_train

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.pool1(x)
        identity = x
        x = self.relu(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += identity
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        if self.for_train:
            x = x.sum()
        return x


class test_model7(nn.Module):

    def __init__(self, for_train=False):
        super(test_model7, self).__init__()
        self.name = 'test_model7'
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.relu = nn.ReLU()
        self.for_train = for_train

    def forward(self, x, y):
        x = self.relu(x)
        y = self.relu(y)
        z = x + y
        x1 = self.conv1(z)
        x2 = self.conv2(z)
        x = x1 + x2
        if self.for_train:
            x = x.sum()
        return x


class test_model8(nn.Module):

    def __init__(self, for_train=False):
        super(test_model8, self).__init__()
        self.name = 'test_model8'
        self.conv1 = nn.Conv2d(3, 128, 8, 8, 0, bias=False)
        self.norm = nn.BatchNorm2d(128, eps=0, momentum=0.5, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(128, 128, 8, 8, 0, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128 * 3 * 3, 1000)
        self.for_train = for_train

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)

        x = x.reshape(1, 128 * 3 * 3)
        x = self.fc(x)
        if self.for_train:
            x = x.sum()
        return x


class test_model9(nn.Module):

    def __init__(self):
        super(test_model9, self).__init__()
        self.name = 'test_model9'
        self.relu = nn.ReLU()
        '''
        d1 = torch.randn((2, 4096, 512))
        d2 = torch.randn((2, 512, 4096))
        d3 = torch.randn((2, 4096, 512))
        d4 = torch.randn((512, 512))
        d5 = torch.randn((1, 512))'''

    def forward(self, d1, d2, d3, d4, d5):
        x = torch.bmm(d1, d2)
        x = torch.nn.functional.softmax(x, dim=2)
        x = torch.bmm(x, d3)
        x = x.reshape(8192, 512)
        x = torch.addmm(d5, x, d4)
        return x


class test_model10(nn.Module):

    def __init__(self):
        super(test_model10, self).__init__()
        self.name = 'test_model10'
        self.gelu = nn.GELU()
        '''
        d1 = torch.randn((2, 4096, 320), dtype = torch.float16)
        d2 = torch.randn((1, 320, 2560), dtype = torch.float16)
        d3 = torch.randn((2560, 320), dtype = torch.float16)
        '''

    def forward(self, d1, d2, d3):
        x = torch.bmm(d1, d2)
        x = self.gelu(x)
        x = x.reshape(8192, 2560)
        x = torch.mm(x, d3)
        return x


class test_model11(nn.Module):

    def __init__(self):
        super(test_model11, self).__init__()
        self.name = 'test_model11'
        self.relu = nn.ReLU()
        '''
        d1 = torch.randn((2, 4096, 320), dtype = torch.float16)
        d2 = torch.randn((1, 320, 2560), dtype = torch.float16)
        d3 = torch.randn((2560, 320), dtype = torch.float16)
        '''

    def forward(self, d1, d2):
        x = torch.mm(x, d3)
        x = self.relu(x)
        return x


# class test_model(nn.Module):
#     def __init__(self, for_train = False):
#         super(test_model, self).__init__()
#         self.name = 'test_model'
#         self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
#         self.norm = nn.BatchNorm2d(64, eps=0, momentum=0.5, affine=True, track_running_stats=True)
#         # self.norm = nn.LayerNorm([64,8,8], elementwise_affine=True)
#         self.pool1 = nn.MaxPool2d(3, 2, 1)
#         # self.relu = nn.ReLU()
#         # self.linear = nn.Linear(1024,10, bias=False)

#     # def forward(self, x, y):
#     #     x = torch.bmm(x, y)
#     def forward(self, x):
#         # x = x + x
#         # x = x.permute(0,2,1,3)
#         # x = x.permute(3,2,1,0)
#         # x = x.transpose(1,2)
#         # x = x.reshape([3,16,16])
#         x = torch.mean(x,1, False)
#         # x = self.conv1(x)
#         # x = 2*x
#         # x = self.norm(x)
#         # x = torch.split(x, 32, dim=1)
#         # x = self.relu(x)
#         # x = self.pool1(x)
#         # x = x.view(-1)
#         # x = self.linear(x)
#         # x = x[0,0,0,0] #aten.select Op not support
#         # x = x.view(-1)[0] #aten.select Op not support
#         # x = x.sum()
#         return x
