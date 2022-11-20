#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: augmentor.py
@time: 2019/09/16
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random

class Augmentor_Displacement(nn.Module):
    def __init__(self, dim):
        super(Augmentor_Displacement, self).__init__()

        self.conv1 = torch.nn.Conv1d(dim+1024+64, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 3, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x


class Augmentor(nn.Module):
    def __init__(self,dim=1024,in_dim=3):
        super(Augmentor, self).__init__()
        self.dim = dim
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.dis = Augmentor_Displacement(self.dim)

    def forward(self, pt, noise):


        B, C, N = pt.size()
        raw_pt = pt[:,:3,:].contiguous()


        x = F.relu(self.bn1(self.conv1(raw_pt)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        feat_d = x.view(-1, 1024, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat([pointfeat, feat_d,noise_d],1)
        displacement = self.dis(feat_d)

        return displacement

