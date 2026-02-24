# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, max_pool2d
import numpy as np
import pdb
from torch.nn.utils import weight_norm as wn
from itertools import chain
from copy import deepcopy
import torch.nn.functional as F
from copy import deepcopy
def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        sizes = [int(x) for x in sizes]
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)
    def forward(self, x):
        return self.net(x)

class block(nn.Module):
    def __init__(self, n_in, n_out):
        super(block, self).__init__()
        self.net = nn.Sequential(*[ nn.Linear(n_in, n_out), nn.ReLU()])
        self.net.apply(Xavier)
    
    def forward(self, x):
        return self.net(x)

       
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class Dualres(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(Dualres, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.s_layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.s_layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.s_layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.s_layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.simclr1=nn.Linear(nf * 8 * block.expansion, 128)



        self.f_layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.f_layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.f_layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.f_layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.simclr2=nn.Linear(nf * 8 * block.expansion, 128)

        self.catconv = nn.Conv2d(nf * 8 * 2, nf * 8, kernel_size=1, bias=False)
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        self.relu = nn.ReLU()

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding = 1):
        layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(out_maps),nn.ReLU()]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def slow_learner(self):
        param = chain(self.conv1.parameters(), self.s_layer1.parameters(), self.s_layer2.parameters(),
                        self.s_layer3.parameters(), self.s_layer4.parameters())
        for p in param:
            yield p

    def fast_learner(self):
        param = chain(self.f_layer1.parameters(), self.f_layer2.parameters(),self.f_layer3.parameters(),
                    self.f_layer4.parameters(), self.linear.parameters())
        for p in param:
            yield p

    def train_slow(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = False

    def train_all(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = True

    def train_fast(self):
        for p in self.slow_learner():
            p.requires_grad = False
        for p in self.fast_learner():
            p.requires_grad = True
    def slow_forward(self, x, use_proj=False):

        # h0 = self.conv1(x)
        # h0 = relu(self.bn1(h0))
        # h0 = self.maxpool(h0)
        h1 = self.s_layer1(x)
        h2 = self.s_layer2(h1)
        h3 = self.s_layer3(h2)
        h4 = self.s_layer4(h3)
        feat = self.avgpool(h4)  # b, 512
        feat = feat.view(feat.size(0),-1)        
        if use_proj:

            simfeat = self.simclr1(feat)   #b, 128
            return feat, simfeat
        return feat
    def fast_forward(self, x, use_proj=False):
        h1 = self.f_layer1(x)
        h2 = self.f_layer2(h1)
        h3 = self.f_layer3(h2)
        h4 = self.f_layer4(h3)
        feat = self.avgpool(h4)  # b, 512
        feat = feat.view(feat.size(0),-1)        
        if use_proj:
            simfeat = self.simclr2(feat)   #b, 128
            return feat, simfeat
        return feat
    
    def forward(self, x, use_proj=False):
        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        # h0 = self.maxpool(h0)
        if use_proj:        
            s_feat, s_simfeat = self.slow_forward(h0, use_proj=use_proj)
            f_feat, f_simfeat = self.fast_learner(h0, use_proj=use_proj)

            x = self.catconv(s_feat, f_feat)
            y = self.linear(x)
            return s_feat, s_simfeat, f_feat, f_simfeat, y
        else:
            s_feat = self.slow_forward(h0, use_proj=use_proj)
            f_feat = self.fast_learner(h0, use_proj=use_proj)

            x = self.catconv(s_feat, f_feat)
            y = self.linear(x)
            return y
    def forward_fast(self, x, use_proj=False):
        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        h1 = self.f_layer1(x)
        h2 = self.f_layer2(h1)
        h3 = self.f_layer3(h2)
        h4 = self.f_layer4(h3)
        feat = self.avgpool(h4)  # b, 512
        feat = feat.view(feat.size(0),-1)        
        if use_proj:
            simfeat = self.simclr1(feat)   #b, 128
            return feat, simfeat
        return feat 
    def forward_slow(self, x, use_proj=False):
        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        h1 = self.s_layer1(x)
        h2 = self.s_layer2(h1)
        h3 = self.s_layer3(h2)
        h4 = self.s_layer4(h3)
        feat = self.avgpool(h4)  # b, 512
        feat = feat.view(feat.size(0),-1)        
        if use_proj:
            simfeat = self.simclr1(feat)   #b, 128
            return feat, simfeat
        return feat
class MaskNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(MaskNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, stride=1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_simclr=nn.Linear(nf * 8 * block.expansion, 128)
        self.s_simclr=nn.Linear(nf * 8 * block.expansion, 128)
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        self.linear2 = nn.Linear(nf * 16 * block.expansion, int(num_classes))
        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.predictor = nn.Sequential(*layers)

        self.f_conv1 = self._make_conv2d_layer(3, nf, max_pool=False, padding = 1)
        # self.f_conv1 = self._make_conv2d_layer(3, nf, max_pool=False, padding = 1, stride=4)
        self.f_conv2 = self._make_conv2d_layer(nf*1, nf*2, padding = 1, max_pool=True)
        self.f_conv3 = self._make_conv2d_layer(nf*2, nf*4, padding = 1, max_pool=True)
        self.f_conv4 = self._make_conv2d_layer(nf*4, nf*8, padding = 1, max_pool=True)
        
        self.compress_conv = nn.Conv2d(nf*8*2, nf*8, kernel_size=1)

        self.relu = nn.ReLU()


        sizes = [nf, nf*2, nf*4, nf*8]
        self.weighter = nn.ModuleList()
        for i in range(len(sizes)):
            layers = []
            layers.append(nn.Linear(sizes[i], 2, bias=False))
            # layers.append(nn.BatchNorm1d(2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Softmax(dim=1))
            self.weighter.append(nn.Sequential(*layers))
   

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding = 1, stride=1):
        layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(out_maps),nn.ReLU()]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def slow_learner(self):
        param = chain(self.conv1.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                        self.layer3.parameters(), self.layer4.parameters(), self.projector.parameters())
        for p in param:
            yield p

    def fast_learner(self):
        param = chain(self.f_conv1.parameters(), self.f_conv2.parameters(),self.f_conv3.parameters(),
                    self.f_conv4.parameters(), self.linear.parameters())
        for p in param:
            yield p

    def train_slow(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = False

    def train_all(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = True

    def train_fast(self):
        for p in self.slow_learner():
            p.requires_grad = False
        for p in self.fast_learner():
            p.requires_grad = True
    
    
    def forward(self, x, return_slow_feat=False, return_fast_feat = False, return_all=False):

        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        # h0 = self.maxpool(h0)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        s_feat = self.avgpool(h4)  # b, 512
        s_feat = s_feat.view(s_feat.size(0),-1)        
        if return_slow_feat:

            simfeat = self.s_simclr(s_feat)   #b, 128
            return s_feat, simfeat
        
        m1_ = self.f_conv1(x)
        m1 = m1_ * h1
        m2_ = self.f_conv2(m1)
        m2 = m2_ * h2
        m3_ = self.f_conv3(m2)
        m3 = m3_ * h3
        m4_ = self.f_conv4(m3)
        m4 = m4_ * h4
        f_feat = self.avgpool(m4)
        #out = self.avgpool(h4)
        f_feat = f_feat.view(f_feat.size(0), -1)
        y = self.linear(f_feat)
        if return_fast_feat:
            fastsim = self.f_simclr(f_feat)
            
            return f_feat, fastsim
        if return_all:
            s_sim = self.s_simclr(s_feat)
            f_sim = self.f_simclr(f_feat)
            # y = self.linear(f_feat)
            return s_feat, s_sim, f_feat, f_sim

        return y
    


    


    
class ISNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ISNet, self).__init__()
        self.in_planes = nf
        self.stemconv = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.s_layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.s_layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.s_layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.s_layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_simclr=nn.Linear(nf * 16 * block.expansion, 128)



        self.conv1 = self._make_conv2d_layer(3, nf*1, padding = 1, max_pool=False)
        self.conv2 = self._make_conv2d_layer(nf*2, nf*2, padding = 1, max_pool=True)
        self.conv3 = self._make_conv2d_layer(nf*4, nf*4, padding = 1, max_pool=True)
        self.conv4 = self._make_conv2d_layer(nf*8, nf*8, padding = 1, max_pool=True)
        self.s_simclr=nn.Linear(nf * 8 * block.expansion, 128)
        self.linear = nn.Linear(nf * 16 * block.expansion, int(num_classes))
        self.relu = nn.ReLU()

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding = 1):
        layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=padding),
                nn.BatchNorm2d(out_maps),nn.ReLU()]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def slow_learner(self):
        param = chain(self.conv1.parameters(), self.s_layer1.parameters(), self.s_layer2.parameters(), self.s_layer3.parameters(),
                        self.s_layer4.parameters())
        for p in param:
            yield p

    def fast_learner(self):
        param = chain(self.f_layer1.parameters(), self.f_layer2.parameters(),self.f_layer3.parameters(),
                    self.f_layer4.parameters(), self.linear.parameters())
        for p in param:
            yield p

    def train_slow(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = False

    def train_all(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = True

    def train_fast(self):
        for p in self.slow_learner():
            p.requires_grad = False
        for p in self.fast_learner():
            p.requires_grad = True
    def forward(self, x, return_slow_feat=False, return_fast_feat = False, return_all=False):

        h0 = self.stemconv(x)
        h0 = relu(self.bn1(h0))
        # h0 = self.maxpool(h0)
        h1 = self.s_layer1(h0)
        h2 = self.s_layer2(h1)
        h3 = self.s_layer3(h2)
        h4 = self.s_layer4(h3)
        s_feat = self.avgpool(h4)  # b, 512
        s_feat = s_feat.view(s_feat.size(0),-1)        
        if return_slow_feat:

            simfeat = self.s_simclr(s_feat)   #b, 128
            return s_feat, simfeat
        
        m1_ = self.conv1(x)
        m1 = torch.cat((m1_, h1), dim=1)
        m2_ = self.conv2(m1)
        m2 = torch.cat((m2_, h2), dim=1)
        m3_ = self.conv3(m2)
        m3 = torch.cat((m3_, h3), dim=1)
        m4_ = self.conv4(m3)
        m4 = torch.cat((m4_, h4), dim=1)
        f_feat = self.avgpool(m4)
        #out = self.avgpool(h4)
        f_feat = f_feat.view(f_feat.size(0), -1)
        y = self.linear(f_feat)
        if return_fast_feat:
            fastsim = self.f_simclr(f_feat)
            
            return f_feat, fastsim
        if return_all:
            s_sim = self.s_simclr(s_feat)
            f_sim = self.f_simclr(f_feat)
            # y = self.linear(f_feat)
            return s_feat, s_sim, f_feat, f_sim

        return y


def MaskNet18(num_classes, nf=20):
    return MaskNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)

def ISNet18(num_classes, nf=20):
    return ISNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)
    


# class MaskNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes, nf):
#         super(MaskNet, self).__init__()
#         self.in_planes = nf
#         self.conv1 = conv3x3(3, nf * 1)
#         self.bn1 = nn.BatchNorm2d(nf * 1)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.simclr1=nn.Linear(nf * 8 * block.expansion, 128)
#         self.simclr2=nn.Linear(nf * 8 * block.expansion, 128)
#         self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        
#         sizes = [nf*8] + [256, nf*8]
#         layers = []
#         for i in range(len(sizes) - 2):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
#             layers.append(nn.BatchNorm1d(sizes[i + 1]))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
#         self.projector = nn.Sequential(*layers)

#         sizes = [nf*8] + [256, nf*8]
#         layers = []
#         for i in range(len(sizes) - 2):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
#             layers.append(nn.BatchNorm1d(sizes[i + 1]))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
#         self.predictor = nn.Sequential(*layers)

#         self.f_conv1 = self._make_conv2d_layer(3, nf, max_pool=True, padding = 1)
#         self.f_conv2 = self._make_conv2d_layer(nf*2, nf*2, padding = 1, max_pool=True)
#         self.f_conv3 = self._make_conv2d_layer(nf*4, nf*4, padding = 1, max_pool=True)
#         self.f_conv4 = self._make_conv2d_layer(nf*8, nf*8, padding = 1, max_pool=True)
#         self.relu = nn.ReLU()

#     @staticmethod
#     def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding = 1):
#         layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=padding),
#                 nn.BatchNorm2d(out_maps),nn.ReLU()]
#         if max_pool:
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
#         return nn.Sequential(*layers)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
    

#     def slow_learner(self):
#         param = chain(self.conv1.parameters(), self.layer1.parameters(), self.layer2.parameters(),
#                         self.layer3.parameters(), self.layer4.parameters(), self.projector.parameters())
#         for p in param:
#             yield p

#     def fast_learner(self):
#         param = chain(self.f_conv1.parameters(), self.f_conv2.parameters(),self.f_conv3.parameters(),
#                     self.f_conv4.parameters(), self.linear.parameters())
#         for p in param:
#             yield p

#     def train_slow(self):
#         for p in self.slow_learner():
#             p.requires_grad = True
#         for p in self.fast_learner():
#             p.requires_grad = False

#     def train_all(self):
#         for p in self.slow_learner():
#             p.requires_grad = True
#         for p in self.fast_learner():
#             p.requires_grad = True

#     def train_fast(self):
#         for p in self.slow_learner():
#             p.requires_grad = False
#         for p in self.fast_learner():
#             p.requires_grad = True
    
#     def forward(self, x, return_slow_feat=False, return_fast_simfeat = False):
#         bsz = x.size(0)

#         h0 = self.conv1(x)
#         h0 = relu(self.bn1(h0))
#         h0 = self.maxpool(h0)
#         h1 = self.layer1(h0)
#         h2 = self.layer2(h1)
#         h3 = self.layer3(h2)
#         h4 = self.layer4(h3)
#         feat = self.avgpool(h4)  # b, 512
#         feat = feat.view(feat.size(0),-1)
#         sim_slow = self.simclr1(feat)
#         if return_slow_feat:
#             return feat, sim_slow
        
#         m1_ = self.f_conv1(x)
#         m1 = torch.cat([m1_, h1], dim=1)
#         m2_ = self.f_conv2(m1)
#         m2 = torch.cat([m2_, h2], dim=1)
#         m3_ = self.f_conv3(m2)
#         m3 = torch.cat([m3_, h3], dim=1)
#         m4_ = self.f_conv4(m3)
#         m4 = m4_
#         out = self.avgpool(m4)
#         #out = self.avgpool(h4)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         if return_fast_simfeat:
#             fastsim = self.simclr2(out)
            
#             return out, fastsim

#         return y