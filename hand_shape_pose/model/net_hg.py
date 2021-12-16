# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Networks for heatmap estimation from RGB images using Hourglass Network
"Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from hand_shape_pose.util.net_util import Residual


# class Hourglass(nn.Module):
#     def __init__(self, n, nModules, nFeats):
#         super(Hourglass, self).__init__()
#         self.n = n
#         self.nModules = nModules
#         self.nFeats = nFeats

#         _up1_, _low1_, _low2_, _low3_ = [], [], [], []
#         for j in range(self.nModules):
#             _up1_.append(Residual(self.nFeats, self.nFeats))
#         self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         for j in range(self.nModules):
#             _low1_.append(Residual(self.nFeats, self.nFeats))

#         if self.n > 1:
#             self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
#         else:
#             for j in range(self.nModules):
#                 _low2_.append(Residual(self.nFeats, self.nFeats))
#             self.low2_ = nn.ModuleList(_low2_)

#         for j in range(self.nModules):
#             _low3_.append(Residual(self.nFeats, self.nFeats))

#         self.up1_ = nn.ModuleList(_up1_)
#         self.low1_ = nn.ModuleList(_low1_)
#         self.low3_ = nn.ModuleList(_low3_)

#         self.up2 = nn.Upsample(scale_factor=2)

#     def forward(self, x):
#         up1 = x
#         for j in range(self.nModules):
#             up1 = self.up1_[j](up1)

#         low1 = self.low1(x)
#         for j in range(self.nModules):
#             low1 = self.low1_[j](low1)

#         if self.n > 1:
#             low2 = self.low2(low1)
#         else:
#             low2 = low1
#             for j in range(self.nModules):
#                 low2 = self.low2_[j](low2)

#         low3 = low2
#         for j in range(self.nModules):
#             low3 = self.low3_[j](low3)
#         up2 = self.up2(low3)

#         return up1 + up2


class Net_HM_HG(nn.Module):
    def __init__(self, num_joints, num_stages=2, num_modules=2, num_feats=256):
        super(Net_HM_HG, self).__init__()

        self.numOutput = num_joints
        self.nStack = num_stages

        self.nModules = num_modules
        self.nFeats = num_feats

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []
        encoding = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
                encoding.append(x)
            else:
                encoding.append(ll)

        return out, encoding
    
    
    
    
##################################### MODIFICATION ###############################    
    
       
    
"""
Used by 3-stage and 4-stage Identity Mapping Hourglass Network.
"""

from torch import nn
from torch.autograd import Function
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class Residual(nn.Module):
    """Residual Block modified by us"""

    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(ins, outs//2, 1, bias=False),
            nn.BatchNorm2d(outs//2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(outs // 2, outs // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outs // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(outs // 2, outs, 1, bias=False),
            nn.BatchNorm2d(outs),
        )
        if ins != outs:
            self.skipConv = nn.Sequential(
                nn.Conv2d(ins, outs, 1, bias=False),
                nn.BatchNorm2d(outs)
            )
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        x = self.relu(x)
        return x


class Conv(nn.Module):
    # conv block used in hourglass
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, dropout=False):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 换成 Leak Relu减缓神经元死亡现象
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False)
            # Different form TF, momentum default in Pytorch is 0.1, which means the decay rate of old running value
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)

    def forward(self, x):
        # examine the input channel equals the conve kernel channel
        assert x.size()[1] == self.inp_dim, "input channel {} dese not fit kernel channel {}".format(x.size()[1],
                                                                                                     self.inp_dim)
        if self.dropout:  # comment these two lines if we do not want to use Dropout layers
            # p: probability of an element to be zeroed
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)  # 直接注释掉这一行，如果我们不想使用Dropout

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# class Backbone(nn.Module):    
#     def __init__(self, nFeat=256, inplanes=3, resBlock=Residual):
#         super(Backbone, self).__init__()        
#         self.nFeat = nFeat
#         self.resBlock = resBlock
#         self.inplanes = inplanes
#         self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
#         self.res1 = self.resBlock(64, 128)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.res2 = self.resBlock(128, 128)
#         self.res3 = self.resBlock(128, self.nFeat)

#     def forward(self, x):
#         # head
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.res1(x)
#         x = self.pool(x)
#         x = self.res2(x)
#         x = self.res3(x)

#         return x


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""
    def __init__(self, depth, nFeat=256, increase=128, bn=False, resBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth  # oder number
        self.nFeat = nFeat  # input and output channels
        self.increase = increase  # increased channels while the depth grows
        self.bn = bn
        self.resBlock = resBlock
        # will execute when instantiate the Hourglass object, prepare network's parameters
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)  # no learning parameters, can be used any times repeatedly
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # no learning parameters  # FIXME: 改成反卷积？

    def _make_single_residual(self, depth_id):
        # the innermost conve layer, return as a layer item
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * (depth_id + 1),
                             bn=self.bn)                            # ###########  Index: 6

    def _make_lower_residual(self, depth_id):
        # return as a list
        pack_layers = [self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                     bn=self.bn, relu=False),                                     # ######### Index: 0
                       self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * (depth_id + 1),
                                                                                                  # ######### Index: 1
                                     bn=self.bn),
                       self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * depth_id,
                                                                                                   # ######### Index: 2
                                     bn=self.bn),
                       self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                                                                                   # ######### Index: 3
                                     bn=self.bn),
                       self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                                                                                   # ######### Index: 4
                                     bn=self.bn, relu=False),
                       nn.LeakyReLU(negative_slope=0.01, inplace=True)                              # ######### Index: 5
                       ]
        return pack_layers

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            #  skip path; up_residual_block; down_residual_block_path,
            # 0 ~ n-2 (except the outermost n-1 order) need 3 residual blocks
            res = self._make_lower_residual(i)  # type:list
            if i == (self.depth - 1):  # the deepest path (i.e. the longest path) need 4 residual blocks
                res.append(self._make_single_residual(i))  # list append an element
            hg.append(nn.ModuleList(res))  # pack conve layers of  every oder of hourglass block
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, depth_id, x, up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):  # except for the highest-order hourglass block
            low2 = self.hg[depth_id][6](low1)
        else:
            # call the lower-order hourglass block recursively
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        # ######################## # if we don't consider 8*8 scale
        # if depth_id < self.depth - 1:
        #     self.up_fms.append(low2)
        up2 = self.upsample(low3)
        deconv1 = self.hg[depth_id][3](up2)
        deconv2 = self.hg[depth_id][4](deconv1)
        up1 += deconv2
        out = self.hg[depth_id][5](up1)  # relu after residual add
        return out

    def forward(self, x):
        """
        :param: x a input tensor warpped wrapped as a list
        :return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
        """
        up_fms = []  # collect feature maps produced by low2 at every scale
        feature_map = self._hour_glass_forward(0, x, up_fms)
        return [feature_map] + up_fms[::-1]





