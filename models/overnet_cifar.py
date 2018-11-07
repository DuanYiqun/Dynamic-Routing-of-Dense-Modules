'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import random

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, drop_rate=0.001):
        super(Bottleneck, self).__init__()
        self.drop_rate=drop_rate
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        #out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

graph_matrix=[[[0,1],[1,2],[2]],[[0,1],[1,2],[2]],[[0,1],[1,2],[2]]]
exp_block = [6,12,24]

class Ovnet(nn.Module):
    def __init__(self, block, nblocks=[6,12,24], graph=[[[0,1],[1,2],[2]],[[0,1],[1,2],[2]],[[0,1],[1,2],[2]]] ,growth_rate=32, reduction=0.5, num_classes=10, core_nums=1):
        super(Ovnet, self).__init__()
        self.growth_rate = growth_rate
        self.core_nums = core_nums
        self.core_list = [64]
        self.graph = graph
        num_planes=sum(self.core_list)
        self.conv1=nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

    def cal_input(self, input_features,input_core, growth, num_blocks):
        outnum = input_features + input_core*(growth*num_blocks)
        return outnum
    
    def cal_output(self, input_features, growth, num_blocks):
        outnum = input_features + growth*num_blocks
        return outnum
    
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def random_vibe(self, x1):
        w = random.randint(0, 4)
        h = random.randint(0, 4)
        a = []
        b = []
        for i in range(28):
            a.append(w+i)
            b.append(h+i)
        x2 = torch.index_select(x1, 2 ,torch.LongTensor(a))
        x2 = torch.index_select(x2, 3, torch.LongTensor(b))
        #x2 = torch.index_select(x1, 2 ,torch.cuda.LongTensor(a))
        #x2 = torch.index_select(x2, 3, torch.cuda.LongTensor(b))
        #when using cuda

        return x2

    
    def forward(self,x):
        x1 = self.random_vibe(x)
        x2 = self.random_vibe(x)
        x3 = self.random_vibe(x)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x3 = self.conv1(x3)
        print(x1.size(),x2.size(),x3.size())
        return x1
        

def test_ovnet():
    cc = Ovnet(Bottleneck)
    #print(cc)
    #x1 = torch.randn(1,3,96,96)
    x1 = torch.randn(1,3,32,32)
    y = cc(Variable(x1))
    #print(y)

test_ovnet()