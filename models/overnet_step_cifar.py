'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import ssp_layer
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
        self.core_list = [128,64,32]
        self.graph = graph
        self.output_sam = [4,2,1]
        #num_planes=sum(self.core_list)
        self.conv1_1=nn.Conv2d(3, self.core_list[0], kernel_size=1, padding=1, bias=False)
        self.conv1_2=nn.Conv2d(3, self.core_list[1], kernel_size=7, padding=1, bias=False)
        self.conv1_3=nn.Conv2d(3, self.core_list[2], kernel_size=13, padding=1, bias=False)

        self.db0_1 = self._make_dense_layers(block, int(self.core_list[0]), nblocks[0])
        self.db0_2 = self._make_dense_layers(block, int(self.core_list[1]), nblocks[0])
        self.db0_3 = self._make_dense_layers(block, int(self.core_list[2]), nblocks[0])

        num_features=[self.core_list[0],self.core_list[1],self.core_list[2]]
        #input_features=num_features
        out_planes=[self.core_list[0],self.core_list[1],self.core_list[2]]

        num_features[0] = self.cal_output(self.core_list[0], growth_rate, nblocks[0])
        out_planes[0] = int(math.floor(num_features[0]*reduction))

        num_features[1] = self.cal_output(self.core_list[1], growth_rate, nblocks[0])
        out_planes[1] = int(math.floor(num_features[1]*reduction))

        num_features[2] = self.cal_output(self.core_list[2], growth_rate, nblocks[0])
        out_planes[2] = int(math.floor(num_features[2]*reduction))

        self.trans0_1 = Transition(num_features[0], out_planes[0])
        self.trans0_2 = Transition(num_features[1], out_planes[1])
        self.trans0_3 = Transition(num_features[2], out_planes[2])

        num_features[0]=out_planes[0]
        num_features[1]=out_planes[1]
        num_features[2]=out_planes[2]

        self.db1_1 = self._make_dense_layers(block, num_features[0], nblocks[1])
        self.db1_2 = self._make_dense_layers(block, num_features[1], nblocks[1])
        self.db1_3 = self._make_dense_layers(block, num_features[2], nblocks[1])


        num_features[0] = self.cal_output(num_features[0], growth_rate, nblocks[1])
        out_planes[0] = int(math.floor(num_features[0]*reduction))

        num_features[1] = self.cal_output(num_features[1], growth_rate, nblocks[1])
        out_planes[1] = int(math.floor(num_features[1]*reduction))

        num_features[2] = self.cal_output(num_features[2], growth_rate, nblocks[1])
        out_planes[2] = int(math.floor(num_features[2]*reduction))

        self.trans1_1 = Transition(num_features[0], out_planes[0])
        self.trans1_2 = Transition(num_features[1], out_planes[1])
        self.trans1_3 = Transition(num_features[2], out_planes[2])

        num_features[0]=out_planes[0]
        num_features[1]=out_planes[1]
        num_features[2]=out_planes[2]

        self.db2_1 = self._make_dense_layers(block, num_features[0], nblocks[2])
        self.db2_2 = self._make_dense_layers(block, num_features[1], nblocks[2])
        self.db2_3 = self._make_dense_layers(block, num_features[2], nblocks[2])

        num_features[0] = self.cal_output(num_features[0], growth_rate, nblocks[2])
        out_planes[0] = int(math.floor(num_features[0]*reduction))

        num_features[1] = self.cal_output(num_features[1], growth_rate, nblocks[2])
        out_planes[1] = int(math.floor(num_features[1]*reduction))

        num_features[2] = self.cal_output(num_features[2], growth_rate, nblocks[2])
        out_planes[2] = int(math.floor(num_features[2]*reduction))

        self.trans2_1 = Transition(num_features[0], out_planes[0])
        self.trans2_2 = Transition(num_features[1], out_planes[1])
        self.trans2_3 = Transition(num_features[2], out_planes[2])

 
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
        x1 = self.conv1_1(x1)
        x2 = self.conv1_2(x2)
        x3 = self.conv1_3(x3)
        x1 = self.db0_1(x1)
        x2 = self.db0_2(x2)
        x3 = self.db0_3(x3)
        x1 = self.trans0_1(x1)
        x2 = self.trans0_2(x2)
        x3 = self.trans0_3(x3)
        print(x1.size(),x2.size(),x3.size())
        x1 = self.db1_1(x1)
        x2 = self.db1_2(x2)
        x3 = self.db1_3(x3)
        x1 = self.trans1_1(x1)
        x2 = self.trans1_2(x2)
        x3 = self.trans1_3(x3)
        print(x1.size(),x2.size(),x3.size())
        #x1 = ssp_layer.spatial_pyramid_pool(x1, 1, [int(x1.size(2)),int(x1.size(3))], self.output_sam)

        x1 = self.db2_1(x1)
        x2 = self.db2_2(x2)
        #x3 = self.db2_3(x3)
        x1 = self.trans2_1(x1)
        x2 = self.trans2_2(x2)
        #x3 = self.trans2_3(x3)
        
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