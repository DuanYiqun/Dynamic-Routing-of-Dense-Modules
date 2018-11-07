'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


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
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class Merge(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Merge, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return out


#graph_matrix=[[[1,2],[1,3],[1,3]],[[1,2],[1],[2,3]]]
graph_matrix=[[[0,1],[1,2],[2]],[[0,1],[1,2],[2]],[[0],[1],[2]]]


class Modulenet(nn.Module):
    def __init__(self, block, nblocks, graph ,growth_rate=12, reduction=0.5, num_classes=100, core_nums=3):
        super(Modulenet, self).__init__()
        self.growth_rate = growth_rate
        self.core_nums = core_nums
        self.core_list = [32,32,32]
        self.graph = graph
        num_planes=sum(self.core_list)

        self.conv1=nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        num_features=[]
        out_planes=[]

        self.module0_1 = self._make_dense_layers(block, int(self.core_list[0]), nblocks[0] ,self.growth_rate)
        self.module0_2 = self._make_dense_layers(block, int(self.core_list[1]), nblocks[0] ,self.growth_rate)
        self.module0_3 = self._make_dense_layers(block, int(self.core_list[2]), nblocks[0] ,self.growth_rate)
        #这里需要遍历一遍系数矩阵日后，目前版本硬编码为给定graph
        #可以写一个篇历grpah
        num_features=[int(self.core_list[0]),int(self.core_list[1]),int(self.core_list[2])]
        #input_features=num_features
        out_planes=[int(self.core_list[0]),int(self.core_list[1]),int(self.core_list[2])]
        
        num_features[0] = self.cal_output(int(self.core_list[0]), self.growth_rate, nblocks[0])
        out_planes[0] = int(math.floor(num_features[0]*reduction))

        num_features[1] = self.cal_output(int(self.core_list[1]), self.growth_rate, nblocks[0])
        out_planes[1] = int(math.floor(num_features[1]*reduction))

        num_features[2] = self.cal_output(int(self.core_list[2]), self.growth_rate, nblocks[0])
        out_planes[2] = int(math.floor(num_features[2]*reduction))

        #num_features[2] = self.cal_output(int(num_planes/core_nums), len(graph[0][2]), growth_rate, nblocks[0])
        #out_planes[2] = int(math.floor(num_features[2]*reduction))
        
        self.trans0_1 = Transition(num_features[0], out_planes[0])
        self.trans0_2 = Transition(num_features[1], out_planes[1])
        self.trans0_3 = Transition(num_features[2], out_planes[2])

        #print(num_features)
        #print(out_planes)
        #print(num_features)
        #print(num_features[0],num_features[1])
        num_features[0]=out_planes[0]
        num_features[1]=out_planes[1]
        num_features[2]=out_planes[2]
        print(num_features)

        #print(num_features[0],num_features[1])

        self.module1_1 = self._make_dense_layers(block, num_features[0], nblocks[1], self.growth_rate)
        self.module1_2 = self._make_dense_layers(block, num_features[1], nblocks[1], self.growth_rate)
        self.module1_3 = self._make_dense_layers(block, num_features[2], nblocks[1], self.growth_rate)

        num_features[0] = self.cal_output(num_features[0],  self.growth_rate, nblocks[1])
        #print(num_features[0])
        out_planes[0] = int(math.floor(num_features[0]*reduction))
        #print(num_features[0])
        self.trans1_1 = Transition(num_features[0], out_planes[0])
        num_features[1] = self.cal_output(num_features[1],  self.growth_rate, nblocks[1])
        out_planes[1] = int(math.floor(num_features[1]*reduction))
        self.trans1_2 = Transition(num_features[1], out_planes[1])
        num_features[2] = self.cal_output(num_features[2],  self.growth_rate, nblocks[1])
        out_planes[2] = int(math.floor(num_features[2]*reduction))
        self.trans1_3 = Transition(num_features[2], out_planes[2])
        #print(num_features[0],num_features[1])
        #print(num_features)
        #print(out_planes)
        num_features[0]=out_planes[0]
        num_features[1]=out_planes[1]
        num_features[2]=out_planes[2]

        print(num_features)
        #print(out_planes)


        #print(num_features)
    
        #out_planes[0]=num_features[0]+num_features[1]
        #out_planes[1]=num_features[0]

        self.module2_1 = self._make_dense_layers(block, num_features[0], nblocks[2],self.growth_rate)
        self.module2_2 = self._make_dense_layers(block, num_features[1], nblocks[2],self.growth_rate)
        self.module2_3 = self._make_dense_layers(block, num_features[2], nblocks[2],self.growth_rate)


        num_features[0] = self.cal_output(num_features[0],  self.growth_rate, nblocks[2])
        out_planes[0] = int(math.floor(num_features[0]*reduction))
        self.trans2_1 = Transition(num_features[0], out_planes[0])
        num_features[1] = self.cal_output(num_features[1],  self.growth_rate, nblocks[2])
        out_planes[1] = int(math.floor(num_features[1]*reduction))
        self.trans2_2 = Transition(num_features[1], out_planes[1])
        num_features[2] = self.cal_output(num_features[2],  self.growth_rate, nblocks[2])
        out_planes[2] = int(math.floor(num_features[2]*reduction))
        self.trans2_3 = Transition(num_features[2], out_planes[2])

        
        num_features[0]=out_planes[0]
        num_features[1]=out_planes[1]
        num_features[2]=out_planes[2]


        print(num_features)

        self.module3_1 = self._make_dense_layers(block, num_features[0], nblocks[3] , self.growth_rate)
        self.module3_2 = self._make_dense_layers(block, num_features[1], nblocks[3] , self.growth_rate)
        self.module3_3 = self._make_dense_layers(block, num_features[2], nblocks[3] , self.growth_rate)

        num_features[0] = self.cal_output(num_features[0],  self.growth_rate, nblocks[3])
        out_planes[0] = int(math.floor(num_features[0]*reduction))
        num_features[1] = self.cal_output(num_features[1],  self.growth_rate, nblocks[3])
        out_planes[1] = int(math.floor(num_features[1]*reduction))
        num_features[2] = self.cal_output(num_features[2],  self.growth_rate, nblocks[3])
        out_planes[2] = int(math.floor(num_features[2]*reduction))
        
        self.trans3_1 = Transition(num_features[0], out_planes[0])
        self.trans3_2 = Transition(num_features[1], out_planes[1])
        self.trans3_3 = Transition(num_features[2], out_planes[2])

        num_features[0]=out_planes[0]
        num_features[1]=out_planes[1]
        num_features[2]=out_planes[2]

        num_planes=num_features[0]+num_features[1]+num_features[2]
        print(num_planes)
        self.mg = Transition(num_planes,1000)

        #self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(1000, num_classes)

    def cal_input(self, input_features,input_core, growth, num_blocks):
        outnum = input_features + input_core*(growth*num_blocks)
        return outnum
    
    def cal_output(self, input_features, growth, num_blocks):
        outnum = input_features + growth*num_blocks
        return outnum
    
    def _make_dense_layers(self, block, in_planes, nblock , growth_rate):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, growth_rate))
            in_planes += growth_rate
        return nn.Sequential(*layers)

    def forward(self,x, graph_matrix):
        out = self.conv1(x)
        outs = torch.split(out,self.core_list,dim=1)
        #print(outs[0].size(),outs[1].size(),outs[2].size())
        outs = [self.trans0_1(self.module0_1(outs[0].contiguous())),self.trans0_2(self.module0_2(outs[1].contiguous())),self.trans0_3(self.module0_3(outs[2].contiguous()))]
        
        out_temp=[]
        for ind, i in enumerate(graph_matrix[0]):
            if i == []:
                out_temp.append([])
                print('all connections are cutted false')
                raise Exception("All connections are cutted false")
            for index,k in enumerate(i):
                if index == 0:
                    out_temp.append(outs[k])
                    #print(out_temp[ind].size())
                    #print(k)
                else:
                    out_temp[ind]=torch.add(out_temp[ind],outs[k])
        outs=out_temp
        
        outs = [self.trans1_1(self.module1_1(outs[0].contiguous())),self.trans1_2(self.module1_2(outs[1].contiguous())),self.trans1_3(self.module1_3(outs[2].contiguous()))]
        
        out_temp=[]
        for ind, i in enumerate(graph_matrix[1]):
            for index,k in enumerate(i):
                if index == 0:
                    out_temp.append(outs[k])
                    #print(out_temp[ind].size())
                    #print(k)
                else:
                    out_temp[ind]=torch.add(out_temp[ind],outs[k])
        outs=out_temp

        outs = [self.trans2_1(self.module2_1(outs[0].contiguous())),self.trans2_2(self.module2_2(outs[1].contiguous())),self.trans2_3(self.module2_3(outs[2].contiguous()))]
        
        out_temp=[]
        for ind, i in enumerate(graph_matrix[2]):
            for index,k in enumerate(i):
                if index == 0:
                    out_temp.append(outs[k])
                    #print(out_temp[ind].size())
                    #print(k)
                else:
                    out_temp[ind]=torch.add(out_temp[ind],outs[k])
        outs=out_temp

        outs = [self.module3_1(outs[0].contiguous()),self.module3_2(outs[1].contiguous()),self.module3_3(outs[2].contiguous())]
        outs = [self.trans3_1(outs[0].contiguous()),self.trans3_2(outs[1].contiguous()),self.trans3_3(outs[2].contiguous())]
        
        out = torch.cat((outs[0],outs[1],outs[2]),dim=1)
        out = self.mg(out)
        #print(out.size(0))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #print(outs[0].size(),outs[1].size(),outs[2].size())

        return out


#net=Modulenet(Bottleneck,[6,12,12],graph_matrix, growth_rate=12)

#print(net)
def modulenet_cifar():
    return Modulenet(Bottleneck,[6,12,24,16],graph_matrix, growth_rate=32,num_classes=100)

def modulenet_c10():
    return Modulenet(Bottleneck,[6,12,24,16],graph_matrix, growth_rate=12,num_classes=10)

nums = []
sparses = []

def extract(m):
    global sparses
    global nums
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nums.append(torch.numel(m.weight.data))
        #cc=m.weight.clone().cpu()
        #sparses.append(torch.mean(cc.abs()).detach().numpy())
        #print(m.weight.data)

print(nums)

def test_modulenet():
    net = modulenet_c10()
    x = torch.randn(2,3,32,32)
    graph_matrix=[[[0],[1,2],[2]],[[0,2],[1,2],[2]],[[0,1],[1,2],[2]]]
    y= net(Variable(x), graph_matrix)
    #print(y[0].size(),y[1].size(),y[2].size())
    print(y.size())
    #print(net)
    net.apply(extract)
    print(nums)
    print('the sum of param is {}'.format(sum(nums)))

test_modulenet()

