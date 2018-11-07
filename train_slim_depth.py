'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import depth_dynamic_slim
from NEAT import connection_gene
#from utils import progress_bar
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mname',default='depth_slim_c10', type=str, help='model name for save')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = alextnet.AlexNet(num_classes=10)
# net = vgg.VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = densenet.DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = depth_dynamic_slim.modulenet_c10_slim()
net = net.to(device)

graph_matrix=[[[0],[1],[2]],[[0],[1],[2]],[[0],[1],[2]]]

"""
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
"""
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./train/'+args.mname+'/checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./train/'+args.mname+'/checkpoints/best_check.plk')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    graph_matrix = checkpoint['graph']
    print(best_acc)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


trainpd = pd.DataFrame({"epoch":"","accuracy":"","loss":""},index=["0"])

savepath='./train/'+str(args.mname)+'/checkpoints/'

if not os.path.exists(savepath):
    os.makedirs(savepath)

# Training
def train(epoch,graph,graph1,graph2,connection):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time=time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs,graph,graph1,graph2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy=100.*correct/total
#       progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), accuracy, correct, total))
    end_time=time.time()
    epoch_time=end_time-start_time
    data=[epoch,accuracy,train_loss/(batch_idx+1),epoch_time]
    print('trainloss:{},accuracy:{},time_used:{},graphinfo:{}'.format(train_loss/(batch_idx+1),accuracy,epoch_time,graph))
    state = {
            'net': net.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
            'graph': graph,
            'connection': connection
        }
    if epoch % 30 == 0:
        savepath='./train/'+str(args.mname)+'/checkpoints/'+str(epoch)+'_check.plk'
        print('system_saving...at {} epoch'.format(epoch))
        torch.save(state, savepath)

    return data
  #  new = pd.DataFrame({"epoch":epoch,"accuracy":accuracy,"loss":train_loss/(batch_idx+1)},index=["0"])
    # trainpd = trainpd.append(new,newignore_index=True)
    

def test(epoch,graph,graph1,graph2,connection):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time=time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs,graph,graph1,graph2)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy=100.*correct/total

    end_time=time.time()
    epoch_time=end_time-start_time   
#        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    data=[epoch,accuracy,test_loss/(batch_idx+1),epoch_time]
    print('testloss:{},accuracy:{},time_used:{}'.format(test_loss/(batch_idx+1),accuracy,epoch_time))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..best_record')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'graph': graph,
            'connections': connection
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        savepath='../outputs/'+str(args.mname)+'best_check.plk'
        torch.save(state, savepath)
        best_acc = acc
    return data

nums=[]
sparses=[]

sparsity_book=[]

def extract(m):
    global sparses
    global nums
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nums.append(torch.numel(m.weight.data))
        cc=m.weight.clone().cpu()
        sparses.append(torch.mean(cc.abs()).detach().numpy())
        #print(m.weight.data)

a=[1,2,3,4]
trainnp=np.array(a)
testnp=np.array(a)
"""
for epoch in range(start_epoch, start_epoch+1):
    nd=train(epoch, graph_matrix)
    trainnp=np.vstack((trainnp,np.array(nd)))
    ed=test(epoch, graph_matrix)
    testnp=np.vstack((testnp,np.array(ed)))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]


savepath='./train/'+str(args.mname)+'train01.csv'
train_data=pd.DataFrame(trainnp,columns=['epoch','accuracy','loss','epoch_time'])
train_data.to_csv(savepath)
savepath='./train/'+str(args.mname)+'test01.csv'
test_data=pd.DataFrame(testnp,columns=['epoch','accuracy','loss','epoch_time'])
test_data.to_csv(savepath)

print('\n\nstarting training with evolution')
"""
#learning rate change
args.lr=0.1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

p1 = connection_gene.population()
p2 = connection_gene.population()
p1.create_radom()
p2.create_radom()

#def population(graph1, graph2):

def saving(graph, ac, epoch, connection, p= 'p1'):
    print('Saving..{}compare_record'.format(graph))
    state = {'net': net.state_dict(),'acc': ac,'epoch': epoch,'graph': graph, 'connections': connection}
    savepath='../outputs/'+str(p)+'compare_check.plk'
    torch.save(state, savepath)

def loading(p = 'p1'):
    savepath='../outputs/'+str(p)+'compare_check.plk'
    checkpoint = torch.load(savepath)
    net.load_state_dict(checkpoint['net'])

def backup(graph,ac, epoch, connection, p= 'init'): 
    print('Saving..{}compare_record'.format(graph))
    state = {'net': net.state_dict(),'acc': ac,'epoch': epoch,'graph': graph,'connections': connection}
    savepath='../outputs/'+str(p)+'compare_check.plk'
    torch.save(state, savepath)

best_graph=[]
best_conenctions =[]

for epoch in range(start_epoch, start_epoch+90):
    best_graph = p1.graph_matrix
    print('\ntraining individual 1')
    backup(best_graph, best_acc,epoch,p1.connections, p='init')
    nd=train(epoch,p1.graph_matrix,p1.graph2,p1.graph3,p1.connections)
    ed=test(epoch,p1.graph_matrix,p1.graph2,p1.graph3,p1.connections)
    ac_p1 = ed[1]
    saving(p1.graph_matrix,ac_p1,p1.connections,epoch,p='p1')
    print('accuracy is {}'.format(ac_p1))
    
    print('\ntraining individual 2')
    print('backroll....')
    loading(p = 'init')
    nd1=train(epoch,p2.graph_matrix,p2.graph2,p2.graph3,p2.connections)
    ed1=test(epoch,p2.graph_matrix,p2.graph2,p2.graph3,p2.connections)
    ac_p2 = ed1[1]
    saving(p2.graph_matrix,ac_p2,epoch,p2.connections,p='p2')
    print('accuracy is {}'.format(ac_p1))
    if ac_p1 > ac_p2:
        p2.connections,p2.graph_matrix,p2.graph2,p2.graph3 = p2.generate(p1.connections)
        loading(p ='p1')
        trainnp=np.vstack((trainnp,np.array(nd)))
        testnp=np.vstack((testnp,np.array(ed)))
        best_graph = p1.graph_matrix
        best_conenctions = p1.connections
        print("choosing individual p1, with graph info:{}, connection info:{}".format(p1.graph_matrix, p1.connections))
    else:
        p1.connections,p1.graph_matrix,p1.graph2,p1.graph3 = p1.generate(p2.connections)
        trainnp=np.vstack((trainnp,np.array(nd1)))
        testnp=np.vstack((testnp,np.array(ed1)))
        best_graph = p2.graph_matrix
        best_conenctions = p2.connections
        print("choosing individual p2, with graph info:{}, connection info:{}".format(p2.graph_matrix, p2.connections))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]

savepath='../outputs/'+str(args.mname)+'train01.csv'
train_data=pd.DataFrame(trainnp,columns=['epoch','accuracy','loss','epoch_time'])
train_data.to_csv(savepath)
savepath='../outputs/'+str(args.mname)+'test01.csv'
test_data=pd.DataFrame(testnp,columns=['epoch','accuracy','loss','epoch_time'])
test_data.to_csv(savepath)

print('\n\nadjust learning rate to 0.01')
#learning rate change
args.lr=0.01

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for epoch in range(90+start_epoch, 90+start_epoch+50):
    best_graph = p1.graph_matrix
    print('\ntraining individual 1')
    backup(best_graph, best_acc,epoch,p1.connections, p='init')
    nd=train(epoch,p1.graph_matrix,p1.graph2,p1.graph3,p1.connections)
    ed=test(epoch,p1.graph_matrix,p1.graph2,p1.graph3,p1.connections)
    ac_p1 = ed[1]
    saving(p1.graph_matrix,ac_p1,p1.connections,epoch,p='p1')
    print('accuracy is {}'.format(ac_p1))
    
    print('\ntraining individual 2')
    print('backroll....')
    loading(p = 'init')
    nd1=train(epoch,p2.graph_matrix,p2.graph2,p2.graph3,p2.connections)
    ed1=test(epoch,p2.graph_matrix,p2.graph2,p2.graph3,p2.connections)
    ac_p2 = ed1[1]
    saving(p2.graph_matrix,ac_p2,epoch,p2.connections,p='p2')
    print('accuracy is {}'.format(ac_p1))
    if ac_p1 > ac_p2:
        p2.connections,p2.graph_matrix,p2.graph2,p2.graph3 = p2.generate(p1.connections)
        loading(p ='p1')
        trainnp=np.vstack((trainnp,np.array(nd)))
        testnp=np.vstack((testnp,np.array(ed)))
        best_graph = p1.graph_matrix
        best_conenctions = p1.connections
        print("choosing individual p1, with graph info:{}, connection info:{}".format(p1.graph_matrix, p1.connections))
    else:
        p1.connections,p1.graph_matrix,p1.graph2,p1.graph3 = p1.generate(p2.connections)
        trainnp=np.vstack((trainnp,np.array(nd1)))
        testnp=np.vstack((testnp,np.array(ed1)))
        best_graph = p2.graph_matrix
        best_conenctions = p2.connections
        print("choosing individual p2, with graph info:{}, connection info:{}".format(p2.graph_matrix, p2.connections))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]

savepath='../outputs/'+str(args.mname)+'train01.csv'
train_data=pd.DataFrame(trainnp,columns=['epoch','accuracy','loss','epoch_time'])
train_data.to_csv(savepath)
savepath='../outputs/'+str(args.mname)+'test01.csv'
test_data=pd.DataFrame(testnp,columns=['epoch','accuracy','loss','epoch_time'])
test_data.to_csv(savepath)

print('\n\nadjust learning rate to 0.001')
#learning rate change
args.lr=0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


for epoch in range(140+start_epoch, 140+start_epoch+30):
    best_graph = p1.graph_matrix
    print('\ntraining individual 1')
    backup(best_graph, best_acc,epoch,p1.connections, p='init')
    nd=train(epoch,p1.graph_matrix,p1.graph2,p1.graph3,p1.connections)
    ed=test(epoch,p1.graph_matrix,p1.graph2,p1.graph3,p1.connections)
    ac_p1 = ed[1]
    saving(p1.graph_matrix,ac_p1,p1.connections,epoch,p='p1')
    print('accuracy is {}'.format(ac_p1))
    
    print('\ntraining individual 2')
    print('backroll....')
    loading(p = 'init')
    nd1=train(epoch,p2.graph_matrix,p2.graph2,p2.graph3,p2.connections)
    ed1=test(epoch,p2.graph_matrix,p2.graph2,p2.graph3,p2.connections)
    ac_p2 = ed1[1]
    saving(p2.graph_matrix,ac_p2,epoch,p2.connections,p='p2')
    print('accuracy is {}'.format(ac_p1))
    if ac_p1 > ac_p2:
        p2.connections,p2.graph_matrix,p2.graph2,p2.graph3 = p2.generate(p1.connections)
        loading(p ='p1')
        trainnp=np.vstack((trainnp,np.array(nd)))
        testnp=np.vstack((testnp,np.array(ed)))
        best_graph = p1.graph_matrix
        best_conenctions = p1.connections
        print("choosing individual p1, with graph info:{}, connection info:{}".format(p1.graph_matrix, p1.connections))
    else:
        p1.connections,p1.graph_matrix,p1.graph2,p1.graph3 = p1.generate(p2.connections)
        trainnp=np.vstack((trainnp,np.array(nd1)))
        testnp=np.vstack((testnp,np.array(ed1)))
        best_graph = p2.graph_matrix
        best_conenctions = p2.connections
        print("choosing individual p2, with graph info:{}, connection info:{}".format(p2.graph_matrix, p2.connections))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]

savepath='../outputs/'+str(args.mname)+'train01.csv'
train_data=pd.DataFrame(trainnp,columns=['epoch','accuracy','loss','epoch_time'])
train_data.to_csv(savepath)
savepath='../outputs/'+str(args.mname)+'test01.csv'
test_data=pd.DataFrame(testnp,columns=['epoch','accuracy','loss','epoch_time'])
test_data.to_csv(savepath)

"""
for epoch in range(120+start_epoch,120+start_epoch+20):
    nd=train(epoch)
    trainnp=np.vstack((trainnp,np.array(nd)))
    ed=test(epoch)
    testnp=np.vstack((testnp,np.array(ed)))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]

print('\n\nadjust learning rate to 0.0001')
#learning rate change
args.lr=0.0001

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

for epoch in range(140+start_epoch,140+start_epoch+10):
    nd=train(epoch)
    trainnp=np.vstack((trainnp,np.array(nd)))
    ed=test(epoch)
    testnp=np.vstack((testnp,np.array(ed)))
    net.apply(extract)
    sparsity_book.append(sparses)
    sparses=[]
    nums=[]



savepath='../outputs/'+str(args.mname)+'train02.csv'
train_data=pd.DataFrame(trainnp,columns=['epoch','accuracy','loss','epoch_time'])
train_data.to_csv(savepath)
savepath='../outputs/'+str(args.mname)+'test02.csv'
test_data=pd.DataFrame(testnp,columns=['epoch','accuracy','loss','epoch_time'])
test_data.to_csv(savepath)

sve=np.array(sparsity_book)
#print(sve)
savepath='../outputs/'+str(args.mname)+'sparsity_book.csv'
sparsebook=pd.DataFrame(sve)
sparsebook.to_csv(savepath)
"""
net.apply(extract)

print('param nums:',nums)
print('best accuracy is :{}'.format(best_acc))
print('best conenctions is {}'.format(best_conenctions))
