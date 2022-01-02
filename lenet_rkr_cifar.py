import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
import pdb
import torch.nn.functional as F
import re, random, collections
import pickle

rd=3473 #torch.randint(1,10000,[1])
torch.manual_seed(rd)
torch.cuda.manual_seed_all(rd)


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]='8'


class args():


    data_path = "./Datasets/CIFAR100/"
    num_class = 100
    class_per_task = 10
    num_task = 10
    test_samples_per_class = 100
    dataset = "cifar100"
    optimizer = "radam"
    train_batch = 128
    test_batch = 100
    workers = 16
    
    random_classes = False
    validation = 0
    overflow=False
    
    n_classes=10
    batch_size=128
    lr=0.01
    resume=False
    total_epoch=100
    model_path='ckpt/lenet_rkr_cifar/'
    
args=args()


import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.optim.lr_scheduler import MultiStepLR
torch.set_printoptions(precision=5,sci_mode=False)


import torch
import torch.nn as nn
import torch.nn.functional as F

######################################### 
class conv_dw(nn.Module):
    def __init__(self, inp, oup):
        super(conv_dw, self).__init__()
        self.dwc = nn.AdaptiveAvgPool2d(1) 
        self.dwc1 = nn.Conv2d(inp, oup, 1, 1, 0, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)         
        self.sig2 = nn.Sigmoid()             
    def forward(self, x): 
        x = self.bn1(self.dwc1(self.dwc(x)))        
        x = self.sig2(x)
        return x
    
class Netm(nn.Module):
    def __init__(self):
        super(Netm, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2, bias=False)
        self.padding = 2
        self.conv1_wt1 = nn.Parameter(torch.zeros(3*5,2))
        self.conv1_wt2 = nn.Parameter(torch.zeros(2,5*20))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=2, bias=False)
        self.conv2_wt1 = nn.Parameter(torch.zeros(20*5,2))
        self.conv2_wt2 = nn.Parameter(torch.zeros(2,5*50))
        self.calib1 = conv_dw(20, 20)
        self.calib2 = conv_dw(50, 50) 
        self.fc1 = nn.Linear(8*8*50, 800)
        self.emb1=nn.Embedding(1,800)
        self.fc2 = nn.Linear(800, 500)
        self.emb2=nn.Embedding(1,500)
        self.fc3 = nn.Linear(500, 100)

        self.fc1_wt1 = nn.Parameter(torch.zeros(8*8*50,2))
        self.fc1_wt2 = nn.Parameter(torch.zeros(2,800))

        self.fc2_wt1 = nn.Parameter(torch.zeros(800,2))
        self.fc2_wt2 = nn.Parameter(torch.zeros(2,500))

        nn.init.xavier_uniform_(self.conv1_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.conv1_wt2,gain=0.5)
        nn.init.xavier_uniform_(self.conv2_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.conv2_wt2,gain=0.5)

        nn.init.xavier_uniform_(self.fc1_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.fc1_wt2,gain=0.5)
        nn.init.xavier_uniform_(self.fc2_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.fc2_wt2,gain=0.5)
         
        
    def forward(self, x):
        h1 = self.conv1(x)
        conv1_wt = torch.mm(self.conv1_wt1, self.conv1_wt2)
        h2 = F.conv2d(x,conv1_wt.view(self.conv1.weight.size()),padding=self.padding)
        out = h1 + h2
        y = self.calib1(out)        
        out = out*y        
        out = F.relu(out)
        out = self.pool(out)
        
        h1 = self.conv2(out)
        conv2_wt = torch.mm(self.conv2_wt1, self.conv2_wt2)
        h2 = F.conv2d(out,conv2_wt.view(self.conv2.weight.size()),padding=self.padding)
        out = h1 + h2

        y = self.calib2(out)        
        out = out*y        
        out = F.relu(out)
        x = self.pool(out)        
           
        x = x.view(x.size(0), -1)
        h1=self.fc1(x)
        fc1_wt = torch.mm(self.fc1_wt1, self.fc1_wt2)
        h2 = torch.mm(x,fc1_wt)
        x=h1+h2

        x=x*self.emb1(torch.cuda.LongTensor([0]))
        x = F.relu(x)
        h1=self.fc2(x)
        fc2_wt = torch.mm(self.fc2_wt1, self.fc2_wt2)
        h2 = torch.mm(x,fc2_wt)
        x=h1+h2

        x=x*self.emb2(torch.cuda.LongTensor([0]))
        x = F.relu(x)
        x = self.fc3(x)
        return x

def NetO():
    return Netm()


def save_model(task,acc,model,order):
    if(type(task)==torch.Tensor):
        task = task.item()
    statem = {
        'net': model.state_dict(),
        'acc': acc,
    }
    fname=args.model_path+str(order.tolist())
    if not os.path.isdir(fname):
        os.makedirs(fname)
    torch.save(statem, fname+'/ckpt_task'+str(task)+'.pth')

        
def load_model(task,model,order):
    fname=args.model_path+str(order.tolist())
    if(type(task)==torch.Tensor):
        task = task.item()
    checkpoint = torch.load(fname+'/ckpt_task'+str(task)+'.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    return best_acc



def train(train_loader,epoch,task,model):
    model.train()
    global best_acc
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets=targets-task*10
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs=outputs[:,task*args.n_classes+np.array(range(args.n_classes))]
        if task>=1:
            loss = criterion(outputs, targets)
        else:  
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total




def test(test_loader,task,model,order):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets=targets-task*10
            outputs = model(inputs)
            outputs=outputs[:,task*args.n_classes+np.array(range(args.n_classes))]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total

    

    save_model(task,acc,model,order)
    save_rect(task,acc,model,order)
    return acc


def testnosave(test_loader,task,model,order):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets=targets-task*10
            outputs = model(inputs)
            outputs=outputs[:,task*args.n_classes+np.array(range(args.n_classes))]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc



def grad_false(model):
    gradf=[8,9,16,17,19,20] # Frozen parameters

    i=0
    for p in model.parameters():
        if i in gradf:
            p.requires_grad=False
        i=i+1


def save_rect(task,acc,model,order):
    if(type(task)==torch.Tensor):
        task = task.item()
    fname=args.model_path+str(order.tolist())
    gradf=[8,9,22,23,25,26] # order of frozen parameters is different in the state dictionary than in the list of parameters
    mdm=model.state_dict()
    statem = []
    i=0
    for p in mdm.keys():
        if i not in gradf:
            statem.append(mdm[p].data)
        else:
            statem.append([])
        i=i+1


    pickle.dump(statem, open(fname+'/ckpt_task'+str(task)+'_rect.pkl','wb'))


def load_rect(task,model,order):
    if(type(task)==torch.Tensor):
        task = task.item()
    fname=args.model_path+str(order.tolist())
    statem=pickle.load(open(fname+'/ckpt_task'+str(task)+'_rect.pkl','rb'))
      
    gradf=[8,9,22,23,25,26]
    i=0
    mdm=model.state_dict()
    for p in mdm.keys():
        if i not in gradf:
            assert mdm[p].shape==statem[i].shape
            mdm[p]=statem[i]
        i=i+1

    model.load_state_dict(mdm)

    


import incremental_dataloader as data
inc_dataset = data.IncrementalDataset(
                                dataset_name=args.dataset,
                                args = args,
                                random_order=args.random_classes,
                                shuffle=True,
                                seed=1,
                                batch_size=args.train_batch,
                                workers=args.workers,
                                validation_split=args.validation,
                                increment=args.class_per_task,
                            )
task_data=[]
for i in range(10):
    task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()
    task_data.append([train_loader,test_loader])



all_task_acc=[]

import tqdm
for ntask in range(1):
    task_acc=[]
    order=torch.tensor([0,1,2,3,4,5,6,7,8,9])#torch.randperm(10)
    f=1
    for task in tqdm.tqdm(order):
        task=task.item()

        if task==order[0]:

            train_loader, test_loader = task_data[task][0],task_data[task][1]
            modelm=NetO().cuda()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(modelm.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
            schedulerG = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
            
            for epoch in tqdm.tqdm(range(args.total_epoch)):
                train(train_loader,epoch,task,modelm)
                
                schedulerG.step()
            
            acc1=test(test_loader,task,modelm,order)


        if task!=order[0]:
            train_loader, test_loader = task_data[task][0],task_data[task][1]
            modelm=NetO().cuda()
            acc1=load_model(prev_task,modelm,order)  
            grad_false(modelm)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(modelm.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
            schedulerG = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
            
            for epoch in range(args.total_epoch):
                train(train_loader,epoch,task,modelm)
                
                schedulerG.step()

            acc1=test(test_loader,task,modelm,order)
        prev_task=task
        task_acc.append(acc1)
        print('Task: '+str(task)+'  Test_accuracy: '+ str(acc1))

    all_task_acc.append(task_acc)

print("Final Session Accuracy")
sess_acc=[]
modelm=NetO().cuda()
load_model(order[0],modelm,order)  
for task in order:
    task=task.item()
    load_rect(task,modelm,order)
    test_loader = task_data[task][1]
    acc=testnosave(test_loader,task,modelm,order)
    sess_acc.append(acc)

print(sum(sess_acc)/len(sess_acc))

