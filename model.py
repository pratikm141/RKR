import torch
import torch.nn as nn
import torch.nn.functional as F


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

class shortct(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(shortct, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False) 
        self.stride = stride
        ##### Rectification Weight Matrices
        self.conv1_wt1 = nn.Parameter(torch.zeros(inp*1,2))
        self.conv1_wt2 = nn.Parameter(torch.zeros(2,1*oup))
        #####
        self.bn1 = nn.BatchNorm2d(oup)

        #### Scaling Factors
        self.calib1 = conv_dw(oup, oup)
        ####

        nn.init.xavier_uniform_(self.conv1_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.conv1_wt2,gain=0.5)

    def forward(self, x):
        h1 = self.conv1(x)
        ##### Rectification Weight
        conv1_wt = torch.mm(self.conv1_wt1, self.conv1_wt2)

        h2 = F.conv2d(x,conv1_wt.view(self.conv1.weight.size()),stride=self.stride)
        out = h1 + h2

        out = self.bn1(out)
        y = self.calib1(out)        
        out = out*y        
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.stride = stride

        ##### Rectification Weight Matrices
        self.conv1_wt1 = nn.Parameter(torch.zeros(in_planes*3,2))
        self.conv1_wt2 = nn.Parameter(torch.zeros(2,3*planes))
        #####

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        ##### Rectification Weight Matrices
        self.conv2_wt1 = nn.Parameter(torch.zeros(planes*3,2))
        self.conv2_wt2 = nn.Parameter(torch.zeros(2,3*planes))
        #####
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = shortct(in_planes, self.expansion*planes, stride)

        ##### Scaling Factors
        self.calib1 = conv_dw(planes, planes)
        self.calib2 = conv_dw(planes, planes) 
        #####

        nn.init.xavier_uniform_(self.conv1_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.conv1_wt2,gain=0.5)
        nn.init.xavier_uniform_(self.conv2_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.conv2_wt2,gain=0.5)

    def forward(self, x):
        import pdb
        h1 = self.conv1(x)
        ##### Rectification Weight
        conv1_wt = torch.mm(self.conv1_wt1, self.conv1_wt2)

        h2 = F.conv2d(x,conv1_wt.view(self.conv1.weight.size()),padding=1,stride=self.stride)
        out = h1 + h2
 
        out = self.bn1(out)
        ##### Scaling
        y = self.calib1(out)        
        out = out*y        
        out = F.relu(out)

        h1 = self.conv2(out)
        ##### Rectification Weight
        conv2_wt = torch.mm(self.conv2_wt1, self.conv2_wt2)

        h2 = F.conv2d(out,conv2_wt.view(self.conv2.weight.size()),padding=1)
        out = h1 + h2

        out = self.bn2(out)
        ##### Scaling
        y = self.calib2(out)        
        out = out*y 
        out += self.shortcut(x)
        out = F.relu(out)
        return out


    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        ##### Rectification Weight Matrices
        self.conv1_wt1 = nn.Parameter(torch.zeros(3*3,2))
        self.conv1_wt2 = nn.Parameter(torch.zeros(2,3*64))
        #####
        self.bn1 = nn.BatchNorm2d(64)

        ##### Scaling Factor
        self.calib1 = conv_dw(64, 64)
        #####

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        nn.init.xavier_uniform_(self.conv1_wt1,gain=0.5)
        nn.init.xavier_uniform_(self.conv1_wt2,gain=0.5)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        h1 = self.conv1(x)
        ##### Rectification Weight
        conv1_wt = torch.mm(self.conv1_wt1, self.conv1_wt2)

        h2 = F.conv2d(x,conv1_wt.view(self.conv1.weight.size()),padding=1)
        out = h1 + h2

        out = self.bn1(out)
        ##### Scaling
        y = self.calib1(out)     
   
        out = out*y        
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def ResNet18_RKR():
    return ResNet(BasicBlock, [2,2,2,2])

print(ResNet18_RKR())
