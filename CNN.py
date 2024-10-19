# pylint: disable=no-member
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
import matplotlib.pyplot as plt

classes = ('0' , '1')
batch_size = 2   
# 0 represents health, 1 represents patient


# This CNN structure comes from Pytorch Tutorial
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size = 5, stride = 2,  padding=0)    # 
        self.pool = nn.MaxPool3d((2, 2, 2),stride=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size = (5,5,5), stride = 2,  padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size = (5,5,5), stride = 2,  padding=0)   #kernel size, 
        self.conv4 = nn.Conv3d(32, 64, kernel_size = (5,5,5), stride = 2,  padding=0)   #kernel size, 
#        self.conv5 = nn.Conv3d(64, 64, kernel_size = (5,5,5), stride = 1,  padding=0)   #kernel size, 
        self.fc1 = nn.Linear(int(25088/batch_size), 500)       
        self.fc2 = nn.Linear(500, 120)       
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        xSize = np.array(x.size())
        xD = xSize[1]
        xH = xSize[2]
        xW = xSize[3]
        x = x.reshape(batch_size, 1, xD, xH, xW)
#        print('x size is',x.size())
        x = self.pool((F.relu(self.conv1(x.float()))))
        x = self.pool((F.relu(self.conv2(x))))
        x = self.pool((F.relu(self.conv3(x))))
        x = self.pool((F.relu(self.conv4(x))))
#        print('x size is',x.size())
#        x = self.pool((F.relu(self.conv5(x))))
        x = x.view(batch_size, int(25088/batch_size))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x

net = Net()
# ===================================================================================
# loss function
criterion = torch.nn.CrossEntropyLoss()  
# optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0005)  
