
import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
# from data.load_data import CHARS

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2 
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, random_split

class CPMStage_x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPMStage_x, self).__init__()

        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        # self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(out_channels, out_channels*2, kernel_size=9, padding=4)
        self.bn4 = nn.BatchNorm2d(out_channels*2)
        self.relu4 = nn.ReLU(inplace=True)
                
        self.conv5 = nn.Conv2d(out_channels*2, out_channels, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):

        # x = self.relu2(self.bn2(self.conv2(x)))

        x = self.relu3(self.bn3(self.pool3(self.conv3(x))))

        x = self.relu4(self.bn4(self.conv4(x)))

        x = self.relu5(self.bn5(self.conv5(x)))
        
        return x
    
class CPMStage_g1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPMStage_g1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels//2, in_channels//4, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels//4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        return x
    
class CPMStage_g2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPMStage_g2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels//2, out_channels*2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        return x


class CPMLicensePlateNet(nn.Module):
    def __init__(self, num_stages=6):
        super(CPMLicensePlateNet, self).__init__()
        #self.stages = nn.ModuleList([CPMStage(3 + 4 * i, 32) for i in range(num_stages)])
        
        self.CPMStage_x_1 = CPMStage_x(3,32)
        self.CPMStage_g1_1 = CPMStage_g1(32,4)

        self.CPMStage_x_2 = CPMStage_x(3,32)
        self.CPMStage_g2_2 = CPMStage_g2(36,4)

        self.CPMStage_x_3 = CPMStage_x(3,32)
        self.CPMStage_g2_3 = CPMStage_g2(36,4)



    def forward(self, x):
        #s1
        s1_x_out =self.CPMStage_x_1(x)
        s1_g1_out =self.CPMStage_g1_1(s1_x_out)

        #s2
        s2_x_out =self.CPMStage_x_2(x)
        s2_cat_out = torch.cat([s1_g1_out, s2_x_out], dim=1)
        s2_g2_out =self.CPMStage_g2_2(s2_cat_out)


        return s1_g1_out,  s2_g2_out