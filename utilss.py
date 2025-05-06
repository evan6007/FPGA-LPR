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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=4)
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
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.relu2(self.bn2(self.conv2(x)))

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

class CPMStage_g3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPMStage_g3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

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

        self.CPMStage_x_4 = CPMStage_x(3,32)
        self.CPMStage_g2_4 = CPMStage_g2(36,4)

        self.CPMStage_x_5 = CPMStage_x(3,32)
        self.CPMStage_g2_5 = CPMStage_g2(36,4)

        self.CPMStage_x_6 = CPMStage_x(3,32)
        self.CPMStage_g2_6 = CPMStage_g2(36,4)

        self.CPMStage_x_7 = CPMStage_x(3,32)
        self.CPMStage_g2_7 = CPMStage_g2(36,4)



    def forward(self, x):
        #s1
        s1_x_out =self.CPMStage_x_1(x)
        s1_g1_out =self.CPMStage_g1_1(s1_x_out)

        #s2
        s2_x_out =self.CPMStage_x_2(x)
        s2_cat_out = torch.cat([s1_g1_out, s2_x_out], dim=1)
        s2_g2_out =self.CPMStage_g2_2(s2_cat_out)


        #s3
        s3_x_out = self.CPMStage_x_3(x)
        s3_cat_out = torch.cat([s2_g2_out, s3_x_out], dim=1)
        s3_g2_out =self.CPMStage_g2_3(s3_cat_out)

        #s4
        s4_x_out =self.CPMStage_x_4(x)
        s4_cat_out = torch.cat([s3_g2_out, s4_x_out], dim=1)
        s4_g2_out =self.CPMStage_g2_4(s4_cat_out)

        #s5
        s5_x_out =self.CPMStage_x_5(x)
        s5_cat_out = torch.cat([s4_g2_out, s5_x_out], dim=1)
        s5_g2_out =self.CPMStage_g2_5(s5_cat_out)

        #s6
        s6_x_out =self.CPMStage_x_6(x)
        s6_cat_out = torch.cat([s5_g2_out, s6_x_out], dim=1)
        s6_g2_out =self.CPMStage_g2_6(s6_cat_out)

        s7_x_out =self.CPMStage_x_7(x)
        s7_cat_out = torch.cat([s6_g2_out, s7_x_out], dim=1)
        s7_g2_out =self.CPMStage_g2_7(s7_cat_out)


        return s1_g1_out,  s2_g2_out,  s3_g2_out,  s4_g2_out,  s5_g2_out,  s6_g2_out,  s7_g2_out


# Instantiate the network with 6 stages
model = CPMLicensePlateNet(num_stages=6)


import torch.nn as nn
import torch

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0

            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool2d(kernel_size=3, stride=(1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),# 7
            small_basic_block(ch_in=128, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)), # 14 倒數第三
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=37, out_channels=37, kernel_size=(4, 1), stride=1, padding=(0, 0), groups=37)
        )
        
        
        self.c0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(9, 9), stride=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(5, 3), stride=(1, 2)),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(9, 9), stride=(1, 2)),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(5, 3), stride=(1, 2)),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=128),

        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(9, 5), stride=(1, 2)),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(5, 3), stride=(1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=256),
        )


        self.l1 = nn.Sequential(nn.Conv2d(37,37,kernel_size=(25,1),stride=1))
    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            # if i in [20,22]:
            #     # print(x.shape)

            if i in [2, 6, 13, 22]: 
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            # if i in [3]:
                # print("f.shape",f.shape)
            #print(f"Tensor {i} size: {f.size()}")
            
            if i in [0]:
                # print("f1.shape",f.shape)
                f = self.c0(f)
                # print("f1轉.shape",f.shape)
            if i in [1]:
                # print("f2.shape",f.shape)
                f = self.c1(f)
                # print("f2轉.shape",f.shape)
            if i in [2]:
                # print("f3.shape",f.shape)
                f = self.c2(f)
                # print("f3轉.shape",f.shape)
            """
            if i in [1]:
                f = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 5))(f)
                f = nn.AvgPool2d(kernel_size=(16, 1), stride=(1, 1))(f)
                
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 2))(f)
                f = nn.AvgPool2d(kernel_size=(14, 5), stride=(1, 1))(f)
            """

            global_context.append(f)
            #print(f"後Tensor {i} size: {f.size()}")

        x = torch.cat(global_context, 1)
        # print("x1=",x.shape) #torch.Size([64, 485, 28, 18])
        x = self.container(x)
        # print("container=",x.shape) #torch.Size([64, 37, 25, 18])
        x = self.l1(x)
        # print("x3=",x.shape) #torch.Size([64, 37, 1, 18])
        x = x.reshape(x.shape[0],37,18)
        # print("x.shape",x.shape)
        

        

        return x


def build_lprnet(lpr_max_len=7, phase=False, class_num=37, dropout_rate=0.5):

    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
