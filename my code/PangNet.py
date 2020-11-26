
import torch.nn as nn
import torch
from tensorboardX import *
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from tensorboardX import SummaryWriter

'''
modified to fit dataset size
'''
NUM_CLASSES = 10
eps = 1e-5
#writter = SummaryWriter(log_dir='Scalar', comment='MyNet')

class PangNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, init_weights = True):
        super(PangNet, self).__init__()
        self.features = nn.Sequential(
            #尺寸变化公式：N=(W-F+2P)/S+1,N为输出大小,W为输入大小,F为卷积核大小,P为填充值大小,S为步长大小
            nn.BatchNorm2d(3, eps=eps),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=eps),
            # 3*32*32->64*16*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # stride默认为kernel_size=2
            # 64*16*16->64*8*8
            nn.BatchNorm2d(64, eps=eps),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            # 64*8*8->192*8*8
            nn.BatchNorm2d(256, eps=eps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 192*8*8->192*4*4
            nn.BatchNorm2d(256, eps=eps),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # 192*4*4->384*4*4
            nn.BatchNorm2d(512, eps=eps),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 384*4*4->256*4*4
            nn.BatchNorm2d(1024, eps=eps),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 256*4*4->256*4*4
            nn.BatchNorm2d(512, eps=eps),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024, eps=eps),

            nn.AvgPool2d(kernel_size=2),

            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            #
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, groups=2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            #
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, groups=2),
            # nn.BatchNorm2d(512),
            # nn.ReLU(True),
            #
            # nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, groups=2),
            # nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            #
            # nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True)



        )

        self.classifier = nn.Sequential(
            #nn.BatchNorm2d(4096, eps=eps),
            # nn.ReLU(inplace=True),
            #nn.BatchNorm2d(4096, eps=eps),
            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            # nn.ReLU(inplace=True),
            nn.Linear(2048, NUM_CLASSES),
            )

        if init_weights:
            self._initialize_weights()


    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), 1024)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

