import torch.nn as nn
import torch
'''
modified to fit dataset size
'''
NUM_CLASSES = 10
eps = 1e-5

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            #尺寸变化公式：N=(W-F+2P)/S+1,N为输出大小,W为输入大小,F为卷积核大小,P为填充值大小,S为步长大小
            nn.BatchNorm2d(3, eps=eps),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=eps),
            #3*32*32->64*16*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),# stride默认为kernel_size=2
            #64*16*16->64*8*8
            nn.BatchNorm2d(64, eps=eps),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            #64*8*8->192*8*8
            nn.BatchNorm2d(256, eps=eps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            #192*8*8->192*4*4
            nn.BatchNorm2d(256, eps=eps),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #192*4*4->384*4*4
            nn.BatchNorm2d(512, eps=eps),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            #384*4*4->256*4*4
            nn.BatchNorm2d(1024, eps=eps),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            #256*4*4->256*4*4
            nn.BatchNorm2d(512, eps=eps),
            nn.Conv2d(512, 256, kernel_size=3,  stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=eps),
            nn.AvgPool2d(kernel_size=2),



        )
        self.classifier = nn.Sequential(
            #nn.BatchNorm2d(4096, eps=eps),
            # nn.ReLU(inplace=True),
            #nn.BatchNorm2d(4096, eps=eps),
            nn.Linear(256, 2048),
            nn.Dropout(),
            # nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=NUM_CLASSES):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             #尺寸变化公式：N=(W-F+2P)/S+1,N为输出大小,W为输入大小,F为卷积核大小,P为填充值大小,S为步长大小
#             nn.BatchNorm2d(3, eps=eps),
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64, eps=eps),
#             #3*32*32->64*16*16
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),# stride默认为kernel_size=2
#             #64*16*16->64*8*8
#             nn.BatchNorm2d(64, eps=eps),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             #64*8*8->192*8*8
#             nn.BatchNorm2d(192, eps=eps),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             #192*8*8->192*4*4
#             nn.BatchNorm2d(192, eps=eps),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             #192*4*4->384*4*4
#             nn.BatchNorm2d(384, eps=eps),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             #384*4*4->256*4*4
#             nn.BatchNorm2d(256, eps=eps),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             #256*4*4->256*4*4
#             nn.BatchNorm2d(256, eps=eps),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             #256*4*4->256*2*2
#         )
#         self.classifier = nn.Sequential(
#
#             nn.Linear(256 * 2 * 2, 4096),
#             #nn.BatchNorm2d(4096, eps=eps),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 4096),
#             #nn.BatchNorm2d(4096, eps=eps),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 2 * 2)
#         x = self.classifier(x)
#         return x
# import torch.nn as nn
# num_classes = 10
# #
# #
# class AlexNet(nn.Module):
#     def __init__(self, num_classes = num_classes):
#         super(AlexNet, self).__init__()
#         self.feaures = nn.Sequential(
#             #从224*224*3 -> 55* 55*96
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(stride=2, kernel_size=3),
#             #从55*55*48 -> 27*27*256
#             nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(stride=2, kernel_size=3),
#             #从27*27*128 -> 13*13*384
#             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
#             nn.ReLU(inplace=True),
#             #从13*13*384 -> 13*13*256  生成图像为256*256的
#             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(in_features=256 * 13 * 13, out_features=4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
# def forward(self, x):
#     x = self.feaures(x)
#     x = x.reshape(x.size(0), 256 * 13 * 13)
#     x = self.classifier(x)
#     return x
