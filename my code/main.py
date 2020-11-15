import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn
from torchvision import transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
import argparse

from AlexNet import AlexNet
from models import *
from misc import progress_bar


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')# 数据集中的类别标签
train_Loss_list = []# 初始化
test_Loss_list = []
train_acc_list = []
test_acc_list = []

def main():
    # 定义命令行和参数解析器三步走：1）创建对象 2）添加参数 3）解析参数
    # 创建ArgumentParser()对象-
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    #添加参数
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')#学习率
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')#训练多少次
    parser.add_argument('--trainBatchSize', default=512, type=int, help='training batch size')#每次训练取多少张
    parser.add_argument('--testBatchSize', default=512, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')#cuda加速
    #实例化，把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()
    solver.plot_result(train_Loss_list,test_Loss_list,train_acc_list,test_acc_list)


class Solver(object):
    def __init__(self, config):#初始化
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
#===============================================================================================================导入数据
    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])#train数据转tensor
        test_transform = transforms.Compose([transforms.ToTensor()])#test数据转tensor
        train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=train_transform)#true表示载入train数据
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)#false表示载入test数据
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
#===============================================================================================================导入网络
    def load_model(self):
        #有cuda就加速，没有拉倒
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #self.model = LeNet().to(self.device)
        self.model = AlexNet().to(self.device)
        #self.model = resnet101().to(self.device)
        #self.model = resnet50().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)#优化器
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.1)#动态调整学习率
        self.criterion = nn.CrossEntropyLoss().to(self.device) #多任务分类训练常用的损失函数，交叉熵损失函数

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            '''
            print(data,data.shape)
            print(target,target.shape)
            print(output,output.shape)
            print(prediction[0],prediction[0].shape)
            print(prediction[1],prediction[1].shape)
            print('--------------------------------------------------------------------------------------')
            '''
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()#获取train中的权重
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    #update plot
    def plot_result(self,train_loss,test_loss,train_acc,test_acc):
        x1 = range(0, 200)
        x2 = range(0, 200)
        x3 = range(0, 200)
        x4 = range(0, 200)
        y1 = train_loss
        y2 = test_loss
        y3 = train_acc
        y4 = test_acc

        plt.subplot(2, 2, 1)
        plt.plot(x1, y1, 'o-')
        plt.title('Train Loss for epochs')
        plt.ylabel('Train Loss')

        plt.subplot(2, 2, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Test Loss for epochs')
        plt.ylabel('Test loss')

        plt.subplot(2, 2, 3)
        plt.plot(x3, y3, 'o-')
        plt.xlabel('Train Acc for epochs')
        plt.ylabel('Train Acc(%)')

        plt.subplot(2, 2, 4)
        plt.plot(x4, y4, '.-')
        plt.xlabel('Test Acc for epochs')
        plt.ylabel('Test Acc(%)')

        plt.show()
        plt.savefig("loss_accuracy.jpg")

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):#遍历200次，取左不取右
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            print(test_result)
            train_Loss_list.append(train_result[0])
            test_Loss_list.append(test_result[0])
            train_acc_list.append(train_result[1]*100)
            test_acc_list.append(test_result[1]*100)
            accuracy = max(accuracy, test_result[1]) #更新准确率
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()
