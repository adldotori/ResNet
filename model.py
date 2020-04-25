import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else None

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(out.shape, x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return self.relu2(out)

class ResNet_CIFAR10(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = [BasicBlock(16, 16, 1)]
        for _ in range(self.n-1):
            self.conv2.append(BasicBlock(16, 16, 1))
        self.conv3 = [BasicBlock(16, 32, 2)]
        for _ in range(self.n-1):
            self.conv3.append(BasicBlock(32, 32, 1))
        self.conv4 = [BasicBlock(32, 64, 2)]
        for _ in range(self.n-1):
            self.conv4.append(BasicBlock(64, 64, 1))

        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Sequential(*self.conv3)
        self.conv4 = nn.Sequential(*self.conv4)

        self.gap = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, 10)

        self.apply(_weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    batch_size = 4

    model = ResNet_CIFAR10(3).cuda()

    x = torch.randn(batch_size, 3, 32, 32).cuda()
    ret = model(x)
    print(ret.shape)
