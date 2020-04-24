import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResNet_CIFAR10(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = []
        for _ in range(self.n-1):
            self.conv2.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.conv2.append(nn.Conv2d(16, 16, 3, 1, padding=1))
        self.conv2.append(nn.Conv2d(16, 16, 3, 1, padding=1))
        self.conv2.append(nn.Conv2d(16, 32, 3, 2, padding=1))
        self.conv3 = []
        for _ in range(self.n-1):
            self.conv3.append(nn.Conv2d(32, 32, 3, 1, padding=1))
            self.conv3.append(nn.Conv2d(32, 32, 3, 1, padding=1))
        self.conv3.append(nn.Conv2d(32, 32, 3, 1, padding=1))
        self.conv3.append(nn.Conv2d(32, 64, 3, 2, padding=1))
        self.conv4 = []
        for _ in range(self.n-1):
            self.conv4.append(nn.Conv2d(64, 64, 3, 1, padding=1))
            self.conv4.append(nn.Conv2d(64, 64, 3, 1, padding=1))
        self.conv4.append(nn.Conv2d(64, 64, 3, 1, padding=1))
        self.conv4.append(nn.Conv2d(64, 64, 3, 2, padding=1))

        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Sequential(*self.conv3)
        self.conv4 = nn.Sequential(*self.conv4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
if __name__ == '__main__':
    batch_size = 4

    model = ResNet_CIFAR10(9).cuda()

    x = torch.randn(batch_size, 3, 32, 32).cuda()
    ret = model(x)
    print(ret.shape)
