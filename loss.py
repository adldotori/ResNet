import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.loss(x,y)

if __name__ == '__main__':
    batch_size = 3

    x = torch.randn(batch_size, 10).cuda()
    y = torch.empty(batch_size, dtype=torch.long).random_(10).cuda()
    print(x.shape, y.shape)
    criterion = Loss()
    loss = criterion(x, y)

    print(loss.shape)
    print(loss)
