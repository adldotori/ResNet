import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from model import *
from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default = 128)
    parser.add_argument('-t', '--test-batch-size', type=int, default = 1)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()
    return opt

def train(opt):
    model = ResNet_CIFAR10(9).cuda()
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.train()

    train_dataset = CIFAR10Dataset('train')
    train_data_loader = CIFAR10Dataloader('train', opt, train_dataset)

    test_dataset = CIFAR10Dataset('test')
    test_data_loader = CIFAR10Dataloader('test', opt, test_dataset)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    criterion = Loss()

    writer = SummaryWriter()

    for epoch in range(opt.epoch):
        for i in range(len(train_data_loader.data_loader)):
            step = epoch * len(train_data_loader.data_loader) + i + 1

            # load data
            image, label = train_data_loader.next_batch()
            image = image.cuda()
            label = label.cuda()

            # train model
            optim.zero_grad()
            result = model(image)
            loss = criterion(result, label)
            loss.backward()
            optim.step()

            writer.add_scalar('loss', loss, step)
            writer.add_images('image', image, step, dataformats="NCHW")
            
            writer.close()

            if step % opt.display_step == 0:
                _, predicted = torch.max(result, 1)
                total = label.size(0)
                correct = (predicted == label).sum().item()
                total_test = 0
                correct_test = 0
                for i in range(len(test_data_loader.data_loader)):
                    image, label = test_data_loader.next_batch()
                    image = image.cuda()
                    label = label.cuda()
                    result = model(image)
                    _, predicted = torch.max(result, 1)
                    total_test += label.size(0)
                    correct_test += (predicted == label).sum().item()
                print('[Epoch {}] Loss : {:.2}, train_acc : {:.2}, test_acc : {:.2}'.format(epoch, loss, correct/total, correct_test/total_test))
        
        torch.save(model.state_dict(), 'checkpoint.pt')

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    opt = get_opt()
    train(opt)