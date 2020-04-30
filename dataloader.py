from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

class CIFAR10Dataset(data.Dataset):
    def __init__(self, mode):
        super(CIFAR10Dataset, self).__init__()
        if mode == 'train':
            self.dataset = datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]))
        else:
            self.dataset = datasets.CIFAR10('./data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])) 

    def name(self):
        return "CIFAR10Dataset"

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

class CIFAR10Dataloader(object):
    def __init__(self, mode, opt, dataset):
        super(CIFAR10Dataloader, self).__init__()
        use_cuda = not torch.cuda.is_available()
        kwargs = {'num_workers': opt.num_workers} if use_cuda else {}

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default = 128)
    opt = parser.parse_args()

    train_dataset = CIFAR10Dataset('train')
    train_data_loader = CIFAR10Dataloader('train', opt, train_dataset)

    test_dataset = CIFAR10Dataset('test')
    test_data_loader = CIFAR10Dataloader('test', opt, test_dataset)

    print('[+] Size of the train dataset: %05d, train dataloader: %04d' \
        % (len(train_dataset), len(train_data_loader.data_loader)))   
    print('[+] Size of the test dataset: %05d, test dataloader: %04d' \
        % (len(test_dataset), len(test_data_loader.data_loader)))