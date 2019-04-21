from __future__ import print_function
import torch
import torch.utils.data as Data
import torchvision

DOWNLOAD_MNIST = False
DOWNLOAD_Fashion_MNIST = False
DOWNLOAD_CIFAR10 = False

class DataLoader():
    def __init__(self, dataset, BATCH_SIZE):
        self.dataset = dataset
        self.BATCH_SIZE = BATCH_SIZE

    def MNIST(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        train_data = torchvision.datasets.MNIST(
            root='data/MNIST',
            download=DOWNLOAD_MNIST,
            transform=transform,
            train=True
        )  # (60000, 28, 28)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.MNIST(
            root='data/MNIST',
            train=False,
            download=DOWNLOAD_MNIST,
            transform=transform,
        )  # (10000, 28, 28)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader, train_data, test_data

    def Fashion_MNIST(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        train_data = torchvision.datasets.FashionMNIST(
            root='data/Fashion_MNIST',
            download=DOWNLOAD_Fashion_MNIST,
            transform=transform,
            train=True
        )  # (60000, 28, 28)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.FashionMNIST(
            root='data/Fashion_MNIST',
            train=False,
            download=DOWNLOAD_Fashion_MNIST,
            transform=transform,
        )  # (10000, 28, 28)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader, train_data, test_data

    def CIFAR10(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),  # to 0-1
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1-1
        ])
        train_data = torchvision.datasets.CIFAR10(
            root='data/CIFAR10',
            download=DOWNLOAD_CIFAR10,
            transform=transform,
            train=True
        )  # (60000, 28, 28)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2,  # how many child process to load data
        )
        test_data = torchvision.datasets.CIFAR10(
            root='data/CIFAR10',
            train=False,
            download=DOWNLOAD_CIFAR10,
            transform=transform,
        )  # (10000, 32, 32)
        test_loader = Data.DataLoader(
            dataset=test_data,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2,  # how many child process to load data
        )
        return train_loader, test_loader, train_data, test_data


    def load(self): # type, form of transform
        if self.dataset == 'MNIST':
            train_loader, test_loader, _, _ = self.MNIST()
            return train_loader, test_loader
        elif self.dataset == 'Fashion_MNIST':
            train_loader, test_loader, _, _ = self.Fashion_MNIST()
            return train_loader, test_loader
        elif self.dataset == 'CIFAR10':
            train_loader, test_loader, _, _ = self.CIFAR10()
            return train_loader, test_loader
        else:
            print('Dataset name is error!')

    def data(self):
        if self.dataset == 'MNIST':
            _, _, train_data, test_data = self.MNIST()
            return train_data, test_data
        elif self.dataset == 'Fashion_MNIST':
            _, _, train_data, test_data = self.Fashion_MNIST()
            return train_data, test_data
        elif self.dataset == 'CIFAR10':
            _, _, train_data, test_data = self.CIFAR10()
            return train_data, test_data
        else:
            print('Dataset name is error!')

if __name__ == '__main__':
    dataloader = DataLoader('MNIST', 50)
    # dataloader = DataLoader('Fashion_MNIST', 50)
    # dataloader = DataLoader('CIFAR10', 50)
    trainloader, testloader = dataloader.load()
    for i, (x, y) in enumerate(trainloader):
        print(1)
