from __future__ import print_function
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import DataLoader
import time
import argparse
from classifier import ResNet, ResNet18, CNN

# Hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--DATANAME', type=str, default='CIFAR10', help='MNIST, Fashion_MNIST, CIFAR10, SVHN')
opt = parser.parse_args()
BATCH_SIZE = 5
EPOCH_C = 50
LR_NET = 2e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 100

def SoftCrossEntropy(inputs, target):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    return loss

def train_distill1():
    model1.train()
    for epoch in range(EPOCH_C):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer1.zero_grad()
            output = F.log_softmax(model1(x)/T, dim=1) # distillation
            loss = F.nll_loss(output, y.squeeze())
            loss.backward()
            optimizer1.step()
        print('|Training Classifier: |Epoch: {}/{} |loss: {:.4f}'
              .format(epoch, EPOCH_C, loss.item()), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        test(epoch, model1)
        print('---------------------------------------------------------------------------------------')
    torch.save(model1.state_dict(), 'models/cnn_' + opt.DATANAME + 'distillation1.pkl')

def train_distill2():
    model2.train()
    model1.eval()
    for epoch in range(EPOCH_C):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer2.zero_grad()
            output = model2(x)/T
            softlabel = F.softmax(model1(x)/T, dim=1)
            loss = SoftCrossEntropy(output, softlabel)
            loss.backward()
            optimizer2.step()
        print('|Training Classifier: |Epoch: {}/{} |loss: {:.4f}'
              .format(epoch, EPOCH_C, loss.item()), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        test(epoch, model2)
        print('---------------------------------------------------------------------------------------')
    torch.save(model2.state_dict(), 'models/cnn_' + opt.DATANAME + 'distillation2.pkl')

def test(epoch, model):
    model.eval()
    total, correct0 = 0, 0
    test_loss = 0.
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            total += x.size(0)
            x, y = x.to(device), y.to(device)
            output = F.log_softmax(model(x), dim=1)
            y0 = output.data.max(1)[1]
            correct0 += y0.eq(y.data).cpu().sum()  # accuracy with no ad
            loss = F.nll_loss(output, y)
            test_loss += loss.item()

        test_loss /= float(len(test_loader))
        print('|Testing  Classifier: |Epoch: {}/{}, |Average loss: {:.4f}, |Acc0: {}/{} ({:.2f}%)'
              .format(epoch, EPOCH_C, test_loss, correct0, total, 100. * float(correct0) / total))


if __name__ == '__main__':
    if opt.DATANAME in ['MNIST', 'Fashion_MNIST']:
        model1 = CNN().to(device)
        model2 = CNN().to(device)
    else:
        model1 = ResNet18(channels=3, num_classes=10).to(device)
        model2 = ResNet18(channels=3, num_classes=10).to(device)
    dataloader = DataLoader(opt.DATANAME, BATCH_SIZE)
    train_loader, test_loader = dataloader.load()
    optimizer1 = optim.Adam(model1.parameters(), lr=LR_NET, betas=(0.5, 0.999))
    optimizer2 = optim.Adam(model2.parameters(), lr=LR_NET, betas=(0.5, 0.999))
    # train and test
    if os.path.exists('models/cnn_' + opt.DATANAME + 'distillation1.pkl'):
        if torch.cuda.is_available():
            model1.load_state_dict(torch.load('models/cnn_' + opt.DATANAME + 'distillation1.pkl'))
        else:
            model1.load_state_dict(torch.load('models/cnn_' + opt.DATANAME + 'distillation1.pkl',  map_location='cpu'))
    else:
        train_distill1()
    train_distill2()

