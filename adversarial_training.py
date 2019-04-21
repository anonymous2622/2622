# adversarial training
from __future__ import print_function
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time, pickle
from attacks import FGSM, test, test_attack, DATANAME_LIST, PGD
from attacks import ATTACKNAME, ATTACK, EPSILON, DATANAME, PGD_ITERATION
from dataloader import DataLoader
from classifier import CNN, ResNet18

Flag_pgd_adv = True
EPOCH = 30
epsilon = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def FGSM_advtrain(EPOCH, model, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0)
    model.train()
    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # generate adversarail examples by FGSM
            fgsm = FGSM(x, y, model, epsilon=epsilon)
            x_ad = fgsm.perturb()
            output = F.log_softmax(model(x_ad), dim=1)
            loss = F.nll_loss(output, y.squeeze())
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print('|Training Classifier: |epoch: {}/{}, |Finish: {}/{}, ({:.2f}%)|loss: {:.4f}'
                      .format(epoch, EPOCH, i * len(x), len(train_loader.dataset), 100. * i * len(x) / len(train_loader.dataset), loss.item()),
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        test(model, test_loader)
        print('---------------------------------------------------------------------------------------')
    torch.save(model.state_dict(), 'models/cnn_' + DATANAME + '_adv'+str(int(epsilon*100))+'.pkl')

def PGD_advtrain(EPOCH, model, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0)
    model.train()
    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # generate adversarail examples by FGSM
            pgd = PGD(x, y, model, iterations=PGD_ITERATION)
            x_ad = pgd.perturb()
            output = F.log_softmax(model(x_ad), dim=1)
            loss = F.nll_loss(output, y.squeeze())
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print('|Training Classifier: |epoch: {}/{}, |Finish: {}/{}, ({:.2f}%)|loss: {:.4f}'
                      .format(epoch, EPOCH, i * len(x), len(train_loader.dataset), 100. * i * len(x) / len(train_loader.dataset), loss.item()),
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        test(model, test_loader)
        print('---------------------------------------------------------------------------------------')
    torch.save(model.state_dict(), 'models/cnn_' + DATANAME + '_adv_PGD.pkl')

if __name__ == '__main__':
    # load data
    dataloader = DataLoader(DATANAME, 50)
    train_loader, test_loader = dataloader.load()
    # define model

    if DATANAME in ['MNIST', 'Fashion_MNIST']:
        model = CNN().to(device)
    else:
        model = ResNet18(channels=3, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=0.0)

    # adversarial training
    if Flag_pgd_adv:
        if os.path.exists('models/cnn_' + DATANAME + '_adv_PGD.pkl'):
            model.load_state_dict(torch.load('models/cnn_' + DATANAME + '_adv_PGD.pkl'))
        else:
            PGD_advtrain(EPOCH, model, train_loader, test_loader)
    else:
        if os.path.exists('models/cnn_'+DATANAME+'_adv'+str(int(epsilon*100))+'.pkl'):
            model.load_state_dict(torch.load('models/cnn_' + DATANAME + '_adv'+str(int(epsilon*100))+'.pkl'))
        else:
            FGSM_advtrain(EPOCH, model, train_loader, test_loader)

    # test adversary
    test(model, test_loader)
    test_attack(ATTACK, model, test_loader)