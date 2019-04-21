# defense adversarial attacks by boundary acgan: fine tune
from __future__ import print_function
import argparse, os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import boundaryCGAN
from attacks import ATTACK, DATANAME, CUDA, device, FGSM, BlackBox, DATANAME_LIST
from attacks import test, test_attack, SubstituteModel2, SubstituteModel1, HOLDOUT_SIZE
from classifier import CNN, ResNet18
from dataloader import DataLoader
from torchvision.utils import save_image


interval = 1000 # 2000
print(interval)
parser = argparse.ArgumentParser()
parser.add_argument('--add', type=int, default=500000, help='add samples of cgan')
parser.add_argument('--BETA', type=float, default=0.1, help='beta add in kl penalty')
parser.add_argument('--EPOCH_G', type=int, default=0, help='fine tune, epoch of acgan, 0: no finetune')
parser.add_argument('--addkl', type=int, default=0, help='add samples finetune boundary contional gan')
parser.add_argument('--LR_GAN', type=float, default=2e-4, help='learning rate of gan')
opt = parser.parse_args()
print(opt)
BETA = opt.BETA
BATCH_SIZE = 50
NUM_CLASS = 10
DIM_Z = 100
PLOT = False
IMG_SIZE = 28 if DATANAME in ['MNIST', 'Fashion_MNIST'] else 32

def train_acgan(EPOCH, BETA, train_loader):
    for epoch in range(EPOCH):
        for iter, (x_, y_) in enumerate(train_loader):
            batch_size = x_.shape[0]
            uniform_dist = torch.Tensor(batch_size, NUM_CLASS).fill_(1. / NUM_CLASS).to(device)
            y_real_, y_fake_ = torch.ones(batch_size, 1).to(device), torch.zeros(batch_size, 1).to(device)
            z_ = torch.rand((batch_size, DIM_Z))
            y_vec_ = torch.zeros((batch_size, NUM_CLASS)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
            x_, z_, y_vec_, y_ = x_.to(device), z_.to(device), y_vec_.to(device), y_.to(device)

            # update D network
            optimizer_D.zero_grad()
            # true x
            D_real, C_realD = discriminator(x_)
            D_real_loss = BCE_loss(D_real, y_real_)
            C_real_loss = CE_loss(C_realD, torch.max(y_vec_, 1)[1])
            # fake x
            G_ = generator(z_, y_vec_)
            D_fake, C_fakeD = discriminator(G_)
            D_fake_loss = BCE_loss(D_fake, y_fake_)
            C_fake_loss = CE_loss(C_fakeD, torch.max(y_vec_, 1)[1])

            D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
            D_loss.backward()
            optimizer_D.step()

            # update G network
            optimizer_G.zero_grad()
            G_ = generator(z_, y_vec_)
            D_fake, C_fakeG = discriminator(G_)

            G_loss = BCE_loss(D_fake, y_real_)
            C_fake_loss = CE_loss(C_fakeG, torch.max(y_vec_, 1)[1])
            G_loss += C_fake_loss
            # add KL divergence
            if BETA != 0.0:
                output_modelG = F.log_softmax(model(G_), dim=1)  # let the gradient backward
                KL_G = F.kl_div(output_modelG, uniform_dist)
                lossG_KL = G_loss + BETA * KL_G
                lossG_KL.backward()
                optimizer_G.step()  # only update generator
            else:
                G_loss.backward()
                optimizer_G.step()
            # calculate the discriminator class accuracy
            pred = torch.cat((C_realD.detach(), C_fakeD.detach()))
            gt = torch.cat((y_, y_))
            d_acc = torch.mean((torch.max(pred, dim=1)[1] == gt).float())
        d_acc = d_acc.item()
        kl_loss = 0
        loss_gan = D_loss.item() + G_loss.item()
        print('|Train: |Epoch:{}/{}, |AccDis: ({:.2f}%)|Loss_GAN: {:.4f}, |Loss_KL： {:.4f} '
              .format(epoch, EPOCH, 100 * d_acc, loss_gan, kl_loss),
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))


def aug(n): # data augmentation by acgan+KL
    model.train()
    for i in range(int(n / BATCH_SIZE)):
        # generate data by acgan:
        labels = torch.from_numpy(np.random.randint(0, NUM_CLASS, (BATCH_SIZE,))).type(torch.LongTensor)
        labels = labels.view(BATCH_SIZE).long().to(device)
        z_ = torch.rand((BATCH_SIZE, 100)).to(device)
        y_vec_ = torch.zeros((BATCH_SIZE, NUM_CLASS)).scatter_(1, labels.type(torch.LongTensor).unsqueeze(1), 1)
        y_vec_ = y_vec_.to(device)
        newdata = generator(z_, y_vec_)
        optimizer.zero_grad()
        output_new = F.log_softmax(model(newdata.detach()), dim=1)
        loss_new = F.nll_loss(output_new, labels.squeeze())
        loss_new.backward()
        optimizer.step()
        if (i+1) % interval == 0: # retrain every 10,0000
            # hybrid clean examples to train one epoch
            for j, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = F.log_softmax(model(x), dim=1)
                loss = F.nll_loss(output, y.squeeze())
                loss.backward()
                optimizer.step()

        if i % max(int(0.2 * (n / BATCH_SIZE)), 1) == 0:
            print('|Adding samples: |Finish: {}/{}, ({:.2f}%, |Loss: {:.4f})'
                  .format(i * BATCH_SIZE, n, 100. * i * BATCH_SIZE / n, loss_new.item()),
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            test2(test_loader, model)

def test2(testloader, model):
    model.eval()
    correct, test_loss = 0, 0.
    sum_testloader = len(testloader.dataset)
    for i, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        output = F.log_softmax(model(x), dim=1)
        y0 = output.max(1)[1]
        correct += y0.eq(y).cpu().sum()
        loss = F.nll_loss(output, y)
        test_loss += loss.item()
    test_loss /= float(sum_testloader)
    print('|Testing  Classifier: |Average loss: {:.4f}, |Acc0: {}/{} ({:.2f}%)'
          .format(test_loss, correct, sum_testloader, 100. * float(correct) / sum_testloader))

if __name__ == '__main__':
    # load data
    dataloader = DataLoader(DATANAME, 50)
    train_loader, test_loader = dataloader.load()
    # define model
    index = DATANAME_LIST.index(DATANAME)
    if DATANAME in ['MNIST', 'Fashion_MNIST']:
        model = CNN().to(device)
    else:
        model = ResNet18(channels=3, num_classes=10).to(device)
    if torch.cuda.is_available() and os.path.exists('models/cnn_'+DATANAME+'.pkl'):
        model.load_state_dict(torch.load('models/cnn_'+DATANAME+'.pkl'))

    # load model
    generator = boundaryCGAN.Generator().to(device)
    discriminator = boundaryCGAN.Discriminator().to(device)
    if CUDA:
        generator.load_state_dict(torch.load('models/generator_'+DATANAME+'.pkl'))
        discriminator.load_state_dict(torch.load('models/discriminator_'+DATANAME+'.pkl'))

    # Initial label
    real_label = 1.0
    fake_label = 0.0
    # Initial loss and optimizer
    BCE_loss = nn.BCELoss()
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    CE_loss = nn.CrossEntropyLoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.LR_GAN, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=opt.LR_GAN, betas=(0.5, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0)

    # attacks
    # for blackbox attack
    blackbox = None
    net_blackbox = None
    if ATTACK == 'BlackBox':
        net_blackbox = SubstituteModel1() if DATANAME in ['MNIST', 'Fashion_MNIST'] else SubstituteModel2()
        net_blackbox.to(device)
        blackbox = BlackBox()
        if CUDA:
            net_blackbox.load_state_dict(torch.load('models/cnn_' + DATANAME + '_sub.pkl'))

    #(1) test model accuracy and adversary in the first time
    # if DATANAME != 'CW':
    #     test(model, test_loader)
    #     test_attack(ATTACK, model, test_loader, blackbox, net_blackbox)

    #(2) add acgan data to classifier
    if opt.add != 0:
        if not os.path.exists('models/cnnaug_' + DATANAME + '_' + str(opt.add) + '_int' + str(interval) + '.pkl'):
            aug(opt.add)
            torch.save(model.state_dict(), 'models/cnnaug_' + DATANAME + '_' + str(opt.add) + '_int' + str(interval) +'.pkl')
        else:
            model.load_state_dict(torch.load('models/cnnaug_' + DATANAME + '_' + str(opt.add) + '_int' + str(interval)+ '.pkl'))

    #(3) test model and adversary after add acgan data
    if opt.add != 0:
        test(model, test_loader)
        test_attack(ATTACK, model, test_loader, blackbox, net_blackbox)

    #(4) fine tune
    if opt.EPOCH_G != 0:
        train_acgan(opt.EPOCH_G, BETA, train_loader)

    #(5) test adversary and clear data test
    if opt.addkl != 0:
        aug(opt.addkl)
        test_attack(ATTACK, model, test_loader, blackbox, net_blackbox)
        test(model, test_loader)
