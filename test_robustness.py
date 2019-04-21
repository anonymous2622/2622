import torch
import torch.nn as nn
from classifier import CNN, ResNet18
from dataloader import DataLoader
import torch.nn.functional as F
import numpy as np
import time
from defenseGAN import Generator
from attacks import L, lr

# Hyperparameter
ALLNAMES = ['MNIST', 'Fashion_MNIST', 'CIFAR10']
DATANAME = ALLNAMES[0]
TYPES = ['BCGAN', 'Defense-GAN', 'FGSM_adv1', 'FGSM_adv2', 'PGD_adv', 'Distillation']
TYPE = TYPES[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deltas = list(np.arange(0.1, 2.1, 0.1))

print('Test robustness: Dataset {}, Type:{}'.format(DATANAME, TYPE))

def defensegan(x):
    z = torch.randn((x.shape[0], 100)).view(-1, 100, 1, 1).to(device)
    z.requires_grad = True
    for l in range(L):
        samples = generator(z)
        loss_mse = MSE_loss(samples, x)
        loss_mse.backward()
        z = z - lr * z.grad
        z = z.detach()  # not leaf
        z.requires_grad = True
    return generator(z)

def getsign(x, y, model):
    model.eval()
    model.zero_grad()
    x.requires_grad = True
    output = F.log_softmax(model(x), dim=1)
    loss = F.nll_loss(output, y)
    loss.backward()
    grad = torch.sign(x.grad.to(device))
    return grad

def calculate_robustness(model):
    delta = 0.0
    datas = 0
    for i, (x, y) in enumerate(train_loader):
        datas += x.shape[0]
        x, y = x.to(device), y.to(device)
        if TYPE == 'Defense-GAN':
            x = defensegan(x.detach()).detach()
        grad = getsign(x, y, model)
        for j in range(x.shape[0]):
            for k in deltas:
                x_ad = torch.clamp(x[j] + float(k) * grad[j], -1, 1)
                output_ad = F.log_softmax(model(x_ad.unsqueeze(0)), dim=1)
                y_ad = output_ad.data.max(1)[1]
                if y[j] != y_ad or k == deltas[-1]:
                    delta += k
                    break
        if i % 100 == 0:
            print('Finish: {}/{}'.format(datas, len(train_loader.dataset)), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    delta /= len(train_loader.dataset)
    return delta


if __name__ == '__main__':
    # load data
    dataloader = DataLoader(DATANAME, 50)
    train_loader, test_loader = dataloader.load()
    # define loss and generator for Defense-GAN
    if TYPE == 'Defense-GAN':
        MSE_loss = nn.MSELoss()
        generator = Generator().to(device)
        if torch.cuda.is_available():
            generator.load_state_dict(torch.load('models/generator_wgan_' + DATANAME + '.pkl'))
    # load model
    if DATANAME in ['MNIST', 'Fashion_MNIST']:
        model1 = CNN().to(device)
        model2 = CNN().to(device)
    else:
        model1 = ResNet18(channels=3, num_classes=10).to(device)
        model2 = ResNet18(channels=3, num_classes=10).to(device)
    if torch.cuda.is_available():
        model1.load_state_dict(torch.load('models/cnn_' + DATANAME + '.pkl'))

    if torch.cuda.is_available():
        if TYPE in ['BCGAN']:
            if DATANAME == 'MNIST':
                model2.load_state_dict(torch.load('models/cnnaug_'+DATANAME+'_500000000.pkl'))
            elif DATANAME == 'Fashion_MNIST':
                model2.load_state_dict(torch.load('models/cnnaug_'+DATANAME+'_500000000.pkl'))
            else:
                model2.load_state_dict(torch.load('models/cnnaug_CIFAR10_100000000_int1000.pkl'))
        elif TYPE == 'FGSM_adv1':
            if DATANAME == 'MNIST':
                model2.load_state_dict(torch.load('models/cnn_'+DATANAME+'_adv20.pkl'))
            elif DATANAME == 'Fashion_MNIST':
                model2.load_state_dict(torch.load('models/cnn_' + DATANAME + '_adv5.pkl'))
            else:
                model2.load_state_dict(torch.load('models/cnn_'+DATANAME+'_adv1.pkl'))
        elif TYPE == 'FGSM_adv2':
            if DATANAME == 'MNIST':
                model2.load_state_dict(torch.load('models/cnn_'+DATANAME+'_adv40.pkl'))
            elif DATANAME == 'Fashion_MNIST':
                model2.load_state_dict(torch.load('models/cnn_' + DATANAME + '_adv15.pkl'))
            else:
                model2.load_state_dict(torch.load('models/cnn_'+DATANAME+'_adv3.pkl'))
        elif TYPE == 'PGD_adv':
            model2.load_state_dict(torch.load('models/cnn_'+DATANAME+'_adv_PGD.pkl'))
        elif TYPE == 'Distillation':
            model2.load_state_dict(torch.load('models/cnn_'+DATANAME+'distillation2.pkl'))
        else: # defense-gan
            model2.load_state_dict(torch.load('models/cnn_' + DATANAME + '.pkl'))
    # d1 = calculate_robustness(model1)
    d2 = calculate_robustness(model2)
    # print(DATANAME+': Robustness of Model1: {:.4f}'.format(d1))
    print(DATANAME+': Robustness of Model2: {:.4f}'.format(d2))
