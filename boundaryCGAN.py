import torch, time, os
import torch.nn as nn
import torch.optim as optim
from dataloader import DataLoader
from torchvision.utils import save_image
from classifier import CNN, ResNet, ResNet18
import torch.nn.functional as F
import argparse

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--DATANAME', type=str, default='MNIST', help='MNIST, Fashion_MNIST, SVHN, CIFAR10')
opt = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=3, input_size=32, class_num=10):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=3, output_dim=1, input_size=32, class_num=10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)
        return d, c

def sample_image(n_row=10, epoches_done=0, saved_img_dir=None):  # draw a picture
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    gen_imgs = generator(sample_z_, sample_y_)
    save_image(gen_imgs.data, "{}/{}.png".format(saved_img_dir, epoches_done), nrow=n_row, normalize=True)

def train_acgan(EPOCH, BETA, train_loader):
    for epoch in range(EPOCH):
        if (epoch + 1) in [50, 100, 150]:
            optimizer_G.param_groups[0]['lr'] /= 10
            optimizer_D.param_groups[0]['lr'] /= 10
            print('learning rate decay')
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
        if PLOT:
            sample_image(n_row=NUM_CLASS, epoches_done=epoch + 1, saved_img_dir=DIRECTORY)

if __name__ == '__main__':
    sample_num = 100
    LR_GAN = 2e-4
    IMG_SIZE = 28 if opt.DATANAME in ['MNIST', 'Fashion_MNIST'] else 32
    EPOCH = 200
    BATCH_SIZE = 50
    CHANNELS = 1 if opt.DATANAME in ['MNIST', 'Fashion_MNIST'] else 3
    DIRECTORY = 'images_'+opt.DATANAME
    DIM_Z = 100
    NUM_CLASS = 10
    BETA = 0.1
    PLOT = False
    # load data
    dataloader = DataLoader(opt.DATANAME, BATCH_SIZE)
    train_loader, test_loader = dataloader.load()
    # define model
    if opt.DATANAME in ['MNIST', 'Fashion_MNIST']:
        model = CNN().to(device)
    else:
        model = ResNet18(channels=3, num_classes=10).to(device)
    if torch.cuda.is_available() and os.path.exists('models/cnn_'+opt.DATANAME+'.pkl'):
        model.load_state_dict(torch.load('models/cnn_'+opt.DATANAME+'.pkl'))
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    # Initial label
    real_label = 1.0
    fake_label = 0.0
    # Initial loss and optimizer
    BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
    # fixed noise & condition
    sample_z_ = torch.zeros((sample_num, DIM_Z)) # (10,100) * 10 repeat, fix noise
    for i in range(NUM_CLASS):
        sample_z_[i*NUM_CLASS] = torch.rand(1, DIM_Z)
        for j in range(1, NUM_CLASS):
            sample_z_[i*NUM_CLASS + j] = sample_z_[i*NUM_CLASS]
    temp = torch.zeros((NUM_CLASS, 1))
    for i in range(NUM_CLASS):
        temp[i, 0] = i  # 0-9
    temp_y = torch.zeros((sample_num, 1))
    for i in range(NUM_CLASS):
        temp_y[i*NUM_CLASS: (i+1)*NUM_CLASS] = temp  # 0-9 * 10
    sample_y_ = torch.zeros((sample_num, NUM_CLASS)).scatter_(1, temp_y.type(torch.LongTensor), 1) # onehot
    sample_z_, sample_y_ = sample_z_.to(device), sample_y_.to(device)

    # start training
    train_acgan(EPOCH, BETA, train_loader)

    if torch.cuda.is_available():
        torch.save(generator.state_dict(), 'models/generator_'+opt.DATANAME+'.pkl')
        torch.save(discriminator.state_dict(), 'models/discriminator_'+opt.DATANAME+'.pkl')