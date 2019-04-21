# white-box attacks: FGSM, PGD, CW
from __future__ import print_function
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from dataloader import DataLoader
from classifier import CNN, ResNet18
import operator as op
import functools as ft
import sys, random

BCGAN = True
DISTILLATION = False
EPSILON = 0.01
ATTACKNAME = ['FGSM', 'PGD', 'CW']
ATTACK = ATTACKNAME[3]
DATANAME_LIST = ['MNIST', 'Fashion_MNIST', 'CIFAR10']
DATANAME = DATANAME_LIST[3]
print(DATANAME)
if DATANAME == 'MNIST':
    CW_maxsteps, CW_seachsteps = 1000, 5
    PGD_ITERATION = 40
    L = 12000
    lr = 500
elif DATANAME == 'Fashion_MNIST':
    CW_maxsteps, CW_seachsteps = 1200, 6
    PGD_ITERATION = 40
    L = 12000
    lr = 500
else:
    CW_maxsteps, CW_seachsteps = 50, 5
    PGD_ITERATION = 5
    L = 12000
    lr = 1000
print('L: {}, lr: {}'.format(L, lr))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = True if torch.cuda.is_available() else False
NUM_CLASS = 10
print('Attack: {}, Epsilon: {}, PGD_iter: {}, CW: {}/{}'.format(ATTACK, EPSILON, PGD_ITERATION, CW_maxsteps, CW_seachsteps))
# white-box attacks: FGSM, PGD, CW
class FGSM():
    def __init__(self, x, y, net, epsilon):
        self.x = x
        self.x.requires_grad = True
        self.y = y
        self.epsilon = epsilon
        self.net = net

    def perturb(self):
        self.net.zero_grad() # empty gradient
        # calculate adversarial examples
        output = F.log_softmax(self.net(self.x), dim=1)
        loss = F.nll_loss(output, self.y)
        loss.backward()  # calculate gradient but not update
        x_grad = torch.sign(self.x.grad.data)
        x_ad = torch.clamp(self.x.data + self.epsilon * x_grad, -1, 1)
        x_ad = x_ad.to(device)
        return x_ad

class PGD():
    # reference: https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py
    def __init__(self, x, y, net, epsilon=0.3, a=0.01, iterations=40):
        '''
        x_n+1 = x_n + a * sign(gradient_x_n)
        x_n+1 = projection(x_n+1 - epsilon, x_n+1 + epsilon), l_infinity
        '''
        self.x = x
        self.x.requires_grad = True
        self.y = y
        self.epsilon = epsilon
        self.net = net
        self.a = a
        self.k = iterations

    def perturb(self):
        self.net.zero_grad() # empty gradient
        x = self.x.detach() # copy
        # calculate adversarial examples
        for i in range(self.k):
            x = x.to(device)
            x_ad = x.detach()
            x_ad.requires_grad = True
            output = F.log_softmax(self.net(x_ad), dim=1)
            loss = F.nll_loss(output, self.y)
            loss.backward()  # calculate gradient but not update
            x_grad = torch.sign(x_ad.grad.data)
            x += self.a * x_grad
            x = torch.from_numpy(np.clip(x.cpu().numpy(), (self.x.data - self.epsilon).cpu().numpy(), (self.x.data + self.epsilon).cpu().numpy()))
            # x = torch.clamp(x, self.x.data - self.epsilon, self.x.data + self.epsilon) # projection with l_infinity
            x = torch.clamp(x, -1, 1) # to ensure the rationality
        x_ad = x.detach().to(device)
        return x_ad

class CW():
    # reference: https://github.com/rwightman/pytorch-nips2017-attack-example/blob/master/run_attack_cwl2.py
    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=CUDA, debug=False):
        self.debug = debug
        self.targeted = targeted
        self.num_classes = NUM_CLASS  # important
        self.confidence = 20  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.1  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.abort_early = True
        self.clip_min = -1.
        self.clip_max = 1.
        self.cuda = cuda
        self.clamp_fn = 'tanh'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()

        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if self.clamp_fn == 'tanh':
            input_adv = self.tanh_rescale(modifier_var + input_var, self.clip_min, self.clip_max)
        else:
            input_adv = torch.clamp(modifier_var + input_var, self.clip_min, self.clip_max)

        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            dist = self.l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = self.l2_dist(input_adv, input_orig, keepdim=False)

        loss = self._loss(output, target_var, dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, input_adv_np

    def reduce_sum(self, x, keepdim=True):
        # silly PyTorch, when will you get proper reducing sums/means?
        for a in reversed(range(1, x.dim())):
            x = x.sum(a, keepdim=keepdim)
        return x

    def reduce_mean(self, x, keepdim=True):
        numel = ft.reduce(op.mul, x.size()[1:])
        x = self.reduce_sum(x, keepdim=keepdim)
        return x / numel

    def reduce_min(self, x, keepdim=True):
        for a in reversed(range(1, x.dim())):
            x = x.min(a, keepdim=keepdim)[0]
        return x

    def reduce_max(self, x, keepdim=True):
        for a in reversed(range(1, x.dim())):
            x = x.max(a, keepdim=keepdim)[0]
        return x

    def torch_arctanh(self, x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def l2r_dist(self, x, y, keepdim=True, eps=1e-8):
        d = (x - y) ** 2
        d = self.reduce_sum(d, keepdim=keepdim)
        d += eps  # to prevent infinite gradient at 0
        return d.sqrt()

    def l2_dist(self, x, y, keepdim=True):
        d = (x - y) ** 2
        return self.reduce_sum(d, keepdim=keepdim)

    def l1_dist(self, x, y, keepdim=True):
        d = torch.abs(x - y)
        return self.reduce_sum(d, keepdim=keepdim)

    def l2_norm(self, x, keepdim=True):
        norm = self.reduce_sum(x * x, keepdim=keepdim)
        return norm.sqrt()

    def l1_norm(self, x, keepdim=True):
        return self.reduce_sum(x.abs(), keepdim=keepdim)

    def rescale(self, x, x_min=-1., x_max=1.):
        return x * (x_max - x_min) + x_min

    def tanh_rescale(self, x, x_min=-1., x_max=1.):
        return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

    def perturb(self, input, target, model, batch_idx=0):
        batch_size = input.size(0)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = self.torch_arctanh(input)
            input_orig = self.tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = input
            input_orig = None

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = target_onehot

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(means=modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = modifier
        modifier_var.requires_grad = True

        optimizer = optim.Adam([modifier_var], lr=0.0005)

        for search_step in range(self.binary_search_steps):
            # print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = scale_const_tensor

            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist, output, adv_img = self._optimize(optimizer,model,input_var,modifier_var,target_var,scale_const_var,input_orig)

                # if step % 100 == 0 or step == self.max_steps - 1:
                #     print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                #         step, loss, dist.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        # print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            # print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop
        x_ad0 = torch.from_numpy(np.array(o_best_attack))
        x_ad0 = x_ad0.permute(0, 3, 1, 2)
        x_ad0 = x_ad0.to(device)
        return x_ad0

# test original accuracy
def test(net, test_loader):
    net.eval()
    total, correct0 = 0, 0
    test_loss = 0.
    for i, (x, y) in enumerate(test_loader):
        total += x.size(0)
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        output = F.log_softmax(net(x), dim=1)
        y0 = output.data.max(1)[1]
        correct0 += y0.eq(y.data).cpu().sum()  # accuracy with no ad
    test_loss /= float(len(test_loader))
    print('|Testing  Classifier:|Average loss: {:.4f}, |Acc0: {}/{} ({:.2f}%)'
          .format(test_loss, correct0, total, 100. * float(correct0) / total), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

# test adversarial examples accuracy
def test_attack(attackname, model, test_loader):
    model.eval()
    total, correct = 0, 0
    cw_total = 0
    random.seed(1234)
    cw_list = random.sample(range(50), 30)  # batch * 30
    # with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):

        total += x.size(0)
        x, y = x.to(device), y.to(device)
        # generate adversarail examples
        if attackname == 'FGSM':
            fgsm = FGSM(x, y, model, epsilon=EPSILON)
            x_ad = fgsm.perturb()
        elif attackname == 'PGD':
            pgd = PGD(x, y, model, iterations=PGD_ITERATION)
            x_ad = pgd.perturb()
        elif attackname == 'CW':
            if cw_total < 1000 and (i in cw_list):
                cw = CW(max_steps=CW_maxsteps, search_steps=CW_seachsteps, targeted=False)
                x_ad = cw.perturb(x, y, model, i)
                cw_total += x.size(0)
            else:
                continue
        # evaluation
        output_ad = F.log_softmax(model(x_ad), dim=1)
        y_ad = output_ad.data.max(1)[1]
        correct += y_ad.eq(y).cpu().sum()  # accuracy with no ad

        if i % 100 == 0:
            print(total, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    if attackname == 'CW':
        total = cw_total
    print('|Testing Adersarial Examples({}):|Acc0: {}/{} ({:.2f}%)'
          .format(attackname, correct, total, 100. * float(correct) / total), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

if __name__ == '__main__':
    # load data
    dataloader = DataLoader(DATANAME, 50)
    train_loader, test_loader = dataloader.load()
    # define model
    if DATANAME in ['MNIST', 'Fashion_MNIST']:
        model = CNN().to(device)
    else:
        model = ResNet18(channels=3, num_classes=10).to(device)
    if DISTILLATION:
        if torch.cuda.is_available() and os.path.exists('models/cnn_'+DATANAME+'.pkl'):
            model.load_state_dict(torch.load('models/cnn_'+DATANAME+'distillation2.pkl'))
    elif BCGAN:
        model.load_state_dict(torch.load('models/boundary_'+DATANAME+'.pkl'))
    else:
        if torch.cuda.is_available() and os.path.exists('models/cnn_'+DATANAME+'.pkl'):
            model.load_state_dict(torch.load('models/cnn_'+DATANAME+'.pkl'))
    # fix paramaters of model
    for p in model.parameters():
        p.requires_grad = False

    # test the orginal accuracy
    test(model, test_loader)
    # test adversary
    test_attack(ATTACK, model, test_loader)
