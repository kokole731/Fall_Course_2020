import matplotlib.pyplot as plt
import numpy as np

import os
from utils import load_data, save_data
from dataset import MNIST
from model import FC

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

n_input, n_hidden, n_output = 784, 500, 10
n_epoch, lr, batch_size = 100, 0.3, 32

device = torch.device('cuda')
# net = FC(n_input, n_hidden, n_output).to(device)
# optimizer = optim.SGD(net.parameters(), lr=lr)

train_loader = DataLoader(MNIST(train=True), batch_size=batch_size)
eval_loader = DataLoader(MNIST(train=False), batch_size=batch_size)

def eval():
    n_total, n_correct = 0, 0
    for idx, (data, label) in enumerate(eval_loader):
        data, label = data.to(device), label.to(device)
        label = label.max(dim=1)[1]
        output = net(data)
        predict = output.max(dim=1)[1]
        n_total += batch_size
        n_correct += torch.sum(predict == label).item()
    return n_correct / n_total


def log(loss, act, lr, rst):
    base_dir = 'out'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    file_name = '_'.join((loss, act, lr)) + '.txt'
    file_name = os.path.join(base_dir, file_name)
    with open(file_name, 'a') as f:
        f.write(rst)


# sys.stdout = Logger('out/a.txt')
# import logging

lr_set = (0.1, 0.3, 0.6)
loss_set = ('MSE', 'CEP')
act_set = ('relu', 'leaky_relu')

# try 3 learning rates
for lr in lr_set:

    net = FC(n_input, n_hidden, n_output).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for epoch in range(n_epoch):
        loss_total, batch = 0., 0
        for idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = F.mse_loss(output, label)
            loss_total += loss.item()

            loss.backward()
            optimizer.step()

            batch += 1
        acc = eval()
        rst = 'Epoch: %d, training loss: %.4f, eval acc: %.4f\n' % (epoch, loss_total / batch, acc)
        log('mse', 'relu', str(lr), rst)
        print('Epoch: %d, training loss: %.4f, eval acc: %.4f' % (epoch, loss_total / batch, acc))
        # logging.debug(rst)
        loss_total, batch = 0., 0

'''
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
'''