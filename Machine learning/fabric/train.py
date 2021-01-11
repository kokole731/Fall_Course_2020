import matplotlib.pyplot as plt
import numpy as np

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from dataset import Fabric
from model import FC
import const as C

from torchvision import transforms, datasets

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from utils import one_hot_encoding

n_input, n_hidden, n_output = 784 * 3, 500, 3
n_epoch, lr, batch_size = 100, 0.001, 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(28),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((28, 28)),  
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# net = FC(n_input, n_hidden, n_output).to(device)
# optimizer = optim.SGD(net.parameters(), lr=lr)

# train_loader = DataLoader(Fabric(train=True), batch_size=batch_size)
# eval_loader = DataLoader(Fabric(train=False), batch_size=batch_size)

train_dataset = datasets.ImageFolder(root=os.path.join(C.PATH_IMAGE, 'train'),
                                     transform=data_transform["train"], target_transform=one_hot_encoding)
test_dataset = datasets.ImageFolder(root=os.path.join(C.PATH_IMAGE, 'test'),
                                     transform=data_transform["val"], target_transform=one_hot_encoding)                             
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(train_dataset, batch_size=batch_size)

def eval():
    n_total, n_correct = 0, 0
    for idx, (data, label) in enumerate(test_loader):
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

# lr = 0.01
# loss_set = ('MSE', 'CEP')
# act_set = ('relu', 'leaky_relu')

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