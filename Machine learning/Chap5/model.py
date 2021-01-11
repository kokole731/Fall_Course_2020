import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):

    def __init__(self,n_input, n_hidden, n_output):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.float()
        # x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)