import numpy as np
import os
import torch

data_root = 'data_processed'

def load_data():
    train_X, train_y = [], []
    for label in os.listdir(data_root):
        cwd = os.path.join(data_root, label)
        for data in os.listdir(cwd):
            data_np = np.load(os.path.join(cwd, data), allow_pickle=True).item()
            train_X.append(data_np['img'])
            train_y.append(data_np['flaw_type'])
    return train_X, train_X, train_y, train_y
            
def one_hot_encoding(label_str, dim_num=10):
    if label_str == '1':
        return torch.Tensor([1, 0, 0])
    elif label_str == '2':
        return torch.Tensor([0, 1, 0])
    else:
        return torch.Tensor([0, 0, 1])
    