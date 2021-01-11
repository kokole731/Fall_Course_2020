import torch
from utils import load_data

class MNIST(torch.utils.data.Dataset):
    
    def __init__(self, train=True):
        super(MNIST, self).__init__()
        # load mnist dataset
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            return self.get_train_sample(idx)
        else:
            return self.get_test_sample(idx)
    
    def __len__(self, ):
        return self.y_train.shape[0] if self.train else self.y_test.shape[0]

    def get_train_sample(self, idx):
        sample_x_train = self.X_train[idx]
        sample_y_train = int(self.y_train[idx])
        # return torch.from_numpy(sample_x_train), torch.tensor(sample_y_train)
        return torch.from_numpy(sample_x_train), one_hot_encoding(sample_y_train)

    def get_test_sample(self, idx):
        sample_x_test = self.X_test[idx]
        sample_y_test = int(self.y_test[idx])
        # return torch.from_numpy(sample_x_test), torch.tensor(sample_y_test)
        return torch.from_numpy(sample_x_test), one_hot_encoding(sample_y_test)

def one_hot_encoding(data, dim_num=10):
    out = torch.zeros(dim_num)
    out[data] = 1
    return out