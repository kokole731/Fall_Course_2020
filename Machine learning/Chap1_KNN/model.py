# from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)

class KNeighborsClassifier:

    def __init__(self, n_neighbors=3):
        """
        :arg:
            n_neighbors {int} -- k as the threshold
        """
        self.n_neighbors = n_neighbors
        self.train_X = None
        self.train_y = None
        self._L = None
        self._T = None
        self._N = None

    def predict(self, X, y):
        """
        :arg
            X{numpy.int} -- input data to be predicted (T, N)
        :return
            y{numpy.int} -- prediction results (T, 1)
        """
        self._T = X.shape[0]
        output = np.zeros((self._T, 1))
        for idx, inX in enumerate(X):
            dist = inX - self.train_X
            dist_square = np.sum(dist ** 2, axis=1) ** 0.5  # (L, )
            # dist_square = dist_square[:, np.newaxis]  # (L, 1)
            sorted_index_list = dist_square.argsort()
            class_count = {}
            for i in range(self.n_neighbors):
                voted_label = self.train_y[sorted_index_list[i], 0]
                if not voted_label in class_count:
                    class_count[voted_label] = 1
                else:
                    class_count[voted_label] += 1
            sorted_class_count = sorted(class_count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            output[idx] = sorted_class_count[0][0]
            print('Ground truth: %d, prediction: %d' % (y[idx], output[idx]))

        return output

    def fit(self, X, y):
        """
        :arg
            X{numpy.int} -- training data reused in prediction (L, N)
            Y{numpy.int} -- training label (L, 1)
        :return
        """
        self.train_X = X
        self.train_y = y
        self._L = X.shape[0]
        self._N = X.shape[1]


knn_classifier = KNeighborsClassifier(n_neighbors=3)
train_X = np.array([
    [0, -1, -2],
    [1, 2, 3],
    [4, 5, 6]
])
train_y = np.array([
    [0],
    [0],
    [1]
])
test_X = np.array([
    [0, 0, 0],
    [5, 5, 5]
])

