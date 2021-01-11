import torch
from model import knn_classifier, CNN
import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from helper.confusion_matrix import plot_confusion_matrix


train_root_dir = 'dataset/mnist_data/train'
test_root_dir = 'dataset/mnist_data/test'

# simple input example
# x = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]

normalizer = Normalizer()
min_max_scalar = MinMaxScaler()

TEST_NUM_EACH_DIGIT = 50

# The length of training and test collection
# len_train, len_test = 60000, 10*TEST_NUM_EACH_DIGIT
len_train, len_test = 60000, 10000

img_size = 28

# numpy array form
train_X = np.empty((len_train, 1, img_size, img_size))
train_y = np.empty((len_train, 1))
test_X = np.empty((len_test, 1, img_size, img_size))
test_y = np.empty((len_test, 1))


def image_to_array(root_dir, train=True):
    """
    transfer to numpy format from image format
    :param root_dir: data to transfer
    :param train: if the data is the training data
    """
    idx = 0  # row index
    print(os.path.abspath(root_dir))
    for label in os.listdir(root_dir):
        test_label_idx = 0
        img_path_label = os.path.join(root_dir, label)
        for file_name in os.listdir(img_path_label):
            img_file_name = os.path.join(img_path_label, file_name)
            img = Image.open(img_file_name)
            arr = np.array(img)
            # if train, manipulate training array.
            if train:
                train_X[idx, 0, :] = arr
                train_y[idx] = int(label)
            else:
                test_X[idx, 0, :] = arr
                test_y[idx] = int(label)
                test_label_idx += 1
            idx += 1


# data pre_processing
image_to_array(train_root_dir, train=True)
image_to_array(test_root_dir, train=False)

train_X = torch.Tensor(train_X)
test_X = torch.Tensor(test_X)

net = CNN()

with torch.no_grad():
    train_X = net(train_X).numpy()
    test_X = net(test_X).numpy()

train_X = normalizer.fit_transform(train_X)
test_X = normalizer.fit_transform(test_X)

print("training sample shape: %s, training label shape: %s" % (train_X.shape, train_y.shape))
print("test sample shape: %s, test label shape: %s" % (test_X.shape, test_y.shape))

# train

print("Data prepared, start training...")
knn_classifier.fit(train_X, train_y)
print("Finished training.")


print('Start predicting...')

prediction = knn_classifier.predict(test_X, test_y)
acc = (prediction == test_y).sum() / len(prediction)

labels = list(range(10))
error_counter = [0] * 10

for idx in range(len(prediction)):
    if prediction[idx] != test_y[idx]:
        # print('Negative prediction: answer: %d prediction: %d' % (test_y[idx], prediction[idx]))
        error_counter[int(test_y[idx])] += 1


classes = [str(item) for item in range(10)]
cm = confusion_matrix(test_y, prediction)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, 'cm.png', classes, 'Results of KNN classifier for digits recognition problem')


print("Prediction accuracy: %.1f%%" % (acc * 100))

# predict

# single image prediction
# test_img = "/Users/apple/Desktop/2020秋季课程/Machine_Learning_2020/mnist_data/test/0/3.png"
# img = Image.open(test_img)
# img_arr = np.array(img).flatten().reshape(1, -1)
# rst = knn_classifier.predict(img_arr)
# print(rst)


# plt.bar(labels, error_counter)
# plt.show()
# plt.savefig("error.png", format='png', dpi=300)