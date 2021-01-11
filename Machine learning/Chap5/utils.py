import os
import numpy as np
from sklearn.preprocessing import Normalizer

def save_data(X_train, X_test, y_train, y_test):
    if not os.path.exists('data'):
        os.mkdir('data')
    print('Start to save data...')
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    print('Save data completed!')

def load_data():
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    return normalize(X_train), normalize(X_test), y_train, y_test

def normalize(X):
    transformer = Normalizer().fit(X)
    return transformer.transform(X)