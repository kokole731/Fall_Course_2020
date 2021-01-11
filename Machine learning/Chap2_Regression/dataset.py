from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

mnist_data = fetch_openml("mnist_784")
data = mnist_data['data']
target = mnist_data['target'].astype(int)

seed = 42

X_train, X_test, y_train, y_test = train_test_split(data, target, \
    test_size=0.2, random_state=seed)

np.save('mnist/X_train.npy', X_train, allow_pickle=True)
np.save('mnist/X_test.npy', X_test, allow_pickle=True)
np.save('mnist/y_train.npy', y_train, allow_pickle=True)
np.save('mnist/y_test.npy', y_test, allow_pickle=True)