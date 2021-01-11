import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

X, y = fetch_openml('mnist_784', return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scalar = StandardScaler()
scalar.fit_transform(X_train)
scalar.fit_transform(X_test)

clf = Perceptron()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))
