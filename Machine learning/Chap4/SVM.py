import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC, NuSVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

X, y = fetch_openml('mnist_784', return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scalar = StandardScaler()
scalar.fit_transform(X_train)

# clf = LinearSVC(penalty='l2', loss='squared_hinge')

clf_rbf = NuSVC(kernel='rbf')
clf_linear = NuSVC(kernel='linear')
clf_poly = NuSVC(kernel='poly')

clf_rbf.fit(X_train, y_train)
print('rbf ok...')

clf_linear.fit(X_train, y_train)
print('linear ok...')

clf_poly.fit(X_train, y_train)
print('poly ok...')


predict_rbf = clf_rbf.predict(X_test)
print('rbf predict ok...')

predict_linear = clf_linear.predict(X_test)
print('linear predict ok...')

predict_poly = clf_poly.predict(X_test)
print('poly predict ok...')


print("SVM with kernal rbf: %.4f" % metrics.accuracy_score(y_test, predict_rbf))
print("SVM with kernal linear: %.4f" % metrics.accuracy_score(y_test, predict_linear))
print("SVM with kernal poly: %.4f" % metrics.accuracy_score(y_test, predict_poly))


