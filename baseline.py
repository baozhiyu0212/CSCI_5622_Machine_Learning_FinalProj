import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')
label_dic = np.load('data/label_dic.npy')


nei = KNeighborsClassifier(n_neighbors=3)
nei.fit(x_train, y_train)
print nei.score(x_test, y_test)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
print tree.score(x_test, y_test)

lr = LogisticRegression(multi_class='multinomial', solver='sag', verbose=10, tol=1e-3)
lr.fit(x_train, y_train)
print('Training Complete')
print lr.score(x_test, y_test)
