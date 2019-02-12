import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

iris_x = iris.data
iris_y = iris.target


np.random.seed(0)

indices = np.random.permutation(len(iris_x))
print(len(iris_x[0]))

iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]

iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

knn = KNeighborsClassifier()

print(knn.fit(iris_x_train,iris_y_train))

print(knn.predict(iris_x_test))

print(iris_y_test)


