from sklearn import datasets
from sklearn import linear_model
import numpy as np

diabetes= datasets.load_diabetes()

diabetes_x_train =diabetes.data[:-20]
diabetes_y_train = diabetes.target[:-20]

diabetes_x_test = diabetes.data[-20:]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()


print(regr.fit(diabetes_x_train,diabetes_y_train))

print(len(diabetes_x_train[0]))

print(regr.coef_)

print(np.mean((regr.predict(diabetes_x_test)- diabetes_y_test)**2))

print(regr.score(diabetes_x_test,diabetes_y_test))