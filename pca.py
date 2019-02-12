from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.decomposition import TruncatedSVD

X = datasets.load_iris()

x = X.data
y = X.target

x_std= StandardScaler().fit_transform(x)


mean_vec= np.mean(x_std, axis=0)

cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec))/(len(x_std)-1)

#Using svd to find out eigen values and vectors
svd= TruncatedSVD()
svd.fit(cov_mat)
print(svd.singular_values_.round(2))
print(svd.components_)

#using numpy eig to find eigen values and vectors
eig_values, eig_vectors= np.linalg.eig(cov_mat)

print('*************************')
print(eig_values)
print(eig_vectors)
eig_pairs = [(eig_values[i], eig_vectors[:,i]) for i in range(len(eig_values))]

eig_pairs.sort()
eig_pairs.reverse()

S = np.hstack((eig_pairs[0][1].reshape(4,1),
               eig_pairs[1][1].reshape(4,1)))

Y = x_std.dot(S)



