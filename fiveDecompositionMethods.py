from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

X = load_iris()

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

Y = x_std.dot(svd.components_.T)
print(np.shape(Y))



from sklearn.decomposition import FastICA

fastIca = FastICA(n_components=2,random_state=0)
x_new= fastIca.fit_transform(x)
print(np.shape(x_new))



from sklearn.decomposition import FactorAnalysis

fact_analysis = FactorAnalysis(n_components=3,random_state=0)
x_new= fact_analysis.fit_transform(x)
print(np.shape(x_new))


from sklearn.decomposition import NMF

model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(x)
H = model.components_
print(np.shape(W))
print(H)



from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

X, _ = make_multilabel_classification(random_state=0)
print(np.shape(X))
lda = LatentDirichletAllocation(n_components=5,random_state=0)
lda.fit(X) 
lda.transform(X[-2:])