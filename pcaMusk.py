from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data =pd.read_csv('clean2data.csv')
print(np.shape(data))

X = data.iloc[:,2:168]
print(np.shape(X))

target = data.iloc[:,168:]
print(np.shape(target))

x_std= StandardScaler().fit_transform(X)

mean_vec= np.mean(x_std, axis=0)


cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec))/(len(x_std)-1)



#using numpy eig to find eigen values and vectors
eig_values, eig_vectors= np.linalg.eig(cov_mat)

for ev in eig_vectors:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print("All good")

eig_pairs = [(eig_values[i], eig_vectors[:,i]) for i in range(len(eig_values))]

eig_pairs.sort()
eig_pairs.reverse()

sum_of_eig_values = sum (eig_values)

explained_var = [(eig_val/sum_of_eig_values)*100 for eig_val in sorted(eig_values, reverse = True)]

cum_explained_var = np.cumsum(explained_var)


import matplotlib.pyplot as plt
import seaborn as sb

sb.set(font_scale=1.2,style="whitegrid")

plt.xlabel('number of features/eigen values')
plt.ylabel('cumulative explained variance');

plt.plot(cum_explained_var)

highest_99_eig_values = [var for var in cum_explained_var if var < 99.1]
#print(highest_99_eig_values)

S = np.hstack((eig_pairs[i][1].reshape(len(eig_values),1) for i in range(len(highest_99_eig_values))))

print(np.shape(S))


Y = x_std.dot(S)

print(np.shape(Y))

##  Combining the new dimension features Y with the taeget column and first two identity column


new_data_set = np.concatenate((data.iloc[:,0:2], Y, target), axis=1)
print(new_data_set)



'''
S = np.hstack((eig_pairs[0][1].reshape(4,1),
               eig_pairs[1][1].reshape(4,1)))'''

