import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

X = np.array([[2,2], [2,6], [3,7], [5,2], [5,5], 
              [5,8], [7,3], [6,6], [8,4], [10,6], [12,8]])

labels = range(1, 12)
plt.figure(figsize=(10, 10))
plt.scatter(X[:,0],X[:,1], label='True Position') 
for label, x, y in zip(labels, X[:, 0], X[:,1]): 
    plt.annotate(label,
                 xy =(x, y),xytext =( -3, 3),
                 textcoords='offset points', ha='right', va ='bottom')
plt.show()

print("1. Single link clustering")
labels = range(1, 12)
plt.figure(figsize=(10, 10))
linked= linkage(X,'single')
dendrogram(linked,labels= labels, distance_sort = 'descending')
plt.show()

print("2. Complete link clustering")
labels = range(1, 12)
plt.figure(figsize=(10, 10))
linked= linkage(X,'complete')
dendrogram(linked,labels= labels, distance_sort = 'descending')
plt.show()

print("3. Average link clustering")
labels = range(1, 12)
plt.figure(figsize=(10, 10))
linked= linkage(X,'average')
dendrogram(linked,labels= labels, distance_sort = 'descending')
plt.show()