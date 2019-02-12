import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

X = np.array([[1,5], [1,7], [2,6], [2,9], [3,3], [3,6], [3,8], [3,8],[3,9],
              [4,5], [4,8], [4,8], [5,4], [5,7], [6,9], [7,2], [7,3],[7,9],
              [8,1],[8,7]])

labels = range(1, 21)
plt.figure(figsize=(10, 7))

plt.scatter(X[:,0],X[:,1], label='True Position') 
for label, x, y in zip(labels, X[:, 0], X[:,1]): 
    plt.annotate(label,
                 xy =(x, y),xytext =( -3, 3),
                 textcoords='offset points', ha='right', va ='bottom')
plt.show()

print("1. Single link clustering")
labels = range(1, 21)
plt.figure(figsize=(10, 7))
linked= linkage(X,'single')
dendrogram(linked,labels= labels, distance_sort = 'descending')
plt.show()

print("2. Complete link clustering")
labels = range(1, 21)
plt.figure(figsize=(10, 7))
linked= linkage(X,'complete')
dendrogram(linked,labels= labels, distance_sort = 'descending')
plt.show()

print("3. Average link clustering")
labels = range(1, 21)
plt.figure(figsize=(10, 7))
linked= linkage(X,'average')
dendrogram(linked,labels= labels, distance_sort = 'descending')
plt.show()