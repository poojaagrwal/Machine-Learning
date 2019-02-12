from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
x1 = np.array([3, 3, 5, 2, 4, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 6, 6, 5, 8, 2, 3, 6, 7, 2, 5, 1, 2, 6, 1, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

plt.figure(figsize=(10, 7))
plt.title("Data Points")
plt.scatter(x1, x2)
plt.show()

print("1. K=2 means clustering")
colors = ['b', 'g']
markers = ['o', 'v']
K = 2
kmeans_model = KMeans(n_clusters=K).fit(X)
plt.figure(figsize=(10, 7))
plt.title("K=2 means clustering")

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
plt.show()

print("2. K=3 means clustering")
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']
K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)
plt.figure(figsize=(10, 7))
plt.title("K=3 means clustering")

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
plt.show()

print("3. K=4 means clustering")
colors = ['b', 'g', 'r', 'y']
markers = ['o', 'v', 's', 'o']
K = 4
kmeans_model = KMeans(n_clusters=K).fit(X)
plt.figure(figsize=(10, 7))
plt.title("K=4 means clustering")

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
plt.show()
