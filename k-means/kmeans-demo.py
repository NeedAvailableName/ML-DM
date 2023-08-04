# Algorithm:
# 1. choose number of clusters (in this demo is 3)
# 2. randomly init 3 centroids
# 3. loop: assign sample to nearest centroid and compute new centroids of each cluster
# until centroid do not change

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate data
# np.random.seed(0)
X = np.vstack((np.random.randn(100, 2) * 0.5 + [1, 1],
               np.random.randn(100, 2) * 0.5 + [-1, -1],
               np.random.randn(100, 2) * 0.5 + [1, -1]))

# Run KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Run Online-Kmeans with 3 clusters
online_kmeans = MiniBatchKMeans(n_clusters=3, random_state=0)
online_kmeans.fit(X)

# Plot the clusters
plt.subplot(1,2,1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title('Kmeans')


plt.subplot(1,2,2)
plt.scatter(X[:, 0], X[:, 1], c=online_kmeans.labels_)
plt.scatter(online_kmeans.cluster_centers_[:, 0], online_kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title('Online Kmeans')
plt.show()

