import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis

# Generate non-spherical data
np.random.seed(0)
X = np.vstack((np.random.randn(100, 2) * 0.5 + [1, 1],
               np.random.randn(100, 2) * 0.5 + [-1, -1],
               np.random.randn(100, 2) * 0.5 + [1, -1]))
# Create a covariance matrix
cov = np.cov(X.T)
# Use inverse of covariance as metric for Mahalanobis distance
inv_cov = np.linalg.inv(cov)

# Cluster the data using KMeans with Mahalanobis distance metric
kmeans = KMeans(n_clusters=3, init='k-means++', algorithm='full',
                metric=mahalanobis, metric_params={'V': inv_cov})
kmeans.fit(X)

# Calculate the distances between data points and cluster centers
distances = cdist(X, kmeans.cluster_centers_, 'mahalanobis', VI=inv_cov)

# Print the results
print("Cluster assignments:", kmeans.labels_)
print("Cluster centers:", kmeans.cluster_centers_)
print("Distances from data points to cluster centers:", distances)
