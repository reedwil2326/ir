# Practical 5: Agglomerative Hierarchical Clustering on the Iris Dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load the Iris dataset (X = measurements, y = real species 0/1/2)
X, y = load_iris(return_X_y=True)

# Step 2: Draw the dendrogram (shows how clusters merge step by step)
dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=20)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("samples")
plt.ylabel("distance")
plt.show()

# Step 3: Cut the tree into 3 clusters
labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)

# Step 4: Compare clusters with the real species
print("Agreement with real species (ARI):", round(adjusted_rand_score(y, labels), 3))

# Step 5: Plot the clusters (petal length vs petal width)
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("Agglomerative clusters (k=3)")
plt.show()