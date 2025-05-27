from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sea 

iris = load_iris()
X = iris.data 

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8,6))
sea.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='Set2', s=60)
plt.scatter(centroids_pca[:,0], centroids_pca[:,1], s=200, c='black', marker='X', label="Centroids")
plt.title("K-Means Clustering on Iris Dataset (PCA-Reduced)")
plt.legend()
plt.grid(True)
plt.show()