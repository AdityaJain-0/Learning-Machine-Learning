# dbscan_example.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate 2D data with a moon shape
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()


#eps=0.2 and min_samples=5 are hyperparameters to tune
#Noise points are labeled as -1 by fit_predict