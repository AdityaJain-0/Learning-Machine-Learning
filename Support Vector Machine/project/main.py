from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Step 1: Generate data
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)

# Step 2: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train SVM
model = SVC(kernel="linear", C=1)
model.fit(X_scaled, y)
