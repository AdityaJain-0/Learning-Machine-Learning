import numpy as np 
from model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

cls = LogisticRegression(learning_rate=0.1, num_iterations=1000)

cls.fit(X_train, y_train)\

predictions = cls.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, predictions))



# Model the decision boundary 
# def plot_decision_boundary(X, y, model):
#     x1 = np.linspace(X[:,0].min() - 1, X[:,0].max() + 1, 100)
#     x2 = -(model.weights[0] * x1 + model.bias) / model.weights[1]

#     plt.figure(figsize=(8,6))
#     plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=0.7)
#     plt.plot(x1, x2, color='black', linewidth=2)
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title("Logistic Regression Decision Boundary")
#     plt.grid(True)
#     plt.show()

# plot_decision_boundary(X, y, cls)