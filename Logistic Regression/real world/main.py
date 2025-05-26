from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import LogisticRegression
import numpy as np

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features (important for gradient descent!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on breast cancer dataset: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=data.target_names))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
