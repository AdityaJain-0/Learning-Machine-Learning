from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. ROC Curve & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)

# Add a few threshold markers on the ROC curve
for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    idx = np.argmin(np.abs(thresholds - i))
    plt.scatter(fpr[idx], tpr[idx], label=f"Threshold={thresholds[idx]:.2f}", s=40)

plt.legend()
plt.show()

# 8. Print AUC Score Separately
roc_auc_score_value = roc_auc_score(y_test, y_proba)
print(f"AUC Score (from roc_auc_score): {roc_auc_score_value:.4f}")

# 9. Show how thresholds affect prediction
example_thresholds = [0.3, 0.5, 0.7]
print("\nHow threshold affects classification:")
for thresh in example_thresholds:
    preds = (y_proba >= thresh).astype(int)
    acc = (preds == y_test).mean()
    print(f"Threshold {thresh:.1f} → Accuracy: {acc:.3f}")
