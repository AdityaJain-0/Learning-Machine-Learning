# Machine Learning From Scratch

A personal study repo documenting my progress learning machine learning fundamentals. Each algorithm is implemented from scratch where possible, then applied using scikit-learn on real datasets. Notes explaining the underlying math are included alongside every model.

---

## Contents

### Supervised Learning
- **Linear Regression** — gradient descent, MSE loss, normal equation, model persistence
- **Logistic Regression** — sigmoid activation, binary cross-entropy, gradient descent (custom NumPy implementation + scikit-learn)
- **K-Nearest Neighbor** — Euclidean distance, error rate vs K visualization
- **Decision Tree** — Gini impurity, entropy, information gain
- **Random Forest** — bagging, bootstrap sampling, feature importance
- **Support Vector Machine** — hyperplane, support vectors, soft-margin, kernel trick
- **Gradient Boosting / XGBoost** — residual fitting, learning rate, L1/L2 regularization
- **Hyperparameter Tuning** — GridSearchCV, RandomizedSearchCV

### Unsupervised Learning
- **K-Means Clustering** — centroid initialization, inertia, PCA visualization
- **Hierarchical Clustering** — agglomerative, Ward's method, dendrogram
- **DBSCAN** — core/border/noise points, density-based cluster detection

### Extras
- Evaluation metrics — confusion matrix, classification report, ROC/AUC, precision-recall
- Data splitting — train/validation/test, overfitting vs underfitting
- Regularization — L1, L2, weight updates
- Feature scaling — when it matters and when it doesn't
- Hyperparameter reference — n_estimators, max_depth, learning rate, subsample, gamma

---

## Structure

Each algorithm follows the same pattern:

```
Algorithm/
├── extra/
│   └── notes.txt       # Math and intuition behind the model
└── project/
    ├── main.py         # Implementation and training
    └── visualizer.py   # Plots and diagnostics (where applicable)
```

---

## Stack

Python, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, pandas
