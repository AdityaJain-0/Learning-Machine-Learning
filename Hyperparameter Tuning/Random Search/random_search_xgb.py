from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Hyperparameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3, scoring='accuracy', random_state=42, verbose=1)
random_search.fit(X_train_scaled, y_train)

# Best results
print("Best Parameters:", random_search.best_params_)

# Evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
