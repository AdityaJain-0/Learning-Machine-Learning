from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pylab as plt
import seaborn as sea

data = load_breast_cancer()
X = data.data 
y = data.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state=42)
model.fit(X_train_scaled, y_train)

if __name__ == "__main__":
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    sea.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels = data.target_names, yticklabels=data.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()