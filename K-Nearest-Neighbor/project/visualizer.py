from main import plt, X_test_scaled, X_train_scaled, y_test, y_train, KNeighborsClassifier
import numpy as np

errors = []
min = 100
chosen_k = 0
k_range = range(1, 100)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    errors.append(1 - score)
    if min > errors[k-1]:
        min = errors[k-1]
        chosen_k = k

plt.plot(k_range, errors, marker='o')
plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()
print(min, chosen_k)