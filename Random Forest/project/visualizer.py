import matplotlib.pyplot as plt
from main import features

features.head(10).plot(kind='barh')
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
