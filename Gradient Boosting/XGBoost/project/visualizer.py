from xgboost import plot_importance
import matplotlib.pyplot as plt
from main import model

# Plot
plot_importance(model)
plt.show()
