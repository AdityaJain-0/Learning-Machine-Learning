import numpy as np
import matplotlib.pyplot as plt

# A sample quadratic loss curve
x = np.linspace(-2, 2, 100)
y = x**2

plt.plot(x, y, label="Loss")
plt.axvline(x=0, color='gray', linestyle='--')
plt.title("Loss Curve: Goal is to reach the bottom")
plt.xlabel("Weight") # Something like the vlaue of w0 or w1
plt.ylabel("Loss") # How bad the model is at predicting the value 
plt.grid()
plt.legend()
plt.show()
