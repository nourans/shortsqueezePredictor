import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score



# x_vals: time
# y_vals: sentiment analysis + frequency
def polynomial_regression(x_vals, y_vals, degree):
    regression = np.poly1d(np.polyfit(x_vals, y_vals, degree))
    return regression


# dummy data
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

# adjust degree as needed
model = polynomial_regression(x, y, 10)
line = np.linspace(1, 22, 100)
plt.scatter(x, y)
plt.plot(line, model(line))
plt.show()


