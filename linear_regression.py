import numpy as np
import matplotlib.pyplot as plt


# Data generating function
def generate_data(n):
    X = np.random.uniform(1, 5, n)
    Y = 3 * X + 2 + np.random.randn(n)  # Here, we're assuming the true model is Y = 3X + 2, but we're adding some noise
    return X, Y


# Regression function
def simple_linear_regression(X, Y):
    n = len(X)
    m = (n * np.sum(X * Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X * X) - np.sum(X) ** 2)
    b = (np.sum(Y) - m * np.sum(X)) / n
    return m, b  # slope and intercept of the fitted line


# Plotting function
def plot_regression(X, Y, m, b):
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X, m * X + b, color='red') # The fitted line
    plt.savefig('./figures/regression/linear_regression.png')
    plt.show()


# Generate some data
X, Y = generate_data(100)

# Fit the model to the data
m, b = simple_linear_regression(X, Y)

# Plot the data and the fitted line
plot_regression(X, Y, m, b)
