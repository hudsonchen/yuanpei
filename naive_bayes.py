import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(0)
X1 = np.random.normal(0, 1, size=(100, 2))  # class 1
X2 = np.random.normal(1, 1, size=(100, 2))  # class 2
X = np.vstack((X1, X2))

y = np.hstack((np.zeros(100), np.ones(100)))  # labels

# Visualize synthetic data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Synthetic Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar()
plt.show()

# Train naive Bayes
means = np.array([np.mean(X[y == class_], axis=0) for class_ in [0, 1]])
stds = np.array([np.std(X[y == class_], axis=0) for class_ in [0, 1]])

# Sample new data from learned distributions
new_X1 = np.random.normal(means[0], stds[0], size=(100, 2))  # class 1
new_X2 = np.random.normal(means[1], stds[1], size=(100, 2))  # class 2

new_X = np.vstack((new_X1, new_X2))
new_y = np.hstack((np.zeros(100), np.ones(100)))  # labels for new samples

# Visualize new samples
plt.figure(figsize=(8, 6))
plt.scatter(new_X[:, 0], new_X[:, 1], c=new_y)
plt.title("New Samples from Learned Distributions")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar()
plt.show()

# Plot the decision boundary
x = np.linspace(-3.5, 3.5, 100)
y = np.linspace(-3.5, 3.5, 100)
X, Y = np.meshgrid(x,y)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv1 = multivariate_normal(means[0], np.diag(stds[0]**2))
rv2 = multivariate_normal(means[1], np.diag(stds[1]**2))

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, rv1.pdf(pos)-rv2.pdf(pos), cmap='RdBu')
plt.scatter(new_X[:, 0], new_X[:, 1], c=new_y, edgecolors='k')
plt.title("Decision Boundary and New Samples with Density Heatmap")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar()
plt.show()
