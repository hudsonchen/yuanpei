import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Logistic regression function
def logistic_regression(X, y, alpha, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - alpha * gradient

        cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_history.append(cost)

    return theta, cost_history


# Load the wine dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
         'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
         'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
data = pd.read_csv(url, names=names)

# Split the data into features X and target y
X = data.drop('Class', axis=1).values
y = data['Class'].values

# PCA starts here. You do not need to understand this part of the code.
# Perform PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

# Convert to DataFrame
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Convert the target to DataFrame
y_df = pd.DataFrame(data=y, columns=['Class'])

# Concatenate with target label
finalDf = pd.concat([principalDf, y_df], axis=1)

# Visualization
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = [1, 2, 3]
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'],
               finalDf.loc[indicesToKeep, 'Principal Component 2'],
               c=color, s=50)

ax.legend(targets)
ax.grid()
plt.show()
# PCA ends here. You do not need to understand this part of the code.

# Only consider classes 1 and 2 for binary logistic regression
data = data[data['Class'] <= 2]

# Split the data into features X and target y
X = data.drop('Class', axis=1).values
y = data['Class'].values

# Add a column of ones for the bias term
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Remap the class labels to 0 and 1
y = np.where(y == 1, 0, 1)

# Set the hyperparameters
alpha = 0.00001
iterations = 100000

# Run logistic regression
theta, cost_history = logistic_regression(X, y, alpha, iterations)

# Plot the cost function
plt.figure()
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
