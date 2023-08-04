import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Splitting the dataset into training and testing sets
# 80% for training, 20% for testing
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Number of classes in the Iris dataset
num_classes = len(np.unique(y))

# Calculate the prior probabilities for each class
prior_probs = np.zeros(num_classes)
for c in range(num_classes):
    prior_probs[c] = np.sum(y_train == c) / len(y_train)

# Calculate the class-wise means and variances for each feature
class_means = np.zeros((num_classes, X_train.shape[1]))
class_variances = np.zeros((num_classes, X_train.shape[1]))
for c in range(num_classes):
    X_c = X_train[y_train == c]
    class_means[c] = np.mean(X_c, axis=0)
    class_variances[c] = np.var(X_c, axis=0)

# Function to calculate the class conditional probabilities
def calculate_class_probs(x):
    class_probs = np.zeros(num_classes)
    for c in range(num_classes):
        class_probs[c] = np.prod((1 / np.sqrt(2 * np.pi * class_variances[c])) * \
                                np.exp(-(x - class_means[c]) ** 2 / (2 * class_variances[c])))
    return class_probs

# Predict function using Maximum A Posteriori estimation
def predict(X):
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i]
        class_probs = calculate_class_probs(x)
        y_pred[i] = np.argmax(prior_probs * class_probs)
    return y_pred

# Make predictions on the test set
y_pred = predict(X_test)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
