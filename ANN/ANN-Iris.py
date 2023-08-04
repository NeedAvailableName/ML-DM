import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_size = X_train.shape[1]  # Number of input features
hidden_size = 16  # Number of neurons in the hidden layer
output_size = len(np.unique(y_train))  # Number of output classes
learning_rate = 0.1
num_epochs = 1000
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
# Initialize the weights and biases
np.random.seed(42)
weights_hidden = np.random.randn(input_size, hidden_size)
biases_hidden = np.zeros(hidden_size)

weights_output = np.random.randn(hidden_size, output_size)
biases_output = np.zeros(output_size)
# Training loop
for epoch in range(num_epochs):
    # Forward propagation
    hidden_layer_output = sigmoid(np.dot(X_train, weights_hidden) + biases_hidden)
    output_layer_output = sigmoid(np.dot(hidden_layer_output, weights_output) + biases_output)
    
    # Backpropagation
    output_error = y_train.reshape(-1, 1) - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)
    
    hidden_error = np.dot(output_delta, weights_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_output += learning_rate * np.dot(hidden_layer_output.T, output_delta)
    biases_output += learning_rate * np.sum(output_delta, axis=0)
    
    weights_hidden += learning_rate * np.dot(X_train.T, hidden_delta)
    biases_hidden += learning_rate * np.sum(hidden_delta, axis=0)
# Forward propagation on the test set
hidden_layer_output = sigmoid(np.dot(X_test, weights_hidden) + biases_hidden)
output_layer_output = sigmoid(np.dot(hidden_layer_output, weights_output) + biases_output)

# Predict the class labels
predictions = np.argmax(output_layer_output, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == y_test) * 100
print("Accuracy: {:.2f}%".format(accuracy))
