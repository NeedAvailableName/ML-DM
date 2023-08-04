from sklearn import datasets
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
num_iterations = 100  # Number of bootstrap iterations
accuracy_scores = []  # List to store accuracy scores

for i in range(num_iterations):
    # Perform bootstrap sampling with replacement
    X_sample, y_sample = resample(X, y, replace=True, random_state=i)

    # Create an SVM classifier and fit it to the bootstrap sample
    svm = SVC()
    svm.fit(X_sample, y_sample)

    # Make predictions on the original dataset
    y_pred = svm.predict(X)

    # Compute accuracy score and store it
    accuracy = accuracy_score(y, y_pred)
    accuracy_scores.append(accuracy)

# Calculate the mean accuracy across all iterations
mean_accuracy = sum(accuracy_scores) / num_iterations

# Print the mean accuracy
print("Mean Accuracy:", mean_accuracy)
