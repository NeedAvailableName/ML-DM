from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
n_repeats = 5  # Number of repetitions for hold-out sampling
test_size = 0.2  # Proportion of data for testing

accuracies = []

for i in range(n_repeats):
    # Split the dataset into training and testing sets using hold-out sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

    # Create an SVM classifier and fit it to the training data
    svm = SVC()
    svm.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm.predict(X_test)

    # Evaluate the performance of the SVM classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Iteration", i+1, "- Accuracy:", accuracy)
    accuracies.append(accuracy)

# Calculate the average accuracy across repeated hold-out sampling
average_accuracy = sum(accuracies) / n_repeats

print("Average Accuracy:", average_accuracy)

