from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
svm = SVC()
loo = LeaveOneOut()

# Initialize an empty array to store the predicted labels
y_pred = []

# Perform leave-one-out cross-validation
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the SVM classifier to the training data
    svm.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred.append(svm.predict(X_test)[0])

# Compute the accuracy of the SVM classifier
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
