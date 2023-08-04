from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
iris = datasets.load_iris()
X = iris.data
y = iris.target
svm = SVC()
k = 5  # Number of folds

# Perform cross-validation and compute accuracy scores for each fold
accuracy_scores = cross_val_score(svm, X, y, cv=k)

# Print the accuracy scores for each fold
for fold, accuracy in enumerate(accuracy_scores):
    print("Fold", fold+1, "- Accuracy:", accuracy)

# Compute the mean accuracy across all folds
mean_accuracy = accuracy_scores.mean()
print("Mean Accuracy:", mean_accuracy)
