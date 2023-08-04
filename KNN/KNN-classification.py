import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# use Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
# split dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# using KNN with neighbor = 3
knn = KNeighborsClassifier(n_neighbors=3)
# fit the model
knn.fit(X_train, y_train)
# predict on test dataset
y_pred = knn.predict(X_test)
# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
