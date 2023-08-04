import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# use Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# split dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# using KNN with neighbor = 3
knn = KNeighborsRegressor(n_neighbors=3)
# fit the model
knn.fit(X_train, y_train)
# predict on test dataset
y_pred = knn.predict(X_test)
# evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
