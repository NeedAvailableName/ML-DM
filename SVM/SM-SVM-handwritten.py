# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Loading the Pen-Based Recognition of Handwritten Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Preprocessing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fitting SVM model
# use small C (C = 0.1) for sort-margin SVM
svm_model = SVC(kernel='linear', C=0.1, random_state=42)
svm_model.fit(X_train, y_train)

# Predicting on test set
y_pred = svm_model.predict(X_test)

# Calculating accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
