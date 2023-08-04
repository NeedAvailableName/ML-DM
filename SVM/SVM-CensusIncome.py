# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loading the Census Income dataset
census_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
# Preprocessing the data
census_data = census_data.apply(LabelEncoder().fit_transform)
X = census_data.iloc[:, :-1]
y = census_data.iloc[:, -1]
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Fitting SVM model
# use large C (C = 1) for hard-margin SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
# Predicting on test set
y_pred = svm_model.predict(X_test)

# Calculating accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
