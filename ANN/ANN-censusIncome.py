import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Load the dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)

# Preprocessing the data
census_data = data.apply(LabelEncoder().fit_transform)
X = census_data.iloc[:, :-1]
y = census_data.iloc[:, -1]

# Convert categorical features to numeric using LabelEncoder
label_encoder = LabelEncoder()
X_categorical = X.select_dtypes(include=['object'])
X_categorical = X_categorical.apply(label_encoder.fit_transform)

# Combine categorical and numeric features
X_numeric = X.select_dtypes(include=['int64', 'float64'])
X = pd.concat([X_numeric, X_categorical], axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the target variable to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Create the model
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
