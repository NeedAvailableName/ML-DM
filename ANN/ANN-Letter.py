import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
# Load the dataset
data = pd.read_csv('letter-recognition.data', header=None)

# Split input features and target variable
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode target variable to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create the ANN model
model = Sequential()

# Add the input layer and the first hidden layer
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Add additional hidden layers
model.add(Dense(units=32, activation='relu'))

# Add the output layer
model.add(Dense(units=26, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
# Evaluate the model on the testing dataset
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss*100:.2f}%')
print(f'Test Accuracy: {accuracy*100:.2f}%')
