#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# Reshape the input images to a 4D tensor with shape (batch_size, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Normalize the pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert the target labels to categorical format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
model = Sequential()

# Add the first convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add the first max-pooling layer with a 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Add the second convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add the second max-pooling layer with a 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Add the third convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Add a dense layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))

# Add a dropout layer with a rate of 0.5 to reduce overfitting
model.add(Dropout(0.5))

# Add the final dense layer with 10 units (one for each digit) and softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.2f}')
print(f'Test accuracy: {accuracy:.2f}')

