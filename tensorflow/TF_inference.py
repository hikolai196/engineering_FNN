import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Define the neural network structure
class SimpleNN(tf.keras.Model):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(12, activation='relu', input_shape=(8,))
        self.fc2 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x

# Define and load test data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path, 'X_test.csv')
y_dir = os.path.join(dir_path, 'y_test.csv')

X_test = pd.read_csv(X_dir).to_numpy().astype(np.float32)
y_test = pd.read_csv(y_dir).to_numpy().astype(np.float32)

# Load trained parameters
w1 = np.loadtxt('Best_W1.txt', dtype=float)
b1 = np.loadtxt('Best_B1.txt', dtype=float)
w2 = np.loadtxt('Best_W2.txt', dtype=float)
b2 = np.loadtxt('Best_B2.txt', dtype=float)

# Initialize the model and set the trained weights
model = SimpleNN()
model.build((None, 8))  # Build the model with the appropriate input shape

# Assign trained weights and biases
model.fc1.set_weights([w1.T, b1])  # Note: TensorFlow expects weights in transposed format
model.fc2.set_weights([w2.T, b2])  # Same for the second layer

# Perform inference
y_pred = model.predict(X_test)

# Reshape predictions and compute results
y_pred = y_pred.flatten()
results = np.column_stack([
    y_pred * 1000, 
    y_test.flatten() * 1000, 
    (y_pred - y_test.flatten()) * 100 / y_test.flatten()
])
accuracy = 1 - np.mean(np.abs((y_pred - y_test.flatten()) / y_test.flatten()))

# Print results
print('Average accuracy on new data: {:.0%}'.format(accuracy))
np.set_printoptions(suppress=True, precision=3)
print(results)