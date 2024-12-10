import os
import numpy as np
import pandas as pd
from DenseLayer import Layer_Dense
from Activations import Activation_ReLU
from MSE_Loss import Loss_MSE

# Define and load test data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_test.csv')
y_dir = os.path.join(dir_path,'y_test.csv')

X_test = pd.read_csv(X_dir).to_numpy()
y_test = pd.read_csv(y_dir).to_numpy()

# Load fixed model parameter
w1 = np.loadtxt('Best_W1.txt', dtype=float)
b1 = np.loadtxt('Best_B1.txt', dtype=float)
w2 = np.loadtxt('Best_W2.txt', dtype=float)
b2 = np.loadtxt('Best_B2.txt', dtype=float)

# extablish model with trained parameter
dense1_trained = Layer_Dense(8, 12)
dense1_trained.weights = w1
dense1_trained.biases = b1

dense2_trained = Layer_Dense(12, 1)
dense2_trained.weights = w2
dense2_trained.biases = b2

activation1 = Activation_ReLU()

# Apply the trained layer to new input data
dense1_trained.forward(X_test)
activation1.forward(dense1_trained.output)
dense2_trained.forward(activation1.output)

y_pred = dense2_trained.output[np.newaxis]

# Compare the predicted labels to the true labels
results = np.column_stack([y_pred.T*1000, y_test*1000, (y_pred.T-y_test)*100/y_test])
accuracy = (1-np.mean(abs((y_pred.T-y_test)/y_test)))

print('Average accuracy on new data: {:.0%}'.format(accuracy))
np.set_printoptions(suppress=True,precision=3)
print(results)