import os
import pandas as pd
import numpy as np
from DenseLayer import Layer_Dense
from Activations import Activation_ReLU
from MSE_Loss import Loss_MSE

# Define and load training data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_train.csv')
y_dir = os.path.join(dir_path,'y_train.csv')

X_train = pd.read_csv(X_dir).to_numpy()
y_train = pd.read_csv(y_dir).to_numpy()

# Define linear regression layers and activation functions
dense1 = Layer_Dense(8,12)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(12,1)

# Call the loss function
loss_function = Loss_MSE()

# Define inital loss and replace with lower loss in iteration
lowest_loss = 9999999

# Define variables to store the best weight and biase
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# Initialize momentum variables
dense1_momentum_weights = np.zeros_like(dense1.weights)
dense1_momentum_biases = np.zeros_like(dense1.biases)
dense2_momentum_weights = np.zeros_like(dense2.weights)
dense2_momentum_biases = np.zeros_like(dense2.biases)

# Set learning rate and momentum rate
momentum_rate = 0.7
lr = 0.05

for i in range(100000):

    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    # Calculate the loss
    loss = loss_function.forward(dense2.output, y_train)

    # Backward pass
    loss_grad = loss_function.backward(dense2.output, y_train)
    dense2_grad = dense2.backward(loss_grad)
    activation1_grad = activation1.backward(dense2_grad)
    dense1_grad = dense1.backward(activation1_grad)

    #predictions = np.argmax(dense2.output, axis=1)
    acc = 1-loss

    # Update weights and biases SGD
    dense1_momentum_weights = momentum_rate * dense1_momentum_weights + (1 - momentum_rate) * dense1.dweights
    dense1_momentum_biases = momentum_rate * dense1_momentum_biases + (1 - momentum_rate) * dense1.dbiases
    dense1.weights -= lr * dense1_momentum_weights
    dense1.biases -= lr * dense1_momentum_biases

    dense2_momentum_weights = momentum_rate * dense2_momentum_weights + (1 - momentum_rate) * dense2.dweights
    dense2_momentum_biases = momentum_rate * dense2_momentum_biases + (1 - momentum_rate) * dense2.dbiases
    dense2.weights -= lr * dense2_momentum_weights
    dense2.biases -= lr * dense2_momentum_biases

    if loss < lowest_loss:
        print('Replaced, iter:', i,
                'loss:', loss,
                'error:', acc)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

print('--------------------done--------------------')
print(acc)
np.savetxt('Best_W1.txt',best_dense1_weights, fmt='%f')
np.savetxt('Best_B1.txt',best_dense1_biases, fmt='%f')
np.savetxt('Best_W2.txt',best_dense2_weights, fmt='%f')
np.savetxt('Best_B2.txt',best_dense2_biases, fmt='%f')