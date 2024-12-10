import os
import pandas as pd
import numpy as np
from DenseLayer import Layer_Dense
from Activations import Activation_ReLU, Activation_Softmax
from LossCal import Loss, Loss_Catagorical_Cross_Entropy

# Define and load training data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_train.csv')
y_dir = os.path.join(dir_path,'y_train.csv')

X_train = pd.read_csv(X_dir).to_numpy()
y_train = pd.read_csv(y_dir).to_numpy()

# Define layers and activation functions
dense1 = Layer_Dense(4,5)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(5,2)
activation2 = Activation_Softmax()

# Define the loss function
loss_function = Loss_Catagorical_Cross_Entropy()

# Define inital loss
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
learning_rate = 0.05
momentum_rate = 0.5

for iteration in range(100000):

    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate the loss
    loss = loss_function.calculate(activation2.output, y_train)

    # Backward pass
    loss_grad = loss_function.backward(activation2.output, y_train)
    activation2_grad = activation2.backward(loss_grad)
    dense2_grad = dense2.backward(activation2_grad)
    activation1_grad = activation1.backward(dense2_grad)
    dense1_grad = dense1.backward(activation1_grad)

    predictions = np.argmax(activation2.output, axis=1)
    acc = np.mean(predictions == y_train)

    # Update weights and biases SGD
    dense1_momentum_weights = momentum_rate * dense1_momentum_weights + (1 - momentum_rate) * dense1.dweights
    dense1_momentum_biases = momentum_rate * dense1_momentum_biases + (1 - momentum_rate) * dense1.dbiases
    dense1.weights -= learning_rate * dense1_momentum_weights
    dense1.biases -= learning_rate * dense1_momentum_biases

    dense2_momentum_weights = momentum_rate * dense2_momentum_weights + (1 - momentum_rate) * dense2.dweights
    dense2_momentum_biases = momentum_rate * dense2_momentum_biases + (1 - momentum_rate) * dense2.dbiases
    dense2.weights -= learning_rate * dense2_momentum_weights
    dense2.biases -= learning_rate * dense2_momentum_biases

    if loss < lowest_loss:
        print('W and B replaced, iteration:', iteration,
                'loss:', loss,
                'accuracy:', acc)
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
print(best_dense1_weights)
print(best_dense1_biases)
print(best_dense2_weights)
print(best_dense2_biases)
print(acc)
np.savetxt('Best_W1.txt',best_dense1_weights, fmt='%f')
np.savetxt('Best_B1.txt',best_dense1_biases, fmt='%f')
np.savetxt('Best_W2.txt',best_dense2_weights, fmt='%f')
np.savetxt('Best_B2.txt',best_dense2_biases, fmt='%f')