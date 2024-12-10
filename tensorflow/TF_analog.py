import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Define and load training data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path, 'X_train.csv')
y_dir = os.path.join(dir_path, 'y_train.csv')

X_train = pd.read_csv(X_dir).to_numpy().astype(np.float32)
y_train = pd.read_csv(y_dir).to_numpy().astype(np.float32)

# Define the model
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(12, activation='relu', input_shape=(8,))
        self.dense2 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = SimpleModel()

# Define the optimizer and loss
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.7)

# Training loop
lowest_loss = float('inf')
best_weights = model.get_weights()

for i in range(100000):
    with tf.GradientTape() as tape:
        predictions = model(X_train, training=True)
        loss = loss_function(y_train, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    acc = 1 - loss.numpy()  # Example accuracy calculation (1 - loss)

    if loss.numpy() < lowest_loss:
        print(f'Replaced, iter: {i}, loss: {loss.numpy()}, error: {acc}')
        best_weights = model.get_weights()
        lowest_loss = loss.numpy()
    else:
        model.set_weights(best_weights)

print('--------------------done--------------------')
print(acc)

# Save best weights
best_dense1_weights, best_dense1_biases, best_dense2_weights, best_dense2_biases = best_weights
np.savetxt('Best_W1.txt', best_dense1_weights, fmt='%f')
np.savetxt('Best_B1.txt', best_dense1_biases, fmt='%f')
np.savetxt('Best_W2.txt', best_dense2_weights, fmt='%f')
np.savetxt('Best_B2.txt', best_dense2_biases, fmt='%f')