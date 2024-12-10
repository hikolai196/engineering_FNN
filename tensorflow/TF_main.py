import os
import tensorflow as tf
import numpy as np
import pandas as pd

# Define and load training data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_train.csv')
y_dir = os.path.join(dir_path,'y_train.csv')

X_train = pd.read_csv(X_dir).to_numpy().astype(np.float32)
y_train = pd.read_csv(y_dir).to_numpy().astype(np.float32)

# Define the model
class SimpleNN(tf.keras.Model):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(5, activation='relu', input_shape=(4,))
        self.fc2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.5)

# Prepare the dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(len(X_train))

# Training loop
lowest_loss = float('inf')
best_weights = None

for iteration in range(100000):
    for batch_X, batch_y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_X, training=True)
            loss = loss_function(batch_y, predictions)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Compute accuracy
        pred_classes = tf.argmax(predictions, axis=1, output_type=tf.int64)
        acc = tf.reduce_mean(tf.cast(pred_classes == batch_y, tf.float32)).numpy()

        # Save the best model
        if loss.numpy() < lowest_loss:
            print(f'W and B replaced, iteration: {iteration}, loss: {loss.numpy()}, accuracy: {acc}')
            best_weights = model.get_weights()
            lowest_loss = loss.numpy()

        else:
            model.set_weights(best_weights)

# Save best weights
print('--------------------done--------------------')
model.set_weights(best_weights)

# Extract weights and biases
best_fc1_weights, best_fc1_biases = model.fc1.get_weights()
best_fc2_weights, best_fc2_biases = model.fc2.get_weights()

# Save weights and biases to text files
np.savetxt('Best_W1.txt', best_fc1_weights, fmt='%f')
np.savetxt('Best_B1.txt', best_fc1_biases, fmt='%f')
np.savetxt('Best_W2.txt', best_fc2_weights, fmt='%f')
np.savetxt('Best_B2.txt', best_fc2_biases, fmt='%f')