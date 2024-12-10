import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Define and load training data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_train.csv')
y_dir = os.path.join(dir_path,'y_train.csv')

X_train = pd.read_csv(X_dir).to_numpy()
y_train = pd.read_csv(y_dir).to_numpy()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(8, 12)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNN()

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.7)

# Track the best loss
lowest_loss = float('inf')
best_weights = None

# Training loop
for i in range(100000):
    # Forward pass
    predictions = model(X_train)
    loss = loss_function(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = 1 - loss.item()

    # Check for best weights
    if loss.item() < lowest_loss:
        print(f'Replaced, iter: {i}, loss: {loss.item()}, error: {acc}')
        lowest_loss = loss.item()
        best_weights = {name: param.clone() for name, param in model.named_parameters()}

    # Rollback to the best weights if the current loss increases
    else:
        for name, param in model.named_parameters():
            param.data = best_weights[name].data.clone()

print('--------------------done--------------------')
print(acc)

# Save the best weights and biases
for name, param in best_weights.items():
    np.savetxt(f'Best_{name}.txt', param.detach().numpy(), fmt='%f')