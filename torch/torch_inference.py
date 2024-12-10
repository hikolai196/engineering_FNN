import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Define the neural network structure
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(8, 12)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define and load test data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_test.csv')
y_dir = os.path.join(dir_path,'y_test.csv')

X_test = pd.read_csv(X_dir).to_numpy()
y_test = pd.read_csv(y_dir).to_numpy()

# Load trained parameters
w1 = np.loadtxt('Best_W1.txt', dtype=float)
b1 = np.loadtxt('Best_B1.txt', dtype=float)
w2 = np.loadtxt('Best_W2.txt', dtype=float)
b2 = np.loadtxt('Best_B2.txt', dtype=float)

# Initialize the model and set the trained weights
model = SimpleNN()
model.fc1.weight.data = torch.tensor(w1, dtype=torch.float32)
model.fc1.bias.data = torch.tensor(b1, dtype=torch.float32)
model.fc2.weight.data = torch.tensor(w2, dtype=torch.float32)
model.fc2.bias.data = torch.tensor(b2, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Perform forward pass
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

# Reshape predictions and compute results
y_pred = y_pred.squeeze()
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