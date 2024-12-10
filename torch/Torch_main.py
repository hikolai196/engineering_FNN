import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define and load training data
dir_path = os.getcwd()
X_dir = os.path.join(dir_path,'X_train.csv')
y_dir = os.path.join(dir_path,'y_train.csv')

X_train = pd.read_csv(X_dir).to_numpy()
y_train = pd.read_csv(y_dir).to_numpy()

X_train = torch.tensor(X_train).to(device)
y_train = torch.tensor(y_train).to(device)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

# Training loop
lowest_loss = float('inf')
best_model_state = None

for iteration in range(100000):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Compute accuracy
    _, predictions = torch.max(outputs, 1)
    acc = (predictions == y_train).float().mean().item()

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the best model
    if loss.item() < lowest_loss:
        print(f'W and B replaced, iteration: {iteration}, loss: {loss.item()}, accuracy: {acc}')
        best_model_state = model.state_dict()
        lowest_loss = loss.item()

    else:
        model.load_state_dict(best_model_state)

# Save best model parameters
print('--------------------done--------------------')
torch.save(best_model_state, "best_model.pth")

# Save the weights and biases to text files
best_fc1_weights = model.fc1.weight.data.cpu().numpy()
best_fc1_biases = model.fc1.bias.data.cpu().numpy()
best_fc2_weights = model.fc2.weight.data.cpu().numpy()
best_fc2_biases = model.fc2.bias.data.cpu().numpy()

np.savetxt('Best_W1.txt', best_fc1_weights, fmt='%f')
np.savetxt('Best_B1.txt', best_fc1_biases, fmt='%f')
np.savetxt('Best_W2.txt', best_fc2_weights, fmt='%f')
np.savetxt('Best_B2.txt', best_fc2_biases, fmt='%f')