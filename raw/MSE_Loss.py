import numpy as np

class Loss_MSE:
    def forward(self, y_pred, y_true):
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, y_pred, y_true):
        dvalues = 2 * (y_pred - y_true) / y_true.size
        return dvalues