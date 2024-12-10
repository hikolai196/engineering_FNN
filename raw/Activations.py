import numpy as np

# ReLU
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        dinputs = dvalues.copy()
        dinputs[self.inputs <= 0] = 0
        return dinputs
  
# Softmap
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
        self.inputs = inputs

    def backward(self, dvalues):
        dinputs = np.empty_like(dvalues)
        for i, output in enumerate(self.output):
            jacobian_matrix = np.diag(output) - np.outer(output, output)
            dinputs[i] = np.dot(jacobian_matrix, dvalues[i])
        return dinputs