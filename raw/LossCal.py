import numpy as np

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_Catagorical_Cross_Entropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        
        #vector array
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #one hot encoded array
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        
        if len(y_true.shape) == 1:
            y_true = np.eye(np.max(y_true) + 1)[y_true]

        dvalues = y_pred - y_true
        dvalues = dvalues / samples

        return dvalues