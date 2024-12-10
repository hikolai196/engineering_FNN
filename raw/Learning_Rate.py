import numpy as np

class StepDecaySchedule:
    def __init__(self, initial_lr, decay_factor, step_size):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size

    def __call__(self, epoch):
        lr = self.initial_lr * (self.decay_factor ** np.floor(epoch / self.step_size))
        return lr