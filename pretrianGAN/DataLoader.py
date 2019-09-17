import numpy as np


class DataLoader:
    def __init__(self, file):
        self.data = np.load(file)

    def next_batch(self, batch_size):
        return self.data[np.random.choice(self.data.shape[0], batch_size)]/255
