import numpy as np
from pandas.io.sas.sas_constants import dataset_length
import tensorflow as tf
from sklearn.utils import shuffle


class DataLoader(object):
    def __init__(self, dataset, batch_size=100, shuffle=False, random_stat=None):
        self.dataset = list(zip(dataset[0], dataset[1]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        if random_state is None:
            random_stat = np.random.RandomState(1234)
        self.random_state = random_state
        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        x, y = zip(*self.dataset[self._idx:(self_idx + self.batch_size)])

        x = tf.convert_to_tensor(x)  # convert np.array to tf.Tensor
        y = tf.convert_to_tensor(y)

        self._idx += self.batch_size

        return x, y

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset, random_state=self.random_state)
        self._idx = 0


