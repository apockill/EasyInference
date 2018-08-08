import random

import numpy as np


def batch_iterator(data: np.ndarray, batch_size, shuffle=True):
    """Return a batch of data for each batch until it's done"""

    num_batches = int(data.shape[0] / batch_size)

    if shuffle:
        np.random.shuffle(data)

    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]