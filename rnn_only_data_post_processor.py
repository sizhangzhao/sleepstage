from typing import List, Tuple
import numpy as np
from functools import reduce
from utils import *


class RNNPostProcessor:

    """
    Post processor for hand RNN model
    main function is to pad the sequence in the batch so they are have the same length
    """

    def __init__(self, device, max_length=50):
        self.device = device
        self.max_length = max_length

    def __call__(self, batches: [List[Tuple]]):
        labels = []
        features = []
        lengths = []
        batches = sorted(batches, key=lambda batch: len(batch[1]), reverse=True)
        for batch in batches:
            label = batch[1] - 1
            feature = batch[0].to_numpy()
            length = len(label)
            lengths.append(length)
            padded_label = np.zeros((1, self.max_length))
            padded_label[0, 0:length] = label
            padded_feature = np.zeros((1, self.max_length, feature.shape[1]))
            padded_feature[0, 0:length, :] = feature
            labels.append(padded_label)
            features.append(padded_feature)
        labels = reduce(lambda x1, x2: np.concatenate((x1, x2), axis=0), labels)
        features = reduce(lambda x1, x2: np.concatenate((x1, x2), axis=0), features)
        return to_tensor(features, self.device, "float"), to_tensor(labels, self.device, "long"), lengths