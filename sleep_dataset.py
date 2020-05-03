from torch.utils.data import Dataset
from data_set import *
from scipy.stats import mode


class SleepDataset(Dataset):

    def __init__(self, train, val, test, weights):
        super(SleepDataset, self).__init__()
        self.train = train
        self.train_size = len(train)
        self.val = val
        self.val_size = len(val)
        self.test = test
        self.test_size = len(test)
        self.weights = 1 / weights
        self._target_split = None
        self._target, self._target_size = None, None
        self._sample_weight = None
        self._lookup_dict = {DataSet.TRAIN: (self.train, self.train_size, self.get_weight(self.train)),
                             DataSet.VAL: (self.val, self.val_size, self.get_weight(self.train)),
                             DataSet.TEST: (self.test, self.test_size, self.get_weight(self.train))}
        self.set_split(DataSet.TRAIN)

    def get_weight(self, data):
        """
        calculate the weights for each data point
        since each data point is a bucket of epochs, the weight is determined by the mode of classes among all epochs
        """
        weights = []
        for single_sample in data:
            labels = single_sample[1]
            mode_label = mode(labels)[0][0]
            weight = self.weights[int(mode_label)-1]
            weights.append(weight)
        return weights

    def set_split(self, split=DataSet.TRAIN):
        """
        switch between training, testing and validation
        """
        self._target_split = split
        self._target, self._target_size, self._sample_weight = self._lookup_dict[split]
        pass

    def get_sample_weight(self):
        return self._sample_weight

    def get_target(self):
        return self._target

    def __len__(self):
        return self._target_size

    def __getitem__(self, item):
        sample = self._target[item]
        return sample

