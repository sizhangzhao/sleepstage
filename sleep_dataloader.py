import numpy as np
import torch
from torch.utils.data import DataLoader
from sleep_dataset import *
from data_set import *


class SleepDataLoader:

    def __init__(self, train, val, test, transformer, weights=None):
        self.weights = weights
        self.transformer = transformer
        self.dataset = SleepDataset(train, val, test, weights)
        self.split = DataSet.TRAIN
        self.set_data_set()

    def set_data_set(self, split=DataSet.TRAIN):
        """
        distinguish whether we are using training, testing or validation data
        """
        self.split = split
        self.dataset.set_split(split)
        return

    def generate_batches(self, batch_size, shuffle=True, drop_last=True):
        """
        To generate batches, for training, we are using weighted sample so small population classes 
        can be sampled more
        For validation or test, no weighted sample is used as we only want to check the performance
        """
        if self.split == DataSet.TRAIN:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(self.dataset.get_sample_weight(), len(self.dataset))
            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, drop_last=drop_last,
                                    num_workers=0, collate_fn=self.transformer, pin_memory=False, sampler=sampler)
        else:
            dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                    num_workers=0, collate_fn=self.transformer, pin_memory=False)
        for data in dataloader:
            yield data

