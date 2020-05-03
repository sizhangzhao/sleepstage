import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch
import numpy as np


class SleepCNN(nn.Module):

    """
    Model for CNN model
    Model structure explanation please check in report
    """

    num_feature = 3000

    def __init__(self, small_kernel, large_kernel, channel_number, output_size, dropout_ratio=0.5, device="cpu", step=50):
        super(SleepCNN, self).__init__()
        self.small_kernel = small_kernel
        self.large_kernel = large_kernel
        self.channel_number = channel_number
        self.output_size = output_size
        self.dropout_ratio = dropout_ratio

        self.small_cnn_1 = torch.nn.Conv1d(2, int(channel_number / 2), small_kernel)
        self.small_pool_1 = torch.nn.MaxPool1d(kernel_size=int(small_kernel/2), stride=2, padding=0)
        self.small_feature_1 = (self.num_feature - self.small_kernel + 1 - int(small_kernel/2)) // 2 + 1

        self.small_cnn_2 = torch.nn.Conv1d(int(channel_number / 2), channel_number, small_kernel * 2)
        self.small_pool_2 = torch.nn.MaxPool1d(kernel_size=small_kernel, stride=small_kernel, padding=0)
        self.small_feature_2 = (self.small_feature_1 - self.small_kernel * 2 + 1 - small_kernel) // small_kernel + 1

        self.large_cnn_1 = torch.nn.Conv1d(2, int(channel_number / 2), large_kernel * 2)
        self.large_pool_1 = torch.nn.MaxPool1d(kernel_size=int(large_kernel / 2), stride=2, padding=0)
        self.large_feature_1 = (self.num_feature - self.large_kernel + 1 - int(large_kernel/2)) // 2 + 1

        self.large_cnn_2 = torch.nn.Conv1d(int(channel_number / 2), channel_number, large_kernel)
        self.large_pool_2 = torch.nn.MaxPool1d(kernel_size=large_kernel, stride=large_kernel, padding=0)
        self.large_feature_2 = (self.large_feature_1 - self.large_kernel * 2 + 1 - large_kernel) // large_kernel + 1

        self.fc1 = torch.nn.Linear(channel_number * (self.small_feature_2 + self.large_feature_2), 64)
        self.fc2 = torch.nn.Linear(64, output_size)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, features):
        batch_size, length, num_channel, num_feature = features.size()
        features = features.view(batch_size * length, num_channel, num_feature)

        small = F.relu(self.small_cnn_1(features))
        small = self.small_pool_1(small)
        small = F.relu(self.small_cnn_2(small))
        small = self.small_pool_2(small)
        large = F.relu(self.large_cnn_1(features))
        large = self.large_pool_1(large)
        large = F.relu(self.large_cnn_2(large))
        large = self.large_pool_2(large)

        concat = torch.cat([small, large], dim=-1)
        concat = concat.view(batch_size * length, -1)

        x = F.relu(self.dropout(self.fc1(concat)))
        x = self.dropout(self.fc2(x))

        x = x.view(batch_size, length, -1)

        return x

