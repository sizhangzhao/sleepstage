import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch
import numpy as np
from sleep_cnn import *
from hand_rnn import *


class SleepCNNOnly(nn.Module):

    """
    Model to pretrain CNN model
    """

    def __init__(self, small_kernel, large_kernel, channel_number, output_size, dropout_ratio=0.5, device="cpu", step=50):
        super(SleepCNNOnly, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.cnn = SleepCNN(small_kernel, large_kernel, channel_number, output_size, dropout_ratio)
        self.final_projection = nn.Linear(output_size, 5, bias=True)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.device = device

    def forward(self, features: torch.Tensor, lengths: List[int]):
        x = self.cnn(features)
        batch_size, seq_length = x.size()[:2]
        zero_mask = torch.from_numpy(np.array(
            [[1.0 if i < lengths[idx] else 0 for i in range(seq_length)] for idx in
             range(batch_size)])).float().to(self.device)
        P = self.dropout(self.final_projection(x))
        return P, zero_mask, lengths

