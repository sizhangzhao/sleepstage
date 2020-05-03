import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch
import numpy as np
from sleep_cnn import *
from hand_rnn import *


class SleepCNNRNN(nn.Module):

	"""
	Wrapper model for CNN RNN model
	"""

    def __init__(self, small_kernel, large_kernel, channel_number, output_size, hidden_size, dropout_ratio=0.5, device="cpu", step=50):
        super(SleepCNNRNN, self).__init__()
        self.cnn = SleepCNN(small_kernel, large_kernel, channel_number, output_size, dropout_ratio)
        self.rnn = HandRNN(output_size, hidden_size, dropout_ratio, device, step)

    def forward(self, features: torch.Tensor, lengths: List[int]):
        x = self.cnn(features)
        x = self.rnn(x, lengths)
        return x

