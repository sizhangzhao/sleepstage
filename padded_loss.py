import torch.nn.functional as F
import numpy as np
import torch
from utils import *


class PaddedCrossEntropyLoss(object):

    """
    This is the padded loss used across all RNN model
    we have to take padded mask into consideration when calculate cross entropy
    as padded sequence shouldn't contribute anything
    """

    def __call__(self, logits, target, mask, length, device):
        length = to_tensor(np.array(length), device)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss
