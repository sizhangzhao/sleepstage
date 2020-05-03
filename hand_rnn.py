import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torch
import numpy as np


class HandRNN(nn.Module):

    """
    Model for Seq2Seq model with encoder, decoder and attention
    Model structure explanation please check in report
    """

    def __init__(self, feature_size, hidden_size, dropout_ratio=0.5, device="cpu", step=50):
        super(HandRNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.device = device
        self.step = step
        self.encoder = nn.LSTM(feature_size, hidden_size, bidirectional=True, batch_first=True, dropout=self.dropout_ratio)
        self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.decoder = nn.LSTMCell(feature_size + hidden_size, hidden_size)
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=True)
        self.final_projection = nn.Linear(hidden_size, 5, bias=True)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, features: torch.Tensor, lengths: List[int]):
        batch_size, seq_length = features.size()[:2]
        enc_hiddens, dec_init_state = self.encode(features, lengths)
        mask = torch.from_numpy(np.array(
            [[0.0 if i < lengths[idx] else 1.0 for i in range(seq_length)] for idx in
             range(batch_size)])).float().to(self.device)
        dec_output = self.decode(features, enc_hiddens, dec_init_state, mask, batch_size)
        zero_mask = torch.from_numpy(np.array(
            [[1.0 if i < lengths[idx] else 0 for i in range(seq_length)] for idx in
             range(batch_size)])).float().to(self.device)
        P = self.dropout(self.final_projection(dec_output))
        '''
        this should also work, with nn.CrossEntropyLoss
        P = F.softmax(self.dropout(self.final_projection(dec_output)), dim=-1)
        dec_output = P * zero_mask.unsqueeze(-1)
        dec_output = dec_output - 1
        dec_output = dec_output * zero_mask.unsqueeze(-1)
        dec_output = dec_output + 1
        return dec_output.transpose(1, 2)
        '''
        return P, zero_mask, lengths

    def encode(self, features, lengths):
        X = nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        enc_hiddens = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True, total_length=self.step)[0]
        last_hidden_reformat = torch.cat([last_hidden[0, :], last_hidden[1, :]], 1)
        init_decoder_hidden = self.dropout(self.h_projection(last_hidden_reformat))
        last_cell_reformat = torch.cat([last_cell[0, :], last_cell[1, :]], 1)
        init_decoder_cell = self.dropout(self.c_projection(last_cell_reformat))
        dec_init_state = [init_decoder_hidden, init_decoder_cell]
        return enc_hiddens, dec_init_state

    def decode(self, features, enc_hiddens, dec_init_state, mask, batch_size):
        enc_hiddens_proj = self.dropout(self.att_projection(enc_hiddens))
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []
        dec_state = dec_init_state
        for i in range(self.step):
            input = features[:, i, :].squeeze(1)
            input_hat = torch.cat([input, o_prev], 1)
            dec_state = self.decoder(input_hat, dec_state)
            dec_hidden, _ = dec_state
            e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2)), 2)
            e_t.data.masked_fill_(mask.byte(), -float('inf'))
            alpha_t = F.softmax(e_t, 1)
            a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens), dim=1)
            U_t = torch.cat([a_t, dec_hidden], 1)
            V_t = self.combined_output_projection(U_t)
            O_t = self.dropout(torch.tanh(V_t))
            combined_outputs.append(O_t)
            # if (O_t != O_t).any():
            #     hi = 1
            o_prev = O_t
        combined_outputs = torch.stack(combined_outputs, dim=1)
        return combined_outputs



