import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HistoryGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(HistoryGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hiddens,lengths=None, inference=False):
        
         # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size)

        if not inference:
            packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, hiddens = self.gru(packed_input,hiddens)
            # hiddens: (num_layers, batch_size, hidden_size)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)

            
            print(output.shape)
            print(lengths.shape)
            print(lengths)

            assert 1==2

        else:
            _, hiddens = self.gru(x,hiddens)

        last_hidden = hiddens[-1]  # (batch_size, hidden_size)

        out = self.fc(last_hidden)
        return out,hiddens