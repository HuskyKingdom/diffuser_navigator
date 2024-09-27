import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HistoryGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(HistoryGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
         # x: (batch_size, seq_len, input_size)
        # lengths: (batch_size) 每个序列的实际长度

        # 使用 pack_padded_sequence 处理不等长序列
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hiddens = self.gru(packed_input)
        # hiddens: (num_layers, batch_size, hidden_size)

        # 提取最后一层的隐藏状态
        last_hidden = hiddens[-1]  # (batch_size, hidden_size)

        # 通过全连接层
        out = self.fc(last_hidden)
        return out