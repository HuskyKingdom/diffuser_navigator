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

        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hiddens = self.gru(packed_input)
        # 解包序列
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))

        # 根据序列长度获取每个序列的最后一个有效时间步的输出
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(output.size(0), 1, output.size(2))
        idx = idx.to(torch.long)
        last_output = output.gather(1, idx).squeeze(1)

        print(hiddens.shape)
        assert 1==2

        out = self.fc(last_output)
        return out
