import torch
import math
from torch import nn
from diffuser_baselines.models.common.position_encodings import PositionalEncoding,RotaryPositionEncoding


# 示例调用
batch_size = 2
seq_length = 10
feature_dim = 128

rotary_pe = RotaryPositionEncoding(feature_dim)

# 生成位置索引 (batch_size, seq_length)
position_indices = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_length]

# 获取位置编码
position_encoding = rotary_pe(position_indices)

# 输出形状
print("Position encoding shape:", position_encoding.shape)
