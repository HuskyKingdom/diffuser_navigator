import torch
import math
from torch import nn
from diffuser_baselines.models.common.position_encodings import PositionalEncoding,RotaryPositionEncoding

# 假设特征维度为 128
feature_dim = 128

# 创建位置编码模块
rotary_pe = RotaryPositionEncoding(feature_dim)

# 假设有一个批量大小为 2，序列长度为 10 的输入
batch_size = 2
seq_length = 10

# 生成位置索引（通常是 [0, 1, 2, ..., seq_length-1]）
position_indices = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_length]

position_indices = position_indices.unsqueeze(-1)

# 传入位置编码模块
position_encoding = rotary_pe(position_indices)

# 输出形状
print("Position encoding shape:", position_encoding.shape)