import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from torch import Tensor

from transformers import BertTokenizer, BertModel
from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule
from diffuser_baselines.models.common.position_encodings import PositionalEncoding

class TrajectoryDecoder(nn.Module):
    def __init__(self, config: Config,embedding_dim,num_attention_heads,num_layers) -> None:

        super().__init__()

        self.config = config

        dec_dim = int(embedding_dim*2+embedding_dim/2)

        self.pe_layer = PositionalEncoding(dec_dim,0.2)
        self.sa_decoder = FFWRelativeSelfAttentionModule(dec_dim,num_attention_heads,num_layers)
        self.ca_decoder = FFWRelativeCrossAttentionModule(dec_dim,num_attention_heads,num_layers)


    def forward(self, dec_input,dec_pad_mask, enc_out, enc_pad_mask, causal_mask) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """

        dec_input = self.pe_layer(dec_input)
        selfatten_out = self.sa_decoder(dec_input.transpose(0,1), diff_ts=None,
                query_pos=None, context=None, context_pos=None,pad_mask=dec_pad_mask,causal_mask=causal_mask)[-1].transpose(0,1)
        
        crossatten_out = self.ca_decoder(query=selfatten_out.transpose(0, 1),
            value=enc_out.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=None,pad_mask=enc_pad_mask)[-1].transpose(0,1)
        

        return crossatten_out