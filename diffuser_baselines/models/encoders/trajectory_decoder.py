import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from torch import Tensor

from transformers import BertTokenizer, BertModel
from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule, FFWRelativeDecoderModule
from diffuser_baselines.models.common.position_encodings import PositionalEncoding,RotaryPositionEncoding

class TrajectoryDecoder(nn.Module):
    def __init__(self, config: Config,embedding_dim,num_attention_heads,num_layers, action_dim) -> None:

        super().__init__()

        self.config = config


        self.pe_layer = PositionalEncoding(embedding_dim,0.1)
        # self.pe_layer = RotaryPositionEncoding(embedding_dim)

        self.decoder = FFWRelativeDecoderModule(embedding_dim,num_attention_heads,num_layers,dropout=0.1)

        self.action_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, action_dim)
        )

        # pg monitor
        self.progress_monitor = nn.Linear(embedding_dim, 1)



    def forward(self, dec_input,dec_pad_mask, enc_out, enc_pad_mask, causal_mask, ins_text=None) -> Tensor:
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """

        # # rotary positional encoding
        # batch_size, seq_length, embedding_dim = dec_input.shape
        # q_position_indices = torch.arange(seq_length, device=dec_input.device).unsqueeze(0).repeat(batch_size, 1)
        # batch_size, seq_length, embedding_dim = enc_out.shape
        # k_position_indices = torch.arange(seq_length, device=dec_input.device).unsqueeze(0).repeat(batch_size, 1)

        

        # q_pos = self.pe_layer(q_position_indices)
        # k_pos = self.pe_layer(k_position_indices)

        # fixed pe
        dec_input = self.pe_layer(dec_input)


        decoder_out, avg_weights = self.decoder(dec_input.transpose(0,1), enc_out.transpose(0,1),
                diff_ts=None,
                query_pos=None, 
                value_pos=None,
                q_pad_mask=dec_pad_mask,
                k_pad_mask=enc_pad_mask,
                causal_mask=causal_mask,
                vis=False,
                ins_text=ins_text)
        

        
        decoder_out = decoder_out[-1].transpose(0,1)
        
        pred_action_logits = self.action_predictor(decoder_out)
        pred_progress = self.progress_monitor(decoder_out)

        print(pred_action_logits.shape, pred_progress.shape)
        assert 1==2
        
        self.avg_weights = avg_weights

        return pred_action_logits