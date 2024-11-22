import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from torch import Tensor

from transformers import BertTokenizer, BertModel
from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule
from diffuser_baselines.models.common.position_encodings import PositionalEncoding

from transformers import BertTokenizer,BertModel



class InstructionEncoder(nn.Module):
    def __init__(self, config: Config,embed_dim) -> None:
        """An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: If True, return just the final state
        """
        super().__init__()

        self.config = config

        # bert pre-train
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert_model.parameters(): # frozon
            param.requires_grad = False



        
        
        self.map_layer = nn.Linear(768, embed_dim)


        # self.embedding_layer = nn.Embedding.from_pretrained(
        #             embeddings=self._load_embeddings(),
        #             freeze=True,
        # )

        # self.pe_layer = PositionalEncoding(embed_dim,0.1)
        # self.language_self_atten = FFWRelativeSelfAttentionModule(embed_dim,4,2,dropout=0.1)

    def _load_embeddings(self) -> Tensor:
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.MODEL.INSTRUCTION_ENCODER.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, observations) -> Tensor:

        

        
        outputs = self.bert_model(**observations).last_hidden_state

        # assert 1==2


        # input = observations.long()


        # out = self.map_layer(self.embedding_layer(input))
        # out = self.pe_layer(out)

        # out,_ = self.language_self_atten(out.transpose(0,1), diff_ts=None,
        #         query_pos=None, context=None, context_pos=None,pad_mask=pad_mask,vis=False,ins_text=ins_text)
        # out = out[-1].transpose(0,1)
        
        return outputs