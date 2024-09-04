import gzip
import json

import torch
import torch.nn as nn
from habitat import Config
from torch import Tensor


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

        self.embedding_layer = nn.Embedding.from_pretrained(
                    embeddings=self._load_embeddings(),
                    freeze=True,
        )

        self.map_layer = nn.Linear(50,embed_dim)


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
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """

        input = observations.long()
        out = self.map_layer(self.embedding_layer(input))
        
        return out