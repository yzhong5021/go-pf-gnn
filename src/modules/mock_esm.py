"""
mock_esm.py

Stand-in for the ESM1b transformer. Creates initial token embeddings with positional encoding via simple learned embedding + 2-layer MLP.
"""

import torch.nn as nn

N_AA = 21

class MockESM(nn.Module):
    
    def __init__(self, seq_len, hidden_len, embed_len, proj_len):
        super().__init__()
        self.embed = nn.Embedding(N_AA, embed_len)
        self.layer1 = nn.Linear(embed_len, hidden_len)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_len, proj_len)

        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.embed.embedding_dim ** -0.5)
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
    
    def forward(self, seqs):
        x = self.embed(seqs)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x