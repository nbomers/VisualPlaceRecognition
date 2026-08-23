import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.linear = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, x):
        x = self.linear(x)
        return F.normalize(x, dim=-1)
