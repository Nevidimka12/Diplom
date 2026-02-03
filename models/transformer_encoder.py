import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class TransformerAudioEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def _make_src_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        device = lengths.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        return idxs >= lengths.unsqueeze(1)  # (B, T)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = self.pos_encoder(x)

        src_key_padding_mask = self._make_src_key_padding_mask(lengths, T) if lengths is not None else None

        seq_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if lengths is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = seq_out.mean(dim=1)

        return seq_out, pooled
