from typing import Optional, Tuple

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaAudioEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_layers: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, d_model)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _make_src_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        device = lengths.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        return idxs >= lengths.unsqueeze(1)  # (B, T)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        seq_out = x

        if lengths is not None:
            pad_mask = self._make_src_key_padding_mask(lengths, T)
            mask = (~pad_mask).unsqueeze(-1).float()
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = seq_out.mean(dim=1)

        return seq_out, pooled
