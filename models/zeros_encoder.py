from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.zeros_attention import ZeroSAttention


class ZeroSEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, block_size: int, init_n_layers: int):
        super().__init__()

        config = SimpleNamespace(
            n_embd=d_model,
            n_head=n_heads,
            bias=False,
            dropout=dropout,
            block_size=block_size,
            is_causal=False,
            init_params=True,
            use_norm=getattr(self, "zeros_use_norm", True),
            use_associative=True,
            init_n_layers=init_n_layers,
        )
        self.attn = ZeroSAttention(config)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x


class ZeroSAudioEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 15000,
        zeros_use_norm: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.d_model = d_model
        self.num_layers = num_layers

        if in_dim != d_model:
            self.input_proj = nn.Linear(in_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.layers = nn.ModuleList(
            [
                ZeroSEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    block_size=max_len,
                    init_n_layers=num_layers,
                )
                for _ in range(num_layers)
            ]
        )
        for layer in self.layers:
            layer.zeros_use_norm = zeros_use_norm

        self.out_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _make_mask(lengths: Optional[torch.Tensor], max_len: int) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        device = lengths.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        mask = idxs < lengths.unsqueeze(1)                        # (B, T)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)

        seq_out = self.out_norm(x)  # (B, T, d_model)

        if lengths is not None:
            mask = self._make_mask(lengths, T)  # (B, T)
            mask_f = mask.unsqueeze(-1).float()
            summed = (seq_out * mask_f).sum(dim=1)            # (B, d_model)
            denom = mask_f.sum(dim=1).clamp(min=1.0)          # (B, 1)
            pooled = summed / denom
        else:
            pooled = seq_out.mean(dim=1)

        return seq_out, pooled
