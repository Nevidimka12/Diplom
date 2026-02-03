from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from models.transformer_encoder import TransformerAudioEncoder
from models.mamba_encoder import MambaAudioEncoder
from models.zeros_encoder import ZeroSAudioEncoder


@dataclass
class TransformerMTLConfig:
    in_dim: int

    encoder_type: str = "transformer"   # "transformer" | "mamba" | "zeros"
    zeros_use_norm: bool = True

    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 5000

    mamba_d_state: int = 16
    mamba_d_conv: int = 3
    mamba_expand: int = 2

    emo_mosei_out_dim: int = 7
    emo_resd_out_dim: int = 7
    pers_out_dim: int = 5
    ah_out_dim: int = 2


class MultiTaskTransformerModel(nn.Module):
    def __init__(self, cfg: TransformerMTLConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.encoder_type == "transformer":
            self.encoder = TransformerAudioEncoder(
                in_dim=cfg.in_dim,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                num_layers=cfg.num_layers,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                max_len=cfg.max_len,
            )
        elif cfg.encoder_type == "mamba":
            self.encoder = MambaAudioEncoder(
                in_dim=cfg.in_dim,
                d_model=cfg.d_model,
                num_layers=cfg.num_layers,
                d_state=cfg.mamba_d_state,
                d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand,
                dropout=cfg.dropout,
                max_len=cfg.max_len,
            )
        elif cfg.encoder_type == "zeros":
            self.encoder = ZeroSAudioEncoder(
                in_dim=cfg.in_dim,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                max_len=cfg.max_len,
                zeros_use_norm=cfg.zeros_use_norm,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")

        hidden_dim = cfg.d_model

        self.emo_mosei_head = nn.Linear(hidden_dim, cfg.emo_mosei_out_dim)
        self.emo_resd_head = nn.Linear(hidden_dim, cfg.emo_resd_out_dim)
        self.personality_head = nn.Linear(hidden_dim, cfg.pers_out_dim)
        self.ah_head = nn.Linear(hidden_dim, cfg.ah_out_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        lengths = batch["lengths"]

        seq, pooled = self.encoder(x, lengths)

        emo_mosei_pred = self.emo_mosei_head(pooled)
        emo_resd_logits = self.emo_resd_head(pooled)
        personality_preds = self.personality_head(pooled)
        ah_logits = self.ah_head(pooled)

        return {
            "emotion_mosei_pred": emo_mosei_pred,
            "emotion_resd_logits": emo_resd_logits,
            "personality_preds": personality_preds,
            "ah_logits": ah_logits,
        }
