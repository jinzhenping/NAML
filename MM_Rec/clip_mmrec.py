#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MM-Rec + CLIP news encoder.

news_vec_t = Linear(CLIP_text(title))
news_vec_v = Linear(CLIP_image(thumbnail))
user_encoder / scoring 은 원본 MM-Rec 과 동일.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import user_encoder


class ClipNewsEncoder(nn.Module):
    """Frozen CLIP matrices + trainable projections to hidden_size."""

    def __init__(
        self,
        clip_t: torch.Tensor,
        clip_v: torch.Tensor,
        hidden_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if clip_t.ndim != 2 or clip_v.ndim != 2:
            raise ValueError("clip_t / clip_v must be [N, D]")
        if clip_t.shape != clip_v.shape:
            raise ValueError(f"clip shape mismatch t={tuple(clip_t.shape)} v={tuple(clip_v.shape)}")
        dim = int(clip_t.shape[1])
        self.hidden_size = int(hidden_size)
        self.register_buffer("clip_t", clip_t.float().contiguous())
        self.register_buffer("clip_v", clip_v.float().contiguous())
        self.proj_t = nn.Linear(dim, self.hidden_size)
        self.proj_v = nn.Linear(dim, self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))

    @property
    def n_news(self) -> int:
        return int(self.clip_t.shape[0])

    def encode_all(self) -> tuple:
        t = self.dropout(self.proj_t(self.clip_t))
        v = self.dropout(self.proj_v(self.clip_v))
        return t, v

    def lookup(self, news_ids: torch.Tensor) -> tuple:
        """news_ids: LongTensor of any shape with global news indices."""
        flat = news_ids.reshape(-1)
        # clamp invalid to 0 (padding)
        flat = flat.clamp(min=0, max=self.n_news - 1)
        t = self.proj_t(F.embedding(flat, self.clip_t))
        v = self.proj_v(F.embedding(flat, self.clip_v))
        t = self.dropout(t).view(*news_ids.shape, self.hidden_size)
        v = self.dropout(v).view(*news_ids.shape, self.hidden_size)
        return t, v


class mmrec_clip(nn.Module):
    def __init__(self, clip_t: torch.Tensor, clip_v: torch.Tensor, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.news_encoder = ClipNewsEncoder(clip_t, clip_v, hidden_size=hidden_size, dropout=dropout)
        self.user_encoder = user_encoder()
        self.criterion = nn.CrossEntropyLoss()
        self.hidden_size = int(hidden_size)

    def forward(self, input_ids, log_ids, log_mask, targets, compute_loss=True):
        """
        input_ids: [B, n_cand] global news index
        log_ids:   [B, hist] global news index
        log_mask:  [B, hist]
        """
        imp_t, imp_v = self.news_encoder.lookup(input_ids)
        his_t, his_v = self.news_encoder.lookup(log_ids)
        user_vector = self.user_encoder(imp_t, imp_v, his_t, his_v, log_mask)
        score = torch.sum((imp_t + imp_v) * user_vector, dim=-1)
        if compute_loss:
            return self.criterion(score, targets), score
        return score

    def encode_news_tables(self) -> tuple:
        self.news_encoder.eval()
        with torch.no_grad():
            # eval: no dropout for stable scoring tables
            t = self.news_encoder.proj_t(self.news_encoder.clip_t)
            v = self.news_encoder.proj_v(self.news_encoder.clip_v)
        return t.detach().cpu().numpy(), v.detach().cpu().numpy()
