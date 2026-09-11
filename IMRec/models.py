#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NRMS-IM and FIM-IM (IMRec, ACM MM 2021)."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, D]
        a = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # [B, L]
        if mask is not None:
            a = a.masked_fill(mask <= 0, -1e9)
            empty = mask.sum(dim=-1, keepdim=True) <= 0
        else:
            empty = None
        w = torch.softmax(a, dim=-1)
        if empty is not None:
            w = torch.where(empty, torch.zeros_like(w), w)
        return (w.unsqueeze(-1) * x).sum(dim=1)


class MemoryImpression(nn.Module):
    """Local impression: memory attend cues → enhance word embeddings."""

    def __init__(self, word_dim: int, cue_dim: int, dropout: float = 0.2):
        super().__init__()
        self.q = nn.Linear(word_dim, word_dim)
        self.k = nn.Linear(cue_dim, word_dim)
        self.v = nn.Linear(cue_dim, word_dim)
        self.v_word = nn.Linear(word_dim, word_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        word_emb: torch.Tensor,
        word_mask: torch.Tensor,
        cues: torch.Tensor,
        cue_mask: torch.Tensor,
    ) -> torch.Tensor:
        # word_emb [B,L,D], cues [B,M,C]
        q = self.q(word_emb)  # [B,L,D]
        k = self.k(cues)  # [B,M,D]
        v = self.v(cues)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))  # [B,L,M]
        scores = scores.masked_fill(cue_mask.unsqueeze(1) <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * word_mask.unsqueeze(-1)
        enhanced = torch.matmul(attn, v) + self.v_word(word_emb)
        return self.drop(enhanced)


class SelfAttnEnhance(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k = self.q(x), self.k(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(mask.unsqueeze(1) <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, x)
        return self.drop(out)


class GlobalGateFusion(nn.Module):
    def __init__(self, news_dim: int, global_dim: int):
        super().__init__()
        self.proj_g = nn.Linear(global_dim, news_dim)
        self.gate = nn.Linear(news_dim * 2, 1)

    def forward(self, e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        o = self.proj_g(g)
        a = torch.sigmoid(self.gate(torch.cat([e, o], dim=-1)))
        return a * e + (1.0 - a) * o


class MultiHeadUserEncoder(nn.Module):
    def __init__(self, dim: int, n_heads: int = 3, head_dim: int = 50, attn_hidden: int = 200, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q = nn.Linear(dim, n_heads * head_dim)
        self.k = nn.Linear(dim, n_heads * head_dim)
        self.v = nn.Linear(dim, n_heads * head_dim)
        self.out = nn.Linear(n_heads * head_dim, dim)
        self.pool = AdditiveAttention(dim, attn_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, news: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # news [B,H,D]
        B, H, D = news.shape
        q = self.q(news).view(B, H, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(news).view(B, H, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(news).view(B, H, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, H, -1)
        ctx = self.drop(self.out(ctx))
        return self.pool(ctx, mask)


class NewsEncoderIM(nn.Module):
    """Shared local+global impression news encoder (NRMS-IM style)."""

    def __init__(
        self,
        vocab_size: int,
        emb_mat: Optional[torch.Tensor],
        word_dim: int = 100,
        cue_dim: int = 512,
        global_dim: int = 2048,
        attn_hidden: int = 200,
        dropout: float = 0.2,
        freeze_emb: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if emb_mat is not None:
            self.embedding.weight.data.copy_(emb_mat)
        self.embedding.weight.requires_grad = not freeze_emb
        self.cue_proj = nn.Linear(cue_dim, cue_dim)
        self.memory = MemoryImpression(word_dim, cue_dim, dropout)
        self.self_attn = SelfAttnEnhance(word_dim, dropout)
        self.word_pool = AdditiveAttention(word_dim, attn_hidden)
        self.global_fuse = GlobalGateFusion(word_dim, global_dim)
        self.drop = nn.Dropout(dropout)
        self.news_dim = word_dim

    def build_cues(self, word_vis, word_vis_mask, cover_regions, category_vis):
        # word_vis [B,L,512], cover [B,9,512], category [B,512]
        B = word_vis.size(0)
        cat = category_vis.unsqueeze(1)
        cues = torch.cat([word_vis, cover_regions, cat], dim=1)
        cues = self.cue_proj(cues)
        cue_mask = torch.cat(
            [
                word_vis_mask,
                torch.ones(B, cover_regions.size(1), device=word_vis.device),
                torch.ones(B, 1, device=word_vis.device),
            ],
            dim=1,
        )
        # zero-pad news: disable all cues
        return cues, cue_mask

    def forward(self, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.drop(self.embedding(pack["title_ids"]))
        mask = pack["title_mask"]
        cues, cue_mask = self.build_cues(
            pack["word_vis"], pack["word_vis_mask"], pack["cover_regions"], pack["category_vis"]
        )
        hat = self.memory(emb, mask, cues, cue_mask)
        star = self.self_attn(hat, mask)
        e = self.word_pool(star, mask)
        e_star = self.global_fuse(e, pack["global_feat"])
        return e_star


class NRMSIM(nn.Module):
    def __init__(self, news_encoder: NewsEncoderIM, n_heads: int = 3, head_dim: int = 50, attn_hidden: int = 200, dropout: float = 0.2):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = MultiHeadUserEncoder(
            news_encoder.news_dim, n_heads=n_heads, head_dim=head_dim, attn_hidden=attn_hidden, dropout=dropout
        )

    def encode_news_batch(self, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pack fields [B, N, ...]
        B, N = pack["title_ids"].shape[:2]
        flat = {k: v.reshape(B * N, *v.shape[2:]) for k, v in pack.items()}
        out = self.news_encoder(flat)
        return out.view(B, N, -1)

    def forward_train(self, history, candidates, hist_mask, labels=None):
        h = self.encode_news_batch(history)  # [B,H,D]
        c = self.encode_news_batch(candidates)  # [B,C,D]
        u = self.user_encoder(h, hist_mask)  # [B,D]
        scores = torch.bmm(c, u.unsqueeze(-1)).squeeze(-1)  # [B,C]
        loss = F.cross_entropy(scores, labels.argmax(dim=-1)) if labels is not None else None
        return scores, loss

    def forward_eval(self, history, candidates, hist_mask):
        # candidates: [B, ...] single candidate each
        h = self.encode_news_batch(history)
        u = self.user_encoder(h, hist_mask)
        flat = candidates
        c = self.news_encoder(flat)  # [B,D]
        scores = (c * u).sum(dim=-1)
        return scores


class HDCEncoder(nn.Module):
    def __init__(self, word_dim: int = 100, filter_num: int = 150, window: int = 3, seq_len: int = 32):
        super().__init__()
        self.filter_num = filter_num
        self.seq_len = seq_len
        pad = (window - 1) // 2
        self.conv1 = nn.Conv1d(word_dim, filter_num, window, padding=pad, dilation=1)
        self.conv2 = nn.Conv1d(filter_num, filter_num, window, padding=pad + 1, dilation=2)
        self.conv3 = nn.Conv1d(filter_num, filter_num, window, padding=pad + 2, dilation=3)
        self.ln1 = nn.LayerNorm([filter_num, seq_len])
        self.ln2 = nn.LayerNorm([filter_num, seq_len])
        self.ln3 = nn.LayerNorm([filter_num, seq_len])

    def forward(self, word_seq: torch.Tensor) -> tuple:
        # word_seq [B, L, D] -> pad/truncate to seq_len
        B, L, D = word_seq.shape
        if L < self.seq_len:
            pad = torch.zeros(B, self.seq_len - L, D, device=word_seq.device, dtype=word_seq.dtype)
            word_seq = torch.cat([word_seq, pad], dim=1)
        else:
            word_seq = word_seq[:, : self.seq_len]
        d0 = word_seq.transpose(1, 2)  # [B,D,S]
        d1 = F.relu(self.ln1(self.conv1(d0)))
        d2 = F.relu(self.ln2(self.conv2(d1)))
        d3 = F.relu(self.ln3(self.conv3(d2)))
        dL = torch.stack([d1, d2, d3], dim=1)  # [B,3,F,S]
        return d0, dL


class FIMIM(nn.Module):
    """
    FIM + local memory on word embeddings + global matching scores (paper FIM-IM).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_mat: Optional[torch.Tensor],
        word_dim: int = 100,
        cue_dim: int = 512,
        global_dim: int = 2048,
        max_title_len: int = 30,
        max_history: int = 50,
        hdc_filters: int = 150,
        conv3d_1: int = 32,
        conv3d_2: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if emb_mat is not None:
            self.embedding.weight.data.copy_(emb_mat)
        self.memory = MemoryImpression(word_dim, cue_dim, dropout)
        self.cue_proj = nn.Linear(cue_dim, cue_dim)
        self.drop = nn.Dropout(dropout)
        self.seq_len = max_title_len  # title only for HDC (simplified vs cat+subcat)
        self.hdc = HDCEncoder(word_dim, hdc_filters, seq_len=self.seq_len)
        self.max_history = max_history
        self.scalar = math.sqrt(float(hdc_filters))
        self.conv3d_a = nn.Conv3d(4, conv3d_1, kernel_size=3, padding=1)
        self.conv3d_b = nn.Conv3d(conv3d_1, conv3d_2, kernel_size=3, padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.adapt = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.global_proj = nn.Linear(global_dim, word_dim)
        fim_dim = conv3d_2 * 4 * 4 * 4
        self.fim_dim = fim_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(fim_dim + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _cues(self, pack):
        B = pack["word_vis"].size(0)
        cues = torch.cat(
            [pack["word_vis"], pack["cover_regions"], pack["category_vis"].unsqueeze(1)],
            dim=1,
        )
        cues = self.cue_proj(cues)
        cue_mask = torch.cat(
            [
                pack["word_vis_mask"],
                torch.ones(B, pack["cover_regions"].size(1), device=cues.device),
                torch.ones(B, 1, device=cues.device),
            ],
            dim=1,
        )
        return cues, cue_mask

    def encode_words(self, pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.drop(self.embedding(pack["title_ids"]))
        cues, cue_mask = self._cues(pack)
        return self.memory(emb, pack["title_mask"], cues, cue_mask)

    def hdc_pair(self, pack: Dict[str, torch.Tensor]):
        words = self.encode_words(pack)
        return self.hdc(words)

    def fim_match(self, cand_d0, cand_dL, hist_d0, hist_dL, hist_mask):
        # cand: [B,C,F,S] / [B,C,3,F,S]; hist: [B,H,F,S]
        B, C = cand_d0.shape[:2]
        H = hist_d0.size(1)
        S = cand_d0.size(-1)
        Fdim = cand_d0.size(2)
        # pad/truncate history to max_history
        if H < self.max_history:
            pad_h = self.max_history - H
            hist_d0 = F.pad(hist_d0, (0, 0, 0, 0, 0, pad_h))
            hist_dL = F.pad(hist_dL, (0, 0, 0, 0, 0, 0, 0, pad_h))
            hist_mask = F.pad(hist_mask, (0, pad_h))
        elif H > self.max_history:
            hist_d0 = hist_d0[:, -self.max_history :]
            hist_dL = hist_dL[:, -self.max_history :]
            hist_mask = hist_mask[:, -self.max_history :]
        H = self.max_history

        cand_d0_t = cand_d0.unsqueeze(2).permute(0, 1, 2, 4, 3)  # B C 1 S F
        hist_d0_t = hist_d0.unsqueeze(1)  # B 1 H F S
        m0 = torch.matmul(cand_d0_t, hist_d0_t) / self.scalar  # B C H S S

        cand_dL_t = cand_dL.unsqueeze(2).permute(0, 1, 2, 3, 5, 4)  # B C 1 3 S F
        hist_dL_t = hist_dL.unsqueeze(1)  # B 1 H 3 F S
        mL = torch.matmul(cand_dL_t, hist_dL_t) / self.scalar  # B C H 3 S S

        matching = torch.cat([m0.unsqueeze(3), mL.permute(0, 1, 2, 3, 4, 5)], dim=3)
        matching = matching.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = matching.view(B * C, 4, H, S, S)
        # mask empty history channels softly by zeroing
        q1 = F.elu(self.conv3d_a(x))
        q1 = self.pool3d(q1)
        q2 = F.elu(self.conv3d_b(q1))
        q2 = self.pool3d(q2)
        q2 = self.adapt(q2)
        salient = q2.view(B, C, -1)
        return salient

    def global_match(self, cand_g, hist_g, hist_mask):
        # cand_g [B,C,Dg], hist_g [B,H,Dg]
        cg = self.global_proj(cand_g)
        hg = self.global_proj(hist_g)
        # max cosine-like match over history
        cg_n = F.normalize(cg, dim=-1)
        hg_n = F.normalize(hg, dim=-1)
        sim = torch.bmm(cg_n, hg_n.transpose(1, 2))  # B C H
        sim = sim.masked_fill(hist_mask.unsqueeze(1) <= 0, -1e9)
        return sim.max(dim=-1).values.unsqueeze(-1)  # B C 1

    def forward_train(self, history, candidates, hist_mask, labels=None):
        B, H = history["title_ids"].shape[:2]
        C = candidates["title_ids"].size(1)

        def flat_encode(pack, n1, n2):
            flat = {k: v.reshape(n1 * n2, *v.shape[2:]) for k, v in pack.items()}
            d0, dL = self.hdc_pair(flat)
            d0 = d0.view(n1, n2, *d0.shape[1:])
            dL = dL.view(n1, n2, *dL.shape[1:])
            g = flat["global_feat"].view(n1, n2, -1)
            return d0, dL, g

        h0, hL, hg = flat_encode(history, B, H)
        c0, cL, cg = flat_encode(candidates, B, C)
        salient = self.fim_match(c0, cL, h0, hL, hist_mask)
        gmatch = self.global_match(cg, hg, hist_mask)
        feats = torch.cat([salient, gmatch], dim=-1)
        scores = self.score_mlp(feats).squeeze(-1)
        loss = F.cross_entropy(scores, labels.argmax(dim=-1)) if labels is not None else None
        return scores, loss

    def forward_eval(self, history, candidates, hist_mask):
        # wrap single candidate as C=1
        cand = {k: v.unsqueeze(1) for k, v in candidates.items()}
        scores, _ = self.forward_train(history, cand, hist_mask, labels=None)
        return scores.squeeze(1)


def build_model(name: str, tables, args) -> nn.Module:
    emb = torch.tensor(tables.embedding_mat, dtype=torch.float32)
    if name.lower() in ("nrms-im", "nrms_im", "nrmsim"):
        news = NewsEncoderIM(
            vocab_size=len(tables.word_dict),
            emb_mat=emb,
            word_dim=args.word_dim,
            cue_dim=512,
            global_dim=2048,
            attn_hidden=args.attn_hidden,
            dropout=args.dropout,
        )
        return NRMSIM(news, n_heads=args.n_heads, head_dim=args.head_dim, attn_hidden=args.attn_hidden, dropout=args.dropout)
    if name.lower() in ("fim-im", "fim_im", "fimim"):
        return FIMIM(
            vocab_size=len(tables.word_dict),
            emb_mat=emb,
            word_dim=args.word_dim,
            max_title_len=args.max_title_len,
            max_history=args.max_history,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model: {name}")
