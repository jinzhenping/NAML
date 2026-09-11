#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np


def mrr_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    rr = y_true / (np.arange(len(y_true)) + 1.0)
    s = float(np.sum(y_true))
    return float(np.sum(rr) / s) if s > 0 else 0.0


def dcg_score(y_true, y_score, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true, y_score, k: int = 5) -> float:
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return float(actual / best) if best > 0 else 0.0


def hit_at_k(y_true, y_score, k: int = 1) -> float:
    order = np.argsort(y_score)[::-1][:k]
    return float(1.0 if np.any(np.asarray(y_true)[order] > 0) else 0.0)


def session_metrics(labels, scores) -> dict:
    return {
        "MRR": mrr_score(labels, scores),
        "NDCG@5": ndcg_score(labels, scores, k=5),
        "Hit@1": hit_at_k(labels, scores, k=1),
    }
