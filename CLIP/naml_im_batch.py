#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NAML-IM 배치 생성 (full-text, S2와 동일 슬롯 순서 + IM 텐서)."""
from __future__ import annotations

from typing import List

import numpy as np

from im_features import ImNewsFeatures
from train_s1_s2 import _split_by_slot


def _im_cand_hist_parts(cand_i: np.ndarray, hist_i: np.ndarray, im: ImNewsFeatures) -> List[np.ndarray]:
    """model 입력 순: cand×5 im fields, then hist×5 (각 field별)."""
    fields = (
        im.word_vis,
        im.word_vis_mask,
        im.cover_regions,
        im.category_vis,
        im.global_feat,
    )
    parts: List[np.ndarray] = []
    for arr in fields:
        parts.extend(_split_by_slot(arr[cand_i]))
        parts.extend(_split_by_slot(arr[hist_i]))
    return parts


def generate_batch_data_train_im(
    all_train_pn,
    all_label,
    all_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    batch_size,
    im: ImNewsFeatures,
):
    n = len(all_label)
    inputid = np.arange(n)
    np.random.shuffle(inputid)
    batches = [
        inputid[range(batch_size * i, min(n, batch_size * (i + 1)))]
        for i in range((n + batch_size - 1) // batch_size)
        if batch_size * i < n
    ]
    while True:
        for idx in batches:
            cand_i = all_train_pn[idx]
            hist_i = all_user_pos[idx]
            parts = _split_by_slot(news_words[cand_i]) + _split_by_slot(news_words[hist_i])
            parts = (
                parts
                + _split_by_slot(news_body[cand_i])
                + _split_by_slot(news_body[hist_i])
                + _split_by_slot(news_v[cand_i])
                + _split_by_slot(news_v[hist_i])
                + _split_by_slot(news_sv[cand_i])
                + _split_by_slot(news_sv[hist_i])
            )
            parts = parts + _im_cand_hist_parts(cand_i, hist_i, im)
            yield (parts, np.asarray(all_label[idx], dtype=np.float32))


def generate_batch_data_test_im(
    all_test_pn,
    all_test_label,
    all_test_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    batch_size,
    im: ImNewsFeatures,
):
    n = len(all_test_label)
    inputid = np.arange(n)
    batches = [
        inputid[range(batch_size * i, min(n, batch_size * (i + 1)))]
        for i in range((n + batch_size - 1) // batch_size)
        if batch_size * i < n
    ]
    while True:
        for idx in batches:
            cand_i = all_test_pn[idx]
            hist_i = all_test_user_pos[idx]
            parts = [news_words[cand_i]] + _split_by_slot(news_words[hist_i])
            parts = (
                parts
                + [news_body[cand_i]]
                + _split_by_slot(news_body[hist_i])
                + [news_v[cand_i]]
                + _split_by_slot(news_v[hist_i])
                + [news_sv[cand_i]]
                + _split_by_slot(news_sv[hist_i])
            )
            im_parts = []
            fields = (
                im.word_vis,
                im.word_vis_mask,
                im.cover_regions,
                im.category_vis,
                im.global_feat,
            )
            for arr in fields:
                im_parts.append(arr[cand_i])
                im_parts.extend(_split_by_slot(arr[hist_i]))
            parts = parts + im_parts
            yield (parts, np.asarray(all_test_label[idx], dtype=np.float32))
