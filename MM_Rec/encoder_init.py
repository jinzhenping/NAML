#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ViLBERT 뉴스 인코더 로드. 체크포인트가 없으면 BERT-base 텍스트 가중치를 복사한다."""
from __future__ import annotations

import logging
import os

import torch

from model import news_encoder

logger = logging.getLogger(__name__)


def copy_bert_text_weights(enc: news_encoder) -> int:
    """bert-base-uncased 의 embedding + 앞쪽 text layer 를 ViLBERT text tower 에 복사."""
    try:
        from transformers import BertModel as HFBert
    except ImportError as exc:
        raise ImportError("transformers 가 필요합니다: pip install transformers") from exc

    hf = HFBert.from_pretrained("bert-base-uncased")
    copied = 0
    with torch.no_grad():
        src_emb = hf.embeddings.state_dict()
        dst_emb = enc.bert.embeddings.state_dict()
        mapped = {}
        for k, v in src_emb.items():
            if k in dst_emb and dst_emb[k].shape == v.shape:
                mapped[k] = v
        missing, unexpected = enc.bert.embeddings.load_state_dict(mapped, strict=False)
        copied += len(mapped)

        n_layers = min(len(hf.encoder.layer), len(enc.bert.encoder.layer))
        for i in range(n_layers):
            src = hf.encoder.layer[i].state_dict()
            dst = enc.bert.encoder.layer[i].state_dict()
            layer_map = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
            enc.bert.encoder.layer[i].load_state_dict(layer_map, strict=False)
            copied += len(layer_map)
    logger.info("copied %s tensors from bert-base-uncased into news_encoder", copied)
    return copied


def build_news_encoder(config, from_pretrained: str, default_gpu: bool = True):
    ckpt = (from_pretrained or "").strip()
    if ckpt and os.path.isfile(ckpt):
        logger.info("loading ViLBERT / news encoder weights from %s", ckpt)
        enc = news_encoder.from_pretrained(ckpt, config, default_gpu=default_gpu)
        if enc is None:
            raise FileNotFoundError(f"failed to load encoder from {ckpt}")
        return enc
    logger.warning(
        "ViLBERT 체크포인트가 없습니다 (%s). bert-base-uncased 텍스트 가중치로 초기화합니다. "
        "논문 재현이 필요하면 vilbert pytorch_model_8.bin 을 --from_pretrained 로 넘기세요.",
        ckpt or "(empty)",
    )
    enc = news_encoder(config, default_gpu=default_gpu)
    copy_bert_text_weights(enc)
    return enc
