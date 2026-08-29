#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
학습셋 실제본문 CLIP text + 썸네일 CLIP image → Δ
학습/테스트 기대본문 CLIP text + Δ → L2 기대이미지 캐시

  python CLIP/build_expected_image_embeds.py --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from clip_embeddings import (
    default_actual_body_text_cache_path,
    default_b1_cache_path,
    default_b1_train_cache_path,
    default_cache_path,
    default_delta_cache_path,
    default_expected_image_test_path,
    default_expected_image_train_path,
    resolve_project_path,
)
from expected_image import (
    apply_delta_to_text_pairs,
    compute_text_image_delta,
    load_pair_dict_normed,
    news_embed_dict,
    save_delta,
    save_pair_embed_dict,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Δ and expected-image CLIP caches")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--actual-body-text-cache", type=str, default=None)
    ap.add_argument("--thumbnail-clip-cache", type=str, default=None)
    ap.add_argument("--train-expected-text-cache", type=str, default=None)
    ap.add_argument("--test-expected-text-cache", type=str, default=None)
    ap.add_argument("--delta-out", type=str, default=None)
    ap.add_argument("--train-out", type=str, default=None)
    ap.add_argument("--test-out", type=str, default=None)
    ap.add_argument("--no-l2", action="store_true", help="Δ 더한 뒤 L2를 하지 않음 (기본은 L2)")
    args = ap.parse_args()

    sub = args.mind_dataset_subdir
    actual_path = resolve_project_path(args.actual_body_text_cache) if args.actual_body_text_cache else default_actual_body_text_cache_path(sub)
    thumb_path = resolve_project_path(args.thumbnail_clip_cache) if args.thumbnail_clip_cache else default_cache_path(sub)
    train_text_path = (
        resolve_project_path(args.train_expected_text_cache)
        if args.train_expected_text_cache
        else default_b1_train_cache_path(sub)
    )
    test_text_path = (
        resolve_project_path(args.test_expected_text_cache)
        if args.test_expected_text_cache
        else default_b1_cache_path(sub)
    )
    delta_path = resolve_project_path(args.delta_out) if args.delta_out else default_delta_cache_path(sub)
    train_out = resolve_project_path(args.train_out) if args.train_out else default_expected_image_train_path(sub)
    test_out = resolve_project_path(args.test_out) if args.test_out else default_expected_image_test_path(sub)
    do_l2 = not bool(args.no_l2)

    for p, name in (
        (actual_path, "실제본문 CLIP text"),
        (thumb_path, "썸네일 CLIP image"),
        (train_text_path, "학습 기대본문 CLIP text"),
    ):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} cache 없음: {p}")

    text_news = news_embed_dict(actual_path)
    image_news = news_embed_dict(thumb_path)
    mu_text, mu_img, delta, mean_ids = compute_text_image_delta(text_news, image_news)
    print(
        f"[delta] n_news={len(mean_ids)} dim={delta.shape[0]}\n"
        f"[delta] ||μ_text||={float(np.linalg.norm(mu_text)):.6f} "
        f"||μ_img||={float(np.linalg.norm(mu_img)):.6f} "
        f"||Δ||={float(np.linalg.norm(delta)):.6f}",
        flush=True,
    )
    save_delta(
        delta_path,
        mu_text,
        mu_img,
        delta,
        mean_ids,
        extra={
            "actual_body_text_cache": os.path.abspath(actual_path),
            "thumbnail_clip_cache": os.path.abspath(thumb_path),
            "l2_after_shift": do_l2,
        },
    )
    print(f"[delta] saved {delta_path}", flush=True)

    train_text = load_pair_dict_normed(train_text_path)
    train_img = apply_delta_to_text_pairs(train_text, delta, l2=do_l2)
    n_nz = sum(1 for v in train_img.values() if np.any(v))
    save_pair_embed_dict(train_out, train_img)
    print(
        f"[expected-image train] pairs={len(train_img)} nonzero={n_nz} → {train_out}",
        flush=True,
    )

    if os.path.isfile(test_text_path):
        test_text = load_pair_dict_normed(test_text_path)
        test_img = apply_delta_to_text_pairs(test_text, delta, l2=do_l2)
        n_nz_te = sum(1 for v in test_img.values() if np.any(v))
        save_pair_embed_dict(test_out, test_img)
        print(
            f"[expected-image test] pairs={len(test_img)} nonzero={n_nz_te} → {test_out}",
            flush=True,
        )
    else:
        print(f"[expected-image test] skip (cache 없음): {test_text_path}", flush=True)


if __name__ == "__main__":
    main()
