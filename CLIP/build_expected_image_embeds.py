#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
학습셋 실제본문 embedding + 썸네일 CLIP image → Δ
학습/테스트 기대본문 embedding + Δ → L2 기대이미지 캐시

--source clip_text (기본):
  Δ = mean(CLIP_image(thumbnail)) − mean(CLIP_text(actual_body))
  expected_image = L2(CLIP_text(expected_body) + Δ)

--source prior:
  Δ_prior = mean(CLIP_image(thumbnail)) − mean(prior(actual_body))
  expected_image = L2(prior(expected_body) + Δ_prior)

CLIP-text 캐시는 --source prior 때 덮어쓰지 않는다.

  python CLIP/build_expected_image_embeds.py --mind-dataset-subdir MIND_2000
  python CLIP/build_expected_image_embeds.py --source prior --mind-dataset-subdir MIND_2000

  # 이미 있는 Δ로 최종 test 기대이미지만 만들기
  python CLIP/build_expected_image_embeds.py --apply-delta-only --mind-dataset-subdir MIND_2000
  python CLIP/build_expected_image_embeds.py --source prior --apply-delta-only --mind-dataset-subdir MIND_2000
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
    default_cache_path,
    expected_image_recipe_paths,
    resolve_project_path,
)
from expected_image import (
    apply_delta_to_text_pairs,
    compute_text_image_delta,
    load_delta,
    load_pair_dict_normed,
    news_embed_dict,
    save_delta,
    save_pair_embed_dict,
)


def _apply_and_save(src_path: str, out_path: str, delta: np.ndarray, do_l2: bool, label: str) -> None:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"{label} source cache 없음: {src_path}")
    src_pairs = load_pair_dict_normed(src_path)
    img_pairs = apply_delta_to_text_pairs(src_pairs, delta, l2=do_l2)
    n_nz = sum(1 for v in img_pairs.values() if np.any(v))
    save_pair_embed_dict(out_path, img_pairs)
    print(
        f"[expected-image {label}] pairs={len(img_pairs)} nonzero={n_nz} → {out_path}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Δ and expected-image CLIP caches")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--source",
        type=str,
        default="clip_text",
        choices=["clip_text", "prior"],
        help="clip_text: CLIP text+Δ. prior: Kandinsky prior+Δ_prior",
    )
    ap.add_argument("--actual-body-text-cache", type=str, default=None)
    ap.add_argument("--thumbnail-clip-cache", type=str, default=None)
    ap.add_argument("--train-expected-text-cache", type=str, default=None)
    ap.add_argument("--test-expected-text-cache", type=str, default=None)
    ap.add_argument("--delta-out", type=str, default=None)
    ap.add_argument("--train-out", type=str, default=None)
    ap.add_argument("--test-out", type=str, default=None)
    ap.add_argument("--no-l2", action="store_true", help="Δ 더한 뒤 L2를 하지 않음 (기본은 L2)")
    ap.add_argument(
        "--apply-delta-only",
        action="store_true",
        help="저장된 Δ만 로드해 기대본문 임베딩에 적용. train/Δ는 다시 쓰지 않음. "
        "기본 입력/출력은 test_final 캐시",
    )
    args = ap.parse_args()

    sub = args.mind_dataset_subdir
    paths = expected_image_recipe_paths(sub, args.source)
    delta_path = resolve_project_path(args.delta_out) if args.delta_out else paths["delta"]
    do_l2 = not bool(args.no_l2)
    src_label = "prior" if args.source == "prior" else "CLIP text"

    if args.apply_delta_only:
        if not os.path.isfile(delta_path):
            raise FileNotFoundError(f"Δ cache 없음: {delta_path}")
        test_src_path = (
            resolve_project_path(args.test_expected_text_cache)
            if args.test_expected_text_cache
            else paths["test_final_src"]
        )
        test_out = (
            resolve_project_path(args.test_out)
            if args.test_out
            else paths["test_final_image"]
        )
        delta = load_delta(delta_path)
        print(
            f"[delta] source={args.source} loaded {delta_path} "
            f"dim={delta.shape[0]} ||Δ||={float(np.linalg.norm(delta)):.6f}",
            flush=True,
        )
        _apply_and_save(test_src_path, test_out, delta, do_l2, "test_final")
        return

    actual_path = (
        resolve_project_path(args.actual_body_text_cache)
        if args.actual_body_text_cache
        else paths["actual_body"]
    )
    thumb_path = (
        resolve_project_path(args.thumbnail_clip_cache)
        if args.thumbnail_clip_cache
        else default_cache_path(sub)
    )
    train_src_path = (
        resolve_project_path(args.train_expected_text_cache)
        if args.train_expected_text_cache
        else paths["train_src"]
    )
    test_src_path = (
        resolve_project_path(args.test_expected_text_cache)
        if args.test_expected_text_cache
        else paths["test_src"]
    )
    train_out = resolve_project_path(args.train_out) if args.train_out else paths["train_image"]
    test_out = resolve_project_path(args.test_out) if args.test_out else paths["test_image"]

    for p, name in (
        (actual_path, f"실제본문 {src_label}"),
        (thumb_path, "썸네일 CLIP image"),
        (train_src_path, f"학습 기대본문 {src_label}"),
    ):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} cache 없음: {p}")

    text_news = news_embed_dict(actual_path)
    image_news = news_embed_dict(thumb_path)
    mu_text, mu_img, delta, mean_ids = compute_text_image_delta(text_news, image_news)
    print(
        f"[delta] source={args.source} n_news={len(mean_ids)} dim={delta.shape[0]}\n"
        f"[delta] ||μ_src||={float(np.linalg.norm(mu_text)):.6f} "
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
            "source": args.source,
            "actual_body_cache": os.path.abspath(actual_path),
            "thumbnail_clip_cache": os.path.abspath(thumb_path),
            "l2_after_shift": do_l2,
        },
    )
    print(f"[delta] saved {delta_path}", flush=True)

    _apply_and_save(train_src_path, train_out, delta, do_l2, "train")
    if os.path.isfile(test_src_path):
        _apply_and_save(test_src_path, test_out, delta, do_l2, "test")
    else:
        print(f"[expected-image test] skip (cache 없음): {test_src_path}", flush=True)


if __name__ == "__main__":
    main()
