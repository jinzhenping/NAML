#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
랜덤 Δ ablation 캐시 생성.

기존 train Δ (= μ_img − μ_text)의 크기 r=‖Δ‖만 유지하고 방향만 무작위인
전역 Δ_rand 를 만든 뒤, 기대본문 embed에 더해 L2 한다.

  Δ_rand = r · u,  u = g/‖g‖,  g ~ N(0, I)  (seed 고정, 전역 1벡터)
  synthetic = L2(CLIP_text/prior(expected_body) + Δ_rand)

  # 기존 Δ가 있어야 함
  python CLIP/build_expected_image_embeds.py --mind-dataset-subdir MIND_2000
  python CLIP/build_random_delta_embeds.py --mind-dataset-subdir MIND_2000

  # 최종 test용 (이미 있는 Δ_rand 적용)
  python CLIP/build_random_delta_embeds.py --apply-delta-only --mind-dataset-subdir MIND_2000

  # prior recipe
  python CLIP/build_random_delta_embeds.py --source prior --mind-dataset-subdir MIND_2000
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

from naml_common import SEED
from clip_embeddings import random_delta_recipe_paths, resolve_project_path
from expected_image import (
    apply_delta_to_text_pairs,
    delta_cosine,
    load_delta,
    load_pair_dict_normed,
    make_random_delta,
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
        f"[random-delta {label}] pairs={len(img_pairs)} nonzero={n_nz} → {out_path}",
        flush=True,
    )


def _load_delta_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {
        "delta": np.asarray(data["delta"], dtype=np.float32).reshape(-1),
    }
    if "mu_text" in data.files:
        out["mu_text"] = np.asarray(data["mu_text"], dtype=np.float32).reshape(-1)
    if "mu_img" in data.files:
        out["mu_img"] = np.asarray(data["mu_img"], dtype=np.float32).reshape(-1)
    if "news_ids" in data.files:
        out["news_ids"] = [str(x) for x in data["news_ids"].tolist()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build global Δ_rand and expected-image caches")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--source",
        type=str,
        default="clip_text",
        choices=["clip_text", "prior"],
        help="clip_text: CLIP text+Δ_rand. prior: Kandinsky prior+Δ_rand",
    )
    ap.add_argument(
        "--delta-seed",
        type=int,
        default=SEED,
        help="Δ_rand 방향 샘플링 시드 (전역 1벡터). 기본 naml SEED=42",
    )
    ap.add_argument("--delta-in", type=str, default=None, help="기존 Δ npz (기본 recipe delta)")
    ap.add_argument("--delta-out", type=str, default=None, help="Δ_rand 저장 경로")
    ap.add_argument("--train-expected-text-cache", type=str, default=None)
    ap.add_argument("--test-expected-text-cache", type=str, default=None)
    ap.add_argument("--train-out", type=str, default=None)
    ap.add_argument("--test-out", type=str, default=None)
    ap.add_argument("--no-l2", action="store_true", help="Δ_rand 더한 뒤 L2를 하지 않음 (기본은 L2)")
    ap.add_argument(
        "--apply-delta-only",
        action="store_true",
        help="저장된 Δ_rand만 로드해 기대본문 임베딩에 적용. 기본 입출력은 test_final",
    )
    args = ap.parse_args()

    paths = random_delta_recipe_paths(args.mind_dataset_subdir, args.source)
    delta_in = resolve_project_path(args.delta_in) if args.delta_in else paths["delta"]
    delta_out = resolve_project_path(args.delta_out) if args.delta_out else paths["delta_random"]
    do_l2 = not bool(args.no_l2)
    source_flag = f" --source {args.source}" if args.source != "clip_text" else ""

    if args.apply_delta_only:
        if not os.path.isfile(delta_out):
            raise FileNotFoundError(
                f"Δ_rand cache 없음: {delta_out}\n"
                f"먼저 python CLIP/build_random_delta_embeds.py{source_flag} "
                f"--mind-dataset-subdir {args.mind_dataset_subdir}"
            )
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
        delta_rand = load_delta(delta_out)
        print(
            f"[delta_rand] source={args.source} loaded {delta_out} "
            f"dim={delta_rand.shape[0]} ||Δ_rand||={float(np.linalg.norm(delta_rand)):.6f}",
            flush=True,
        )
        _apply_and_save(test_src_path, test_out, delta_rand, do_l2, "test_final")
        return

    if not os.path.isfile(delta_in):
        raise FileNotFoundError(
            f"기존 Δ cache 없음: {delta_in}\n"
            f"먼저 python CLIP/build_expected_image_embeds.py{source_flag} "
            f"--mind-dataset-subdir {args.mind_dataset_subdir}"
        )

    packed = _load_delta_npz(delta_in)
    delta = packed["delta"]
    mu_text = packed.get("mu_text", np.zeros_like(delta))
    mu_img = packed.get("mu_img", np.zeros_like(delta))
    mean_ids = packed.get("news_ids", [])
    delta_rand = make_random_delta(delta, int(args.delta_seed))
    cos = delta_cosine(delta, delta_rand)
    print(
        f"[delta_rand] source={args.source} seed={int(args.delta_seed)} dim={delta.shape[0]}\n"
        f"[delta_rand] ||Δ||={float(np.linalg.norm(delta)):.6f} "
        f"||Δ_rand||={float(np.linalg.norm(delta_rand)):.6f} "
        f"cos(Δ,Δ_rand)={cos:.6f}",
        flush=True,
    )
    save_delta(
        delta_out,
        mu_text,
        mu_img,
        delta_rand,
        mean_ids,
        extra={
            "ablation": "random_delta",
            "source": args.source,
            "delta_seed": int(args.delta_seed),
            "source_delta_path": os.path.abspath(delta_in),
            "source_delta_norm": float(np.linalg.norm(delta)),
            "random_delta_norm": float(np.linalg.norm(delta_rand)),
            "cosine_with_source_delta": cos,
            "l2_after_shift": do_l2,
            "note": "npz 'delta' key stores Δ_rand (same ||·|| as source Δ, random direction)",
        },
    )
    print(f"[delta_rand] saved {delta_out}", flush=True)

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

    _apply_and_save(train_src_path, train_out, delta_rand, do_l2, "train")
    if os.path.isfile(test_src_path):
        _apply_and_save(test_src_path, test_out, delta_rand, do_l2, "test")
    else:
        print(f"[random-delta test] skip (cache 없음): {test_src_path}", flush=True)


if __name__ == "__main__":
    main()
