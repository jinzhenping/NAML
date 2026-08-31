#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기대본문 → Kandinsky 2.2 prior → CLIP image_embeds (user×news). B2와 같은 encoder.

기본은 학습셋:
  user_preference/expected_body/<subdir>/train_3cluster_11_13_8_rawtitle
  + MIND_train TSV 후보 pair

  conda activate clip_cu128
  python CLIP/extract_expected_body_prior_embeds.py --split train --mind-dataset-subdir MIND_2000
  python CLIP/extract_expected_body_prior_embeds.py --split test --mind-dataset-subdir MIND_2000
  python CLIP/extract_expected_body_prior_embeds.py --split test_final --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from naml_dataset_env import DATASET_FILE_PRESETS, apply_dataset_env_from_argv

from clip_embeddings import (
    default_b2_cache_path,
    default_b2_test_final_cache_path,
    default_b2_train_cache_path,
    default_test_final_tsv,
    resolve_project_path,
)
from route_embeddings import (
    build_pairs_and_texts,
    collect_candidate_pairs_from_tsv,
    default_expected_body_dir,
    extract_prior_image_embeddings,
    load_expected_bodies_from_dir,
)


def _split_tsv(mind_dataset_subdir: str, split: str) -> str:
    if split == "test_final":
        return default_test_final_tsv(mind_dataset_subdir)
    names = DATASET_FILE_PRESETS.get(mind_dataset_subdir)
    news_name, train_name, test_name = names if names else (
        "MIND_news.tsv",
        "MIND_train_(2000).tsv",
        "MIND_test_(2000).tsv",
    )
    fname = train_name if split == "train" else test_name
    return str(_ROOT / "dataset" / mind_dataset_subdir / fname)


def main() -> None:
    apply_dataset_env_from_argv()
    ap = argparse.ArgumentParser(description="Expected body → Kandinsky prior embeds")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test", "test_final"])
    ap.add_argument("--expected-body-dir", type=str, default=None)
    ap.add_argument("--tsv", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-inference-steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    body_split = "test" if args.split in ("test", "test_final") else "train"
    expected_dir = (
        resolve_project_path(args.expected_body_dir)
        if args.expected_body_dir
        else default_expected_body_dir(str(_ROOT), args.mind_dataset_subdir, split=body_split)
    )
    tsv_path = resolve_project_path(args.tsv) if args.tsv else _split_tsv(args.mind_dataset_subdir, args.split)
    if args.out:
        out_path = resolve_project_path(args.out)
    elif args.split == "train":
        out_path = default_b2_train_cache_path(args.mind_dataset_subdir)
    elif args.split == "test_final":
        out_path = default_b2_test_final_cache_path(args.mind_dataset_subdir)
    else:
        out_path = default_b2_cache_path(args.mind_dataset_subdir)

    if not os.path.isdir(expected_dir):
        raise FileNotFoundError(f"기대본문 폴더 없음: {expected_dir}")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"interaction tsv 없음: {tsv_path}")

    bodies = load_expected_bodies_from_dir(expected_dir)
    pairs = collect_candidate_pairs_from_tsv(tsv_path)
    items, n_hit, n_missing = build_pairs_and_texts(pairs, bodies)
    print(
        f"[expected-body prior] split={args.split}\n"
        f"[expected-body prior] dir={expected_dir} files={len(bodies)}\n"
        f"[expected-body prior] tsv={tsv_path} pairs={len(pairs)} "
        f"hit={n_hit} missing={n_missing}\n"
        f"[expected-body prior] out={out_path}",
        flush=True,
    )
    extract_prior_image_embeddings(
        items,
        out_path,
        device=args.device,
        batch_size=int(args.batch_size),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        seed=int(args.seed),
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()
