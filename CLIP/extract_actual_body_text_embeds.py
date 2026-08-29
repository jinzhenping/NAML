#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
학습셋 뉴스 실제 본문 → CLIP text embedding (뉴스 단위).

MIND_news.tsv body 컬럼을 쓰고, 기본은 train TSV에 등장한 unique 뉴스만 인코딩한다.
CLIP 77토큰 truncation, L2 없음. B1(기대본문)과 같은 text encoder.

  conda activate clip_cu128
  python CLIP/extract_actual_body_text_embeds.py --mind-dataset-subdir MIND_2000
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

from clip_embeddings import default_actual_body_text_cache_path, resolve_project_path
from route_embeddings import (
    collect_news_ids_from_interaction_tsv,
    extract_clip_text_actual_bodies,
    load_actual_bodies_from_news_tsv,
)


def _dataset_paths(mind_dataset_subdir: str):
    names = DATASET_FILE_PRESETS.get(mind_dataset_subdir)
    news_name = names[0] if names else "MIND_news.tsv"
    train_name = names[1] if names else "MIND_train_(2000).tsv"
    base = _ROOT / "dataset" / mind_dataset_subdir
    return str(base / news_name), str(base / train_name)


def main() -> None:
    apply_dataset_env_from_argv()
    ap = argparse.ArgumentParser(description="Train-set actual news body → CLIP text embeds")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--news-tsv", type=str, default=None)
    ap.add_argument("--train-tsv", type=str, default=None)
    ap.add_argument(
        "--scope",
        type=str,
        default="train",
        choices=["train", "catalog"],
        help="train: train TSV에 등장한 뉴스만. catalog: MIND_news.tsv 전체",
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    news_tsv, train_tsv = _dataset_paths(args.mind_dataset_subdir)
    if args.news_tsv:
        news_tsv = resolve_project_path(args.news_tsv)
    if args.train_tsv:
        train_tsv = resolve_project_path(args.train_tsv)
    out_path = (
        resolve_project_path(args.out)
        if args.out
        else default_actual_body_text_cache_path(args.mind_dataset_subdir)
    )

    if not os.path.isfile(news_tsv):
        raise FileNotFoundError(f"news tsv 없음: {news_tsv}")
    bodies = load_actual_bodies_from_news_tsv(news_tsv)
    if args.scope == "catalog":
        news_ids = list(bodies.keys())
    else:
        if not os.path.isfile(train_tsv):
            raise FileNotFoundError(f"train tsv 없음: {train_tsv}")
        news_ids = collect_news_ids_from_interaction_tsv(train_tsv)
        news_ids = [nid for nid in news_ids if nid in bodies]
    print(
        f"[actual-body text] news_tsv={news_tsv}\n"
        f"[actual-body text] scope={args.scope} n_ids={len(news_ids)} "
        f"body_files={len(bodies)} out={out_path}",
        flush=True,
    )
    extract_clip_text_actual_bodies(
        news_ids,
        bodies,
        out_path,
        device=args.device,
        batch_size=int(args.batch_size),
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()
