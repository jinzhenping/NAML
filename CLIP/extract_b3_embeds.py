#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3 개인화 픽셀 → CLIP image embedding.

  conda activate clip_cu128
  python CLIP/extract_b3_embeds.py --mind-dataset-subdir MIND_2000
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

from clip_embeddings import (
    default_b3_cache_path,
    default_b3_image_dir,
    resolve_project_path,
)
from route_embeddings import extract_b3_pixel_clip_embeddings


def main() -> None:
    ap = argparse.ArgumentParser(description="B3 PNG → CLIP image encoder")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--image-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    image_dir = (
        resolve_project_path(args.image_dir)
        if args.image_dir
        else default_b3_image_dir(args.mind_dataset_subdir)
    )
    out_path = (
        resolve_project_path(args.out)
        if args.out
        else default_b3_cache_path(args.mind_dataset_subdir)
    )
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"B3 이미지 폴더 없음: {image_dir}\n"
            f"먼저 python CLIP/generate_b3_images.py 를 실행하세요."
        )
    extract_b3_pixel_clip_embeddings(
        image_dir,
        out_path,
        device=args.device,
        batch_size=int(args.batch_size),
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()
