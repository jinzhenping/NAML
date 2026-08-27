#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3 개인화 픽셀 생성.

B2 prior image embedding → Kandinsky 2.2 decoder → PNG
  out/user_<uid>/news_<nid>.png

  conda activate clip_cu128
  python CLIP/generate_b3_images.py --mind-dataset-subdir MIND_2000
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
    default_b2_cache_path,
    default_b3_image_dir,
    load_pair_embed_dict,
    resolve_project_path,
)
from route_embeddings import generate_images_from_prior_embeds


def main() -> None:
    ap = argparse.ArgumentParser(description="B2 prior embed → Kandinsky 2.2 decoder PNG")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--b2-cache", type=str, default=None)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="기본: <ours 상위>/MIND_image_b3/<mind-dataset-subdir>",
    )
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0, help="0이면 전체. 테스트는 예: 8")
    ap.add_argument(
        "--negative",
        type=str,
        default="prior",
        choices=["prior", "zeros"],
        help="decoder CFG negative. prior=empty prompt uncond, zeros=0벡터",
    )
    ap.add_argument("--prior-steps", type=int, default=25)
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    b2_cache = (
        resolve_project_path(args.b2_cache)
        if args.b2_cache
        else default_b2_cache_path(args.mind_dataset_subdir)
    )
    if not os.path.isfile(b2_cache):
        raise FileNotFoundError(
            f"B2 prior cache 없음: {b2_cache}\n"
            f"먼저 python CLIP/extract_route_embeds.py --routes b2 를 실행하세요."
        )
    out_dir = (
        resolve_project_path(args.out_dir)
        if args.out_dir
        else default_b3_image_dir(args.mind_dataset_subdir)
    )
    pair_embeds = load_pair_embed_dict(b2_cache)
    print(f"[generate B3] b2_cache={b2_cache} pairs={len(pair_embeds)}", flush=True)
    generate_images_from_prior_embeds(
        pair_embeds,
        out_dir,
        device=args.device,
        batch_size=int(args.batch_size),
        height=int(args.height),
        width=int(args.width),
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance),
        seed=int(args.seed),
        resume=not bool(args.no_resume),
        max_images=int(args.max_images),
        negative_mode=str(args.negative),
        cpu_offload=bool(args.cpu_offload),
        prior_steps=int(args.prior_steps),
    )


if __name__ == "__main__":
    main()
