#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frozen teacher 파일럿용 임베딩 추출.

B1: 기대본문 → CLIP text encoder (Route T)
B2: 기대본문 → Kandinsky prior → CLIP image embed (Route E)
B4: MIND_image/{id}.png → CLIP image encoder (비개인화 생성 픽셀)

B0 썸네일은 CLIP/clip_embeddings.py 캐시를 그대로 쓴다.
B3 개인화 픽셀은 이번 실험에서 제외.

  conda activate clip_cu128
  python CLIP/extract_route_embeds.py --routes b1,b2,b4 --mind-dataset-subdir MIND_2000
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
    DEFAULT_GENERATED_IMAGE_DIR,
    default_b1_cache_path,
    default_b2_cache_path,
    default_b4_cache_path,
    extract_clip_embeddings,
    load_news_ids_from_tsv,
    print_missing_image_report,
    resolve_project_path,
)
from route_embeddings import (
    build_pairs_and_texts,
    collect_candidate_pairs_from_tsv,
    default_expected_body_dir,
    extract_clip_text_embeddings,
    extract_prior_image_embeddings,
    load_expected_bodies_from_dir,
)


def _parse_routes(raw: str) -> list:
    allowed = {"b1", "b2", "b4"}
    routes = []
    for part in (raw or "").split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in allowed:
            raise ValueError(f"unknown route {part!r}. use b1,b2,b4")
        if key not in routes:
            routes.append(key)
    if not routes:
        raise ValueError("--routes 가 비어 있습니다. 예: b1,b2,b4")
    return routes


def _test_tsv_path(mind_dataset_subdir: str) -> str:
    names = DATASET_FILE_PRESETS.get(mind_dataset_subdir)
    test_name = names[2] if names else "MIND_test_(2000).tsv"
    return str(_ROOT / "dataset" / mind_dataset_subdir / test_name)


def _news_tsv_path(mind_dataset_subdir: str) -> str:
    names = DATASET_FILE_PRESETS.get(mind_dataset_subdir)
    news_name = names[0] if names else "MIND_news.tsv"
    return str(_ROOT / "dataset" / mind_dataset_subdir / news_name)


def main() -> None:
    apply_dataset_env_from_argv()
    ap = argparse.ArgumentParser(description="B1/B2/B4 CLIP 임베딩 추출")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--routes", type=str, default="b1,b2,b4", help="comma: b1,b2,b4")
    ap.add_argument("--expected-body-dir", type=str, default=None)
    ap.add_argument("--test-tsv", type=str, default=None)
    ap.add_argument("--generated-image-dir", type=str, default=DEFAULT_GENERATED_IMAGE_DIR)
    ap.add_argument("--b1-out", type=str, default=None)
    ap.add_argument("--b2-out", type=str, default=None)
    ap.add_argument("--b4-out", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--b1-batch-size", type=int, default=32)
    ap.add_argument("--b2-batch-size", type=int, default=4)
    ap.add_argument("--b4-batch-size", type=int, default=16)
    ap.add_argument("--prior-steps", type=int, default=25)
    ap.add_argument("--prior-guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true", help="기존 pair cache를 무시하고 다시 추출")
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    routes = _parse_routes(args.routes)
    resume = not bool(args.no_resume)

    expected_dir = (
        resolve_project_path(args.expected_body_dir)
        if args.expected_body_dir
        else default_expected_body_dir(str(_ROOT), args.mind_dataset_subdir, split="test")
    )
    test_tsv = resolve_project_path(args.test_tsv) if args.test_tsv else _test_tsv_path(args.mind_dataset_subdir)
    news_tsv = _news_tsv_path(args.mind_dataset_subdir)
    gen_dir = resolve_project_path(args.generated_image_dir)
    b1_out = resolve_project_path(args.b1_out) if args.b1_out else default_b1_cache_path(args.mind_dataset_subdir)
    b2_out = resolve_project_path(args.b2_out) if args.b2_out else default_b2_cache_path(args.mind_dataset_subdir)
    b4_out = resolve_project_path(args.b4_out) if args.b4_out else default_b4_cache_path(args.mind_dataset_subdir)

    print(
        f"[extract] dataset={args.mind_dataset_subdir} routes={routes}\n"
        f"[extract] expected_body_dir={expected_dir}\n"
        f"[extract] test_tsv={test_tsv}",
        flush=True,
    )

    if any(r in routes for r in ("b1", "b2")):
        if not os.path.isdir(expected_dir):
            raise FileNotFoundError(f"기대본문 폴더 없음: {expected_dir}")
        if not os.path.isfile(test_tsv):
            raise FileNotFoundError(f"test tsv 없음: {test_tsv}")
        bodies = load_expected_bodies_from_dir(expected_dir)
        pairs = collect_candidate_pairs_from_tsv(test_tsv)
        items, n_hit, n_missing = build_pairs_and_texts(pairs, bodies)
        print(
            f"[extract] test candidate pairs={len(pairs)}  "
            f"expected_body hit={n_hit} missing={n_missing}  "
            f"body files={len(bodies)}",
            flush=True,
        )
        if "b1" in routes:
            extract_clip_text_embeddings(
                items,
                b1_out,
                device=args.device,
                batch_size=int(args.b1_batch_size),
                resume=resume,
            )
        if "b2" in routes:
            extract_prior_image_embeddings(
                items,
                b2_out,
                device=args.device,
                batch_size=int(args.b2_batch_size),
                num_inference_steps=int(args.prior_steps),
                guidance_scale=float(args.prior_guidance),
                seed=int(args.seed),
                resume=resume,
            )

    if "b4" in routes:
        if not os.path.isfile(news_tsv):
            raise FileNotFoundError(f"news tsv 없음: {news_tsv}")
        news_ids = load_news_ids_from_tsv(news_tsv)
        print_missing_image_report(
            news_ids, gen_dir, suffixes=(".png", ".jpg"), label="generated_image"
        )
        extract_clip_embeddings(
            news_ids,
            gen_dir,
            b4_out,
            device=args.device,
            batch_size=int(args.b4_batch_size),
            suffixes=(".png", ".jpg"),
            source_label="generated_image",
        )


if __name__ == "__main__":
    main()
