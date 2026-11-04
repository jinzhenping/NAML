#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ours with Δ: 뉴스 단위 synthetic embed 캐시.

  Δ = mean(CLIP_image(thumbnail)) − mean(CLIP_text(actual_body))
  synth = L2(CLIP_text(caption(actual_body)) + Δ)

기대본문이 아니라 실제본문 CLIP text에 Δ를 더한다 (S5의 with-Δ 대응).

  # 필요 캐시
  python CLIP/clip_embeddings.py --mind-dataset-subdir MIND_2000
  python CLIP/extract_actual_body_text_embeds.py --scope catalog --mind-dataset-subdir MIND_2000

  python CLIP/build_actual_body_delta_embeds.py --mind-dataset-subdir MIND_2000
  # 이미 있는 Δ만 재사용:
  python CLIP/build_actual_body_delta_embeds.py --reuse-delta --mind-dataset-subdir MIND_2000
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
    _save_clip_cache,
    default_actual_body_text_cache_path,
    default_actual_body_text_delta_cache_path,
    default_cache_path,
    default_delta_cache_path,
    load_clip_npz,
    resolve_project_path,
)
from expected_image import (
    apply_delta_to_news_embeds,
    compute_text_image_delta,
    load_delta,
    news_embed_dict,
    save_delta,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build L2(CLIP_text(actual_body)+Δ) news cache")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--actual-body-text-cache", type=str, default=None)
    ap.add_argument("--thumbnail-clip-cache", type=str, default=None)
    ap.add_argument("--delta-in", type=str, default=None, help="기존 Δ npz (기본 recipe delta)")
    ap.add_argument("--delta-out", type=str, default=None, help="Δ 저장 경로 (기본=delta-in)")
    ap.add_argument("--out", type=str, default=None, help="synthetic news embed 출력")
    ap.add_argument(
        "--reuse-delta",
        action="store_true",
        help="저장된 Δ만 로드 (썸네일/재계산 생략). actual_body text cache는 필요",
    )
    ap.add_argument("--no-l2", action="store_true")
    args = ap.parse_args()

    sub = args.mind_dataset_subdir
    text_path = (
        resolve_project_path(args.actual_body_text_cache)
        if args.actual_body_text_cache
        else default_actual_body_text_cache_path(sub)
    )
    thumb_path = (
        resolve_project_path(args.thumbnail_clip_cache)
        if args.thumbnail_clip_cache
        else default_cache_path(sub)
    )
    delta_path = resolve_project_path(args.delta_in) if args.delta_in else default_delta_cache_path(sub)
    delta_out = resolve_project_path(args.delta_out) if args.delta_out else delta_path
    out_path = (
        resolve_project_path(args.out)
        if args.out
        else default_actual_body_text_delta_cache_path(sub)
    )
    do_l2 = not bool(args.no_l2)

    if not os.path.isfile(text_path):
        raise FileNotFoundError(
            f"CLIP_text(actual_body) cache 없음: {text_path}\n"
            "python CLIP/extract_actual_body_text_embeds.py --scope catalog "
            f"--mind-dataset-subdir {sub}"
        )

    text_news = news_embed_dict(text_path)

    if args.reuse_delta:
        if not os.path.isfile(delta_path):
            raise FileNotFoundError(
                f"Δ cache 없음: {delta_path}\n"
                "python CLIP/build_actual_body_delta_embeds.py "
                f"--mind-dataset-subdir {sub}"
            )
        delta = load_delta(delta_path)
        print(
            f"[delta] reused {delta_path} dim={delta.shape[0]} "
            f"||Δ||={float(np.linalg.norm(delta)):.6f}",
            flush=True,
        )
    else:
        if not os.path.isfile(thumb_path):
            raise FileNotFoundError(
                f"썸네일 CLIP cache 없음: {thumb_path}\n"
                f"python CLIP/clip_embeddings.py --mind-dataset-subdir {sub}"
            )
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
            delta_out,
            mu_text,
            mu_img,
            delta,
            mean_ids,
            extra={
                "source": "clip_text",
                "applied_to": "actual_body",
                "actual_body_cache": os.path.abspath(text_path),
                "thumbnail_clip_cache": os.path.abspath(thumb_path),
                "l2_after_shift": do_l2,
            },
        )
        print(f"[delta] saved {delta_out}", flush=True)

    synth = apply_delta_to_news_embeds(text_news, delta, l2=do_l2)
    ids = sorted(synth.keys())
    if not ids:
        raise ValueError("synthetic embed가 비어 있습니다.")
    emb = np.stack([synth[i] for i in ids], axis=0).astype(np.float32)
    _save_clip_cache(out_path, emb, ids)
    n_nz = int(np.sum(np.any(emb, axis=1)))
    print(
        f"[actual-body+Δ] news={len(ids)} nonzero={n_nz} l2={do_l2} → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
