#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ours with Δ: 뉴스 단위 synthetic embed 캐시.

  Δ = mean(CLIP_image(thumbnail)) − mean(CLIP_text(actual_body))
  synth = L2(CLIP_text(caption(actual_body)) + Δ)

기본: Δ는 train impression에 등장한 뉴스만으로 추정 (val/test 썸네일 leakage 방지).
추정된 Δ는 고정해 catalog 전체 actual_body text에 적용한다.

  python CLIP/clip_embeddings.py --mind-dataset-subdir MIND_2000
  python CLIP/extract_actual_body_text_embeds.py --scope catalog --mind-dataset-subdir MIND_2000
  python CLIP/build_actual_body_delta_embeds.py --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

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
    _save_clip_cache,
    default_actual_body_text_cache_path,
    default_actual_body_text_delta_cache_path,
    default_cache_path,
    default_delta_cache_path,
    resolve_project_path,
)
from expected_image import (
    apply_delta_to_news_embeds,
    compute_text_image_delta,
    load_delta,
    news_embed_dict,
    save_delta,
)
from route_embeddings import collect_news_ids_from_interaction_tsv


def _train_tsv_path(mind_dataset_subdir: str) -> str:
    names = DATASET_FILE_PRESETS.get(mind_dataset_subdir)
    train_name = names[1] if names else "MIND_train_(2000).tsv"
    return str(_ROOT / "dataset" / mind_dataset_subdir / train_name)


def _filter_news_dict(
    embeds: Dict[str, np.ndarray],
    keep: Set[str],
) -> Dict[str, np.ndarray]:
    return {nid: vec for nid, vec in embeds.items() if str(nid) in keep}


def main() -> None:
    apply_dataset_env_from_argv()
    ap = argparse.ArgumentParser(description="Build L2(CLIP_text(actual_body)+Δ) news cache")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--actual-body-text-cache", type=str, default=None)
    ap.add_argument("--thumbnail-clip-cache", type=str, default=None)
    ap.add_argument("--train-tsv", type=str, default=None, help="Δ 추정용 train impression TSV")
    ap.add_argument("--delta-in", type=str, default=None, help="기존 Δ npz (기본 recipe delta)")
    ap.add_argument("--delta-out", type=str, default=None, help="Δ 저장 경로 (기본=delta-in)")
    ap.add_argument("--out", type=str, default=None, help="synthetic news embed 출력")
    ap.add_argument(
        "--delta-scope",
        type=str,
        default="train",
        choices=["train", "catalog"],
        help="train(기본): train impression 뉴스만으로 Δ 추정. catalog: text/image cache 교집합 전체",
    )
    ap.add_argument(
        "--reuse-delta",
        action="store_true",
        help="저장된 Δ만 로드 (썸네일/재계산 생략). actual_body text cache는 필요",
    )
    ap.add_argument("--no-l2", action="store_true")
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
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
        delta_scope_used = "reused"
        train_tsv_used = None
        n_train_ids = None
    else:
        if not os.path.isfile(thumb_path):
            raise FileNotFoundError(
                f"썸네일 CLIP cache 없음: {thumb_path}\n"
                f"python CLIP/clip_embeddings.py --mind-dataset-subdir {sub}"
            )
        image_news = news_embed_dict(thumb_path)
        text_for_mean = text_news
        image_for_mean = image_news
        train_tsv_used = None
        n_train_ids = None
        delta_scope_used = args.delta_scope
        if args.delta_scope == "train":
            train_tsv_used = (
                resolve_project_path(args.train_tsv)
                if args.train_tsv
                else _train_tsv_path(sub)
            )
            if not os.path.isfile(train_tsv_used):
                raise FileNotFoundError(f"train TSV 없음: {train_tsv_used}")
            train_ids = set(collect_news_ids_from_interaction_tsv(train_tsv_used))
            n_train_ids = len(train_ids)
            text_for_mean = _filter_news_dict(text_news, train_ids)
            image_for_mean = _filter_news_dict(image_news, train_ids)
            print(
                f"[delta] scope=train train_tsv={train_tsv_used} "
                f"train_news_ids={n_train_ids} "
                f"text_hit={len(text_for_mean)} image_hit={len(image_for_mean)}",
                flush=True,
            )
        else:
            print(
                f"[delta] scope=catalog text={len(text_for_mean)} image={len(image_for_mean)}",
                flush=True,
            )

        mu_text, mu_img, delta, mean_ids = compute_text_image_delta(text_for_mean, image_for_mean)
        print(
            f"[delta] n_news_in_mean={len(mean_ids)} dim={delta.shape[0]}\n"
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
                "delta_scope": delta_scope_used,
                "train_tsv": os.path.abspath(train_tsv_used) if train_tsv_used else None,
                "n_train_impression_news_ids": n_train_ids,
                "actual_body_cache": os.path.abspath(text_path),
                "thumbnail_clip_cache": os.path.abspath(thumb_path),
                "l2_after_shift": do_l2,
                "note": "Δ estimated on train impression news only by default; "
                "applied to full actual_body text cache",
            },
        )
        print(f"[delta] saved {delta_out}", flush=True)

    # Apply fixed Δ to full text cache (val/test coverage).
    synth = apply_delta_to_news_embeds(text_news, delta, l2=do_l2)
    ids = sorted(synth.keys())
    if not ids:
        raise ValueError("synthetic embed가 비어 있습니다.")
    emb = np.stack([synth[i] for i in ids], axis=0).astype(np.float32)
    _save_clip_cache(out_path, emb, ids)
    n_nz = int(np.sum(np.any(emb, axis=1)))
    print(
        f"[actual-body+Δ] apply_scope=catalog news={len(ids)} nonzero={n_nz} "
        f"delta_scope={delta_scope_used} l2={do_l2} → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
