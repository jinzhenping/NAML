#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MM-Rec용 CLIP 뉴스 임베딩 추출.

  - clip_image_embeds.npz : 썸네일 → CLIP image embed
  - clip_title_embeds.npz : 타이틀 → CLIP text embed

기본 모델은 CLIP/clip_embeddings.py 와 동일 (Kandinsky 2.2 prior CLIP).

  conda activate clip_cu128
  python MM_Rec/extract_clip_features.py --mind-dataset-subdir MIND_2000
  python MM_Rec/extract_clip_features.py --mind-dataset-subdir MIND_2000 --reuse-image-cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_MMREC = Path(__file__).resolve().parent
_ROOT = _MMREC.parent
_CLIP = _ROOT / "CLIP"
for p in (_ROOT, _MMREC, _CLIP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_paths import DEFAULT_THUMBNAIL_DIR, prepared_dir, resolve_project_path
from prepare_mind_dataset import prepare

CLIP_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
CLIP_IMAGE_ENCODER_SUBFOLDER = "image_encoder"
CLIP_IMAGE_PROCESSOR_SUBFOLDER = "image_processor"
CLIP_TEXT_ENCODER_SUBFOLDER = "text_encoder"
CLIP_TEXT_MAX_LENGTH = 77


def clip_image_cache_path(mind_dataset_subdir: str) -> Path:
    return prepared_dir(mind_dataset_subdir) / "clip_image_embeds.npz"


def clip_title_cache_path(mind_dataset_subdir: str) -> Path:
    return prepared_dir(mind_dataset_subdir) / "clip_title_embeds.npz"


def default_clip_project_image_cache(mind_dataset_subdir: str) -> Path:
    return _CLIP / "cache" / f"{mind_dataset_subdir}_clip_image_embeds.npz"


def read_news_ids_titles(news_tsv: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    titles: List[str] = []
    with open(news_tsv, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            if line_i == 0 and parts[0].strip().lower() in {"news_id", "id"}:
                continue
            nid = parts[0].strip()
            title = parts[3].strip() if len(parts) > 3 else ""
            ids.append(nid)
            titles.append(title)
    return ids, titles


def _save_news_cache(path: Path, embeddings: np.ndarray, news_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_len = max((len(str(x)) for x in news_ids), default=8)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        news_ids=np.asarray([str(x) for x in news_ids], dtype=f"U{max(n_len, 8)}"),
    )
    print(f"[clip-mmrec] saved {path} shape={embeddings.shape}", flush=True)


def extract_title_embeds(
    news_ids: Sequence[str],
    titles: Sequence[str],
    out_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> None:
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    print(f"[clip-mmrec] load text_encoder device={device}", flush=True)
    try:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID, subfolder="tokenizer")
    except Exception:
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        CLIP_MODEL_ID, subfolder=CLIP_TEXT_ENCODER_SUBFOLDER
    )
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    emb_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = list(titles[i : i + batch_size])
            toks = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=CLIP_TEXT_MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = toks.input_ids.to(device)
            attn = toks.attention_mask.to(device)
            out = text_encoder(input_ids=input_ids, attention_mask=attn)
            vecs = out.text_embeds.float().cpu().numpy().astype(np.float32)
            emb_list.append(vecs)
            if (i // batch_size) % 20 == 0:
                print(f"[clip-mmrec] title {min(i + batch_size, len(titles))}/{len(titles)}", flush=True)
    emb = np.concatenate(emb_list, axis=0) if emb_list else np.zeros((0, 1), dtype=np.float32)
    _save_news_cache(out_path, emb, news_ids)


def extract_or_reuse_image_embeds(
    news_ids: Sequence[str],
    out_path: Path,
    thumbnail_dir: str,
    *,
    reuse_image_cache: bool = True,
    reuse_path: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
) -> None:
    from clip_embeddings import (
        default_cache_path,
        extract_clip_embeddings,
        load_clip_npz,
        resolve_project_path as clip_resolve,
    )

    src = None
    if reuse_path:
        cand = clip_resolve(reuse_path) if not os.path.isabs(reuse_path) else reuse_path
        if os.path.isfile(cand):
            src = cand
    elif reuse_image_cache:
        subdir = out_path.parent.name
        for cand in (default_clip_project_image_cache(subdir), Path(default_cache_path(subdir))):
            if Path(cand).is_file():
                src = str(cand)
                break

    if src:
        print(f"[clip-mmrec] reuse image cache {src}", flush=True)
        emb, cached_ids = load_clip_npz(src)
        id_to_row = {str(n): i for i, n in enumerate(cached_ids)}
        dim = int(emb.shape[1])
        out = np.zeros((len(news_ids), dim), dtype=np.float32)
        n_hit = 0
        for i, nid in enumerate(news_ids):
            r = id_to_row.get(str(nid))
            if r is None:
                continue
            out[i] = emb[r]
            if np.any(emb[r]):
                n_hit += 1
        print(f"[clip-mmrec] image reuse hit={n_hit}/{len(news_ids)}", flush=True)
        _save_news_cache(out_path, out, news_ids)
        return

    print(f"[clip-mmrec] extract thumbnail CLIP → {out_path}", flush=True)
    extract_clip_embeddings(
        news_ids,
        thumbnail_dir,
        str(out_path),
        device=device,
        batch_size=batch_size,
        source_label="thumbnail",
    )


def build_mmrec_clip_matrices(
    news_index: Dict[str, int],
    n_rows: int,
    image_cache: str,
    title_cache: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """news_index 행에 scatter. row0 = padding."""
    from clip_embeddings import load_clip_npz

    img_emb, img_ids = load_clip_npz(image_cache)
    txt_emb, txt_ids = load_clip_npz(title_cache)
    if int(img_emb.shape[1]) != int(txt_emb.shape[1]):
        raise ValueError(
            f"CLIP dim mismatch image={img_emb.shape[1]} title={txt_emb.shape[1]}"
        )
    dim = int(img_emb.shape[1])
    mat_v = np.zeros((int(n_rows), dim), dtype=np.float32)
    mat_t = np.zeros((int(n_rows), dim), dtype=np.float32)
    img_map = {str(n): i for i, n in enumerate(img_ids)}
    txt_map = {str(n): i for i, n in enumerate(txt_ids)}
    n_v = n_t = 0
    for nid, idx in news_index.items():
        i = int(idx)
        if i <= 0 or i >= n_rows:
            continue
        ri = img_map.get(str(nid))
        rt = txt_map.get(str(nid))
        if ri is not None:
            mat_v[i] = img_emb[ri]
            if np.any(img_emb[ri]):
                n_v += 1
        if rt is not None:
            mat_t[i] = txt_emb[rt]
            if np.any(txt_emb[rt]):
                n_t += 1
    return mat_t, mat_v, n_t, n_v


def main() -> None:
    ap = argparse.ArgumentParser(description="MM-Rec CLIP title/image feature extract")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--news-tsv", type=str, default=None)
    ap.add_argument(
        "--reuse-image-cache",
        action="store_true",
        help="CLIP/cache/<subdir>_clip_image_embeds.npz 가 있으면 재사용 (기본은 항상 시도)",
    )
    ap.add_argument("--no-reuse-image-cache", action="store_true")
    ap.add_argument("--image-cache-path", type=str, default=None)
    ap.add_argument("--skip-image", action="store_true")
    ap.add_argument("--skip-title", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force-prepare", action="store_true")
    args = ap.parse_args()

    data_root = prepared_dir(args.mind_dataset_subdir)
    news_tsv = (
        resolve_project_path(args.news_tsv)
        if args.news_tsv
        else data_root / "subnews.tsv"
    )
    if not news_tsv.is_file() or args.force_prepare:
        prepare(args.mind_dataset_subdir)
    if not news_tsv.is_file():
        raise FileNotFoundError(f"news tsv 없음: {news_tsv}")

    news_ids, titles = read_news_ids_titles(str(news_tsv))
    print(f"[clip-mmrec] news={len(news_ids)} from {news_tsv}", flush=True)

    thumb = str(resolve_project_path(args.thumbnail_dir))
    img_out = clip_image_cache_path(args.mind_dataset_subdir)
    title_out = clip_title_cache_path(args.mind_dataset_subdir)

    if not args.skip_image:
        extract_or_reuse_image_embeds(
            news_ids,
            img_out,
            thumb,
            reuse_image_cache=not bool(args.no_reuse_image_cache),
            reuse_path=args.image_cache_path,
            device=args.device,
            batch_size=min(int(args.batch_size), 16),
        )
    if not args.skip_title:
        extract_title_embeds(
            news_ids,
            titles,
            title_out,
            device=args.device,
            batch_size=int(args.batch_size),
        )
    print("[clip-mmrec] done", flush=True)


if __name__ == "__main__":
    main()
