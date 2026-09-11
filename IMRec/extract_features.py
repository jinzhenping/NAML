#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""뉴스 카드에서 ResNet-101 local/global impression feature 추출."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from paths import cards_dir, dataset_raw_dir, DATASET_FILE_PRESETS, features_path, prepared_dir
from render_cards import wrap_title, load_news_tsv

# card layout crops (matches render_cards.py)
TITLE_BOX = (220, 10, 610, 140)  # left, top, right, bottom
COVER_BOX = (0, 0, 210, 195)
CAT_BOX = (220, 140, 610, 195)

LOCAL_DIM = 512
GLOBAL_DIM = 2048
N_COVER_REGIONS = 9
MAX_TITLE_WORDS = 30


def word_tokenize(sent: str) -> List[str]:
    return re.findall(r"[\w]+|[.,!?;|]", (sent or "").lower())


def _build_resnet(device: str):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import transforms

    try:
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        full = models.resnet101(weights=weights)
    except Exception:
        full = models.resnet101(pretrained=True)

    # local: remove last 4 children → ~512x28x28 after 224 input
    local = nn.Sequential(*list(full.children())[:-4]).to(device).eval()
    for p in local.parameters():
        p.requires_grad = False

    # global: remove last FC → 2048
    glob = nn.Sequential(*list(full.children())[:-1]).to(device).eval()
    for p in glob.parameters():
        p.requires_grad = False

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return local, glob, tfm, device


def _forward_map(model, tfm, img: Image.Image, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
        out = model(x).squeeze(0).detach().cpu().numpy()
    return out


def pool_word_region(feat: np.ndarray, row: int, col: int, col_len: int, last: bool) -> np.ndarray:
    """feat: (512, H, W), H=W≈28. Vertical thirds for title lines."""
    c, h, w = feat.shape
    avg = max(1, w // max(1, col_len))
    if row == 0:
        rs, re = 0, h // 3
    elif row == 1:
        rs, re = h // 3, 2 * h // 3
    else:
        rs, re = 2 * h // 3, h
    cs = col * avg
    ce = w if last else min(w, (col + 1) * avg)
    region = feat[:, rs:re, cs:ce]
    if region.size == 0:
        return np.zeros(c, dtype=np.float32)
    return region.reshape(c, -1).mean(axis=1).astype(np.float32)


def pool_cover_regions(feat: np.ndarray, n: int = N_COVER_REGIONS) -> np.ndarray:
    """3x3 regions on cover feature map."""
    c, h, w = feat.shape
    out = np.zeros((n, c), dtype=np.float32)
    rh, rw = max(1, h // 3), max(1, w // 3)
    for i in range(3):
        for j in range(3):
            rs, re = i * rh, h if i == 2 else (i + 1) * rh
            cs, ce = j * rw, w if j == 2 else (j + 1) * rw
            region = feat[:, rs:re, cs:ce]
            out[i * 3 + j] = region.reshape(c, -1).mean(axis=1)
    return out


def extract_one(
    card: Image.Image,
    title: str,
    local_model,
    glob_model,
    tfm,
    device: str,
) -> Dict[str, np.ndarray]:
    title_crop = card.crop(TITLE_BOX)
    cover_crop = card.crop(COVER_BOX)
    cat_crop = card.crop(CAT_BOX)

    title_map = _forward_map(local_model, tfm, title_crop, device)  # C,H,W
    cover_map = _forward_map(local_model, tfm, cover_crop, device)
    cat_map = _forward_map(local_model, tfm, cat_crop, device)
    global_vec = _forward_map(glob_model, tfm, card, device).reshape(-1).astype(np.float32)

    lines = [ln for ln in wrap_title(title).split("\n") if ln.strip()]
    word_feats = np.zeros((MAX_TITLE_WORDS, LOCAL_DIM), dtype=np.float32)
    word_mask = np.zeros(MAX_TITLE_WORDS, dtype=np.float32)
    idx = 0
    for row, line in enumerate(lines[:3]):
        words = re.findall(r"\b\w+\b", line.lower())
        for j, _w in enumerate(words):
            if idx >= MAX_TITLE_WORDS:
                break
            word_feats[idx] = pool_word_region(title_map, row, j, max(1, len(words)), j == len(words) - 1)
            word_mask[idx] = 1.0
            idx += 1
        if idx >= MAX_TITLE_WORDS:
            break

    cover_regions = pool_cover_regions(cover_map)
    category_feat = cat_map.reshape(cat_map.shape[0], -1).mean(axis=1).astype(np.float32)

    return {
        "word_vis": word_feats,
        "word_vis_mask": word_mask,
        "cover_regions": cover_regions,
        "category_vis": category_feat,
        "global_feat": global_vec,
    }


def extract_all(
    mind_dataset_subdir: str,
    force: bool = False,
    device: str = "auto",
    max_news: int = 0,
) -> Path:
    import torch

    out = features_path(mind_dataset_subdir)
    if out.is_file() and not force:
        print(f"[feat] cache exists: {out}", flush=True)
        return out

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = dataset_raw_dir(mind_dataset_subdir)
    news_name = DATASET_FILE_PRESETS.get(mind_dataset_subdir, (None,))[0] or "MIND_news.tsv"
    news = load_news_tsv(raw / news_name)
    cdir = cards_dir(mind_dataset_subdir)
    if not cdir.is_dir():
        raise FileNotFoundError(f"cards missing: {cdir}. Run render_cards.py first.")

    local_m, glob_m, tfm, device = _build_resnet(device)
    prepared_dir(mind_dataset_subdir).mkdir(parents=True, exist_ok=True)

    ids = sorted(news.keys())
    if max_news > 0:
        ids = ids[:max_news]

    word_vis, word_mask, cover, catv, globv = [], [], [], [], []
    id_list = []
    for nid in tqdm(ids, desc="extract"):
        cp = cdir / f"{nid}.jpg"
        if not cp.is_file():
            continue
        try:
            card = Image.open(cp).convert("RGB")
            feats = extract_one(card, news[nid][1], local_m, glob_m, tfm, device)
        except Exception as exc:
            print(f"[feat] skip {nid}: {exc}", flush=True)
            continue
        id_list.append(nid)
        word_vis.append(feats["word_vis"])
        word_mask.append(feats["word_vis_mask"])
        cover.append(feats["cover_regions"])
        catv.append(feats["category_vis"])
        globv.append(feats["global_feat"])

    if not id_list:
        raise RuntimeError("no features extracted; check cards/thumbnails")

    np.savez_compressed(
        out,
        news_ids=np.array(id_list),
        word_vis=np.stack(word_vis),
        word_vis_mask=np.stack(word_mask),
        cover_regions=np.stack(cover),
        category_vis=np.stack(catv),
        global_feat=np.stack(globv),
    )
    print(f"[feat] saved {len(id_list)} news → {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind-dataset-subdir", default="MIND_2000")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-news", type=int, default=0)
    args = ap.parse_args()
    extract_all(args.mind_dataset_subdir, args.force, args.device, args.max_news)


if __name__ == "__main__":
    main()
