#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MIND_2000 impressions + impression feature tables for IMRec."""
from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from extract_features import LOCAL_DIM, GLOBAL_DIM, N_COVER_REGIONS, MAX_TITLE_WORDS
from paths import DATASET_FILE_PRESETS, dataset_raw_dir, features_path

try:
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
except Exception:
    nltk_word_tokenize = None


def word_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    if nltk_word_tokenize is not None:
        try:
            return nltk_word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[\w]+|[.,!?;|]", text)


@dataclass
class NewsTables:
    news_ids: List[str]
    news_index: Dict[str, int]
    title_ids: np.ndarray  # [N, L]
    title_mask: np.ndarray
    category_ids: np.ndarray
    word_vis: np.ndarray  # [N, L, 512]
    word_vis_mask: np.ndarray
    cover_regions: np.ndarray  # [N, 9, 512]
    category_vis: np.ndarray  # [N, 512]
    global_feat: np.ndarray  # [N, 2048]
    word_dict: Dict[str, int]
    category_dict: Dict[str, int]
    embedding_mat: Optional[np.ndarray]


def load_glove(word_dict: Dict[str, int], glove_path: str, dim: int = 100) -> np.ndarray:
    mat = np.random.uniform(-0.1, 0.1, size=(len(word_dict), dim)).astype(np.float32)
    mat[0] = 0.0
    path = Path(glove_path)
    if not path.is_file():
        print(f"[data] GloVe missing ({glove_path}); random {dim}-d embeddings", flush=True)
        return mat
    # Prefer 100d; if 300d file, take first 100
    found = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 50:
                continue
            w = parts[0]
            if w not in word_dict:
                continue
            vec = np.asarray([float(x) for x in parts[1 : 1 + dim]], dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            mat[word_dict[w]] = vec
            found += 1
    print(f"[data] GloVe hit {found}/{len(word_dict)-1}", flush=True)
    return mat


def build_news_tables(
    mind_dataset_subdir: str,
    max_title_len: int = 30,
    glove_path: str = "",
    emb_dim: int = 100,
) -> NewsTables:
    raw = dataset_raw_dir(mind_dataset_subdir)
    news_name = DATASET_FILE_PRESETS[mind_dataset_subdir][0]
    news_path = raw / news_name

    print("[data] reading news tsv...", flush=True)
    category_dict = {"PADDING": 0}
    word_dict = {"PADDING": 0}
    rows: List[Tuple[str, str, List[str]]] = []

    with news_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            nid, cat, title = parts[0], parts[1], parts[3]
            if nid.lower() in ("news_id", "id"):
                continue
            if cat not in category_dict:
                category_dict[cat] = len(category_dict)
            toks = word_tokenize(title)[:max_title_len]
            for w in toks:
                if w not in word_dict:
                    word_dict[w] = len(word_dict)
            rows.append((nid, cat, toks))

    n = len(rows) + 1  # index 0 = padding news
    L = max_title_len
    title_ids = np.zeros((n, L), dtype=np.int64)
    title_mask = np.zeros((n, L), dtype=np.float32)
    category_ids = np.zeros(n, dtype=np.int64)
    word_vis = np.zeros((n, MAX_TITLE_WORDS, LOCAL_DIM), dtype=np.float32)
    word_vis_mask = np.zeros((n, MAX_TITLE_WORDS), dtype=np.float32)
    cover_regions = np.zeros((n, N_COVER_REGIONS, LOCAL_DIM), dtype=np.float32)
    category_vis = np.zeros((n, LOCAL_DIM), dtype=np.float32)
    global_feat = np.zeros((n, GLOBAL_DIM), dtype=np.float32)

    news_ids = ["PADDING"]
    news_index = {"PADDING": 0}
    for nid, cat, toks in rows:
        idx = len(news_ids)
        news_ids.append(nid)
        news_index[nid] = idx
        category_ids[idx] = category_dict[cat]
        for t, w in enumerate(toks):
            title_ids[idx, t] = word_dict[w]
            title_mask[idx, t] = 1.0

    feat_file = features_path(mind_dataset_subdir)
    if feat_file.is_file():
        print(f"[data] mmap features ← {feat_file}", flush=True)
        z = np.load(feat_file, allow_pickle=True, mmap_mode="r")
        feat_ids = [str(x) for x in z["news_ids"].tolist()]
        # load arrays once (mmap may still page in); map by id without per-row dict copies
        src_word = np.asarray(z["word_vis"])
        src_wmask = np.asarray(z["word_vis_mask"])
        src_cover = np.asarray(z["cover_regions"])
        src_cat = np.asarray(z["category_vis"])
        src_glob = np.asarray(z["global_feat"])
        hit = 0
        for i, nid in enumerate(feat_ids):
            idx = news_index.get(nid)
            if idx is None:
                continue
            word_vis[idx] = src_word[i]
            word_vis_mask[idx] = src_wmask[i]
            cover_regions[idx] = src_cover[i]
            category_vis[idx] = src_cat[i]
            global_feat[idx] = src_glob[i]
            hit += 1
        del src_word, src_wmask, src_cover, src_cat, src_glob, z
        print(f"[data] features aligned hit={hit}/{len(feat_ids)}", flush=True)
    else:
        print(f"[data] WARNING: no features at {feat_file}; using zeros", flush=True)

    print(f"[data] loading GloVe ({glove_path or 'random'}) ...", flush=True)
    emb = load_glove(word_dict, glove_path, emb_dim) if glove_path else None
    if emb is None:
        emb = np.random.uniform(-0.1, 0.1, size=(len(word_dict), emb_dim)).astype(np.float32)
        emb[0] = 0.0
    print(f"[data] tables ready news={n-1} vocab={len(word_dict)}", flush=True)

    return NewsTables(
        news_ids=news_ids,
        news_index=news_index,
        title_ids=title_ids,
        title_mask=title_mask,
        category_ids=category_ids,
        word_vis=word_vis,
        word_vis_mask=word_vis_mask,
        cover_regions=cover_regions,
        category_vis=category_vis,
        global_feat=global_feat,
        word_dict=word_dict,
        category_dict=category_dict,
        embedding_mat=emb,
    )


def _parse_ids(s: str) -> List[str]:
    return [x for x in str(s).strip().split() if x]


def load_impressions(
    path: Path,
    news_index: Dict[str, int],
    *,
    has_header: bool,
    has_labels: bool,
    npratio: int,
    max_history: int,
    rng: random.Random,
) -> Tuple[List[dict], Optional[List[Tuple[int, int]]]]:
    """
    Returns list of impressions.
    Train: each item has history [H], candidates [1+npratio], labels.
    Val/Test: flattened candidates with session spans in all_test_index.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        if has_header:
            next(reader, None)
        for line in reader:
            if len(line) < 3:
                continue
            hist = [news_index[x] for x in _parse_ids(line[1]) if x in news_index]
            cands_raw = _parse_ids(line[2])
            cands = [news_index[x] for x in cands_raw if x in news_index]
            if not hist or len(cands) < 2:
                continue
            if has_labels:
                if len(line) < 4:
                    continue
                labels = [int(float(x)) for x in _parse_ids(line[3])]
                # align labels to filtered cands by original order
                pairs = []
                for i, cid_str in enumerate(cands_raw):
                    if cid_str not in news_index:
                        continue
                    lab = labels[i] if i < len(labels) else 0
                    pairs.append((news_index[cid_str], lab))
                if not any(l for _, l in pairs):
                    continue
                pos = [c for c, l in pairs if l == 1]
                neg = [c for c, l in pairs if l == 0]
                if not pos:
                    continue
                p = rng.choice(pos)
                rng.shuffle(neg)
                chosen = [p] + neg[:npratio]
                while len(chosen) < 1 + npratio:
                    chosen.append(0)
                labs = [1] + [0] * npratio
                order = list(range(len(chosen)))
                rng.shuffle(order)
                chosen = [chosen[i] for i in order]
                labs = [labs[i] for i in order]
                # history: remove candidates, take last max_history
                cand_set = set(chosen)
                h = [x for x in hist if x not in cand_set][-max_history:]
                if not h:
                    continue
                h = ([0] * (max_history - len(h))) + h
                rows.append({"history": h, "candidates": chosen, "labels": labs})
            else:
                # first candidate positive
                labels = [1] + [0] * (len(cands) - 1)
                order = list(range(len(cands)))
                rng.shuffle(order)
                cands = [cands[i] for i in order]
                labels = [labels[i] for i in order]
                cand_set = set(cands)
                h = [x for x in hist if x not in cand_set][-max_history:]
                if not h:
                    continue
                h = ([0] * (max_history - len(h))) + h
                rows.append({"history": h, "candidates": cands, "labels": labels})

    if has_labels:
        return rows, None

    # flatten for eval
    flat = []
    spans = []
    for imp in rows:
        start = len(flat)
        for c, lab in zip(imp["candidates"], imp["labels"]):
            flat.append({"history": imp["history"], "candidate": c, "label": lab})
        spans.append((start, len(flat)))
    return flat, spans


class TrainBatcher:
    def __init__(
        self,
        impressions: List[dict],
        tables: NewsTables,
        batch_size: int,
        max_title_len: int,
        device: str,
        seed: int = 42,
    ):
        self.impressions = impressions
        self.tables = tables
        self.batch_size = batch_size
        self.max_title_len = max_title_len
        self.device = device
        self.rng = random.Random(seed)
        self.indices = list(range(len(impressions)))

    def __len__(self) -> int:
        return max(1, (len(self.impressions) + self.batch_size - 1) // self.batch_size)

    def _pack_news(self, idxs: Sequence[int]) -> dict:
        import torch

        t = self.tables
        L = self.max_title_len
        idxs = np.asarray(idxs, dtype=np.int64)
        return {
            "title_ids": torch.as_tensor(t.title_ids[idxs, :L], device=self.device),
            "title_mask": torch.as_tensor(t.title_mask[idxs, :L], device=self.device),
            "word_vis": torch.as_tensor(t.word_vis[idxs, :L, :], device=self.device),
            "word_vis_mask": torch.as_tensor(t.word_vis_mask[idxs, :L], device=self.device),
            "cover_regions": torch.as_tensor(t.cover_regions[idxs], device=self.device),
            "category_vis": torch.as_tensor(t.category_vis[idxs], device=self.device),
            "global_feat": torch.as_tensor(t.global_feat[idxs], device=self.device),
        }

    def iter_epoch(self):
        import torch

        self.rng.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            batch_ids = self.indices[start : start + self.batch_size]
            imps = [self.impressions[i] for i in batch_ids]
            B = len(imps)
            H = len(imps[0]["history"])
            C = len(imps[0]["candidates"])
            hist = np.zeros((B, H), dtype=np.int64)
            cand = np.zeros((B, C), dtype=np.int64)
            labels = np.zeros((B, C), dtype=np.float32)
            for i, imp in enumerate(imps):
                hist[i] = imp["history"]
                cand[i] = imp["candidates"]
                labels[i] = imp["labels"]
            hist_pack = self._pack_news(hist.reshape(-1))
            cand_pack = self._pack_news(cand.reshape(-1))
            # reshape packs to [B,H,*] / [B,C,*]
            def reshape_pack(pack, n1, n2):
                out = {}
                for k, v in pack.items():
                    out[k] = v.view(n1, n2, *v.shape[1:])
                return out

            yield {
                "history": reshape_pack(hist_pack, B, H),
                "candidates": reshape_pack(cand_pack, B, C),
                "labels": torch.as_tensor(labels, device=self.device),
                "hist_mask": torch.as_tensor((hist != 0).astype(np.float32), device=self.device),
            }


class EvalBatcher:
    def __init__(self, flat: List[dict], tables: NewsTables, batch_size: int, max_title_len: int, device: str):
        self.flat = flat
        self.tables = tables
        self.batch_size = batch_size
        self.max_title_len = max_title_len
        self.device = device

    def __len__(self) -> int:
        return max(1, (len(self.flat) + self.batch_size - 1) // self.batch_size)

    def iter_all(self):
        import torch

        t = self.tables
        L = self.max_title_len
        for start in range(0, len(self.flat), self.batch_size):
            chunk = self.flat[start : start + self.batch_size]
            B = len(chunk)
            H = len(chunk[0]["history"])
            hist = np.zeros((B, H), dtype=np.int64)
            cand = np.zeros(B, dtype=np.int64)
            for i, row in enumerate(chunk):
                hist[i] = row["history"]
                cand[i] = row["candidate"]

            def pack_multi(idxs, n1, n2=None):
                idxs = np.asarray(idxs, dtype=np.int64)
                flat_idx = idxs.reshape(-1)
                pack = {
                    "title_ids": torch.as_tensor(t.title_ids[flat_idx, :L], device=self.device),
                    "title_mask": torch.as_tensor(t.title_mask[flat_idx, :L], device=self.device),
                    "word_vis": torch.as_tensor(t.word_vis[flat_idx, :L, :], device=self.device),
                    "word_vis_mask": torch.as_tensor(t.word_vis_mask[flat_idx, :L], device=self.device),
                    "cover_regions": torch.as_tensor(t.cover_regions[flat_idx], device=self.device),
                    "category_vis": torch.as_tensor(t.category_vis[flat_idx], device=self.device),
                    "global_feat": torch.as_tensor(t.global_feat[flat_idx], device=self.device),
                }
                if n2 is not None:
                    for k, v in pack.items():
                        pack[k] = v.view(n1, n2, *v.shape[1:])
                return pack

            yield {
                "history": pack_multi(hist, B, H),
                "candidates": pack_multi(cand, B, None),
                "hist_mask": torch.as_tensor((hist != 0).astype(np.float32), device=self.device),
            }
