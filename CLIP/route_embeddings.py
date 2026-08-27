#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B1 (CLIP text) / B2 (Kandinsky prior → CLIP image) 임베딩 추출.
B4 픽셀은 clip_embeddings.extract_clip_embeddings 를 사용한다.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clip_embeddings import (
    CLIP_MODEL_ID,
    _save_pair_cache,
    load_pair_embed_dict,
    resolve_torch_device,
)


def norm_pair_key(uid, nid) -> Tuple[str, str]:
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def load_expected_bodies_from_dir(expected_dir: str) -> Dict[Tuple[str, str], str]:
    expected_bodies: Dict[Tuple[str, str], str] = {}
    if not expected_dir or not os.path.isdir(expected_dir):
        return expected_bodies
    for user_folder in os.listdir(expected_dir):
        user_path = os.path.join(expected_dir, user_folder)
        if not os.path.isdir(user_path) or not user_folder.startswith("user_"):
            continue
        user_id = user_folder.replace("user_", "")
        for filename in os.listdir(user_path):
            if not (filename.startswith("news_") and filename.endswith(".json")):
                continue
            news_id = filename.replace("news_", "").replace(".json", "")
            fpath = os.path.join(user_path, filename)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                body = data.get("generated_body") if isinstance(data, dict) else None
                if isinstance(body, str) and body.strip():
                    expected_bodies[norm_pair_key(user_id, news_id)] = body.strip()
            except Exception:
                continue
    return expected_bodies


def collect_candidate_pairs_from_tsv(tsv_path: str) -> List[Tuple[str, str]]:
    """테스트 impression TSV의 (user, candidate_news) 전부. 헤더는 건너뛴다."""
    pairs: List[Tuple[str, str]] = []
    seen = set()
    if not tsv_path or not os.path.isfile(tsv_path):
        return pairs
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            uid_raw, _hist, cand_str = parts[0], parts[1], parts[2]
            if line_i == 0 and uid_raw.strip().lower() in ("user", "userid", "user_id"):
                continue
            uid = norm_pair_key(uid_raw, "x")[0]
            if not uid:
                continue
            for cid in cand_str.split():
                nid = str(cid).strip()
                if not nid:
                    continue
                key = norm_pair_key(uid, nid)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
    return pairs


def default_expected_body_dir(project_root: str, mind_dataset_subdir: str, split: str = "test") -> str:
    name = f"{split}_3cluster_11_13_8_rawtitle"
    return os.path.join(
        project_root, "user_preference", "expected_body", mind_dataset_subdir, name
    )


def _pair_seed(uid: str, nid: str, base_seed: int) -> int:
    h = hashlib.md5(f"{uid}\t{nid}".encode("utf-8")).hexdigest()
    return (int(base_seed) + int(h[:8], 16)) % (2**31 - 1)


def _merge_and_save_pairs(
    out_path: str,
    existing: Dict[Tuple[str, str], np.ndarray],
    new_pairs: Sequence[Tuple[str, str]],
    new_vecs: Sequence[np.ndarray],
    dim: int,
) -> Dict[Tuple[str, str], np.ndarray]:
    merged = dict(existing)
    for key, vec in zip(new_pairs, new_vecs):
        merged[key] = np.asarray(vec, dtype=np.float32)
    keys = sorted(merged.keys(), key=lambda x: (x[0], x[1]))
    users = [k[0] for k in keys]
    news = [k[1] for k in keys]
    emb = np.zeros((len(keys), int(dim)), dtype=np.float32)
    for i, k in enumerate(keys):
        emb[i] = merged[k]
    _save_pair_cache(out_path, emb, users, news)
    return merged


def _load_existing_pair_cache(out_path: str) -> Dict[Tuple[str, str], np.ndarray]:
    if not os.path.isfile(out_path):
        return {}
    try:
        return load_pair_embed_dict(out_path)
    except Exception as e:
        print(f"[CLIP] 기존 pair cache를 읽지 못해 다시 추출합니다 ({out_path}): {e}", flush=True)
        return {}


def extract_clip_text_embeddings(
    pairs_and_texts: Sequence[Tuple[str, str, str]],
    out_path: str,
    *,
    model_id: str = CLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 32,
    resume: bool = True,
) -> dict:
    """
    Route T: expected body → CLIP text encoder → text_embeds.
    CLIP 77토큰 truncation=True (앞을 남기고 뒤를 자름). L2 정규화 없음.
    """
    try:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "B1 추출에는 torch, transformers 가 필요합니다. pip install -r CLIP/requirements.txt"
        ) from e

    device = resolve_torch_device(device)
    if device == "cpu" and batch_size > 16:
        print(f"[CLIP B1] CPU라 batch-size를 {batch_size} → 16 으로 낮춥니다.", flush=True)
        batch_size = 16
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    existing = _load_existing_pair_cache(out_path) if resume else {}
    todo: List[Tuple[str, str, str]] = []
    seen = set()
    for uid, nid, text in pairs_and_texts:
        key = norm_pair_key(uid, nid)
        if key in seen or key in existing:
            continue
        seen.add(key)
        if not (text or "").strip():
            continue
        todo.append((key[0], key[1], text.strip()))

    print(
        f"[CLIP B1] load text_encoder from {model_id} device={device}\n"
        f"[CLIP B1] todo={len(todo)} resume_cached={len(existing)}",
        flush=True,
    )
    if not todo and existing:
        clip_dim = int(next(iter(existing.values())).shape[0])
        meta = {
            "route": "B1",
            "model_id": model_id,
            "encoder": "CLIPTextModelWithProjection.text_embeds",
            "l2_normalize": False,
            "out_path": os.path.abspath(out_path),
            "clip_dim": int(clip_dim),
            "n_pairs": int(len(existing)),
            "n_encoded_this_run": 0,
            "space": "clip_text",
        }
        print(f"[CLIP B1] 새로 추출할 pair가 없어 cache를 유지합니다: {out_path}", flush=True)
        return meta

    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder")
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    fallback_dim = int(getattr(encoder.config, "projection_dim", 1280) or 1280)
    max_len = int(getattr(tokenizer, "model_max_length", 77) or 77)

    clip_dim: Optional[int] = None
    if existing:
        clip_dim = int(next(iter(existing.values())).shape[0])
    n_done = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            texts = [t for _u, _n, t in chunk]
            keys = [(u, n) for u, n, _t in chunk]
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = encoder(**encoded)
            vecs = out.text_embeds.float().cpu().numpy().astype(np.float32)
            if clip_dim is None:
                clip_dim = int(vecs.shape[1])
            elif int(vecs.shape[1]) != int(clip_dim):
                raise ValueError(f"B1 dim mismatch: got {vecs.shape[1]} expected {clip_dim}")
            existing = _merge_and_save_pairs(out_path, existing, keys, vecs, clip_dim)
            n_done += len(chunk)
            n_batches += 1
            if n_batches == 1 or n_batches % 20 == 0 or n_done >= len(todo):
                print(f"[CLIP B1] encoded {n_done}/{len(todo)}", flush=True)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    if clip_dim is None:
        clip_dim = fallback_dim
        if existing:
            clip_dim = int(next(iter(existing.values())).shape[0])
        _merge_and_save_pairs(out_path, existing, [], [], clip_dim)

    meta = {
        "route": "B1",
        "model_id": model_id,
        "encoder": "CLIPTextModelWithProjection.text_embeds",
        "max_length": max_len,
        "l2_normalize": False,
        "out_path": os.path.abspath(out_path),
        "clip_dim": int(clip_dim),
        "n_pairs": int(len(existing)),
        "n_encoded_this_run": int(n_done),
        "space": "clip_text",
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[CLIP B1] saved {out_path}\n"
        f"[CLIP B1] dim={clip_dim} pairs={len(existing)} encoded_this_run={n_done}\n"
        f"[CLIP B1] meta {meta_path}",
        flush=True,
    )
    return meta


def extract_prior_image_embeddings(
    pairs_and_texts: Sequence[Tuple[str, str, str]],
    out_path: str,
    *,
    model_id: str = CLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 4,
    num_inference_steps: int = 25,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resume: bool = True,
) -> dict:
    """
    Route E: expected body → Kandinsky 2.2 prior → CLIP image_embeds.
    L2 정규화 없음. pair마다 고정 시드라 배치 크기와 무관하게 재현된다.
    """
    try:
        import torch
        from diffusers import KandinskyV22PriorPipeline
    except ImportError as e:
        raise ImportError(
            "B2 추출에는 torch, diffusers 가 필요합니다. pip install -r CLIP/requirements.txt"
        ) from e

    device = resolve_torch_device(device)
    if device == "cpu":
        print("[CLIP B2] CPU prior는 매우 느립니다. GPU 환경을 권장합니다.", flush=True)
        if batch_size > 2:
            print(f"[CLIP B2] CPU라 batch-size를 {batch_size} → 1 로 낮춥니다.", flush=True)
            batch_size = 1
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    existing = _load_existing_pair_cache(out_path) if resume else {}
    todo: List[Tuple[str, str, str]] = []
    seen = set()
    for uid, nid, text in pairs_and_texts:
        key = norm_pair_key(uid, nid)
        if key in seen or key in existing:
            continue
        seen.add(key)
        if not (text or "").strip():
            continue
        todo.append((key[0], key[1], text.strip()))

    print(
        f"[CLIP B2] load KandinskyV22PriorPipeline {model_id} device={device}\n"
        f"[CLIP B2] todo={len(todo)} resume_cached={len(existing)} "
        f"steps={num_inference_steps} guidance={guidance_scale}",
        flush=True,
    )
    if not todo and existing:
        clip_dim = int(next(iter(existing.values())).shape[0])
        print(f"[CLIP B2] 새로 추출할 pair가 없어 cache를 유지합니다: {out_path}", flush=True)
        return {
            "route": "B2",
            "model_id": model_id,
            "encoder": "KandinskyV22PriorPipeline.image_embeds",
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "l2_normalize": False,
            "out_path": os.path.abspath(out_path),
            "clip_dim": int(clip_dim),
            "n_pairs": int(len(existing)),
            "n_encoded_this_run": 0,
            "space": "clip_image",
        }
    pipe = KandinskyV22PriorPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    clip_dim: Optional[int] = None
    if existing:
        clip_dim = int(next(iter(existing.values())).shape[0])
    n_done = 0
    n_batches = 0
    fallback_dim = 1280
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        texts = [t for _u, _n, t in chunk]
        keys = [(u, n) for u, n, _t in chunk]
        gens = [
            torch.Generator(device="cpu").manual_seed(_pair_seed(u, n, seed))
            for u, n in keys
        ]
        out = pipe(
            prompt=texts,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gens,
        )
        embeds = out.image_embeds
        if hasattr(embeds, "detach"):
            vecs = embeds.detach().float().cpu().numpy().astype(np.float32)
        else:
            vecs = np.asarray(embeds, dtype=np.float32)
        if clip_dim is None:
            clip_dim = int(vecs.shape[1])
        elif int(vecs.shape[1]) != int(clip_dim):
            raise ValueError(f"B2 dim mismatch: got {vecs.shape[1]} expected {clip_dim}")
        existing = _merge_and_save_pairs(out_path, existing, keys, vecs, clip_dim)
        n_done += len(chunk)
        n_batches += 1
        if n_batches == 1 or n_batches % 5 == 0 or n_done >= len(todo):
            print(f"[CLIP B2] encoded {n_done}/{len(todo)}", flush=True)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    if clip_dim is None:
        clip_dim = fallback_dim
        if existing:
            clip_dim = int(next(iter(existing.values())).shape[0])
        _merge_and_save_pairs(out_path, existing, [], [], clip_dim)

    meta = {
        "route": "B2",
        "model_id": model_id,
        "encoder": "KandinskyV22PriorPipeline.image_embeds",
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "l2_normalize": False,
        "out_path": os.path.abspath(out_path),
        "clip_dim": int(clip_dim),
        "n_pairs": int(len(existing)),
        "n_encoded_this_run": int(n_done),
        "space": "clip_image",
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[CLIP B2] saved {out_path}\n"
        f"[CLIP B2] dim={clip_dim} pairs={len(existing)} encoded_this_run={n_done}\n"
        f"[CLIP B2] meta {meta_path}",
        flush=True,
    )
    return meta


def build_pairs_and_texts(
    pairs: Sequence[Tuple[str, str]],
    expected_bodies: Dict[Tuple[str, str], str],
) -> Tuple[List[Tuple[str, str, str]], int, int]:
    items: List[Tuple[str, str, str]] = []
    n_missing = 0
    for uid, nid in pairs:
        key = norm_pair_key(uid, nid)
        body = expected_bodies.get(key)
        if not body:
            n_missing += 1
            continue
        items.append((key[0], key[1], body))
    return items, len(items), n_missing
