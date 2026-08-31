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
    CLIP_DECODER_MODEL_ID,
    CLIP_IMAGE_ENCODER_SUBFOLDER,
    CLIP_IMAGE_PROCESSOR_SUBFOLDER,
    CLIP_MODEL_ID,
    CLIP_TEXT_MAX_LENGTH,
    _save_clip_cache,
    _save_pair_cache,
    load_clip_npz,
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
    # HF CLIP tokenizer.model_max_length 가 2^100 같은 값으로 저장돼
    # Rust tokenizer enable_truncation 에서 OverflowError 가 난다. CLIP 은 77 고정.
    raw_max = getattr(tokenizer, "model_max_length", CLIP_TEXT_MAX_LENGTH)
    try:
        raw_max_int = int(raw_max)
    except (TypeError, ValueError, OverflowError):
        raw_max_int = CLIP_TEXT_MAX_LENGTH
    if raw_max_int <= 0 or raw_max_int > CLIP_TEXT_MAX_LENGTH:
        print(
            f"[CLIP B1] tokenizer.model_max_length={raw_max} → {CLIP_TEXT_MAX_LENGTH} 로 고정",
            flush=True,
        )
    tokenizer.model_max_length = CLIP_TEXT_MAX_LENGTH
    encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder")
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    fallback_dim = int(getattr(encoder.config, "projection_dim", 1280) or 1280)
    max_len = CLIP_TEXT_MAX_LENGTH

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
    if getattr(pipe, "tokenizer", None) is not None:
        raw_max = getattr(pipe.tokenizer, "model_max_length", CLIP_TEXT_MAX_LENGTH)
        try:
            raw_max_int = int(raw_max)
        except (TypeError, ValueError, OverflowError):
            raw_max_int = CLIP_TEXT_MAX_LENGTH
        if raw_max_int <= 0 or raw_max_int > CLIP_TEXT_MAX_LENGTH:
            print(
                f"[CLIP B2] tokenizer.model_max_length={raw_max} → {CLIP_TEXT_MAX_LENGTH} 로 고정",
                flush=True,
            )
        pipe.tokenizer.model_max_length = CLIP_TEXT_MAX_LENGTH

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


def load_actual_bodies_from_news_tsv(news_tsv: str) -> Dict[str, str]:
    """MIND_news.tsv: news_id, category, subcategory, title, body."""
    bodies: Dict[str, str] = {}
    if not news_tsv or not os.path.isfile(news_tsv):
        return bodies
    with open(news_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            nid = parts[0].strip()
            if not nid or nid.lower() in ("news_id", "clicked_news", "id"):
                continue
            body = parts[4].strip() if len(parts) > 4 else ""
            bodies[nid] = body
    return bodies


def collect_news_ids_from_interaction_tsv(tsv_path: str) -> List[str]:
    """train/test TSV의 clicked_news + candidate_news unique ID (등장 순)."""
    ids: List[str] = []
    seen = set()
    if not tsv_path or not os.path.isfile(tsv_path):
        return ids
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            if line_i == 0 and parts[0].strip().lower() in ("user", "userid", "user_id"):
                continue
            for col in (1, 2):
                for nid in str(parts[col] or "").split():
                    ns = nid.strip()
                    if not ns or ns in seen:
                        continue
                    seen.add(ns)
                    ids.append(ns)
    return ids


def extract_clip_text_actual_bodies(
    news_ids: Sequence[str],
    bodies: Dict[str, str],
    out_path: str,
    *,
    model_id: str = CLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 32,
    resume: bool = True,
) -> dict:
    """
    실제 뉴스 본문 → CLIP text encoder → text_embeds (뉴스 단위).
    77토큰 truncation, L2 없음. 빈 본문은 0벡터.
    """
    try:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "CLIP text 추출에는 torch, transformers 가 필요합니다. pip install -r CLIP/requirements.txt"
        ) from e

    catalog = [str(n) for n in news_ids if n and n != "0"]
    if not catalog:
        raise ValueError("인코딩할 뉴스 ID가 없습니다.")

    device = resolve_torch_device(device)
    if device == "cpu" and batch_size > 16:
        print(f"[CLIP actual-body] CPU라 batch-size를 {batch_size} → 16 으로 낮춥니다.", flush=True)
        batch_size = 16
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    id_to_vec: Dict[str, np.ndarray] = {}
    if resume and os.path.isfile(out_path):
        try:
            emb, cached_ids = load_clip_npz(out_path)
            for nid, vec in zip(cached_ids, emb):
                id_to_vec[str(nid)] = np.asarray(vec, dtype=np.float32)
            print(f"[CLIP actual-body] resume cache {len(id_to_vec)} ids from {out_path}", flush=True)
        except Exception as e:
            print(f"[CLIP actual-body] 기존 cache를 읽지 못해 다시 추출합니다: {e}", flush=True)
            id_to_vec = {}

    todo: List[str] = []
    n_empty = 0
    n_missing_body_key = 0
    for nid in catalog:
        if nid in id_to_vec:
            continue
        if nid not in bodies:
            n_missing_body_key += 1
            continue
        if not (bodies.get(nid) or "").strip():
            n_empty += 1
            continue
        todo.append(nid)

    print(
        f"[CLIP actual-body] catalog={len(catalog)} todo={len(todo)} "
        f"resume={len(id_to_vec)} empty_body={n_empty} not_in_news_tsv={n_missing_body_key}",
        flush=True,
    )

    clip_dim: Optional[int] = None
    if id_to_vec:
        clip_dim = int(next(iter(id_to_vec.values())).reshape(-1).shape[0])

    def _flush(dim: int) -> None:
        merged = dict(id_to_vec)
        _save_news_text_cache(out_path, merged, catalog, dim)

    if not todo:
        dim = int(clip_dim or 1280)
        for nid in catalog:
            if nid in id_to_vec:
                continue
            id_to_vec[nid] = np.zeros((dim,), dtype=np.float32)
        _flush(dim)
        print(f"[CLIP actual-body] 새로 인코딩할 본문이 없어 cache를 갱신합니다: {out_path}", flush=True)
        return {
            "encoder": "CLIPTextModelWithProjection.text_embeds",
            "source": "actual_body",
            "out_path": os.path.abspath(out_path),
            "clip_dim": dim,
            "n_news": len(catalog),
            "n_encoded_this_run": 0,
            "n_empty_body": n_empty,
            "n_missing_in_news_tsv": n_missing_body_key,
            "space": "clip_text",
            "max_length": CLIP_TEXT_MAX_LENGTH,
            "l2_normalize": False,
        }

    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    _fix_clip_tokenizer_max_length(tokenizer)
    encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder")
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    fallback_dim = int(getattr(encoder.config, "projection_dim", 1280) or 1280)
    max_len = CLIP_TEXT_MAX_LENGTH

    n_done = 0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            texts = [bodies[nid].strip() for nid in chunk]
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
                raise ValueError(f"actual-body dim mismatch: got {vecs.shape[1]} expected {clip_dim}")
            for nid, vec in zip(chunk, vecs):
                id_to_vec[nid] = vec
            n_done += len(chunk)
            n_batches += 1
            if n_batches == 1 or n_batches % 20 == 0 or n_done >= len(todo):
                _flush(clip_dim)
                print(f"[CLIP actual-body] encoded {n_done}/{len(todo)}", flush=True)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    if clip_dim is None:
        clip_dim = fallback_dim
    zero = np.zeros((int(clip_dim),), dtype=np.float32)
    for nid in catalog:
        if nid not in id_to_vec:
            id_to_vec[nid] = zero
    _flush(int(clip_dim))

    meta = {
        "encoder": "CLIPTextModelWithProjection.text_embeds",
        "source": "actual_body",
        "model_id": model_id,
        "out_path": os.path.abspath(out_path),
        "clip_dim": int(clip_dim),
        "n_news": len(catalog),
        "n_encoded_this_run": int(n_done),
        "n_nonzero": int(sum(1 for nid in catalog if np.any(id_to_vec[nid]))),
        "n_empty_body": n_empty,
        "n_missing_in_news_tsv": n_missing_body_key,
        "space": "clip_text",
        "max_length": CLIP_TEXT_MAX_LENGTH,
        "l2_normalize": False,
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[CLIP actual-body] saved {out_path}\n"
        f"[CLIP actual-body] dim={clip_dim} news={len(catalog)} encoded_this_run={n_done} "
        f"nonzero={meta['n_nonzero']}\n"
        f"[CLIP actual-body] meta {meta_path}",
        flush=True,
    )
    return meta


def _save_news_text_cache(
    out_path: str,
    id_to_vec: Dict[str, np.ndarray],
    catalog: Sequence[str],
    dim: int,
) -> None:
    emb = np.zeros((len(catalog), int(dim)), dtype=np.float32)
    for i, nid in enumerate(catalog):
        vec = id_to_vec.get(nid)
        if vec is not None:
            emb[i] = np.asarray(vec, dtype=np.float32).reshape(-1)
    _save_clip_cache(out_path, emb, catalog)


def extract_prior_actual_bodies(
    news_ids: Sequence[str],
    bodies: Dict[str, str],
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
    실제 뉴스 본문 → Kandinsky 2.2 prior → CLIP image_embeds (뉴스 단위).
    L2 없음. 빈 본문은 0벡터. 뉴스마다 고정 시드.
    """
    try:
        import torch
        from diffusers import KandinskyV22PriorPipeline
    except ImportError as e:
        raise ImportError(
            "actual-body prior 추출에는 torch, diffusers 가 필요합니다. pip install -r CLIP/requirements.txt"
        ) from e

    catalog = [str(n) for n in news_ids if n and n != "0"]
    if not catalog:
        raise ValueError("인코딩할 뉴스 ID가 없습니다.")

    device = resolve_torch_device(device)
    if device == "cpu":
        print("[CLIP actual-body prior] CPU prior는 매우 느립니다.", flush=True)
        if batch_size > 2:
            print(f"[CLIP actual-body prior] CPU라 batch-size를 {batch_size} → 1 로 낮춥니다.", flush=True)
            batch_size = 1
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    id_to_vec: Dict[str, np.ndarray] = {}
    if resume and os.path.isfile(out_path):
        try:
            emb, cached_ids = load_clip_npz(out_path)
            for nid, vec in zip(cached_ids, emb):
                id_to_vec[str(nid)] = np.asarray(vec, dtype=np.float32)
            print(f"[CLIP actual-body prior] resume cache {len(id_to_vec)} ids from {out_path}", flush=True)
        except Exception as e:
            print(f"[CLIP actual-body prior] 기존 cache를 읽지 못해 다시 추출합니다: {e}", flush=True)
            id_to_vec = {}

    todo: List[str] = []
    n_empty = 0
    n_missing_body_key = 0
    for nid in catalog:
        if nid in id_to_vec:
            continue
        if nid not in bodies:
            n_missing_body_key += 1
            continue
        if not (bodies.get(nid) or "").strip():
            n_empty += 1
            continue
        todo.append(nid)

    print(
        f"[CLIP actual-body prior] catalog={len(catalog)} todo={len(todo)} "
        f"resume={len(id_to_vec)} empty_body={n_empty} not_in_news_tsv={n_missing_body_key} "
        f"steps={num_inference_steps} guidance={guidance_scale}",
        flush=True,
    )

    clip_dim: Optional[int] = None
    if id_to_vec:
        clip_dim = int(next(iter(id_to_vec.values())).reshape(-1).shape[0])

    def _flush(dim: int) -> None:
        _save_news_text_cache(out_path, dict(id_to_vec), catalog, dim)

    if not todo:
        dim = int(clip_dim or 1280)
        for nid in catalog:
            if nid in id_to_vec:
                continue
            id_to_vec[nid] = np.zeros((dim,), dtype=np.float32)
        _flush(dim)
        print(f"[CLIP actual-body prior] 새로 인코딩할 본문이 없어 cache를 갱신합니다: {out_path}", flush=True)
        return {
            "encoder": "KandinskyV22PriorPipeline.image_embeds",
            "source": "actual_body",
            "out_path": os.path.abspath(out_path),
            "clip_dim": dim,
            "n_news": len(catalog),
            "n_encoded_this_run": 0,
            "n_empty_body": n_empty,
            "n_missing_in_news_tsv": n_missing_body_key,
            "space": "clip_image",
            "l2_normalize": False,
        }

    pipe = KandinskyV22PriorPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe, "tokenizer", None) is not None:
        raw_max = getattr(pipe.tokenizer, "model_max_length", CLIP_TEXT_MAX_LENGTH)
        try:
            raw_max_int = int(raw_max)
        except (TypeError, ValueError, OverflowError):
            raw_max_int = CLIP_TEXT_MAX_LENGTH
        if raw_max_int <= 0 or raw_max_int > CLIP_TEXT_MAX_LENGTH:
            print(
                f"[CLIP actual-body prior] tokenizer.model_max_length={raw_max} → {CLIP_TEXT_MAX_LENGTH} 로 고정",
                flush=True,
            )
        pipe.tokenizer.model_max_length = CLIP_TEXT_MAX_LENGTH

    n_done = 0
    n_batches = 0
    fallback_dim = 1280
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        texts = [bodies[nid].strip() for nid in chunk]
        gens = [
            torch.Generator(device="cpu").manual_seed(_pair_seed("actual_body", nid, seed))
            for nid in chunk
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
            raise ValueError(f"actual-body prior dim mismatch: got {vecs.shape[1]} expected {clip_dim}")
        for nid, vec in zip(chunk, vecs):
            id_to_vec[nid] = vec
        n_done += len(chunk)
        n_batches += 1
        if n_batches == 1 or n_batches % 5 == 0 or n_done >= len(todo):
            _flush(clip_dim)
            print(f"[CLIP actual-body prior] encoded {n_done}/{len(todo)}", flush=True)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    if clip_dim is None:
        clip_dim = fallback_dim
    zero = np.zeros((int(clip_dim),), dtype=np.float32)
    for nid in catalog:
        if nid not in id_to_vec:
            id_to_vec[nid] = zero
    _flush(int(clip_dim))

    meta = {
        "encoder": "KandinskyV22PriorPipeline.image_embeds",
        "source": "actual_body",
        "model_id": model_id,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "out_path": os.path.abspath(out_path),
        "clip_dim": int(clip_dim),
        "n_news": len(catalog),
        "n_encoded_this_run": int(n_done),
        "n_nonzero": int(sum(1 for nid in catalog if np.any(id_to_vec[nid]))),
        "n_empty_body": n_empty,
        "n_missing_in_news_tsv": n_missing_body_key,
        "space": "clip_image",
        "l2_normalize": False,
        "max_length": CLIP_TEXT_MAX_LENGTH,
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[CLIP actual-body prior] saved {out_path}\n"
        f"[CLIP actual-body prior] dim={clip_dim} news={len(catalog)} encoded_this_run={n_done} "
        f"nonzero={meta['n_nonzero']}\n"
        f"[CLIP actual-body prior] meta {meta_path}",
        flush=True,
    )
    return meta


def b3_image_path(out_dir: str, uid: str, nid: str) -> str:
    return os.path.join(out_dir, f"user_{uid}", f"news_{nid}.png")


def _fix_clip_tokenizer_max_length(tokenizer) -> None:
    if tokenizer is None:
        return
    raw_max = getattr(tokenizer, "model_max_length", CLIP_TEXT_MAX_LENGTH)
    try:
        raw_max_int = int(raw_max)
    except (TypeError, ValueError, OverflowError):
        raw_max_int = CLIP_TEXT_MAX_LENGTH
    if raw_max_int <= 0 or raw_max_int > CLIP_TEXT_MAX_LENGTH:
        tokenizer.model_max_length = CLIP_TEXT_MAX_LENGTH
    else:
        tokenizer.model_max_length = CLIP_TEXT_MAX_LENGTH


def _uncond_prior_image_embed(
    *,
    model_id: str,
    device: str,
    dtype,
    seed: int,
    num_inference_steps: int,
):
    import torch
    from diffusers import KandinskyV22PriorPipeline

    print(f"[CLIP B3] uncond prior embed from empty prompt device={device}", flush=True)
    pipe = KandinskyV22PriorPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    _fix_clip_tokenizer_max_length(getattr(pipe, "tokenizer", None))
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = pipe(
        prompt="",
        num_inference_steps=int(num_inference_steps),
        guidance_scale=1.0,
        generator=gen,
    )
    neg = out.image_embeds.detach().float().cpu()
    if neg.ndim == 1:
        neg = neg.unsqueeze(0)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()
    return neg


def generate_images_from_prior_embeds(
    pair_embeds: Dict[Tuple[str, str], np.ndarray],
    out_dir: str,
    *,
    decoder_id: str = CLIP_DECODER_MODEL_ID,
    prior_id: str = CLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 1,
    height: int = 768,
    width: int = 768,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resume: bool = True,
    max_images: int = 0,
    negative_mode: str = "prior",
    cpu_offload: bool = False,
    prior_steps: int = 25,
) -> dict:
    """
    B2 prior image embed → Kandinsky 2.2 decoder → PNG.
    저장: out_dir/user_<uid>/news_<nid>.png
    """
    try:
        import torch
        from diffusers import KandinskyV22Pipeline
    except ImportError as e:
        raise ImportError(
            "B3 이미지 생성에는 torch, diffusers, Pillow 가 필요합니다. "
            "pip install -r CLIP/requirements.txt"
        ) from e

    device = resolve_torch_device(device)
    if device == "cpu":
        print("[CLIP B3] CPU decoder는 매우 느립니다. GPU를 권장합니다.", flush=True)
        batch_size = 1
        dtype = torch.float32
    else:
        dtype = torch.float16

    pair_embeds = {
        norm_pair_key(uid, nid): np.asarray(vec, dtype=np.float32).reshape(-1)
        for (uid, nid), vec in pair_embeds.items()
    }

    keys: List[Tuple[str, str]] = []
    for (uid, nid), vec in pair_embeds.items():
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if not np.any(arr):
            continue
        path = b3_image_path(out_dir, uid, nid)
        if resume and os.path.isfile(path):
            continue
        keys.append((uid, nid))
    keys.sort(key=lambda x: (x[0], x[1]))
    if max_images and max_images > 0:
        keys = keys[: int(max_images)]

    n_existing = 0
    if resume:
        for (uid, nid), vec in pair_embeds.items():
            if os.path.isfile(b3_image_path(out_dir, uid, nid)):
                n_existing += 1

    print(
        f"[CLIP B3] decoder={decoder_id} device={device} size={width}x{height} "
        f"steps={num_inference_steps} guidance={guidance_scale}\n"
        f"[CLIP B3] out_dir={out_dir}\n"
        f"[CLIP B3] prior_pairs={len(pair_embeds)} existing_png={n_existing} todo={len(keys)}",
        flush=True,
    )
    if not keys:
        print("[CLIP B3] 새로 생성할 이미지가 없습니다.", flush=True)
        return {
            "n_todo": 0,
            "n_generated": 0,
            "n_existing": n_existing,
            "out_dir": os.path.abspath(out_dir),
        }

    if negative_mode == "zeros":
        sample_dim = int(np.asarray(next(iter(pair_embeds.values()))).reshape(-1).shape[0])
        negative = torch.zeros((1, sample_dim), dtype=torch.float32)
        print("[CLIP B3] negative_image_embeds = zeros", flush=True)
    else:
        negative = _uncond_prior_image_embed(
            model_id=prior_id,
            device=device,
            dtype=dtype,
            seed=seed,
            num_inference_steps=prior_steps,
        )

    print(f"[CLIP B3] load decoder {decoder_id}", flush=True)
    pipe = KandinskyV22Pipeline.from_pretrained(decoder_id, torch_dtype=dtype)
    if cpu_offload and device.startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    n_done = 0
    n_batches = 0
    os.makedirs(out_dir, exist_ok=True)
    for i in range(0, len(keys), batch_size):
        chunk = keys[i : i + batch_size]
        vecs = []
        for uid, nid in chunk:
            vecs.append(np.asarray(pair_embeds[(uid, nid)], dtype=np.float32).reshape(-1))
        emb = torch.from_numpy(np.stack(vecs, axis=0)).to(device=device, dtype=dtype)
        neg = negative.to(device=device, dtype=dtype)
        if neg.shape[0] == 1 and emb.shape[0] > 1:
            neg = neg.expand(emb.shape[0], -1)
        elif neg.shape[0] != emb.shape[0]:
            neg = neg[:1].expand(emb.shape[0], -1)
        gens = [
            torch.Generator(device="cpu").manual_seed(_pair_seed(uid, nid, seed))
            for uid, nid in chunk
        ]
        out = pipe(
            image_embeds=emb,
            negative_image_embeds=neg,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=gens,
        )
        for (uid, nid), img in zip(chunk, out.images):
            path = b3_image_path(out_dir, uid, nid)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)
        n_done += len(chunk)
        n_batches += 1
        if n_batches == 1 or n_batches % 10 == 0 or n_done >= len(keys):
            print(f"[CLIP B3] generated {n_done}/{len(keys)}", flush=True)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    meta = {
        "route": "B3",
        "decoder_id": decoder_id,
        "prior_id": prior_id,
        "out_dir": os.path.abspath(out_dir),
        "height": int(height),
        "width": int(width),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "negative_mode": negative_mode,
        "n_prior_pairs": int(len(pair_embeds)),
        "n_existing": int(n_existing),
        "n_generated_this_run": int(n_done),
        "n_todo": int(len(keys)),
    }
    meta_path = os.path.join(out_dir, "generate_b3_log.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[CLIP B3] saved images → {out_dir}\n[CLIP B3] meta {meta_path}", flush=True)
    return meta


def list_b3_image_pairs(image_dir: str) -> List[Tuple[str, str, str]]:
    """(uid, nid, path) from user_<uid>/news_<nid>.png|jpg."""
    items: List[Tuple[str, str, str]] = []
    if not image_dir or not os.path.isdir(image_dir):
        return items
    for user_folder in os.listdir(image_dir):
        user_path = os.path.join(image_dir, user_folder)
        if not os.path.isdir(user_path) or not user_folder.startswith("user_"):
            continue
        uid_raw = user_folder[len("user_") :]
        for filename in os.listdir(user_path):
            lower = filename.lower()
            if not (filename.startswith("news_") and lower.endswith((".png", ".jpg", ".jpeg"))):
                continue
            stem = filename[len("news_") :]
            nid_raw = os.path.splitext(stem)[0]
            uid, nid = norm_pair_key(uid_raw, nid_raw)
            if not uid or not nid:
                continue
            items.append((uid, nid, os.path.join(user_path, filename)))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def extract_b3_pixel_clip_embeddings(
    image_dir: str,
    out_path: str,
    *,
    model_id: str = CLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 16,
    resume: bool = True,
) -> dict:
    """
    Route P 후단: B3 PNG → CLIP image encoder → image_embeds.
    L2 정규화 없음. pair cache 저장.
    """
    try:
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    except ImportError as e:
        raise ImportError(
            "B3 CLIP 추출에는 torch, transformers, Pillow 가 필요합니다. "
            "pip install -r CLIP/requirements.txt"
        ) from e

    items = list_b3_image_pairs(image_dir)
    if not items:
        raise FileNotFoundError(
            f"B3 생성 이미지가 없습니다: {image_dir}\n"
            f"먼저 python CLIP/generate_b3_images.py 로 PNG를 만드세요. "
            f"예: {os.path.join(image_dir, 'user_1', 'news_N1.png')}"
        )

    device = resolve_torch_device(device)
    if device == "cpu" and batch_size > 8:
        print(f"[CLIP B3 emb] CPU라 batch-size를 {batch_size} → 8 로 낮춥니다.", flush=True)
        batch_size = 8
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    existing = _load_existing_pair_cache(out_path) if resume else {}
    todo: List[Tuple[str, str, str]] = []
    seen = set()
    for uid, nid, path in items:
        key = (uid, nid)
        if key in seen or key in existing:
            continue
        seen.add(key)
        todo.append((uid, nid, path))

    print(
        f"[CLIP B3 emb] load image_encoder from {model_id}/{CLIP_IMAGE_ENCODER_SUBFOLDER} device={device}\n"
        f"[CLIP B3 emb] image_dir={image_dir} png_pairs={len(items)} "
        f"todo={len(todo)} resume_cached={len(existing)}",
        flush=True,
    )
    if not todo and existing:
        clip_dim = int(next(iter(existing.values())).reshape(-1).shape[0])
        print(f"[CLIP B3 emb] 새로 추출할 pair가 없어 cache를 유지합니다: {out_path}", flush=True)
        return {
            "route": "B3",
            "encoder": "CLIPVisionModelWithProjection.image_embeds",
            "image_dir": os.path.abspath(image_dir),
            "out_path": os.path.abspath(out_path),
            "clip_dim": clip_dim,
            "n_pairs": int(len(existing)),
            "n_encoded_this_run": 0,
            "space": "clip_image",
        }

    try:
        processor = CLIPImageProcessor.from_pretrained(
            model_id, subfolder=CLIP_IMAGE_PROCESSOR_SUBFOLDER
        )
    except Exception:
        processor = CLIPImageProcessor.from_pretrained(model_id)
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder=CLIP_IMAGE_ENCODER_SUBFOLDER
    )
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    fallback_dim = int(getattr(encoder.config, "projection_dim", 1280) or 1280)

    def _open_rgb(path: str):
        img = Image.open(path)
        img.load()
        return img.convert("RGB")

    clip_dim: Optional[int] = None
    if existing:
        clip_dim = int(next(iter(existing.values())).reshape(-1).shape[0])
    n_done = 0
    n_batches = 0
    n_unreadable = 0
    with torch.no_grad():
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            images = []
            ok_keys = []
            for uid, nid, path in chunk:
                try:
                    images.append(_open_rgb(path))
                    ok_keys.append((uid, nid))
                except Exception:
                    n_unreadable += 1
            if not images:
                continue
            pixel = processor(images=images, return_tensors="pt").pixel_values
            pixel = pixel.to(device=device, dtype=dtype)
            out = encoder(pixel_values=pixel)
            vecs = out.image_embeds.float().cpu().numpy().astype(np.float32)
            if clip_dim is None:
                clip_dim = int(vecs.shape[1])
            elif int(vecs.shape[1]) != int(clip_dim):
                raise ValueError(f"B3 dim mismatch: got {vecs.shape[1]} expected {clip_dim}")
            existing = _merge_and_save_pairs(out_path, existing, ok_keys, vecs, clip_dim)
            n_done += len(ok_keys)
            n_batches += 1
            if n_batches == 1 or n_batches % 20 == 0 or n_done >= len(todo):
                print(f"[CLIP B3 emb] encoded {n_done}/{len(todo)}", flush=True)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    if clip_dim is None:
        clip_dim = fallback_dim
        if existing:
            clip_dim = int(next(iter(existing.values())).reshape(-1).shape[0])
        _merge_and_save_pairs(out_path, existing, [], [], clip_dim)

    meta = {
        "route": "B3",
        "model_id": model_id,
        "encoder": "CLIPVisionModelWithProjection.image_embeds",
        "image_dir": os.path.abspath(image_dir),
        "l2_normalize": False,
        "out_path": os.path.abspath(out_path),
        "clip_dim": int(clip_dim),
        "n_png": int(len(items)),
        "n_pairs": int(len(existing)),
        "n_encoded_this_run": int(n_done),
        "n_unreadable": int(n_unreadable),
        "space": "clip_image",
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[CLIP B3 emb] saved {out_path}\n"
        f"[CLIP B3 emb] dim={clip_dim} pairs={len(existing)} encoded_this_run={n_done} "
        f"unreadable={n_unreadable}\n"
        f"[CLIP B3 emb] meta {meta_path}",
        flush=True,
    )
    return meta
