#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kandinsky 2.2 prior의 CLIP image encoder로 썸네일 임베딩을 추출한다.

  python CLIP/clip_embeddings.py \
    --mind-dataset-subdir MIND_2000 \
    --thumbnail-dir dataset/MIND_thumbnail \
    --out CLIP/cache/MIND_2000_clip_image_embeds.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

CLIP_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
CLIP_IMAGE_ENCODER_SUBFOLDER = "image_encoder"
CLIP_IMAGE_PROCESSOR_SUBFOLDER = "image_processor"
DEFAULT_THUMBNAIL_DIR = "dataset/MIND_thumbnail"
DEFAULT_GENERATED_IMAGE_DIR = "dataset/MIND_image"
DEFAULT_CACHE_NAME = "{subdir}_clip_image_embeds.npz"
DEFAULT_B4_CACHE_NAME = "{subdir}_clip_b4_mind_image.npz"
DEFAULT_B1_CACHE_NAME = "{subdir}_clip_b1_text_expected.npz"
DEFAULT_B2_CACHE_NAME = "{subdir}_clip_b2_prior_expected.npz"

_HEADER_IDS = frozenset({"news_id", "clicked_news", "id"})


def resolve_project_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    return os.path.normpath(p) if os.path.isabs(p) else os.path.normpath(str(_ROOT / p))


def default_cache_path(mind_dataset_subdir: str) -> str:
    name = DEFAULT_CACHE_NAME.format(subdir=mind_dataset_subdir)
    return str(_ROOT / "CLIP" / "cache" / name)


def thumbnail_path(thumbnail_dir: str, news_id: str) -> str:
    return os.path.join(thumbnail_dir, f"{news_id}.jpg")


def resolve_news_image_path(
    image_dir: str,
    news_id: str,
    suffixes: Sequence[str] = (".jpg",),
) -> Optional[str]:
    for sfx in suffixes:
        path = os.path.join(image_dir, f"{news_id}{sfx}")
        if os.path.isfile(path):
            return path
    return None


def default_b4_cache_path(mind_dataset_subdir: str) -> str:
    return str(_ROOT / "CLIP" / "cache" / DEFAULT_B4_CACHE_NAME.format(subdir=mind_dataset_subdir))


def default_b1_cache_path(mind_dataset_subdir: str) -> str:
    return str(_ROOT / "CLIP" / "cache" / DEFAULT_B1_CACHE_NAME.format(subdir=mind_dataset_subdir))


def default_b2_cache_path(mind_dataset_subdir: str) -> str:
    return str(_ROOT / "CLIP" / "cache" / DEFAULT_B2_CACHE_NAME.format(subdir=mind_dataset_subdir))


def load_news_ids_from_tsv(news_tsv: str) -> List[str]:
    ids: List[str] = []
    seen = set()
    with open(news_tsv, "r", encoding="utf-8") as f:
        for line in f:
            nid = line.split("\t", 1)[0].strip()
            if not nid or nid.lower() in _HEADER_IDS:
                continue
            if nid in seen:
                continue
            seen.add(nid)
            ids.append(nid)
    return ids


def count_missing_images(
    news_ids: Iterable[str],
    image_dir: str,
    suffixes: Sequence[str] = (".jpg",),
) -> Tuple[List[str], List[str]]:
    catalog: List[str] = []
    missing: List[str] = []
    for nid in news_ids:
        if not nid or nid == "0":
            continue
        catalog.append(nid)
        if resolve_news_image_path(image_dir, nid, suffixes) is None:
            missing.append(nid)
    return catalog, missing


def count_missing_thumbnails(
    news_ids: Iterable[str],
    thumbnail_dir: str,
) -> Tuple[List[str], List[str]]:
    return count_missing_images(news_ids, thumbnail_dir, suffixes=(".jpg",))


def print_missing_image_report(
    news_ids: Sequence[str],
    image_dir: str,
    *,
    suffixes: Sequence[str] = (".jpg",),
    label: str = "image",
    sample: int = 20,
) -> List[str]:
    catalog, missing = count_missing_images(news_ids, image_dir, suffixes=suffixes)
    print(
        f"[CLIP] {label}_dir={image_dir} suffixes={list(suffixes)}\n"
        f"[CLIP] news catalog (padding 제외)={len(catalog)}\n"
        f"[CLIP] missing {label}={len(missing)} / {len(catalog)}",
        flush=True,
    )
    if missing:
        shown = missing[:sample]
        extra = "" if len(missing) <= sample else f" ... (+{len(missing) - sample})"
        print(f"[CLIP] missing sample: {shown}{extra}", flush=True)
    return missing


def print_missing_thumbnail_report(
    news_ids: Sequence[str],
    thumbnail_dir: str,
    *,
    sample: int = 20,
) -> List[str]:
    return print_missing_image_report(
        news_ids,
        thumbnail_dir,
        suffixes=(".jpg",),
        label="thumbnail",
        sample=sample,
    )


def _news_ids_from_index(news_index: Dict[str, int]) -> List[str]:
    rows = [(idx, nid) for nid, idx in news_index.items() if nid != "0" and int(idx) != 0]
    rows.sort()
    return [nid for _idx, nid in rows]


def _news_ids_sidecar_path(npz_path: str) -> str:
    return os.path.splitext(npz_path)[0] + "_news_ids.json"


def _save_clip_cache(out_path: str, embeddings: np.ndarray, catalog: Sequence[str]) -> None:
    """numpy 1.x / 2.x 모두 읽히도록 object(pickle) 배열을 쓰지 않는다."""
    catalog_list = [str(x) for x in catalog]
    max_len = max((len(x) for x in catalog_list), default=8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        news_ids=np.asarray(catalog_list, dtype=f"U{max(max_len, 8)}"),
    )
    with open(_news_ids_sidecar_path(out_path), "w", encoding="utf-8") as f:
        json.dump(catalog_list, f, ensure_ascii=False)


def _read_npz_npy(path: str, key: str, allow_pickle: bool = False) -> np.ndarray:
    from numpy.lib.format import read_array

    with zipfile.ZipFile(path, "r") as z:
        name = key if key.endswith(".npy") else f"{key}.npy"
        with z.open(name) as f:
            return read_array(f, allow_pickle=allow_pickle)


def load_clip_npz(
    path: str,
    news_ids_fallback: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    emb = np.asarray(_read_npz_npy(path, "embeddings", allow_pickle=False), dtype=np.float32)

    news_ids: Optional[List[str]] = None
    sidecar = _news_ids_sidecar_path(path)
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list) and loaded:
            news_ids = [str(x) for x in loaded]

    if news_ids is None:
        try:
            raw_ids = _read_npz_npy(path, "news_ids", allow_pickle=False)
            news_ids = [str(x) for x in np.asarray(raw_ids).reshape(-1).tolist()]
        except Exception:
            news_ids = None

    if news_ids is None:
        try:
            raw_ids = _read_npz_npy(path, "news_ids", allow_pickle=True)
            news_ids = [str(x) for x in np.asarray(raw_ids).reshape(-1).tolist()]
        except Exception:
            news_ids = None

    if news_ids is None and news_ids_fallback is not None:
        fb = [str(x) for x in news_ids_fallback]
        if len(fb) == emb.shape[0]:
            print(
                "[CLIP] npz의 news_ids가 다른 numpy 버전으로 pickle되어 읽을 수 없습니다. "
                "뉴스 TSV 순서로 ID를 복구합니다. 캐시를 numpy1 호환으로 다시 저장합니다.",
                flush=True,
            )
            news_ids = fb
            _save_clip_cache(path, emb, news_ids)
        else:
            raise ValueError(
                f"CLIP cache news_ids를 읽을 수 없고 fallback 길이도 다릅니다 "
                f"(emb={emb.shape[0]}, fallback={len(fb)}). "
                f"CLIP/clip_embeddings.py 로 캐시를 다시 추출하세요: {path}"
            )

    if news_ids is None:
        raise ValueError(
            f"CLIP cache news_ids를 읽을 수 없습니다 (numpy 버전 불일치 가능). "
            f"tf28gpu가 아닌 추출 환경에서 다시 저장하거나, 이 코드로 재추출하세요: {path}"
        )

    if emb.ndim != 2 or len(news_ids) != emb.shape[0]:
        raise ValueError(
            f"CLIP cache shape mismatch: embeddings={emb.shape} news_ids={len(news_ids)} ({path})"
        )
    return emb, news_ids


def build_news_image_matrix(
    news_index: Dict[str, int],
    n_rows: int,
    cache_path: str,
    news_ids_fallback: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, int]:
    """
    news_words 와 같은 행 인덱스의 이미지 임베딩 행렬.
    캐시에 없거나 추출 당시 파일이 없었던 ID는 0벡터.
    Returns: (news_image, n_nonzero)
    """
    emb, news_ids = load_clip_npz(cache_path, news_ids_fallback=news_ids_fallback)
    dim = int(emb.shape[1])
    id_to_row = {nid: i for i, nid in enumerate(news_ids)}
    news_image = np.zeros((int(n_rows), dim), dtype=np.float32)
    n_hit = 0
    for nid, idx in news_index.items():
        i = int(idx)
        if i <= 0 or i >= n_rows:
            continue
        row = id_to_row.get(str(nid))
        if row is None:
            continue
        vec = emb[row]
        news_image[i] = vec
        if np.any(vec):
            n_hit += 1
    return news_image, n_hit


def _pairs_sidecar_path(npz_path: str) -> str:
    return os.path.splitext(npz_path)[0] + "_pairs.json"


def _save_pair_cache(
    out_path: str,
    embeddings: np.ndarray,
    user_ids: Sequence[str],
    news_ids: Sequence[str],
) -> None:
    users = [str(x) for x in user_ids]
    news = [str(x) for x in news_ids]
    if len(users) != len(news) or len(users) != int(np.asarray(embeddings).shape[0]):
        raise ValueError(
            f"pair cache length mismatch: users={len(users)} news={len(news)} "
            f"emb={np.asarray(embeddings).shape}"
        )
    u_len = max((len(x) for x in users), default=8)
    n_len = max((len(x) for x in news), default=8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        user_ids=np.asarray(users, dtype=f"U{max(u_len, 8)}"),
        news_ids=np.asarray(news, dtype=f"U{max(n_len, 8)}"),
    )
    with open(_pairs_sidecar_path(out_path), "w", encoding="utf-8") as f:
        json.dump([[u, n] for u, n in zip(users, news)], f, ensure_ascii=False)


def load_pair_npz(
    path: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    emb = np.asarray(_read_npz_npy(path, "embeddings", allow_pickle=False), dtype=np.float32)
    users: Optional[List[str]] = None
    news: Optional[List[str]] = None
    sidecar = _pairs_sidecar_path(path)
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list) and loaded:
            users = [str(x[0]) for x in loaded]
            news = [str(x[1]) for x in loaded]

    def _try_key(key: str, allow_pickle: bool) -> Optional[List[str]]:
        try:
            raw = _read_npz_npy(path, key, allow_pickle=allow_pickle)
            return [str(x) for x in np.asarray(raw).reshape(-1).tolist()]
        except Exception:
            return None

    if users is None:
        users = _try_key("user_ids", False) or _try_key("user_ids", True)
    if news is None:
        news = _try_key("news_ids", False) or _try_key("news_ids", True)
    if users is None or news is None:
        raise ValueError(f"pair cache user/news ids를 읽을 수 없습니다: {path}")
    if emb.ndim != 2 or len(users) != emb.shape[0] or len(news) != emb.shape[0]:
        raise ValueError(
            f"pair cache shape mismatch: embeddings={emb.shape} users={len(users)} "
            f"news={len(news)} ({path})"
        )
    return emb, users, news


def load_pair_embed_dict(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    emb, users, news = load_pair_npz(path)
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for vec, uid, nid in zip(emb, users, news):
        out[(str(uid), str(nid))] = np.asarray(vec, dtype=np.float32)
    return out


def _chunks(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_torch_device(requested: str) -> str:
    """
    Blackwell(sm_120) 등 현재 PyTorch 휠이 지원하지 않는 GPU면 CUDA를 쓰지 않는다.
    torch.cuda.is_available() 만으로는 커널이 없어서 런타임에 터진다.
    """
    import torch

    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return "cpu"
    if req not in ("auto", "cuda"):
        return "cpu"
    if not torch.cuda.is_available():
        if req == "cuda":
            print("[CLIP] CUDA를 쓸 수 없어 CPU로 추출합니다.", flush=True)
        return "cpu"

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    archs = []
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            archs = [str(a) for a in torch.cuda.get_arch_list()]
        except Exception:
            archs = []
    supported = sm in archs
    if not supported:
        print(
            f"[CLIP] {name} ({sm}) 은 현재 PyTorch CUDA 빌드와 호환되지 않습니다.\n"
            f"[CLIP] 이 설치가 지원하는 arch: {archs or '(unknown)'}\n"
            f"[CLIP] CUDA 커널이 없어 GPU 추출을 건너뛰고 CPU로 진행합니다.\n"
            f"[CLIP] GPU로 추출하려면 sm_120 지원 PyTorch(CUDA 12.8+)를 별도 환경에 설치하세요.",
            flush=True,
        )
        return "cpu"
    try:
        x = torch.zeros(1, device="cuda")
        _ = x + 1
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[CLIP] CUDA 테스트 실패 ({type(e).__name__}: {e}) → CPU로 추출합니다.", flush=True)
        return "cpu"
    return "cuda"


def extract_clip_embeddings(
    news_ids: Sequence[str],
    thumbnail_dir: str,
    out_path: str,
    *,
    model_id: str = CLIP_MODEL_ID,
    device: str = "auto",
    batch_size: int = 16,
    suffixes: Sequence[str] = (".jpg",),
    source_label: str = "thumbnail",
) -> dict:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    except ImportError as e:
        raise ImportError(
            "CLIP 추출에는 torch, transformers, Pillow 가 필요합니다. "
            "pip install -r CLIP/requirements.txt"
        ) from e

    device = resolve_torch_device(device)
    if device == "cpu" and batch_size > 8:
        print(
            f"[CLIP] CPU 추출이라 batch-size를 {batch_size} → 8 로 낮춥니다 "
            f"(유지하려면 --batch-size 8 이하로 지정).",
            flush=True,
        )
        batch_size = 8
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(
        f"[CLIP] load image_encoder from {model_id}/{CLIP_IMAGE_ENCODER_SUBFOLDER} device={device}",
        flush=True,
    )
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

    catalog, missing = count_missing_images(news_ids, thumbnail_dir, suffixes=suffixes)
    missing_set = set(missing)
    to_encode = [nid for nid in catalog if nid not in missing_set]
    print(
        f"[CLIP] encode {len(to_encode)} {source_label} images, skip missing={len(missing)}",
        flush=True,
    )

    clip_dim: Optional[int] = None
    id_to_vec: Dict[str, np.ndarray] = {}
    unreadable: List[str] = []

    def _open_rgb(path: str):
        img = Image.open(path)
        img.load()
        return img.convert("RGB")

    n_done = 0
    n_batches = 0
    with torch.no_grad():
        for batch_ids in _chunks(to_encode, batch_size):
            images = []
            ok_ids = []
            for nid in batch_ids:
                path = resolve_news_image_path(thumbnail_dir, nid, suffixes)
                if not path:
                    unreadable.append(nid)
                    continue
                try:
                    images.append(_open_rgb(path))
                    ok_ids.append(nid)
                except Exception:
                    unreadable.append(nid)
            if not images:
                continue
            pixel = processor(images=images, return_tensors="pt").pixel_values
            pixel = pixel.to(device=device, dtype=dtype)
            out = encoder(pixel_values=pixel)
            vecs = out.image_embeds.float().cpu().numpy().astype(np.float32)
            if clip_dim is None:
                clip_dim = int(vecs.shape[1])
            for nid, vec in zip(ok_ids, vecs):
                id_to_vec[nid] = vec
            n_done += len(ok_ids)
            n_batches += 1
            if n_batches == 1 or n_batches % 20 == 0 or n_done >= len(to_encode):
                print(f"[CLIP] encoded {n_done}/{len(to_encode)}", flush=True)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    if clip_dim is None:
        clip_dim = fallback_dim
        print(
            f"[CLIP] 경고: 인코딩된 이미지가 없어 0벡터만 저장합니다 (dim={clip_dim})",
            flush=True,
        )

    embeddings = np.zeros((len(catalog), clip_dim), dtype=np.float32)
    for i, nid in enumerate(catalog):
        vec = id_to_vec.get(nid)
        if vec is not None:
            embeddings[i] = vec

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _save_clip_cache(out_path, embeddings, catalog)
    meta = {
        "model_id": model_id,
        "image_encoder_subfolder": CLIP_IMAGE_ENCODER_SUBFOLDER,
        "image_dir": thumbnail_dir,
        "suffixes": list(suffixes),
        "source_label": source_label,
        "out_path": os.path.abspath(out_path),
        "clip_dim": int(clip_dim),
        "n_news": len(catalog),
        "n_encoded": int(len(id_to_vec)),
        "n_missing": len(missing),
        "n_unreadable": len(unreadable),
        "missing_sample": missing[:50],
        "unreadable_sample": unreadable[:50],
    }
    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[CLIP] saved {out_path}\n"
        f"[CLIP] dim={clip_dim} encoded={len(id_to_vec)} "
        f"missing={len(missing)} unreadable={len(unreadable)}\n"
        f"[CLIP] meta {meta_path}",
        flush=True,
    )
    return meta


def main() -> None:
    from naml_dataset_env import apply_dataset_env_from_argv

    apply_dataset_env_from_argv()

    ap = argparse.ArgumentParser(description="Kandinsky 2.2 CLIP image encoder로 썸네일 임베딩 추출")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--thumbnail-dir", type=str, default=DEFAULT_THUMBNAIL_DIR)
    ap.add_argument("--news-tsv", type=str, default=None, help="미지정 시 dataset/<subdir>/MIND_news.tsv")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    apply_dataset_env_from_argv(["--mind-dataset-subdir", args.mind_dataset_subdir])
    from naml_common import MIND_NEWS_FILENAME, mind_data_path

    news_tsv = resolve_project_path(args.news_tsv) if args.news_tsv else mind_data_path(MIND_NEWS_FILENAME)
    thumb_dir = resolve_project_path(args.thumbnail_dir)
    out_path = resolve_project_path(args.out) if args.out else default_cache_path(args.mind_dataset_subdir)

    if not os.path.isfile(news_tsv):
        raise FileNotFoundError(f"news tsv 없음: {news_tsv}")
    if not os.path.isdir(thumb_dir):
        print(f"[CLIP] 경고: thumbnail 폴더 없음: {thumb_dir} (전부 0벡터로 저장)", flush=True)

    news_ids = load_news_ids_from_tsv(news_tsv)
    print_missing_thumbnail_report(news_ids, thumb_dir)
    extract_clip_embeddings(
        news_ids,
        thumb_dir,
        out_path,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
