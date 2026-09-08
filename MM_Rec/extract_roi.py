#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
썸네일에서 Mask R-CNN COCO 박스 + ResNet-50 ROI/전체 피처를 뽑아 MM-Rec용 npz를 만든다.

  conda activate clip_cu128
  python MM_Rec/extract_roi.py --mind-dataset-subdir MIND_2000 \\
      --thumbnail-dir dataset/MIND_thumbnail

출력: MM_Rec/data/<subdir>/image_size.tsv, rois.npz
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset_paths import (
    DEFAULT_THUMBNAIL_DIR,
    prepared_dir,
    resolve_project_path,
    thumbnail_path,
)


ROI_NUM = 30
FEAT_DIM = 2048
SCORE_THRESH = 0.2


def _load_detector(device):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    except Exception:
        model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model


def _load_resnet(device):
    import torch
    from torch import nn
    from torchvision import models
    from torchvision import transforms

    try:
        from torchvision.models import ResNet50_Weights

        res50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        res50 = models.resnet50(pretrained=True)
    res50 = nn.Sequential(*list(res50.children())[:-1])
    res50.eval()
    res50.to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return res50, preprocess


def _read_news_ids(news_tsv: Path) -> List[str]:
    ids: List[str] = []
    with news_tsv.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            if i == 0 and parts[0].strip().lower() in {"news_id", "id"}:
                continue
            ids.append(parts[0].strip())
    return ids


def _detect_rois(detector, image_np: np.ndarray, device, score_thresh: float):
    import torch
    from torchvision.transforms.functional import to_tensor
    from PIL import Image as PILImage

    h, w = image_np.shape[0], image_np.shape[1]
    tensor = to_tensor(PILImage.fromarray(image_np)).to(device)
    with torch.no_grad():
        out = detector([tensor])[0]
    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    cands = []
    for box, score in zip(boxes, scores):
        if float(score) < score_thresh:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        area = float((x2 - x1) * (y2 - y1))
        # Matterport 순서: y1, x1, y2, x2
        cands.append((area, [y1, x1, y2, x2]))
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[:ROI_NUM]


def extract_roi(
    mind_dataset_subdir: str,
    thumbnail_dir: str,
    news_tsv: Optional[str] = None,
    out_dir: Optional[str] = None,
    device: str = "auto",
    score_thresh: float = SCORE_THRESH,
    max_news: int = 0,
) -> Path:
    import torch

    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    out = Path(out_dir) if out_dir else prepared_dir(mind_dataset_subdir)
    out.mkdir(parents=True, exist_ok=True)
    news_path = Path(news_tsv) if news_tsv else out / "subnews.tsv"
    if not news_path.is_file():
        raise FileNotFoundError(
            f"뉴스 TSV 없음: {news_path}  (먼저 prepare_mind_dataset.py 를 실행하세요)"
        )
    thumb_dir = resolve_project_path(str(thumbnail_dir))
    if not thumb_dir.is_dir():
        raise FileNotFoundError(
            f"썸네일 폴더 없음: {thumb_dir}  (서버의 dataset/MIND_thumbnail 을 확인하세요)"
        )

    news_ids = _read_news_ids(news_path)
    if max_news > 0:
        news_ids = news_ids[:max_news]

    print(f"[roi] device={device_t} news={len(news_ids)} thumbnail_dir={thumb_dir}", flush=True)
    detector = _load_detector(device_t)
    res50, preprocess = _load_resnet(device_t)

    kept_ids: List[str] = []
    sizes: List[Tuple[int, int]] = []
    features_list = []
    location_list = []
    mask_list = []
    whole_list = []

    missing = 0
    with torch.no_grad():
        for nid in tqdm(news_ids, desc="extract_roi"):
            path = thumbnail_path(thumb_dir, nid)
            if path is None:
                missing += 1
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                missing += 1
                continue
            image_np = np.array(img)
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                missing += 1
                continue
            image_h, image_w = int(image_np.shape[0]), int(image_np.shape[1])
            image_area = float(image_w * image_h) or 1.0

            whole_t = preprocess(img).unsqueeze(0).to(device_t)
            whole_feat = res50(whole_t).squeeze(-1).squeeze(-1).cpu().numpy().reshape(1, FEAT_DIM)

            rois = _detect_rois(detector, image_np, device_t, score_thresh)
            feat = np.zeros((ROI_NUM, FEAT_DIM), dtype=np.float32)
            loc = np.zeros((ROI_NUM, 5), dtype=np.float32)
            mask = np.zeros((ROI_NUM,), dtype=np.int32)
            if rois:
                crops = []
                for roi_idx, (area, roi) in enumerate(rois):
                    loc[roi_idx, 0] = roi[0] / image_h
                    loc[roi_idx, 1] = roi[1] / image_w
                    loc[roi_idx, 2] = roi[2] / image_h
                    loc[roi_idx, 3] = roi[3] / image_w
                    loc[roi_idx, 4] = float(area) / image_area
                    crop = Image.fromarray(image_np[roi[0] : roi[2], roi[1] : roi[3], :]).convert("RGB")
                    crops.append(preprocess(crop))
                    mask[roi_idx] = 1
                stacked = torch.stack(crops, 0).to(device_t)
                roi_feat = res50(stacked).squeeze(-1).squeeze(-1).cpu().numpy()
                feat[: len(rois)] = roi_feat

            kept_ids.append(nid)
            sizes.append((image_w, image_h))
            features_list.append(feat)
            location_list.append(loc)
            mask_list.append(mask)
            whole_list.append(whole_feat)

    n = len(kept_ids)
    news_image_len = n + 1
    features = np.zeros((news_image_len, ROI_NUM, FEAT_DIM), dtype=np.float32)
    location = np.zeros((news_image_len, ROI_NUM, 5), dtype=np.float32)
    mask_arr = np.zeros((news_image_len, ROI_NUM), dtype=np.int32)
    whole = np.zeros((news_image_len, 1, FEAT_DIM), dtype=np.float32)
    for i in range(n):
        features[i + 1] = features_list[i]
        location[i + 1] = location_list[i]
        mask_arr[i + 1] = mask_list[i]
        whole[i + 1] = whole_list[i]

    size_path = out / "image_size.tsv"
    with size_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for nid, (w, h) in zip(kept_ids, sizes):
            writer.writerow([nid, w, h])

    npz_path = out / "rois.npz"
    np.savez(npz_path, features=features, location=location, mask=mask_arr, whole=whole)
    print(
        f"[roi] kept={n} missing_or_bad={missing} → {npz_path} / {size_path}",
        flush=True,
    )
    return npz_path


def main() -> None:
    ap = argparse.ArgumentParser(description="썸네일 → MM-Rec ROI npz")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--news-tsv", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--score-thresh", type=float, default=SCORE_THRESH)
    ap.add_argument("--max-news", type=int, default=0, help="디버그용 상한 (0=전체)")
    args = ap.parse_args()
    extract_roi(
        args.mind_dataset_subdir,
        args.thumbnail_dir,
        news_tsv=args.news_tsv,
        out_dir=args.out_dir,
        device=args.device,
        score_thresh=args.score_thresh,
        max_news=args.max_news,
    )


if __name__ == "__main__":
    main()
