#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MIND 뉴스 → IMRec visual impression 카드(610x195) 합성."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from paths import (
    DEFAULT_THUMBNAIL_DIR,
    DATASET_FILE_PRESETS,
    cards_dir,
    dataset_raw_dir,
    prepared_dir,
    resolve_project_path,
    thumbnail_path,
)

CARD_W, CARD_H = 610, 195
IMG_W = 210
TITLE_XY = (227, 15)
CAT_XY = (227, 142)
MAX_LINE = 27


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_title(text: str, max_len: int = MAX_LINE, max_rows: int = 3) -> str:
    text = (text or "").strip() + " "
    ans, tmp, numcnt, row, prelen = "", "", 0, 1, 0
    for ch in text:
        numcnt += 1
        if ch != " ":
            tmp += ch
            continue
        if numcnt - 1 > max_len:
            if row == max_rows:
                if numcnt + 3 > max_len:
                    ans = ans[: len(ans) - 1 - prelen] + "..."
                else:
                    ans += "..."
                break
            row += 1
            tmp = tmp.strip() + " "
            ans += "\n" + tmp
            numcnt = len(tmp)
            prelen = len(tmp)
            tmp = ""
        else:
            tmp += " "
            ans += tmp
            prelen = len(tmp)
            tmp = ""
    return ans.strip()


def _fmt_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    if c == "foodanddrink":
        return "food and drink"
    return c.replace("_", " ")


def render_card(
    title: str,
    category: str,
    thumb: Optional[Path],
    title_font: ImageFont.ImageFont,
    cat_font: ImageFont.ImageFont,
) -> Image.Image:
    base = Image.new("RGB", (CARD_W, CARD_H), (255, 255, 255))
    if thumb is not None and thumb.is_file():
        try:
            im = Image.open(thumb).convert("RGB")
            im = im.resize((IMG_W, CARD_H), Image.Resampling.BILINEAR)
            base.paste(im, (0, 0))
        except Exception:
            pass
    else:
        # placeholder gray panel
        panel = Image.new("RGB", (IMG_W, CARD_H), (230, 230, 230))
        base.paste(panel, (0, 0))

    draw = ImageDraw.Draw(base)
    draw.text(TITLE_XY, wrap_title(title), font=title_font, fill="#2b2b2b", spacing=10)
    draw.text(CAT_XY, _fmt_category(category), font=cat_font, fill="#666666")
    return base


def load_news_tsv(path: Path) -> Dict[str, Tuple[str, str]]:
    """news_id -> (category, title)"""
    out: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            nid = parts[0].strip()
            if nid.lower() in ("news_id", "id"):
                continue
            out[nid] = (parts[1], parts[3])
    return out


def render_all(
    mind_dataset_subdir: str,
    thumbnail_dir: str | Path,
    force: bool = False,
    max_news: int = 0,
) -> Path:
    raw = dataset_raw_dir(mind_dataset_subdir)
    news_name = DATASET_FILE_PRESETS.get(mind_dataset_subdir, (None,))[0] or "MIND_news.tsv"
    news_path = raw / news_name
    if not news_path.is_file():
        raise FileNotFoundError(news_path)

    out_dir = cards_dir(mind_dataset_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir(mind_dataset_subdir).mkdir(parents=True, exist_ok=True)

    news = load_news_tsv(news_path)
    thumb_root = resolve_project_path(str(thumbnail_dir))
    if not thumb_root.is_dir():
        raise FileNotFoundError(
            f"썸네일 폴더 없음: {thumb_root}  (서버의 dataset/MIND_thumbnail 을 확인하세요)"
        )
    title_font = _load_font(27, bold=True)
    cat_font = _load_font(24, bold=False)

    n_ok, n_miss, n_skip = 0, 0, 0
    items = list(news.items())
    if max_news > 0:
        items = items[:max_news]

    for i, (nid, (cat, title)) in enumerate(items, 1):
        dst = out_dir / f"{nid}.jpg"
        if dst.is_file() and not force:
            n_skip += 1
            continue
        tp = thumbnail_path(thumb_root, nid)
        if tp is None:
            n_miss += 1
        card = render_card(title, cat, tp, title_font, cat_font)
        card.save(dst, quality=92)
        n_ok += 1
        if i % 2000 == 0:
            print(f"[cards] {i}/{len(items)} written={n_ok} miss_thumb={n_miss} skip={n_skip}", flush=True)

    print(
        f"[cards] done → {out_dir}  new={n_ok} skip={n_skip} missing_thumbnail={n_miss}",
        flush=True,
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind-dataset-subdir", default="MIND_2000")
    ap.add_argument("--thumbnail-dir", default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-news", type=int, default=0)
    args = ap.parse_args()
    render_all(args.mind_dataset_subdir, args.thumbnail_dir, args.force, args.max_news)


if __name__ == "__main__":
    main()
