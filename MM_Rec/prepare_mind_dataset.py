#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset/<MIND_2000> TSV → MM-Rec behaviors 형식.

  python MM_Rec/prepare_mind_dataset.py --mind-dataset-subdir MIND_2000

출력 (MM_Rec/data/MIND_2000/):
  subnews.tsv
  MIND_2000/train/train_0.tsv
  MIND_2000/dev/test_0.tsv     # MIND_test_(2000).tsv (val)
  MIND_2000/test/test_0.tsv    # MIND_test_2000_final.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dataset_paths import (
    dataset_raw_dir,
    discover_tsv_names,
    prepared_dir,
)

# impression_id, user, time, history, unused, unused, positives, negatives
BEH_COLS = 8


def _is_header(first_cell: str) -> bool:
    return first_cell.strip().lower() in {
        "user",
        "userid",
        "user_id",
        "news_id",
        "clicked_news",
        "id",
    }


def write_news_tsv(src: Path, dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if i == 0 and _is_header(parts[0]):
                continue
            if len(parts) < 4:
                continue
            news_id = parts[0]
            category = parts[1] if len(parts) > 1 else ""
            subcategory = parts[2] if len(parts) > 2 else ""
            title = parts[3]
            fout.write("\t".join([news_id, category, subcategory, title]) + "\n")
            n += 1
    return n


def parse_behavior_rows(
    src: Path,
    has_header: bool,
    first_is_positive: bool,
) -> Iterable[Tuple[str, List[str], List[str], List[str]]]:
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if idx == 0 and (has_header or _is_header(parts[0])):
                continue
            if len(parts) < 3:
                continue
            user_id = parts[0]
            history = parts[1].split() if parts[1].strip() else []
            candidates = parts[2].split() if parts[2].strip() else []
            if not candidates:
                continue
            if len(parts) >= 4 and parts[3].strip() and not first_is_positive:
                labels = parts[3].split()
                if len(labels) != len(candidates):
                    continue
            elif first_is_positive:
                labels = ["1" if i == 0 else "0" for i in range(len(candidates))]
            else:
                labels = ["1" if i == 0 else "0" for i in range(len(candidates))]
            yield user_id, history, candidates, labels


def write_behaviors(
    rows: Iterable[Tuple[str, List[str], List[str], List[str]]],
    dst: Path,
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dst.open("w", encoding="utf-8") as f:
        for impression_id, (user_id, history, candidates, labels) in enumerate(rows, start=1):
            poss = [c for c, lab in zip(candidates, labels) if str(lab) == "1"]
            neg = [c for c, lab in zip(candidates, labels) if str(lab) != "1"]
            if not poss:
                continue
            if not neg:
                neg = ["0"]
            cols = [
                str(impression_id),
                str(user_id),
                "0",
                " ".join(history),
                "",
                "",
                " ".join(poss),
                " ".join(neg),
            ]
            assert len(cols) == BEH_COLS
            f.write("\t".join(cols) + "\n")
            n += 1
    return n


def prepare(mind_dataset_subdir: str, raw_dir: Optional[str] = None) -> Path:
    raw = Path(raw_dir) if raw_dir else dataset_raw_dir(mind_dataset_subdir)
    if not raw.is_dir():
        raise FileNotFoundError(f"원본 데이터 폴더 없음: {raw}")
    news_name, train_name, val_name, test_final_name = discover_tsv_names(mind_dataset_subdir)
    out = prepared_dir(mind_dataset_subdir)
    out.mkdir(parents=True, exist_ok=True)

    news_src = raw / news_name
    if not news_src.is_file():
        raise FileNotFoundError(f"뉴스 TSV 없음: {news_src}")
    n_news = write_news_tsv(news_src, out / "subnews.tsv")
    print(f"[prepare] news {n_news} → {out / 'subnews.tsv'}", flush=True)

    dataset_name = mind_dataset_subdir
    train_src = raw / train_name
    n_train = write_behaviors(
        parse_behavior_rows(train_src, has_header=True, first_is_positive=False),
        out / dataset_name / "train" / "train_0.tsv",
    )
    print(f"[prepare] train {n_train} → {out / dataset_name / 'train' / 'train_0.tsv'}", flush=True)

    val_src = raw / val_name
    n_val = write_behaviors(
        parse_behavior_rows(val_src, has_header=False, first_is_positive=True),
        out / dataset_name / "dev" / "test_0.tsv",
    )
    print(f"[prepare] val(dev) {n_val} → {out / dataset_name / 'dev' / 'test_0.tsv'}", flush=True)

    if test_final_name and (raw / test_final_name).is_file():
        n_test = write_behaviors(
            parse_behavior_rows(raw / test_final_name, has_header=False, first_is_positive=True),
            out / dataset_name / "test" / "test_0.tsv",
        )
        print(f"[prepare] test {n_test} → {out / dataset_name / 'test' / 'test_0.tsv'}", flush=True)
    else:
        print("[prepare] test_final TSV 없음 → val을 test로 복사", flush=True)
        import shutil
        test_dst = out / dataset_name / "test" / "test_0.tsv"
        test_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / dataset_name / "dev" / "test_0.tsv", test_dst)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="MIND_2000 → MM-Rec 데이터 변환")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--raw-dir", type=str, default=None)
    args = ap.parse_args()
    prepare(args.mind_dataset_subdir, args.raw_dir)


if __name__ == "__main__":
    main()
