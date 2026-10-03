#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MM-Rec + CLIP (title text embed + thumbnail image embed).

뉴스 인코더만 CLIP으로 교체. user_encoder / (t+v)·user 점수는 원본 MM-Rec 동일.

  conda activate clip_cu128
  python MM_Rec/extract_clip_features.py --mind-dataset-subdir MIND_2000 --reuse-image-cache
  python MM_Rec/train_eval_clip.py --mind-dataset-subdir MIND_2000

  python MM_Rec/tune_clip.py --mind-dataset-subdir MIND_2000 --two-phase \\
      --trials 24 --screening-epochs 3 --refine-top-k 5 --epochs-per-trial 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MMREC = Path(__file__).resolve().parent
if str(_MMREC) not in sys.path:
    sys.path.insert(0, str(_MMREC))

from dataset_paths import DEFAULT_THUMBNAIL_DIR, default_config_file, prepared_dir, saved_dir
from extract_clip_features import clip_image_cache_path, clip_title_cache_path
from parameters import parse_args
from prepare_mind_dataset import prepare
from run_clip import test as mmrec_clip_test
from run_clip import train as mmrec_clip_train
import utils


def _build_run_args(
    *,
    mode: str,
    mind_dataset_subdir: str,
    data_root: Path,
    save_root: Path,
    eval_split: str,
    epochs: int,
    batch_size: int,
    lr: float,
    npratio: int,
    load_ckpt: str | None,
    debug: bool,
    enable_gpu: bool,
    selection_metric: str = "MRR",
    clip_hidden_size: int = 1024,
    clip_dropout: float = 0.1,
) -> argparse.Namespace:
    split_dir = "test" if eval_split == "test" else "dev"
    argv = [
        "run_clip.py",
        "--mode",
        mode,
        "--root_data_dir",
        str(data_root),
        "--dataset",
        mind_dataset_subdir,
        "--news_file",
        "subnews.tsv",
        "--train_dir",
        "train",
        "--test_dir",
        split_dir,
        "--valid_dir",
        "dev",
        "--selection_metric",
        selection_metric,
        "--filename_pat",
        "train_*.tsv",
        "--model_dir",
        str(save_root / "ckpts"),
        "--log_dir",
        str(save_root / "logs"),
        "--config_file",
        str(default_config_file()),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--npratio",
        str(npratio),
        "--enable_hvd",
        "False",
        "--enable_prefetch",
        "False",
        "--enable_gpu",
        "True" if enable_gpu else "False",
        "--hvd_size",
        "1",
        "--news_attributes",
        "title",
        "--exp_name",
        f"mmrec_clip_{mind_dataset_subdir}",
        "--debug",
        "True" if debug else "False",
        "--num_workers",
        "2",
        # ROI paths unused but parse_args requires defaults
        "--roi_npz_file",
        str(data_root / "rois.npz"),
        "--roi_file",
        str(data_root / "rois.npz"),
        "--whole_file",
        str(data_root / "whole_features.tsv"),
        "--image_size_file",
        str(data_root / "image_size.tsv"),
    ]
    if load_ckpt:
        argv.extend(["--load_ckpt_name", load_ckpt])
    old = sys.argv
    sys.argv = argv
    try:
        args = parse_args()
    finally:
        sys.argv = old
    args.clip_image_cache = str(clip_image_cache_path(mind_dataset_subdir))
    args.clip_title_cache = str(clip_title_cache_path(mind_dataset_subdir))
    args.clip_hidden_size = int(clip_hidden_size)
    args.clip_dropout = float(clip_dropout)
    return args


def main() -> None:
    ap = argparse.ArgumentParser(description="MM-Rec + CLIP train/eval")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["prepare", "extract-clip", "train", "test", "all"],
    )
    ap.add_argument("--eval-split", type=str, default="test", choices=["test", "dev"])
    ap.add_argument("--selection-metric", type=str, default="MRR", choices=["MRR", "NDCG@5", "nDCG@5"])
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--npratio", type=int, default=4)
    ap.add_argument("--clip-hidden-size", type=int, default=1024)
    ap.add_argument("--clip-dropout", type=float, default=0.1)
    ap.add_argument("--reuse-image-cache", action="store_true", help="CLIP/cache 썸네일 npz 재사용")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--load-ckpt", type=str, default="best.pt")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch

    data_root = prepared_dir(args.mind_dataset_subdir)
    # CLIP 변형은 원본과 분리된 저장 경로
    save_root = saved_dir(args.mind_dataset_subdir) / "clip"
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "ckpts").mkdir(parents=True, exist_ok=True)
    (save_root / "logs").mkdir(parents=True, exist_ok=True)

    enable_gpu = (not args.cpu) and torch.cuda.is_available()
    print(
        f"[mmrec-clip] subdir={args.mind_dataset_subdir} stage={args.stage} "
        f"gpu={enable_gpu} data={data_root} save={save_root}",
        flush=True,
    )

    if args.stage in ("prepare", "all"):
        prepare(args.mind_dataset_subdir)

    img_cache = clip_image_cache_path(args.mind_dataset_subdir)
    title_cache = clip_title_cache_path(args.mind_dataset_subdir)
    if args.stage in ("extract-clip", "all"):
        need = args.force_extract or (not img_cache.is_file()) or (not title_cache.is_file())
        if need:
            from extract_clip_features import main as extract_main

            argv = [
                "extract_clip_features.py",
                "--mind-dataset-subdir",
                args.mind_dataset_subdir,
                "--thumbnail-dir",
                args.thumbnail_dir,
                "--device",
                "cpu" if args.cpu else "auto",
            ]
            if args.reuse_image_cache:
                argv.append("--reuse-image-cache")
            old = sys.argv
            sys.argv = argv
            try:
                extract_main()
            finally:
                sys.argv = old
        else:
            print(f"[mmrec-clip] CLIP cache ok: {img_cache.name}, {title_cache.name}", flush=True)

    if args.stage in ("train", "all"):
        if not img_cache.is_file() or not title_cache.is_file():
            raise FileNotFoundError(
                f"CLIP cache 필요: {img_cache} / {title_cache}\n"
                "python MM_Rec/extract_clip_features.py --mind-dataset-subdir "
                f"{args.mind_dataset_subdir} --reuse-image-cache"
            )
        run_args = _build_run_args(
            mode="train",
            mind_dataset_subdir=args.mind_dataset_subdir,
            data_root=data_root,
            save_root=save_root,
            eval_split=args.eval_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            npratio=args.npratio,
            load_ckpt=None,
            debug=args.debug,
            enable_gpu=enable_gpu,
            selection_metric=args.selection_metric,
            clip_hidden_size=args.clip_hidden_size,
            clip_dropout=args.clip_dropout,
        )
        utils.setuplogger(os.path.join(run_args.log_dir, "log_train.txt"))
        summary = mmrec_clip_train(run_args)
        if summary:
            print(
                f"[mmrec-clip] best val epoch={summary.get('best_epoch')}  "
                f"MRR={(summary.get('best_metrics') or {}).get('MRR')}  "
                f"NDCG@5={(summary.get('best_metrics') or {}).get('NDCG@5')}  "
                f"Hit@1={(summary.get('best_metrics') or {}).get('Hit@1')}",
                flush=True,
            )

    if args.stage in ("test", "all"):
        ckpt_name = "best.pt" if args.stage == "all" else args.load_ckpt
        eval_split = "test" if args.stage == "all" else args.eval_split
        run_args = _build_run_args(
            mode="test",
            mind_dataset_subdir=args.mind_dataset_subdir,
            data_root=data_root,
            save_root=save_root,
            eval_split=eval_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            npratio=args.npratio,
            load_ckpt=ckpt_name,
            debug=args.debug,
            enable_gpu=enable_gpu,
            selection_metric=args.selection_metric,
            clip_hidden_size=args.clip_hidden_size,
            clip_dropout=0.0,
        )
        utils.setuplogger(os.path.join(run_args.log_dir, "log_test.txt"))
        mmrec_clip_test(run_args)
        result_path = Path(run_args.log_dir) / "final_result.txt"
        if result_path.is_file():
            print(f"[mmrec-clip] metrics → {result_path}", flush=True)
            print(result_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
