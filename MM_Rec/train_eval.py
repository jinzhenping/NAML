#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIND_2000 (또는 Adressa_2000)으로 MM-Rec 학습/평가.

서버에서:

  conda activate clip_cu128
  python MM_Rec/train_eval.py --mind-dataset-subdir MIND_2000

에폭마다 val(MIND_test_(2000).tsv) MRR을 보고 최고 체크포인트(best.pt)를
MIND_test_2000_final.tsv 로 평가한다. 선택 지표를 nDCG@5로 바꾸려면:

  python MM_Rec/train_eval.py --selection-metric NDCG@5 --mind-dataset-subdir MIND_2000


단계만 따로:

  python MM_Rec/train_eval.py --stage prepare --mind-dataset-subdir MIND_2000
  python MM_Rec/train_eval.py --stage extract-roi --mind-dataset-subdir MIND_2000
  python MM_Rec/train_eval.py --stage train --mind-dataset-subdir MIND_2000
  python MM_Rec/train_eval.py --stage test --mind-dataset-subdir MIND_2000 --eval-split test

val(MIND_test_(2000).tsv) 평가:

  python MM_Rec/train_eval.py --stage test --eval-split dev --mind-dataset-subdir MIND_2000

ViLBERT 가중치가 있으면:

  python MM_Rec/train_eval.py --from-pretrained /path/to/pytorch_model_8.bin
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MMREC = Path(__file__).resolve().parent
if str(_MMREC) not in sys.path:
    sys.path.insert(0, str(_MMREC))

from dataset_paths import (
    DEFAULT_THUMBNAIL_DIR,
    default_config_file,
    prepared_dir,
    saved_dir,
)
from extract_roi import extract_roi
from prepare_mind_dataset import prepare
from parameters import parse_args
from run import test as mmrec_test
from run import train as mmrec_train
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
    from_pretrained: str,
    config_file: str,
    load_ckpt: str | None,
    debug: bool,
    enable_gpu: bool,
    selection_metric: str = "MRR",
) -> argparse.Namespace:
    split_dir = "test" if eval_split == "test" else "dev"
    argv = [
        "run.py",
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
        "--roi_npz_file",
        str(data_root / "rois.npz"),
        "--image_size_file",
        str(data_root / "image_size.tsv"),
        "--roi_file",
        str(data_root / "rois.npz"),
        "--whole_file",
        str(data_root / "whole_features.tsv"),
        "--model_dir",
        str(save_root / "ckpts"),
        "--log_dir",
        str(save_root / "logs"),
        "--config_file",
        config_file,
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
        f"mmrec_{mind_dataset_subdir}",
        "--debug",
        "True" if debug else "False",
        "--num_workers",
        "2",
    ]
    if from_pretrained:
        argv.extend(["--from_pretrained", from_pretrained])
    if load_ckpt:
        argv.extend(["--load_ckpt_name", load_ckpt])
    old = sys.argv
    sys.argv = argv
    try:
        args = parse_args()
    finally:
        sys.argv = old
    return args


def main() -> None:
    ap = argparse.ArgumentParser(description="MIND_2000 MM-Rec train/eval")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["prepare", "extract-roi", "train", "test", "all"],
    )
    ap.add_argument("--eval-split", type=str, default="test", choices=["test", "dev"])
    ap.add_argument(
        "--selection-metric",
        type=str,
        default="MRR",
        choices=["MRR", "NDCG@5", "nDCG@5"],
        help="val에서 best epoch를 고를 지표 (CLIP/NAML과 같이 기본 MRR)",
    )
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--from-pretrained", type=str, default="")
    ap.add_argument("--config-file", type=str, default=str(default_config_file()))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--npratio", type=int, default=4)
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--load-ckpt", type=str, default="best.pt")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch

    data_root = prepared_dir(args.mind_dataset_subdir)
    save_root = saved_dir(args.mind_dataset_subdir)
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "ckpts").mkdir(parents=True, exist_ok=True)
    (save_root / "logs").mkdir(parents=True, exist_ok=True)

    enable_gpu = (not args.cpu) and torch.cuda.is_available()
    print(
        f"[mmrec] subdir={args.mind_dataset_subdir} stage={args.stage} "
        f"gpu={enable_gpu} data={data_root}",
        flush=True,
    )

    if args.stage in ("prepare", "all"):
        prepare(args.mind_dataset_subdir)

    npz_path = data_root / "rois.npz"
    if args.stage in ("extract-roi", "all"):
        if npz_path.is_file() and not args.force_extract:
            print(f"[mmrec] ROI 캐시 사용: {npz_path}  (--force-extract 로 다시 추출)", flush=True)
        else:
            extract_roi(
                args.mind_dataset_subdir,
                args.thumbnail_dir,
                news_tsv=str(data_root / "subnews.tsv"),
                out_dir=str(data_root),
                device="cpu" if args.cpu else "auto",
            )

    if args.stage in ("train", "all"):
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"ROI 없음: {npz_path}. --stage extract-roi 를 먼저 실행하세요."
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
            from_pretrained=args.from_pretrained,
            config_file=args.config_file,
            load_ckpt=None,
            debug=args.debug,
            enable_gpu=enable_gpu,
            selection_metric=args.selection_metric,
        )
        utils.setuplogger(os.path.join(run_args.log_dir, "log_train.txt"))
        summary = mmrec_train(run_args)
        if summary:
            print(
                f"[mmrec] best val epoch={summary.get('best_epoch')}  "
                f"MRR={(summary.get('best_metrics') or {}).get('MRR')}  "
                f"NDCG@5={(summary.get('best_metrics') or {}).get('NDCG@5')}  "
                f"Hit@1={(summary.get('best_metrics') or {}).get('Hit@1')}",
                flush=True,
            )

    if args.stage in ("test", "all"):
        ckpt_name = args.load_ckpt
        if args.stage == "all":
            ckpt_name = "best.pt"
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
            from_pretrained=args.from_pretrained,
            config_file=args.config_file,
            load_ckpt=ckpt_name,
            debug=args.debug,
            enable_gpu=enable_gpu,
            selection_metric=args.selection_metric,
        )
        utils.setuplogger(os.path.join(run_args.log_dir, "log_test.txt"))
        mmrec_test(run_args)
        result_path = Path(run_args.log_dir) / "final_result.txt"
        if result_path.is_file():
            print(f"[mmrec] metrics → {result_path}", flush=True)
            print(result_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
