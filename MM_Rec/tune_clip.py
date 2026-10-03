#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MM-Rec + CLIP 하이퍼파라미터 탐색 (원본 tune.py 와 동일 프로토콜).

  conda activate clip_cu128
  python MM_Rec/extract_clip_features.py --mind-dataset-subdir MIND_2000 --reuse-image-cache
  python MM_Rec/tune_clip.py --mind-dataset-subdir MIND_2000 --two-phase \\
      --trials 24 --screening-epochs 3 --refine-top-k 5 --epochs-per-trial 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MMREC = Path(__file__).resolve().parent
if str(_MMREC) not in sys.path:
    sys.path.insert(0, str(_MMREC))

from dataset_paths import DEFAULT_THUMBNAIL_DIR, prepared_dir, saved_dir
from extract_clip_features import clip_image_cache_path, clip_title_cache_path
from prepare_mind_dataset import prepare
from train_eval_clip import _build_run_args
from run_clip import test as mmrec_clip_test
from run_clip import train as mmrec_clip_train
import utils

HPARAM_CHOICES: Dict[str, list] = {
    "lr": [1e-5, 2e-5, 5e-5, 1e-4],
    "batch_size": [4, 8, 16],
    "npratio": [1, 4],
}
_HP_KEYS = tuple(HPARAM_CHOICES.keys())


def _hp_key(hp: dict) -> tuple:
    return tuple((k, hp[k]) for k in _HP_KEYS)


def sample_hparams(rng: random.Random) -> dict:
    return {k: rng.choice(HPARAM_CHOICES[k]) for k in _HP_KEYS}


def plan_hparam_trials(rng: random.Random, n_trials: int) -> List[dict]:
    vals = [HPARAM_CHOICES[k] for k in _HP_KEYS]
    combos = [dict(zip(_HP_KEYS, prod)) for prod in itertools.product(*vals)]
    rng.shuffle(combos)
    if n_trials <= len(combos):
        return combos[:n_trials]
    out = list(combos)
    for _ in range(n_trials - len(combos)):
        out.append(sample_hparams(rng))
    return out


def _reset_logger() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def _load_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _seen_keys(log_path: Path) -> set:
    seen = set()
    data = _load_json(log_path)
    if not data:
        return seen
    for row in data.get("trials") or []:
        if isinstance(row, dict) and isinstance(row.get("hparams"), dict):
            hp = row["hparams"]
            if all(k in hp for k in _HP_KEYS):
                seen.add(_hp_key(hp))
    return seen


def _ensure_clip(subdir: str, thumbnail_dir: str, force: bool, reuse_image: bool) -> Tuple[Path, Path]:
    data_root = prepared_dir(subdir)
    save_root = saved_dir(subdir) / "clip"
    save_root.mkdir(parents=True, exist_ok=True)
    prepare(subdir)
    img = clip_image_cache_path(subdir)
    title = clip_title_cache_path(subdir)
    if force or not img.is_file() or not title.is_file():
        argv = [
            "extract_clip_features.py",
            "--mind-dataset-subdir",
            subdir,
            "--thumbnail-dir",
            thumbnail_dir,
        ]
        if reuse_image:
            argv.append("--reuse-image-cache")
        old = sys.argv
        sys.argv = argv
        try:
            from extract_clip_features import main as extract_main

            extract_main()
        finally:
            sys.argv = old
    else:
        print(f"[tune-clip] CLIP cache ok: {img}, {title}", flush=True)
    return data_root, save_root


def run_trial(
    hp: dict,
    *,
    trial_id: str,
    epochs: int,
    mind_dataset_subdir: str,
    data_root: Path,
    save_root: Path,
    enable_gpu: bool,
    debug: bool,
) -> Dict[str, Any]:
    trial_root = save_root / "tune" / trial_id
    (trial_root / "ckpts").mkdir(parents=True, exist_ok=True)
    (trial_root / "logs").mkdir(parents=True, exist_ok=True)
    run_args = _build_run_args(
        mode="train",
        mind_dataset_subdir=mind_dataset_subdir,
        data_root=data_root,
        save_root=trial_root,
        eval_split="dev",
        epochs=epochs,
        batch_size=int(hp["batch_size"]),
        lr=float(hp["lr"]),
        npratio=int(hp["npratio"]),
        load_ckpt=None,
        debug=debug,
        enable_gpu=enable_gpu,
        selection_metric="MRR",
    )
    _reset_logger()
    utils.setuplogger(str(trial_root / "logs" / "log_train.txt"))
    summary = mmrec_clip_train(run_args) or {}
    best_metrics = summary.get("best_metrics") or {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}
    return {
        "trial_id": trial_id,
        "hparams": hp,
        "epochs": epochs,
        "best_epoch": summary.get("best_epoch"),
        "best_metrics": best_metrics,
        "best_mrr_in_trial": float(best_metrics.get("MRR") or 0.0),
        "ckpt": str(trial_root / "ckpts" / "best.pt"),
        "epoch_logs": summary.get("epoch_logs"),
    }


def _write_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="MM-Rec + CLIP 하이퍼파라미터 탐색")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs-per-trial", type=int, default=10)
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--screening-epochs", type=int, default=3)
    ap.add_argument("--refine-top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--reuse-image-cache", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--resume-log", type=str, default=None)
    ap.add_argument("--out-log", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-final-test", action="store_true")
    args = ap.parse_args()

    import torch

    enable_gpu = (not args.cpu) and torch.cuda.is_available()
    data_root, save_root = _ensure_clip(
        args.mind_dataset_subdir,
        args.thumbnail_dir,
        args.force_extract,
        args.reuse_image_cache,
    )
    out_log = Path(args.out_log) if args.out_log else save_root / "mmrec_clip_tune_log.json"
    rng = random.Random(args.seed)
    planned = plan_hparam_trials(rng, args.trials)
    seen = _seen_keys(Path(args.resume_log)) if args.resume_log else set()
    if seen:
        planned = [hp for hp in planned if _hp_key(hp) not in seen]
        print(f"[tune-clip] resume: skip {len(seen)} seen, remain {len(planned)}", flush=True)

    trials: List[dict] = []
    prev = _load_json(Path(args.resume_log)) if args.resume_log else None
    if prev and isinstance(prev.get("trials"), list):
        trials.extend(prev["trials"])

    global_best_mrr = -1.0
    global_best: Optional[dict] = None
    if prev:
        try:
            global_best_mrr = float(prev.get("global_best_mrr", -1))
            if isinstance(prev.get("global_best"), dict):
                global_best = prev["global_best"]
        except Exception:
            pass

    def _run_list(hp_list: List[dict], epochs: int, prefix: str) -> None:
        nonlocal global_best_mrr, global_best
        for i, hp in enumerate(hp_list, start=1):
            trial_id = f"{prefix}_{i:03d}"
            print(f"\n[tune-clip] {trial_id} epochs={epochs} hp={hp}", flush=True)
            try:
                row = run_trial(
                    hp,
                    trial_id=trial_id,
                    epochs=epochs,
                    mind_dataset_subdir=args.mind_dataset_subdir,
                    data_root=data_root,
                    save_root=save_root,
                    enable_gpu=enable_gpu,
                    debug=args.debug,
                )
            except Exception:
                traceback.print_exc()
                row = {
                    "trial_id": trial_id,
                    "hparams": hp,
                    "epochs": epochs,
                    "error": traceback.format_exc(),
                    "best_mrr_in_trial": -1.0,
                    "best_metrics": {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0},
                }
            trials.append(row)
            mrr = float(row.get("best_mrr_in_trial") or 0.0)
            print(
                f"[tune-clip] {trial_id} val MRR={mrr:.6f}  "
                f"NDCG@5={(row.get('best_metrics') or {}).get('NDCG@5')}  "
                f"Hit@1={(row.get('best_metrics') or {}).get('Hit@1')}",
                flush=True,
            )
            if mrr > global_best_mrr and Path(row.get("ckpt") or "").is_file():
                global_best_mrr = mrr
                global_best = row
                dst = save_root / "ckpts" / "best.pt"
                (save_root / "ckpts").mkdir(parents=True, exist_ok=True)
                shutil.copy2(row["ckpt"], dst)
                print(f"[tune-clip] global best → {dst}  MRR={global_best_mrr:.6f}", flush=True)
            _write_log(
                out_log,
                {
                    "variant": "mmrec_clip",
                    "selection_metric": "MRR",
                    "hparam_choices": HPARAM_CHOICES,
                    "global_best_mrr": global_best_mrr,
                    "global_best": global_best,
                    "trials": trials,
                },
            )
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not planned:
        print("[tune-clip] 실행할 trial 없음", flush=True)
    elif args.two_phase:
        _run_list(planned, args.screening_epochs, "screen")
        ranked = sorted(
            [t for t in trials if t.get("trial_id", "").startswith("screen_") and "error" not in t],
            key=lambda r: -float(r.get("best_mrr_in_trial") or 0.0),
        )
        top = []
        seen_hp = set()
        for row in ranked:
            k = _hp_key(row["hparams"])
            if k in seen_hp:
                continue
            seen_hp.add(k)
            top.append(row["hparams"])
            if len(top) >= args.refine_top_k:
                break
        print(f"[tune-clip] refine top-{len(top)}", flush=True)
        _run_list(top, args.epochs_per_trial, "refine")
    else:
        _run_list(planned, args.epochs_per_trial, "trial")

    payload = {
        "variant": "mmrec_clip",
        "selection_metric": "MRR",
        "hparam_choices": HPARAM_CHOICES,
        "global_best_mrr": global_best_mrr,
        "global_best": global_best,
        "trials": trials,
    }
    _write_log(out_log, payload)
    print(f"[tune-clip] log → {out_log}", flush=True)
    if not global_best:
        print("[tune-clip] 유효한 trial이 없습니다.", flush=True)
        return
    print(
        f"[tune-clip] best val MRR={global_best_mrr:.6f}  hp={global_best.get('hparams')}  "
        f"epoch={global_best.get('best_epoch')}",
        flush=True,
    )
    if args.skip_final_test:
        return

    run_args = _build_run_args(
        mode="test",
        mind_dataset_subdir=args.mind_dataset_subdir,
        data_root=data_root,
        save_root=save_root,
        eval_split="test",
        epochs=args.epochs_per_trial,
        batch_size=int(global_best["hparams"]["batch_size"]),
        lr=float(global_best["hparams"]["lr"]),
        npratio=int(global_best["hparams"]["npratio"]),
        load_ckpt="best.pt",
        debug=args.debug,
        enable_gpu=enable_gpu,
        selection_metric="MRR",
        clip_dropout=0.0,
    )
    _reset_logger()
    utils.setuplogger(str(save_root / "logs" / "log_test.txt"))
    metrics = mmrec_clip_test(run_args) or {}
    payload["test"] = metrics
    _write_log(out_log, payload)
    if metrics:
        print(
            f"[tune-clip] TEST MRR={metrics['MRR']:.6f}  "
            f"NDCG@5={metrics['NDCG@5']:.6f}  Hit@1={metrics['Hit@1']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
