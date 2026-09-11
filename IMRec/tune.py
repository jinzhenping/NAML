#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMRec 하이퍼파라미터 탐색 (MM_Rec/tune.py 와 동일 프로토콜).

각 trial: 지정 에폭 학습, val(MIND_test_(2000).tsv) MRR 최고 에폭 선택.
전역 최고 best.pt 를 MIND_test_2000_final.tsv 로 최종 평가.

  python IMRec/tune.py --model nrms-im --mind-dataset-subdir MIND_2000 --two-phase \\
      --trials 24 --screening-epochs 3 --refine-top-k 5 --epochs-per-trial 30 \\
      --glove-path glove/glove.6B.100d.txt

  python IMRec/tune.py --model fim-im --mind-dataset-subdir MIND_2000 --two-phase \\
      --trials 24 --screening-epochs 3 --refine-top-k 5 --epochs-per-trial 30 \\
      --glove-path glove/glove.6B.100d.txt
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_IMREC = Path(__file__).resolve().parent
if str(_IMREC) not in sys.path:
    sys.path.insert(0, str(_IMREC))

from paths import DEFAULT_THUMBNAIL_DIR, features_path, saved_dir
from train_eval import train_one

HPARAM_CHOICES: Dict[str, list] = {
    "lr": [1e-4, 2e-4, 5e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "dropout": [0.1, 0.2, 0.3],
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


def _write_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_args(base: argparse.Namespace, hp: dict, *, epochs: int, save_root: Path) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(base))
    ns.stage = "train"
    ns.epochs = epochs
    ns.lr = float(hp["lr"])
    ns.batch_size = int(hp["batch_size"])
    ns.dropout = float(hp["dropout"])
    ns.npratio = int(hp["npratio"])
    ns.skip_final_test = True
    ns.save_root = str(save_root)
    ns.force_cards = False
    ns.force_extract = False
    return ns


def main() -> None:
    ap = argparse.ArgumentParser(description="IMRec hyperparameter search")
    ap.add_argument("--model", type=str, default="nrms-im", choices=["nrms-im", "fim-im"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--epochs-per-trial", type=int, default=30)
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--screening-epochs", type=int, default=3)
    ap.add_argument("--refine-top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--glove-path", type=str, default="")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--max-history", type=int, default=50)
    ap.add_argument("--max-title-len", type=int, default=30)
    ap.add_argument("--word-dim", type=int, default=100)
    ap.add_argument("--attn-hidden", type=int, default=200)
    ap.add_argument("--n-heads", type=int, default=3)
    ap.add_argument("--head-dim", type=int, default=50)
    ap.add_argument("--resume-log", type=str, default=None)
    ap.add_argument("--out-log", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-final-test", action="store_true")
    args = ap.parse_args()

    import torch

    model_tag = args.model.replace("-", "_")
    save_base = saved_dir(args.mind_dataset_subdir) / model_tag
    save_base.mkdir(parents=True, exist_ok=True)
    out_log = Path(args.out_log) if args.out_log else save_base / "imrec_tune_log.json"

    # ensure cards/features once
    prep = argparse.Namespace(
        model=args.model,
        mind_dataset_subdir=args.mind_dataset_subdir,
        stage="all" if not features_path(args.mind_dataset_subdir).is_file() else "prepare",
        thumbnail_dir=args.thumbnail_dir,
        glove_path=args.glove_path,
        epochs=1,
        batch_size=16,
        eval_batch_size=args.eval_batch_size,
        lr=1e-4,
        dropout=0.2,
        npratio=4,
        max_history=args.max_history,
        max_title_len=args.max_title_len,
        word_dim=args.word_dim,
        attn_hidden=args.attn_hidden,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        seed=args.seed,
        force_cards=False,
        force_extract=False,
        load_ckpt="best.pt",
        debug=args.debug,
        cpu=args.cpu,
        skip_final_test=True,
        save_root=str(save_base / "tune" / "_prep"),
    )
    # only prepare+extract; avoid train
    if not features_path(args.mind_dataset_subdir).is_file():
        print("[tune] preparing cards + features...", flush=True)
        prep.stage = "all"
        # train_one with stage all would train — split stages
        prep.stage = "prepare"
        train_one(prep)
        prep.stage = "extract"
        train_one(prep)
    else:
        print(f"[tune] feature cache ok: {features_path(args.mind_dataset_subdir)}", flush=True)

    rng = random.Random(args.seed)
    planned = plan_hparam_trials(rng, args.trials)
    seen = _seen_keys(Path(args.resume_log)) if args.resume_log else set()
    if seen:
        planned = [hp for hp in planned if _hp_key(hp) not in seen]
        print(f"[tune] resume: skip {len(seen)} seen, remain {len(planned)}", flush=True)

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

    base = argparse.Namespace(
        model=args.model,
        mind_dataset_subdir=args.mind_dataset_subdir,
        stage="train",
        thumbnail_dir=args.thumbnail_dir,
        glove_path=args.glove_path,
        epochs=args.epochs_per_trial,
        batch_size=32,
        eval_batch_size=args.eval_batch_size,
        lr=1e-4,
        dropout=0.2,
        npratio=4,
        max_history=args.max_history,
        max_title_len=args.max_title_len,
        word_dim=args.word_dim,
        attn_hidden=args.attn_hidden,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        seed=args.seed,
        force_cards=False,
        force_extract=False,
        load_ckpt="best.pt",
        debug=args.debug,
        cpu=args.cpu,
        skip_final_test=True,
        save_root=None,
    )

    def _run_list(hp_list: List[dict], epochs: int, prefix: str) -> None:
        nonlocal global_best_mrr, global_best
        for i, hp in enumerate(hp_list, start=1):
            trial_id = f"{prefix}_{i:03d}"
            trial_root = save_base / "tune" / trial_id
            print(f"\n[tune] {trial_id} epochs={epochs} hp={hp}", flush=True)
            try:
                run_args = _build_args(base, hp, epochs=epochs, save_root=trial_root)
                summary = train_one(run_args) or {}
                best_val = summary.get("best_val") or {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}
                row = {
                    "trial_id": trial_id,
                    "hparams": hp,
                    "epochs": epochs,
                    "best_epoch": summary.get("best_epoch"),
                    "best_metrics": best_val,
                    "best_mrr_in_trial": float(best_val.get("MRR") or summary.get("best_mrr") or 0.0),
                    "ckpt": summary.get("ckpt") or str(trial_root / "ckpts" / "best.pt"),
                }
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
                f"[tune] {trial_id} val MRR={mrr:.6f}  "
                f"NDCG@5={(row.get('best_metrics') or {}).get('NDCG@5')}  "
                f"Hit@1={(row.get('best_metrics') or {}).get('Hit@1')}",
                flush=True,
            )
            ckpt = Path(row.get("ckpt") or "")
            if mrr > global_best_mrr and ckpt.is_file():
                global_best_mrr = mrr
                global_best = row
                dst = save_base / "ckpts" / "best.pt"
                (save_base / "ckpts").mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt, dst)
                print(f"[tune] global best → {dst}  MRR={global_best_mrr:.6f}", flush=True)
            _write_log(
                out_log,
                {
                    "model": args.model,
                    "selection_metric": "MRR",
                    "hparam_choices": HPARAM_CHOICES,
                    "global_best_mrr": global_best_mrr,
                    "global_best": global_best,
                    "trials": trials,
                },
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.two_phase:
        _run_list(planned, args.screening_epochs, "screen")
        ranked = sorted(
            [t for t in trials if t.get("best_mrr_in_trial", -1) >= 0 and "hparams" in t],
            key=lambda r: float(r.get("best_mrr_in_trial") or -1),
            reverse=True,
        )
        screen_rows = [t for t in ranked if str(t.get("trial_id", "")).startswith("screen_")]
        top = screen_rows[: args.refine_top_k]
        print(f"[tune] refine top-{len(top)} of {len(screen_rows)} screen trials", flush=True)
        _run_list([t["hparams"] for t in top], args.epochs_per_trial, "refine")
    else:
        _run_list(planned, args.epochs_per_trial, "trial")

    payload = {
        "model": args.model,
        "selection_metric": "MRR",
        "hparam_choices": HPARAM_CHOICES,
        "global_best_mrr": global_best_mrr,
        "global_best": global_best,
        "trials": trials,
    }
    _write_log(out_log, payload)
    print(f"[tune] log → {out_log}", flush=True)
    if not global_best:
        print("[tune] 유효한 trial이 없습니다.", flush=True)
        return

    print(
        f"[tune] best val MRR={global_best_mrr:.6f}  hp={global_best.get('hparams')}  "
        f"epoch={global_best.get('best_epoch')}",
        flush=True,
    )
    if args.skip_final_test:
        return

    # final test with global best.pt
    test_args = argparse.Namespace(
        model=args.model,
        mind_dataset_subdir=args.mind_dataset_subdir,
        stage="test",
        thumbnail_dir=args.thumbnail_dir,
        glove_path=args.glove_path,
        epochs=args.epochs_per_trial,
        batch_size=int(global_best["hparams"]["batch_size"]),
        eval_batch_size=args.eval_batch_size,
        lr=float(global_best["hparams"]["lr"]),
        dropout=float(global_best["hparams"]["dropout"]),
        npratio=int(global_best["hparams"]["npratio"]),
        max_history=args.max_history,
        max_title_len=args.max_title_len,
        word_dim=args.word_dim,
        attn_hidden=args.attn_hidden,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        seed=args.seed,
        force_cards=False,
        force_extract=False,
        load_ckpt="best.pt",
        debug=args.debug,
        cpu=args.cpu,
        skip_final_test=False,
        save_root=str(save_base),
    )
    summary = train_one(test_args) or {}
    payload["test"] = summary.get("test")
    _write_log(out_log, payload)
    if summary.get("test"):
        m = summary["test"]
        print(
            f"[tune] TEST MRR={m['MRR']:.6f}  NDCG@5={m['NDCG@5']:.6f}  Hit@1={m['Hit@1']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
