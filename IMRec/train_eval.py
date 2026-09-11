#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMRec (NRMS-IM / FIM-IM) on MIND_2000.

프로토콜 (NAML/CLIP/MM-Rec과 동일):
  - 매폭마다 val(MIND_test_(2000).tsv) 평가
  - val MRR 최고 에폭 저장
  - best.pt 로 held-out test(MIND_test_2000_final.tsv) 평가

  conda activate clip_cu128
  python IMRec/train_eval.py --model nrms-im --mind-dataset-subdir MIND_2000
  python IMRec/train_eval.py --model fim-im --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_IMREC = Path(__file__).resolve().parent
if str(_IMREC) not in sys.path:
    sys.path.insert(0, str(_IMREC))

from dataset import EvalBatcher, TrainBatcher, build_news_tables, load_impressions
from extract_features import extract_all
from metrics import session_metrics
from models import build_model
from paths import (
    DEFAULT_THUMBNAIL_DIR,
    DATASET_FILE_PRESETS,
    dataset_raw_dir,
    features_path,
    saved_dir,
)
from render_cards import render_all


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, flat, spans, tables, args, device) -> dict:
    import torch

    model.eval()
    batcher = EvalBatcher(flat, tables, args.eval_batch_size, args.max_title_len, device)
    scores = []
    with torch.no_grad():
        for batch in batcher.iter_all():
            s = model.forward_eval(batch["history"], batch["candidates"], batch["hist_mask"])
            scores.append(s.detach().cpu().numpy())
    scores = np.concatenate(scores, axis=0)
    labels = np.asarray([r["label"] for r in flat], dtype=np.float32)
    mrrs, ndcgs, hits = [], [], []
    for a, b in spans:
        if b > len(scores) or np.sum(labels[a:b]) == 0:
            continue
        m = session_metrics(labels[a:b], scores[a:b])
        mrrs.append(m["MRR"])
        ndcgs.append(m["NDCG@5"])
        hits.append(m["Hit@1"])
    if not mrrs:
        return {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0, "n": 0}
    return {
        "MRR": float(np.mean(mrrs)),
        "NDCG@5": float(np.mean(ndcgs)),
        "Hit@1": float(np.mean(hits)),
        "n": len(mrrs),
    }


def train_one(args) -> dict:
    import torch
    import torch.optim as optim

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    set_seed(args.seed)

    # prepare cards + features
    if args.stage in ("prepare", "all"):
        render_all(args.mind_dataset_subdir, args.thumbnail_dir, force=args.force_cards)
    if args.stage in ("extract", "all"):
        if features_path(args.mind_dataset_subdir).is_file() and not args.force_extract:
            print(f"[imrec] feature cache ok: {features_path(args.mind_dataset_subdir)}", flush=True)
        else:
            extract_all(args.mind_dataset_subdir, force=args.force_extract, device=device)
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.stage not in ("train", "test", "all"):
        return {}

    print("[imrec] building news tables...", flush=True)
    news_name, train_name, val_name, test_name = DATASET_FILE_PRESETS[args.mind_dataset_subdir]
    raw = dataset_raw_dir(args.mind_dataset_subdir)
    title_len = 15 if "nrms" in args.model.lower() else args.max_title_len
    # keep args.max_title_len for FIM; NRMS paper uses 15
    if "nrms" in args.model.lower():
        args.max_title_len = title_len
    hist_len = 60 if "nrms" in args.model.lower() else args.max_history
    if "nrms" in args.model.lower():
        args.max_history = hist_len

    tables = build_news_tables(
        args.mind_dataset_subdir,
        max_title_len=max(args.max_title_len, 30),
        glove_path=args.glove_path,
        emb_dim=args.word_dim,
    )

    print("[imrec] loading impressions...", flush=True)
    rng = random.Random(args.seed)
    train_imps, _ = load_impressions(
        raw / train_name,
        tables.news_index,
        has_header=True,
        has_labels=True,
        npratio=args.npratio,
        max_history=args.max_history,
        rng=rng,
    )
    val_flat, val_spans = load_impressions(
        raw / val_name,
        tables.news_index,
        has_header=False,
        has_labels=False,
        npratio=args.npratio,
        max_history=args.max_history,
        rng=random.Random(args.seed + 1),
    )
    test_flat, test_spans = load_impressions(
        raw / test_name,
        tables.news_index,
        has_header=False,
        has_labels=False,
        npratio=args.npratio,
        max_history=args.max_history,
        rng=random.Random(args.seed + 2),
    )
    print(
        f"[imrec] model={args.model} train={len(train_imps)} val_sess={len(val_spans)} "
        f"test_sess={len(test_spans)} device={device}",
        flush=True,
    )

    if getattr(args, "save_root", None):
        save_root = Path(args.save_root)
    else:
        save_root = saved_dir(args.mind_dataset_subdir) / args.model.replace("-", "_")
    (save_root / "ckpts").mkdir(parents=True, exist_ok=True)
    (save_root / "logs").mkdir(parents=True, exist_ok=True)

    print("[imrec] building model...", flush=True)
    model = build_model(args.model, tables, args).to(device)
    if args.stage == "test":
        ckpt = save_root / "ckpts" / args.load_ckpt
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload["model"])
        metrics = evaluate(model, test_flat, test_spans, tables, args, device)
        print(f"[imrec] TEST {metrics}", flush=True)
        (save_root / "logs" / "final_result.txt").write_text(
            f"MRR={metrics['MRR']}\nNDCG@5={metrics['NDCG@5']}\nHit@1={metrics['Hit@1']}\n",
            encoding="utf-8",
        )
        return {"test": metrics}

    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = TrainBatcher(
        train_imps, tables, args.batch_size, args.max_title_len, device, seed=args.seed
    )

    best_mrr = -1.0
    best_epoch = -1
    best_metrics = None
    epoch_logs = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader.iter_epoch(), 1):
            opt.zero_grad()
            _, loss = model.forward_train(
                batch["history"], batch["candidates"], batch["hist_mask"], batch["labels"]
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if args.debug and step >= 5:
                break
        val_metrics = evaluate(model, val_flat, val_spans, tables, args, device)
        row = {"epoch": ep, "train_loss": float(np.mean(losses) if losses else 0), "val": val_metrics}
        epoch_logs.append(row)
        print(
            f"[imrec] epoch {ep}/{args.epochs} loss={row['train_loss']:.4f}  "
            f"val MRR={val_metrics['MRR']:.6f} NDCG@5={val_metrics['NDCG@5']:.6f} "
            f"Hit@1={val_metrics['Hit@1']:.6f}",
            flush=True,
        )
        torch.save(
            {"model": model.state_dict(), "epoch": ep, "val": val_metrics, "args": vars(args)},
            save_root / "ckpts" / f"epoch-{ep}.pt",
        )
        if val_metrics["MRR"] > best_mrr:
            best_mrr = val_metrics["MRR"]
            best_epoch = ep
            best_metrics = val_metrics
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val": val_metrics, "args": vars(args)},
                save_root / "ckpts" / "best.pt",
            )
            print(f"[imrec] best ← epoch {ep} MRR={best_mrr:.6f}", flush=True)

    summary = {
        "best_epoch": best_epoch,
        "best_val": best_metrics,
        "best_mrr": best_mrr,
        "epoch_logs": epoch_logs,
        "ckpt": str(save_root / "ckpts" / "best.pt"),
    }
    (save_root / "logs" / "val_epoch_log.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if getattr(args, "skip_final_test", False):
        print(
            f"[imrec] skip final test  best_epoch={best_epoch} val_MRR={best_mrr:.6f}",
            flush=True,
        )
        return summary

    # final test with best
    ckpt = torch.load(save_root / "ckpts" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_flat, test_spans, tables, args, device)
    summary["test"] = test_metrics
    (save_root / "logs" / "final_result.txt").write_text(
        f"best_epoch={best_epoch}\n"
        f"val_MRR={best_metrics['MRR']}\nval_NDCG@5={best_metrics['NDCG@5']}\nval_Hit@1={best_metrics['Hit@1']}\n"
        f"test_MRR={test_metrics['MRR']}\ntest_NDCG@5={test_metrics['NDCG@5']}\ntest_Hit@1={test_metrics['Hit@1']}\n",
        encoding="utf-8",
    )
    (save_root / "logs" / "test_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[imrec] DONE best_epoch={best_epoch}  "
        f"TEST MRR={test_metrics['MRR']:.6f} NDCG@5={test_metrics['NDCG@5']:.6f} "
        f"Hit@1={test_metrics['Hit@1']:.6f}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="IMRec NRMS-IM / FIM-IM train+eval")
    ap.add_argument("--model", type=str, default="nrms-im", choices=["nrms-im", "fim-im"])
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["prepare", "extract", "train", "test", "all"],
    )
    ap.add_argument("--thumbnail-dir", type=str, default=str(DEFAULT_THUMBNAIL_DIR))
    ap.add_argument("--glove-path", type=str, default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--npratio", type=int, default=4)
    ap.add_argument("--max-history", type=int, default=50)
    ap.add_argument("--max-title-len", type=int, default=30)
    ap.add_argument("--word-dim", type=int, default=100)
    ap.add_argument("--attn-hidden", type=int, default=200)
    ap.add_argument("--n-heads", type=int, default=3)
    ap.add_argument("--head-dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-cards", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--load-ckpt", type=str, default="best.pt")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-final-test", action="store_true")
    ap.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="체크포인트/로그 저장 루트 (튜닝 trial용)",
    )
    args = ap.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
