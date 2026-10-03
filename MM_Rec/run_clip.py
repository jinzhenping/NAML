#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MM-Rec + CLIP train / test (user_encoder 동일, news = CLIP title/image)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import utils
from clip_mmrec import mmrec_clip
from dataloader import DataLoaderTest
from dataloader_clip import DataLoaderTrainClip
from extract_clip_features import (
    build_mmrec_clip_matrices,
    clip_image_cache_path,
    clip_title_cache_path,
)
from metrics import hit_at_k, mrr_score, ndcg_score
from preprocess import read_news_bert


def _device(args):
    if args.enable_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _news_tsv_path(args):
    if os.path.isabs(args.news_file):
        return args.news_file
    return os.path.join(args.root_data_dir, args.news_file)


def _selection_key(name: str) -> str:
    n = (name or "MRR").upper().replace("NDCG@5", "NDCG@5")
    if n in ("NDCG@5", "NDCG5"):
        return "NDCG@5"
    if n in ("HIT@1", "HIT1"):
        return "Hit@1"
    return "MRR"


def _load_clip_tables(args, news_index):
    n_rows = max(news_index.values()) + 1 if news_index else 1
    img_cache = getattr(args, "clip_image_cache", None) or str(
        clip_image_cache_path(args.dataset)
    )
    title_cache = getattr(args, "clip_title_cache", None) or str(
        clip_title_cache_path(args.dataset)
    )
    if not os.path.isfile(img_cache):
        raise FileNotFoundError(
            f"CLIP image cache 없음: {img_cache}\n"
            "python MM_Rec/extract_clip_features.py --mind-dataset-subdir "
            f"{args.dataset}"
        )
    if not os.path.isfile(title_cache):
        raise FileNotFoundError(
            f"CLIP title cache 없음: {title_cache}\n"
            "python MM_Rec/extract_clip_features.py --mind-dataset-subdir "
            f"{args.dataset}"
        )
    mat_t, mat_v, n_t, n_v = build_mmrec_clip_matrices(
        news_index, n_rows, img_cache, title_cache
    )
    logging.info(
        "CLIP tables rows=%s dim=%s title_nonzero=%s image_nonzero=%s",
        n_rows,
        mat_t.shape[1],
        n_t,
        n_v,
    )
    return (
        torch.from_numpy(mat_t),
        torch.from_numpy(mat_v),
        os.path.abspath(img_cache),
        os.path.abspath(title_cache),
    )


def score_impressions_clip(args, model, news_index, news_scoring_t, news_scoring_v, data_dir):
    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring_t=news_scoring_t,
        news_scoring_v=news_scoring_v,
        word_dict=None,
        news_bias_scoring=None,
        data_dir=data_dir,
        filename_pat="test_*.tsv",
        args=args,
        world_size=1,
        news_id2image_id={},
        worker_rank=0,
        cuda_device_idx=0,
        enable_prefetch=False,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )
    MRR, nDCG5, HIT1 = [], [], []
    device = _device(args)
    model.eval()
    with torch.no_grad():
        for log_vecs_t, log_vecs_v, log_masks, news_vecs_t, news_vecs_v, _bias, labels in dataloader:
            for user_vec_t, user_vec_v, news_vec_t, news_vec_v, label, log_mask in zip(
                log_vecs_t, log_vecs_v, news_vecs_t, news_vecs_v, labels, log_masks
            ):
                if label.sum() == 0:
                    continue
                user_vec_t = torch.as_tensor(user_vec_t, dtype=torch.float32, device=device).unsqueeze(0)
                user_vec_v = torch.as_tensor(user_vec_v, dtype=torch.float32, device=device).unsqueeze(0)
                news_vec_t = torch.as_tensor(news_vec_t, dtype=torch.float32, device=device).unsqueeze(0)
                news_vec_v = torch.as_tensor(news_vec_v, dtype=torch.float32, device=device).unsqueeze(0)
                log_mask = torch.as_tensor(log_mask, dtype=torch.float32, device=device).unsqueeze(0)
                user_vecs = model.user_encoder(
                    news_vec_t, news_vec_v, user_vec_t, user_vec_v, log_mask
                )
                score = torch.sum((news_vec_t + news_vec_v) * user_vecs, -1).squeeze(0)
                score = score.detach().cpu().numpy()
                label = np.asarray(label)
                MRR.append(mrr_score(label, score))
                nDCG5.append(ndcg_score(label, score, k=5))
                HIT1.append(hit_at_k(label, score, k=1))
    try:
        dataloader.join()
    except Exception:
        pass
    n = len(MRR)
    return {
        "MRR": float(np.mean(MRR)) if n else 0.0,
        "NDCG@5": float(np.mean(nDCG5)) if n else 0.0,
        "Hit@1": float(np.mean(HIT1)) if n else 0.0,
        "n": n,
    }


def train(args):
    utils.init_hvd_cuda(False, args.enable_gpu)
    # news_index only (tokenizer unused for CLIP path but keeps identical ID order)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        _news_tsv_path(args), args, tokenizer
    )
    del news

    clip_t, clip_v, img_cache, title_cache = _load_clip_tables(args, news_index)
    hidden = int(getattr(args, "clip_hidden_size", 1024))
    dropout = float(getattr(args, "clip_dropout", 0.1))
    model = mmrec_clip(clip_t, clip_v, hidden_size=hidden, dropout=dropout)

    start_epoch = 0
    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            logging.info("Model loaded from %s epoch=%s", ckpt_path, start_epoch)
            del checkpoint

    if args.enable_gpu:
        model = model.cuda()

    # CLIP buffers frozen; train proj + user_encoder
    for name, p in model.named_parameters():
        p.requires_grad = True
    model.news_encoder.clip_t.requires_grad = False
    model.news_encoder.clip_v.requires_grad = False

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    dataloader = DataLoaderTrainClip(
        news_index=news_index,
        data_dir=os.path.join(args.root_data_dir, f"{args.dataset}/{args.train_dir}"),
        filename_pat=args.filename_pat,
        args=args,
        world_size=1,
        worker_rank=0,
        cuda_device_idx=0,
        enable_prefetch=False,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )

    world = 1
    args.max_steps_per_epoch = args.max_steps_per_epoch // (world * args.batch_size)
    val_dir_name = args.valid_dir or "dev"
    val_data_dir = os.path.join(args.root_data_dir, args.dataset, val_dir_name)
    sel_key = _selection_key(getattr(args, "selection_metric", "MRR"))
    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    epoch_logs = []

    logging.info(
        "MM-Rec+CLIP train hidden=%s img=%s title=%s",
        hidden,
        img_cache,
        title_cache,
    )

    for ep in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader, start=1):
            if cnt > args.max_steps_per_epoch or (args.debug and cnt > 10):
                break
            bz_loss, _ = model(input_ids, log_ids, log_mask, targets)
            loss_sum += float(bz_loss.detach().cpu())
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            if cnt % args.log_steps == 0:
                logging.info(
                    "epoch [%s] Ed: %s train_avg_loss: %.5f",
                    ep,
                    cnt * args.batch_size,
                    loss_sum / cnt,
                )

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "category_dict": category_dict,
            "domain_dict": domain_dict,
            "subcategory_dict": subcategory_dict,
            "epoch": ep + 1,
            "clip_image_cache": img_cache,
            "clip_title_cache": title_cache,
            "clip_hidden_size": hidden,
            "backend": "clip",
        }
        ckpt_path = os.path.join(args.model_dir, f"epoch-{ep + 1}.pt")
        torch.save(ckpt_payload, ckpt_path)
        logging.info("Model saved to %s", ckpt_path)

        val_metrics = None
        if os.path.isdir(val_data_dir):
            news_scoring_t, news_scoring_v = model.encode_news_tables()
            val_metrics = score_impressions_clip(
                args, model, news_index, news_scoring_t, news_scoring_v, val_data_dir
            )
            logging.info(
                "val epoch %s MRR=%.6f NDCG@5=%.6f Hit@1=%.6f (n=%s)",
                ep + 1,
                val_metrics["MRR"],
                val_metrics["NDCG@5"],
                val_metrics["Hit@1"],
                val_metrics["n"],
            )
            score = float(val_metrics[sel_key])
            if score > best_score:
                best_score = score
                best_epoch = ep + 1
                best_metrics = dict(val_metrics)
                ckpt_payload["val_metrics"] = best_metrics
                best_path = os.path.join(args.model_dir, "best.pt")
                torch.save(ckpt_payload, best_path)
                logging.info("best.pt updated epoch=%s %s=%.6f", best_epoch, sel_key, best_score)

        epoch_logs.append({"epoch": ep + 1, "loss": loss_sum, "val": val_metrics})

    summary = {
        "backend": "clip",
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "selection_metric": sel_key,
        "epoch_logs": epoch_logs,
        "clip_image_cache": img_cache,
        "clip_title_cache": title_cache,
        "clip_hidden_size": hidden,
    }
    return summary


def test(args):
    utils.init_hvd_cuda(False, args.enable_gpu)
    from transformers import AutoTokenizer

    ckpt_path = None
    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        if ckpt_path is None and args.load_ckpt_name != "best.pt":
            ckpt_path = utils.get_checkpoint(args.model_dir, "best.pt")
    else:
        ckpt_path = utils.get_checkpoint(args.model_dir, "best.pt") or utils.latest_checkpoint(
            args.model_dir
        )
    assert ckpt_path is not None, "No ckpt found"
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    news, news_index = read_news_bert(_news_tsv_path(args), args, tokenizer, "test")
    del news

    if getattr(args, "clip_image_cache", None) is None:
        args.clip_image_cache = checkpoint.get("clip_image_cache")
    if getattr(args, "clip_title_cache", None) is None:
        args.clip_title_cache = checkpoint.get("clip_title_cache")
    hidden = int(
        getattr(args, "clip_hidden_size", None)
        or checkpoint.get("clip_hidden_size")
        or 1024
    )
    clip_t, clip_v, img_cache, title_cache = _load_clip_tables(args, news_index)
    model = mmrec_clip(clip_t, clip_v, hidden_size=hidden, dropout=0.0)
    if args.enable_gpu:
        model = model.cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(
        "Model loaded from %s epoch=%s val=%s",
        ckpt_path,
        checkpoint.get("epoch"),
        checkpoint.get("val_metrics"),
    )
    del checkpoint

    model.eval()
    news_scoring_t, news_scoring_v = model.encode_news_tables()
    test_data_dir = os.path.join(args.root_data_dir, f"{args.dataset}/{args.test_dir}")
    metrics = score_impressions_clip(
        args, model, news_index, news_scoring_t, news_scoring_v, test_data_dir
    )
    logging.info(
        "TEST MRR=%.6f NDCG@5=%.6f Hit@1=%.6f (n=%s)",
        metrics["MRR"],
        metrics["NDCG@5"],
        metrics["Hit@1"],
        metrics["n"],
    )
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.log_dir, "final_result.txt"), "w", encoding="utf-8") as fout:
        fout.write(
            f"MRR={metrics['MRR']:.6f}\nNDCG@5={metrics['NDCG@5']:.6f}\n"
            f"Hit@1={metrics['Hit@1']:.6f}\nn={metrics['n']}\n"
            f"backend=clip\nimage={img_cache}\ntitle={title_cache}\n"
        )
    return metrics
