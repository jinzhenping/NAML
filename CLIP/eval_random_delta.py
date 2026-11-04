#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
랜덤 Δ ablation 기대이미지 NAML 최종 test 평가.

  conda activate clip_cu128
  python CLIP/extract_expected_body_text_embeds.py --split test_final --mind-dataset-subdir MIND_2000
  python CLIP/build_random_delta_embeds.py --apply-delta-only --mind-dataset-subdir MIND_2000

  conda activate tf28gpu
  python CLIP/eval_random_delta.py --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR))

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()

from tensorflow.keras import backend as K

from clip_embeddings import (
    default_test_final_tsv,
    random_delta_recipe_paths,
    resolve_project_path,
)
from expected_image import (
    apply_delta_to_text_pairs,
    build_test_candidate_image,
    load_delta,
    load_pair_dict_normed,
    pair_embed_dim,
    save_pair_embed_dict,
)
from naml_common import (
    SEED,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
    sync_max_history_clicks_from_env,
)
import naml_common
from naml_expected_image_model import build_naml_models_expected_image, load_h5_weights_by_name
from train_expected_image import _max_history_from_log, _set_seed, evaluate_metrics
from train_s1_s2 import load_hparams


def ensure_test_final_image_cache(
    paths: dict,
    *,
    expected_image_cache: Optional[str] = None,
    test_src_cache: Optional[str] = None,
    do_l2: bool = True,
) -> str:
    """test_final synthetic image cache가 없으면 Δ_rand로 만든다."""
    cache_path = (
        resolve_project_path(expected_image_cache)
        if expected_image_cache
        else paths["test_final_image"]
    )
    if os.path.isfile(cache_path):
        return cache_path

    delta_path = paths["delta_random"]
    src_path = (
        resolve_project_path(test_src_cache) if test_src_cache else paths["test_final_src"]
    )
    source_flag = f" --source {paths['recipe']}" if paths["recipe"] != "clip_text" else ""
    extract_cmd = (
        "python CLIP/extract_expected_body_prior_embeds.py --split test_final"
        if paths["recipe"] == "prior"
        else "python CLIP/extract_expected_body_text_embeds.py --split test_final"
    )
    if not os.path.isfile(delta_path):
        raise FileNotFoundError(
            f"Δ_rand cache 없음: {delta_path}\n"
            f"python CLIP/build_random_delta_embeds.py{source_flag} "
            f"--mind-dataset-subdir <subdir>"
        )
    if not os.path.isfile(src_path):
        raise FileNotFoundError(
            f"최종 test 기대본문 cache 없음: {src_path}\n"
            "conda activate clip_cu128\n"
            f"{extract_cmd} --mind-dataset-subdir <subdir>"
        )
    delta_rand = load_delta(delta_path)
    src_pairs = load_pair_dict_normed(src_path)
    img_pairs = apply_delta_to_text_pairs(src_pairs, delta_rand, l2=do_l2)
    save_pair_embed_dict(cache_path, img_pairs)
    n_nz = sum(1 for v in img_pairs.values() if np.any(v))
    print(
        f"[eval] built test_final random-delta cache pairs={len(img_pairs)} "
        f"nonzero={n_nz} → {cache_path}",
        flush=True,
    )
    return cache_path


def run_final_test_eval(
    *,
    mind_dataset_subdir: str,
    recipe: str = "clip_text",
    mind_test_tsv: Optional[str] = None,
    weights: Optional[str] = None,
    tune_log: Optional[str] = None,
    expected_image_cache: Optional[str] = None,
    batch_size: int = 64,
    seed: int = SEED,
    max_history_clicks: Optional[int] = None,
    out: Optional[str] = None,
    build_missing_cache: bool = True,
) -> Dict[str, Any]:
    paths = random_delta_recipe_paths(mind_dataset_subdir, recipe)
    weights_arg = weights or str(
        _CLIP_DIR / "saved_models" / mind_dataset_subdir / paths["tuned_weights_name"]
    )
    tune_log_arg = tune_log or str(
        _CLIP_DIR / "saved_models" / mind_dataset_subdir / paths["tune_log_name"]
    )

    argv = ["--mind-dataset-subdir", mind_dataset_subdir]
    hist = max_history_clicks
    if hist is None:
        hist = _max_history_from_log(tune_log_arg)
    if hist is not None:
        argv += ["--max-history-clicks", str(hist)]
    apply_dataset_env_from_argv(argv)
    sync_max_history_clicks_from_env()

    _set_seed(int(seed))
    hp = load_hparams(tune_log_arg)

    weights_path = resolve_project_path(weights_arg)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"가중치 없음: {weights_path}")

    test_tsv = (
        resolve_project_path(mind_test_tsv)
        if mind_test_tsv
        else default_test_final_tsv(mind_dataset_subdir)
    )
    if not os.path.isfile(test_tsv):
        raise FileNotFoundError(f"test TSV 없음: {test_tsv}")

    if build_missing_cache:
        cache_path = ensure_test_final_image_cache(
            paths, expected_image_cache=expected_image_cache
        )
    else:
        cache_path = (
            resolve_project_path(expected_image_cache)
            if expected_image_cache
            else paths["test_final_image"]
        )
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"최종 test random-delta cache 없음: {cache_path}")

    print(
        f"[eval random-delta] recipe={recipe}\n"
        f"[eval random-delta] weights={weights_path}\n"
        f"[eval random-delta] test_tsv={test_tsv}\n"
        f"[eval random-delta] cache={cache_path}",
        flush=True,
    )
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = (
        preprocess_news_file(
            expected_bodies_train=None,
            expected_bodies_test=None,
            expected_bodies_vocab_extra=None,
        )
    )
    (
        _userid_dict,
        _all_train_pn,
        _all_label,
        _all_train_id,
        all_test_pn,
        all_test_label,
        all_test_id,
        _all_user_pos,
        all_test_user_pos,
        all_test_index,
        _c1,
        _c2,
        _tr_u,
        _tr_n,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        test_file=test_tsv,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )
    embedding_mat = get_embedding(word_dict)

    pair_dict = load_pair_dict_normed(cache_path)
    clip_dim = pair_embed_dim(pair_dict)
    cand_img, cov = build_test_candidate_image(
        all_test_userid_str,
        all_test_newsid_str,
        pair_dict,
        clip_dim,
    )
    print(
        f"[eval random-delta] rows={len(all_test_id)} sessions={len(all_test_index)} "
        f"history={naml_common.MAX_HISTORY_CLICKS} clip_dim={clip_dim} coverage={cov}",
        flush=True,
    )

    built = build_naml_models_expected_image(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        hp["learning_rate"],
        clip_dim=clip_dim,
        clear_session=True,
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )
    model = built["model"]
    model_test = built["model_test"]
    load_h5_weights_by_name(model, weights_path)

    metrics = evaluate_metrics(
        model_test,
        all_test_pn,
        all_test_label,
        all_test_id,
        all_test_user_pos,
        all_test_index,
        news_words,
        news_body,
        news_v,
        news_sv,
        cand_img,
        int(batch_size),
    )
    print(
        f"[eval random-delta] MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
        f"Hit@1={metrics['Hit@1']:.6f}",
        flush=True,
    )

    out_path = (
        resolve_project_path(out)
        if out
        else str(_CLIP_DIR / "saved_models" / mind_dataset_subdir / paths["eval_json_name"])
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    summary = {
        "variant": "expected_image_random_delta",
        "recipe": recipe,
        "split": "test_final",
        "weights": os.path.abspath(weights_path),
        "tune_log": os.path.abspath(resolve_project_path(tune_log_arg)),
        "test_tsv": os.path.abspath(test_tsv),
        "expected_image_cache": os.path.abspath(cache_path),
        "delta_random": os.path.abspath(paths["delta_random"]),
        "hparams": hp,
        "max_history_clicks": int(naml_common.MAX_HISTORY_CLICKS),
        "n_rows": int(len(all_test_id)),
        "n_sessions": int(len(all_test_index)),
        "coverage": cov,
        "metrics": metrics,
    }
    summary["out_path"] = os.path.abspath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval random-delta] log → {out_path}", flush=True)
    K.clear_session()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="랜덤 Δ 기대이미지 NAML 최종 test 평가")
    ap.add_argument("--mind-dataset-subdir", type=str, default="MIND_2000")
    ap.add_argument(
        "--recipe",
        type=str,
        default="clip_text",
        choices=["clip_text", "prior"],
    )
    ap.add_argument("--mind-test-tsv", type=str, default=None)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--tune-log", type=str, default=None)
    ap.add_argument("--expected-image-cache", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-history-clicks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--no-build-missing-cache",
        action="store_true",
        help="test_final cache가 없을 때 자동 생성하지 않음",
    )
    args = ap.parse_args()
    run_final_test_eval(
        mind_dataset_subdir=args.mind_dataset_subdir,
        recipe=args.recipe,
        mind_test_tsv=args.mind_test_tsv,
        weights=args.weights,
        tune_log=args.tune_log,
        expected_image_cache=args.expected_image_cache,
        batch_size=args.batch_size,
        seed=args.seed,
        max_history_clicks=args.max_history_clicks,
        out=args.out,
        build_missing_cache=not bool(args.no_build_missing_cache),
    )


if __name__ == "__main__":
    main()
