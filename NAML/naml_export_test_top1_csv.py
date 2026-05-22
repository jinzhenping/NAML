#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
NAML 테스트 impression별 1위 예측 뉴스를 CSV로 저장.

- TSV 한 줄 = CSV 한 줄 (`id` = readlines() 0-based 줄 번호)
- `generate` / `real`: 뉴스 ID가 아니라
  `Category: {cat}, Subcategory: {subcat}, Title: {title}` 문자열
- 전처리 스킵·메타 없음: 해당 칸은 빈 문자열 (`7,,` 형태)


  python NAML/naml_export_test_top1_csv.py \
    --weights saved_models/Adressa_2000/NAML_adressa_2000_actual.h5 \
    --tune-log saved_models/Adressa_2000/naml_tune_actual_log.json \
    --mind-dataset-subdir Adressa_2000 \
    --mind-test-tsv dataset/Adressa_2000/Adressa_test_2000_final.tsv \
    --actual-only \
    --out-csv NAML/export/Adressa_prediction_result_export.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()

from naml_batch_generators import generate_batch_data_test
from naml_eval_test import (
    _DEFAULT_ARCH,
    _arch_from_tune_log,
    _expected_prep_from_tune_log,
    _resolve_dir_like_training,
    calc_metrics_from_scores,
    load_expected_bodies_from_dir,
)


def _tsv_first_candidate(parts: list[str]) -> str:
    if len(parts) < 3:
        return ""
    cands = parts[2].split()
    return cands[0] if cands else ""


def load_news_meta_from_tsv(news_tsv_path: str) -> dict[str, tuple[str, str, str]]:
    """news_id -> (category, subCategory, title) from MIND/Adressa news TSV."""
    from naml_dataset_env import news_tsv_skiprows

    meta: dict[str, tuple[str, str, str]] = {}
    if not news_tsv_path or not os.path.isfile(news_tsv_path):
        return meta
    skip = news_tsv_skiprows(news_tsv_path)
    with open(news_tsv_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < skip:
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            news_id = parts[0].strip()
            category = (parts[1] or "").strip()
            subcategory = (parts[2] or "").strip()
            title = (parts[3] if len(parts) > 3 else "").replace("\t", " ").replace("\n", " ").strip()
            meta[news_id] = (category, subcategory, title)
    return meta


def format_news_cell(news_id: str, meta: dict[str, tuple[str, str, str]]) -> str:
    """`Category: news, Subcategory: x, Title: ...` (메타 없으면 빈 문자열)."""
    if not news_id:
        return ""
    row = meta.get(str(news_id).strip())
    if not row:
        return ""
    category, subcategory, title = row
    return f"Category: {category}, Subcategory: {subcategory}, Title: {title}"


def _resolve_test_tsv(path_opt: str | None, mind_data_path, default_name: str) -> str:
    if path_opt and str(path_opt).strip():
        mt = str(path_opt).strip()
        if os.path.isabs(mt) and os.path.isfile(mt):
            return os.path.normpath(mt)
        cand = os.path.normpath(str(_ROOT / mt))
        if os.path.isfile(cand):
            return cand
        cand2 = mind_data_path(os.path.basename(mt))
        if os.path.isfile(cand2):
            return cand2
        raise FileNotFoundError(f"--mind-test-tsv not found: {path_opt}")
    return mind_data_path(default_name)


def top1_by_session(
    click_score: np.ndarray,
    all_test_index: list,
    all_test_newsid_str: list,
) -> dict[int, str]:
    """session index -> top-1 news_id"""
    out: dict[int, str] = {}
    for sess_i, bounds in enumerate(all_test_index):
        start, end = bounds
        if end > len(click_score):
            continue
        scores = click_score[start:end, 0]
        if len(scores) == 0:
            continue
        top_local = int(np.argmax(scores))
        out[sess_i] = str(all_test_newsid_str[start + top_local])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NAML test top-1 → CSV (Category/Subcategory/Title)")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tune-log", type=str, default="", help="global_best_hparams 아키텍처")
    parser.add_argument("--mind-dataset-subdir", type=str, default="Adressa_2000")
    parser.add_argument("--mind-test-tsv", type=str, default="dataset/Adressa_2000/Adressa_test_2000_final.tsv")
    parser.add_argument("--out-csv", type=str, default="NAML/export/test_top1.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--actual-only", action="store_true", help="실제본문만 (기본 동작과 동일)")
    parser.add_argument(
        "--cq-user-encoder",
        action="store_true",
        help="build_naml_models_candidate_query_user (CQ 교사/학생)",
    )
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--cnn-filters", type=int, default=None)
    parser.add_argument("--cnn-kernel-size", type=int, default=None)
    parser.add_argument("--attention-dense-dim", type=int, default=None)
    parser.add_argument("--category-emb-dim", type=int, default=None)
    parser.add_argument("--print-metrics", action="store_true", help="MRR/NDCG@5/Hit@1 출력")
    args = parser.parse_args()

    if args.mind_dataset_subdir:
        os.environ["MIND_DATASET_SUBDIR"] = args.mind_dataset_subdir

    from naml_common import (
        MIND_NEWS_FILENAME,
        MIND_TEST_FILENAME,
        SEED,
        get_embedding,
        mind_data_path,
        preprocess_news_file,
        preprocess_user_file,
    )
    from naml_model_builder import build_naml_models, build_naml_models_candidate_query_user

    np.random.seed(SEED)

    test_tsv_path = _resolve_test_tsv(args.mind_test_tsv, mind_data_path, MIND_TEST_FILENAME)
    news_tsv_path = mind_data_path(MIND_NEWS_FILENAME)
    news_meta = load_news_meta_from_tsv(news_tsv_path)
    print(f"test TSV: {test_tsv_path}", flush=True)
    print(f"news TSV: {news_tsv_path}  ({len(news_meta)} articles)", flush=True)

    weights_path = _ROOT / args.weights
    if not weights_path.is_file():
        print(f"오류: 가중치 없음: {weights_path}", file=sys.stderr)
        sys.exit(1)

    prep_expected_train = None
    prep_expected_test = None
    if args.tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            prep_cfg = _expected_prep_from_tune_log(tl)
            if bool(prep_cfg.get("use_expected_body", False)):
                tr = _resolve_dir_like_training(str(prep_cfg.get("expected_train_dir") or ""))
                te = _resolve_dir_like_training(str(prep_cfg.get("expected_test_dir") or ""))
                if tr and te:
                    prep_expected_train = load_expected_bodies_from_dir(tr)
                    prep_expected_test = load_expected_bodies_from_dir(te)

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=prep_expected_train,
        expected_bodies_test=prep_expected_test,
    )

    test_line_indices: list[int] = []
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
        _cand_tr,
        _cand_te,
        _all_train_userid_str,
        _all_train_newsid_str,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=prep_expected_train,
        expected_bodies_test=prep_expected_test,
        word_dict=word_dict,
        test_file=test_tsv_path,
        test_impression_tsv_line_index_out=test_line_indices,
    )

    embedding_mat = get_embedding(word_dict)

    arch: dict[str, float | int] = dict(_DEFAULT_ARCH)
    if args.tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            arch.update(_arch_from_tune_log(tl))
    if args.dropout_rate is not None:
        arch["dropout_rate"] = args.dropout_rate
    if args.cnn_filters is not None:
        arch["cnn_filters"] = args.cnn_filters
    if args.cnn_kernel_size is not None:
        arch["cnn_kernel_size"] = args.cnn_kernel_size
    if args.attention_dense_dim is not None:
        arch["attention_dense_dim"] = args.attention_dense_dim
    if args.category_emb_dim is not None:
        arch["category_emb_dim"] = args.category_emb_dim

    build_fn = build_naml_models_candidate_query_user if args.cq_user_encoder else build_naml_models
    built = build_fn(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        args.learning_rate,
        dropout_rate=float(arch["dropout_rate"]),
        cnn_filters=int(arch["cnn_filters"]),
        cnn_kernel_size=int(arch["cnn_kernel_size"]),
        attention_dense_dim=int(arch["attention_dense_dim"]),
        category_emb_dim=int(arch["category_emb_dim"]),
    )
    model_test = built["model_test"]
    model_test.load_weights(str(weights_path))
    print(f"가중치 로드: {weights_path}", flush=True)
    if args.cq_user_encoder:
        print("encoder: CQ (candidate_query_user)", flush=True)

    news_index_reverse = {v: k for k, v in news_index.items()}
    bs = args.batch_size
    n_samples = len(all_test_id)
    test_steps = (n_samples + bs - 1) // bs if n_samples else 0

    testgen = generate_batch_data_test(
        word_dict,
        news_words,
        news_body,
        news_v,
        news_sv,
        news_index,
        all_test_pn,
        all_test_label,
        all_test_id,
        all_test_user_pos,
        bs,
        expected_bodies=None,
        all_userid_str=all_test_userid_str,
        all_newsid_str=all_test_newsid_str,
        news_index_reverse=news_index_reverse,
    )
    print(f"예측 중... samples={n_samples} steps={test_steps}", flush=True)
    click_score = model_test.predict(testgen, steps=test_steps, verbose=0)
    if len(click_score) < n_samples:
        print(
            f"경고: click_score={len(click_score)} < samples={n_samples} — 일부 세션 누락 가능",
            file=sys.stderr,
            flush=True,
        )

    if args.print_metrics:
        metrics = calc_metrics_from_scores(click_score, all_test_label, all_test_index)
        print(
            f"MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
            f"Hit@1={metrics['Hit@1']:.6f}  sessions={metrics['evaluated_sessions']}",
            flush=True,
        )

    top1_by_sess = top1_by_session(click_score, all_test_index, all_test_newsid_str)
    line_to_generate_id: dict[int, str] = {}
    for sess_i, line_idx in enumerate(test_line_indices):
        if sess_i in top1_by_sess:
            line_to_generate_id[line_idx] = top1_by_sess[sess_i]

    with open(test_tsv_path, "r", encoding="utf-8") as f:
        tsv_lines = f.readlines()

    out_path = Path(args.out_csv)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_exported = 0
    n_inferred = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "generate", "real"],
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for line_idx, line in enumerate(tsv_lines):
            parts = line.strip().split("\t")
            real_id = _tsv_first_candidate(parts)
            gen_id = line_to_generate_id.get(line_idx, "")
            generate = format_news_cell(gen_id, news_meta)
            real = format_news_cell(real_id, news_meta)
            if generate:
                n_inferred += 1
            w.writerow({"id": line_idx, "generate": generate, "real": real})
            n_exported += 1

    print(
        f"저장: {out_path}\n"
        f"  TSV 줄 수: {len(tsv_lines)}  CSV 행: {n_exported}  "
        f"추론 포함: {n_inferred}  스킵( generate 빈칸): {n_exported - n_inferred}",
        flush=True,
    )


if __name__ == "__main__":
    main()
