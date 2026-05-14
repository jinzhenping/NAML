#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
"""
테스트 TSV를 `preprocess_user_file` 과 동일하게 읽은 뒤, `all_test_index` 순서(= 전처리에서
살아남은 impression 순서, 원본 TSV의 위에서부터 스킵되지 않은 행 순서와 동일)로
세션마다 모델 1위 후보를 기록한다.

출력 CSV 형식은 `NAML/MIND_prediction_result.csv` 와 동일:
  id,generate,real
  - id: impression 순번 (0부터)
  - generate: 1위 예측 뉴스 `Category: ..., Subcategory: ..., Title: ...`
  - real: 클릭(양성) 뉴스 동일 형식

주의: `preprocess_user_file` 은 후보 순서를 세션마다 random.shuffle 하므로,
`naml_eval_test.py` 와 동일한 결과를 내려면 같은 `--seed`(기본 naml_common.SEED)로
전처리 전에 시드를 고정해야 한다.

예:
  python NAML/naml_export_test_top1_csv.py \
  --weights saved_models/MIND_2000/NAML_mind_2000.h5 \
  --tune-log saved_models/MIND_2000/naml_tune_actual_log.json \
  --mind-dataset-subdir MIND_2000 \
  --mind-test-tsv dataset/MIND_2000/MIND_test_(2000)_final.tsv \
  --out-csv NAML/MIND_prediction_result_export.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()


def _resolve_test_tsv(mt: str) -> str:
    from naml_common import MIND_TEST_FILENAME, mind_data_path

    mt = str(mt).strip()
    if os.path.isabs(mt) and os.path.isfile(mt):
        return os.path.normpath(mt)
    cand = os.path.normpath(str(_ROOT / mt))
    if os.path.isfile(cand):
        return cand
    cand2 = mind_data_path(os.path.basename(mt))
    if os.path.isfile(cand2):
        return cand2
    raise FileNotFoundError(f"--mind-test-tsv 파일을 찾을 수 없습니다: {mt}")


def _fmt_news_line(meta: Dict[str, tuple[str, str, str]], news_id: str) -> str:
    nid = str(news_id).strip() if news_id else ""
    if not nid or nid not in meta:
        return ""
    cat, sub, title = meta[nid]
    return f"Category: {cat}, Subcategory: {sub}, Title: {title}"


def _load_news_display_meta(news_tsv: str) -> Dict[str, tuple[str, str, str]]:
    """news_id -> (category, subcategory, raw_title_string)"""
    from naml_dataset_env import news_tsv_skiprows

    out: Dict[str, tuple[str, str, str]] = {}
    if not os.path.isfile(news_tsv):
        return out
    with open(news_tsv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    skip = news_tsv_skiprows(news_tsv)
    for line in lines[skip:]:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        nid = parts[0]
        if str(nid).strip().lower() in ("news_id", "clicked_news", "id"):
            continue
        cat = parts[1] if parts[1] else "None"
        sub = parts[2] if parts[2] else "None"
        title = parts[3] if len(parts) > 3 else ""
        out[nid] = (cat, sub, title)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="테스트 세션별 모델 1위 후보를 MIND_prediction_result 형식 CSV로 저장")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--tune-log", type=str, default=None)
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument("--mind-test-tsv", type=str, required=True, help="평가용 behaviors TSV (전처리 test_file)")
    ap.add_argument("--out-csv", type=str, required=True, help="저장할 CSV 경로 (프로젝트 루트 기준 상대 가능)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=None, help="미지정 시 naml_common.SEED")
    ap.add_argument(
        "--max-history-clicks",
        type=int,
        default=None,
        metavar="N",
    )
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--cnn-filters", type=int, default=None)
    ap.add_argument("--cnn-kernel-size", type=int, default=None)
    ap.add_argument("--attention-dense-dim", type=int, default=None)
    ap.add_argument("--category-emb-dim", type=int, default=None)
    ap.add_argument(
        "--cq-user-encoder",
        action="store_true",
        help="후보 쿼리 사용자 인코더 모델(build_naml_models_candidate_query_user). 가중치·tune-log 가 CQ와 맞아야 함",
    )
    ap.add_argument(
        "--disable-auto-expected-preprocess-from-tune-log",
        action="store_true",
        help="naml_eval_test.py 와 동일: 튜닝 로그 기반 기대본문 어휘 확장 비활성화",
    )
    args = ap.parse_args()

    if args.mind_dataset_subdir:
        os.environ["MIND_DATASET_SUBDIR"] = str(args.mind_dataset_subdir)

    import tensorflow as tf

    from naml_eval_test import (
        _DEFAULT_ARCH,
        _arch_from_tune_log,
        _expected_prep_from_tune_log,
        _resolve_dir_like_training,
        load_expected_bodies_from_dir,
    )
    from naml_common import (
        MIND_NEWS_FILENAME,
        SEED,
        get_embedding,
        mind_data_path,
        preprocess_news_file,
        preprocess_user_file,
    )
    from naml_model_builder import build_naml_models, build_naml_models_candidate_query_user
    from naml_batch_generators import generate_batch_data_test

    seed = int(args.seed) if args.seed is not None else int(SEED)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    test_tsv = _resolve_test_tsv(args.mind_test_tsv)
    print(f"테스트 TSV: {test_tsv}", flush=True)

    weights_path = _ROOT / args.weights
    if not weights_path.is_file():
        print(f"오류: 가중치 없음: {weights_path}", file=sys.stderr)
        sys.exit(1)

    prep_train = None
    prep_test = None
    if args.tune_log and not args.disable_auto_expected_preprocess_from_tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            prep_cfg = _expected_prep_from_tune_log(tl)
            if bool(prep_cfg.get("use_expected_body", False)):
                tr = _resolve_dir_like_training(str(prep_cfg.get("expected_train_dir") or ""))
                te = _resolve_dir_like_training(str(prep_cfg.get("expected_test_dir") or ""))
                if tr and te:
                    prep_train = load_expected_bodies_from_dir(tr)
                    prep_test = load_expected_bodies_from_dir(te)
                    print(
                        f"tune-log 기반 어휘 확장: train={len(prep_train)} test={len(prep_test)}",
                        flush=True,
                    )

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=prep_train,
        expected_bodies_test=prep_test,
    )
    (
        _u,
        _tpn,
        _tl,
        _tid,
        all_test_pn,
        all_test_label,
        all_test_id,
        _up,
        all_test_user_pos,
        all_test_index,
        _a,
        _b,
        _tsu,
        _tsn,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=prep_train,
        expected_bodies_test=prep_test,
        word_dict=word_dict,
        test_file=test_tsv,
    )

    embedding_mat = get_embedding(word_dict)
    news_index_reverse = {v: k for k, v in news_index.items()}
    news_tsv_path = mind_data_path(MIND_NEWS_FILENAME)
    display_meta = _load_news_display_meta(news_tsv_path)
    print(f"뉴스 메타: {len(display_meta)}개 ({news_tsv_path})", flush=True)

    arch: Dict[str, Any] = dict(_DEFAULT_ARCH)
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

    _kw = dict(
        dropout_rate=float(arch["dropout_rate"]),
        cnn_filters=int(arch["cnn_filters"]),
        cnn_kernel_size=int(arch["cnn_kernel_size"]),
        attention_dense_dim=int(arch["attention_dense_dim"]),
        category_emb_dim=int(arch["category_emb_dim"]),
    )

    if args.cq_user_encoder:
        built = build_naml_models_candidate_query_user(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            float(args.learning_rate),
            clear_session=True,
            **_kw,
        )
    else:
        built = build_naml_models(
            word_dict,
            embedding_mat,
            category,
            subcategory,
            float(args.learning_rate),
            **_kw,
        )
    model_test = built["model_test"]
    try:
        model_test.load_weights(str(weights_path))
    except Exception as e:
        print(
            "오류: 가중치 로드 실패. --tune-log / --cq-user-encoder 가 가중치와 일치하는지 확인하세요.",
            file=sys.stderr,
        )
        raise e

    n = len(all_test_id)
    bs = max(1, int(args.batch_size))
    steps = (n + bs - 1) // bs
    gen = generate_batch_data_test(
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
    print(f"예측 중... rows={n}, batch_size={bs}, steps={steps}", flush=True)
    scores = model_test.predict(gen, steps=steps, verbose=1)

    out_path = str(_ROOT / args.out_csv) if not os.path.isabs(args.out_csv) else args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows_out: List[tuple[int, str, str]] = []
    imp_id = 0
    for m in all_test_index:
        start, end = int(m[0]), int(m[1])
        if end > len(scores):
            continue
        sess_scores = scores[start:end, 0]
        sess_labels = np.asarray(all_test_label[start:end], dtype=np.float32)
        if np.sum(sess_labels) == 0:
            continue
        off = int(np.argmax(sess_scores))
        pred_j = start + off
        pred_nid = str(all_test_newsid_str[pred_j]).strip()
        if not pred_nid:
            pi = int(all_test_pn[pred_j])
            pred_nid = str(news_index_reverse.get(pi, "")).strip() if pi else ""

        true_nid = ""
        for j in range(start, end):
            if float(all_test_label[j]) == 1.0:
                tn = str(all_test_newsid_str[j]).strip()
                if not tn:
                    ti = int(all_test_pn[j])
                    tn = str(news_index_reverse.get(ti, "")).strip() if ti else ""
                true_nid = tn
                break

        gen_s = _fmt_news_line(display_meta, pred_nid)
        real_s = _fmt_news_line(display_meta, true_nid)
        rows_out.append((imp_id, gen_s, real_s))
        imp_id += 1

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "generate", "real"])
        for rid, gs, rs in rows_out:
            w.writerow([rid, gs, rs])

    print(f"저장 완료: {os.path.abspath(out_path)}  (impressions={len(rows_out)})", flush=True)


if __name__ == "__main__":
    main()
