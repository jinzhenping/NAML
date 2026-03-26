#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프리트레인 NAML 가중치를 로드해 테스트셋 성능을 비교:
- 실제본문
- 지정한 기대본문 폴더

지표: MRR, NDCG@5, Hit@1 (NAML 기존 3개 지표)

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/eval_test_expected.py \
  --expected-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
  --weights saved_models/NAML_mind_2000.h5 \
  --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_batch_generators import generate_batch_data_test


def _norm_expected_body_key(uid, nid):
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def load_expected_bodies_from_dir(expected_dir: str) -> Dict[Tuple[str, str], str]:
    expected_bodies: Dict[Tuple[str, str], str] = {}
    if not expected_dir or not os.path.isdir(expected_dir):
        return expected_bodies
    for user_folder in os.listdir(expected_dir):
        user_path = os.path.join(expected_dir, user_folder)
        if not os.path.isdir(user_path) or not user_folder.startswith("user_"):
            continue
        user_id = user_folder.replace("user_", "")
        for filename in os.listdir(user_path):
            if not (filename.startswith("news_") and filename.endswith(".json")):
                continue
            news_id = filename.replace("news_", "").replace(".json", "")
            fpath = os.path.join(user_path, filename)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "generated_body" in data:
                    expected_bodies[_norm_expected_body_key(user_id, news_id)] = data["generated_body"]
            except Exception:
                continue
    return expected_bodies


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    denom = np.sum(y_true)
    if denom == 0:
        return 0.0
    return np.sum(rr_score) / denom


def hit_at_k(y_true, y_score, k=1):
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    y_score = np.array(y_score).flatten()
    y_true = np.array(y_true).flatten()
    sorted_indices = np.argsort(y_score)[::-1]
    top_k_indices = sorted_indices[:k]
    return 1.0 if np.any(y_true[top_k_indices] == 1) else 0.0


def calc_metrics_from_scores(click_score, all_test_label, all_test_index):
    all_mrr: List[float] = []
    all_ndcg: List[float] = []
    all_hit1: List[float] = []
    for m in all_test_index:
        start, end = m
        if end > len(click_score):
            continue
        session_scores = click_score[start:end, 0]
        session_labels = all_test_label[start:end]
        if np.sum(session_labels) == 0:
            continue
        all_mrr.append(mrr_score(session_labels, session_scores))
        all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
        all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
    return {
        "MRR": float(np.mean(all_mrr)) if all_mrr else 0.0,
        "NDCG@5": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
        "Hit@1": float(np.mean(all_hit1)) if all_hit1 else 0.0,
        "evaluated_sessions": len(all_mrr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="프리트레인 NAML 테스트셋 실제본문 vs 기대본문 평가")
    parser.add_argument("--expected-dir", type=str, required=True, help="기대본문 폴더 (user_*/news_*.json)")
    parser.add_argument("--weights", type=str, default="saved_models/NAML_mind_2000.h5")
    parser.add_argument("--mind-dataset-subdir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--out", type=str, default=None, help="결과 JSON 저장 경로(선택)")
    args = parser.parse_args()

    if args.mind_dataset_subdir:
        os.environ["MIND_DATASET_SUBDIR"] = args.mind_dataset_subdir

    from naml_common import SEED, get_embedding, preprocess_news_file, preprocess_user_file
    from naml_model_builder import build_naml_models

    weights_path = _ROOT / args.weights
    if not weights_path.is_file():
        print(f"오류: 가중치 파일 없음: {weights_path}")
        sys.exit(1)

    expected_dir_abs = os.path.normpath(str(_ROOT / args.expected_dir)) if not os.path.isabs(args.expected_dir) else args.expected_dir
    if not os.path.isdir(expected_dir_abs):
        print(f"오류: 기대본문 폴더 없음: {expected_dir_abs}")
        sys.exit(1)

    np.random.seed(SEED)
    expected_bodies = load_expected_bodies_from_dir(expected_dir_abs)
    print(f"기대본문 로드: {len(expected_bodies)}개 ({expected_dir_abs})")

    # 사전학습 가중치의 embedding 행 수 = len(word_dict)와 일치해야 함.
    # 기대본문을 word_dict에 넣으면 어휘 크기가 달라져 load_weights가 실패하므로,
    # 전처리는 뉴스 TSV만 사용하고, 기대본문 토큰은 기존 word_dict에 있는 단어만 반영(OOV는 제외).
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=None,
        expected_bodies_test=None,
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
        _cand_tr,
        _cand_te,
        _all_train_userid_str,
        _all_train_newsid_str,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    embedding_mat = get_embedding(word_dict)
    built = build_naml_models(word_dict, embedding_mat, category, subcategory, args.learning_rate)
    model = built["model"]
    model_test = built["model_test"]
    model.load_weights(str(weights_path))

    news_index_reverse = {v: k for k, v in news_index.items()}
    bs = args.batch_size
    test_steps = (len(all_test_id) + bs - 1) // bs

    testgen_real = generate_batch_data_test(
        word_dict, news_words, news_body, news_v, news_sv, news_index,
        all_test_pn, all_test_label, all_test_id, all_test_user_pos, bs,
        expected_bodies=None,
        all_userid_str=all_test_userid_str,
        all_newsid_str=all_test_newsid_str,
        news_index_reverse=news_index_reverse,
    )
    testgen_exp = generate_batch_data_test(
        word_dict, news_words, news_body, news_v, news_sv, news_index,
        all_test_pn, all_test_label, all_test_id, all_test_user_pos, bs,
        expected_bodies=expected_bodies,
        all_userid_str=all_test_userid_str,
        all_newsid_str=all_test_newsid_str,
        news_index_reverse=news_index_reverse,
    )

    print("테스트셋 예측 중... (실제본문)")
    score_real = model_test.predict(testgen_real, steps=test_steps, verbose=0)
    print("테스트셋 예측 중... (기대본문)")
    score_exp = model_test.predict(testgen_exp, steps=test_steps, verbose=0)

    metrics_real = calc_metrics_from_scores(score_real, all_test_label, all_test_index)
    metrics_exp = calc_metrics_from_scores(score_exp, all_test_label, all_test_index)

    # 커버리지
    total_slots = 0
    matched_slots = 0
    for i in range(len(all_test_pn)):
        if int(all_test_pn[i]) == 0:
            continue
        total_slots += 1
        k = _norm_expected_body_key(all_test_userid_str[i], all_test_newsid_str[i])
        if k in expected_bodies:
            matched_slots += 1
    match_rate = (matched_slots / total_slots) if total_slots else 0.0

    out_obj = {
        "weights": str(weights_path),
        "expected_dir": expected_dir_abs,
        "coverage": {
            "json_entries_loaded": len(expected_bodies),
            "test_candidate_slots_non_padding": total_slots,
            "test_slots_matched_expected_body": matched_slots,
            "test_match_rate": round(match_rate, 6),
        },
        "metrics_real_body": {
            "MRR": round(metrics_real["MRR"], 6),
            "NDCG@5": round(metrics_real["NDCG@5"], 6),
            "Hit@1": round(metrics_real["Hit@1"], 6),
            "evaluated_sessions": metrics_real["evaluated_sessions"],
        },
        "metrics_expected_body": {
            "MRR": round(metrics_exp["MRR"], 6),
            "NDCG@5": round(metrics_exp["NDCG@5"], 6),
            "Hit@1": round(metrics_exp["Hit@1"], 6),
            "evaluated_sessions": metrics_exp["evaluated_sessions"],
        },
    }

    print("\n=== 테스트셋 성능 비교 ===")
    print(
        f"[실제본문]   MRR={out_obj['metrics_real_body']['MRR']:.6f}  "
        f"NDCG@5={out_obj['metrics_real_body']['NDCG@5']:.6f}  "
        f"Hit@1={out_obj['metrics_real_body']['Hit@1']:.6f}"
    )
    print(
        f"[기대본문]   MRR={out_obj['metrics_expected_body']['MRR']:.6f}  "
        f"NDCG@5={out_obj['metrics_expected_body']['NDCG@5']:.6f}  "
        f"Hit@1={out_obj['metrics_expected_body']['Hit@1']:.6f}"
    )
    print(
        f"[매칭율]     {out_obj['coverage']['test_slots_matched_expected_body']}/"
        f"{out_obj['coverage']['test_candidate_slots_non_padding']} "
        f"({out_obj['coverage']['test_match_rate']:.2%})"
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
