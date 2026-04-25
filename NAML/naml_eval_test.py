#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
프리트레인 NAML 가중치를 로드해 테스트셋 성능을 비교:
- 실제본문
- 지정한 기대본문 폴더

지표: MRR, NDCG@5, Hit@1 (NAML 기존 3개 지표)

python NAML/naml_eval_test.py \
  --weights saved_models/Adressa_2000/NAML_adressa_2000_actual.h5 \
  --tune-log saved_models/Adressa_2000/naml_tune_actual_log.json \
  --actual-only \
  --mind-dataset-subdir Adressa_2000

python NAML/naml_eval_test.py \
  --expected-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
  --weights saved_models/NAML_mind_2000_expected.h5 \
  --tune-log saved_models/naml_tune_expected_log.json \
  --mind-dataset-subdir MIND_2000 \
  --expected-body-first-n-sentences 3 \
  --mind-test-tsv dataset/MIND_2000/MIND_test_(2000)_final.tsv

# 학습(튜닝)과 동일하게 평가에서도 앞 N문장만 쓰려면 명시: --expected-body-first-n-sentences 3
# 미지정이면 평가·OOV는 전체 기대본문 문자열 사용(튜닝 로그의 N은 자동 반영하지 않음)

# naml_tune_actual.py 로 튜닝·저장한 가중치는 CNN 폭 등 구조가 다를 수 있음 → 같은 로그를 넘겨야 함:
#   --tune-log saved_models/naml_tune_actual_log.json
#
# 다른 테스트 split TSV (예: 후반 절반):
#   --mind-test-tsv dataset/MIND_2000/MIND_test_(2000)_final.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from nltk.tokenize import word_tokenize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

# naml_common 은 import 시점에 MIND_DATASET_SUBDIR·뉴스/TSV 파일명을 고정한다.
# main() 이후에 --mind-dataset-subdir 를 넣어도 이미 늦으므로, 배치 제너레이터 import 전에 argv 반영.
from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()

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


def _token_oov_counts(word_dict: dict, text: str) -> Tuple[int, int, int]:
    """NAML 제너레이터와 동일: lower + word_tokenize, word in word_dict 이면 in-vocab."""
    if not text or not str(text).strip():
        return 0, 0, 0
    tokens = word_tokenize(str(text).lower())
    total = len(tokens)
    in_vocab = sum(1 for w in tokens if w in word_dict)
    oov = total - in_vocab
    return total, in_vocab, oov


def aggregate_oov_from_texts(word_dict: dict, texts: List[str]) -> Dict[str, float]:
    tot = inv = oov = 0
    for t in texts:
        a, b, c = _token_oov_counts(word_dict, t)
        tot += a
        inv += b
        oov += c
    rate = (oov / tot) if tot else 0.0
    return {
        "total_tokens": tot,
        "in_vocab_tokens": inv,
        "oov_tokens": oov,
        "oov_token_rate": round(rate, 6),
    }


def load_news_id_to_body_from_tsv(news_tsv: str) -> Dict[str, str]:
    """뉴스 TSV: news_id -> 원문 body (5번째 칼럼). Adressa 등 헤더 행은 건너뜀."""
    from naml_dataset_env import news_tsv_skiprows

    out: Dict[str, str] = {}
    if not news_tsv or not os.path.isfile(news_tsv):
        return out
    with open(news_tsv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    skip = news_tsv_skiprows(news_tsv)
    for line in lines[skip:]:
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            out[parts[0]] = parts[4]
    return out


# build_naml_models 튜닝 인자와 동일 키 (기본값 = naml_model_builder 기본)
_DEFAULT_ARCH: Dict[str, float | int] = {
    "dropout_rate": 0.3,
    "cnn_filters": 400,
    "cnn_kernel_size": 3,
    "attention_dense_dim": 200,
    "category_emb_dim": 50,
}
_ARCH_KEYS = tuple(_DEFAULT_ARCH.keys())


def _arch_from_tune_log(log_path: str) -> Dict[str, float | int]:
    """naml_tune_actual_log.json 의 global_best_hparams 에서 아키텍처만 추출."""
    out: Dict[str, float | int] = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out
    gb = data.get("global_best_hparams")
    if not isinstance(gb, dict):
        return out
    for k in _ARCH_KEYS:
        if k in gb:
            out[k] = gb[k]
    return out


def _expected_prep_from_tune_log(log_path: str) -> Dict[str, object]:
    """
    naml_tune_expected.py 로그에서 기대본문 전처리 재현용 설정 추출.
    (어휘·임베딩 행 수 맞추기용 train/test 디렉터리)
    평가 시 앞 N문장 컷은 이 로그 값을 쓰지 않고 --expected-body-first-n-sentences 로만 적용한다.
    반환 예:
      {
        "use_expected_body": True/False,
        "expected_train_dir": "...",
        "expected_test_dir": "...",
        "expected_body_first_n_sentences": 3,  # 기록용; eval_test_expected 는 자동 미적용
      }
    """
    out: Dict[str, object] = {"use_expected_body": False}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out
    if not isinstance(data, dict):
        return out
    out["use_expected_body"] = bool(data.get("use_expected_body", False))
    out["expected_train_dir"] = data.get("expected_train_dir")
    out["expected_test_dir"] = data.get("expected_test_dir")
    out["expected_body_first_n_sentences"] = data.get("expected_body_first_n_sentences")
    return out


def _resolve_dir_like_training(path_option: str | None) -> str | None:
    """
    NAML 학습 스크립트들과 동일 규칙:
    절대경로 -> _ROOT 상대 -> _ROOT/body_generation/output 상대 순으로 해석.
    """
    if not path_option or not str(path_option).strip():
        return None
    p = str(path_option).strip()
    if os.path.isabs(p) and os.path.isdir(p):
        return os.path.normpath(p)
    cand = os.path.normpath(str(_ROOT / p))
    if os.path.isdir(cand):
        return cand
    legacy = os.path.normpath(str(_ROOT / "body_generation" / "output" / p))
    if os.path.isdir(legacy):
        return legacy
    return None


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
    parser.add_argument("--expected-dir", type=str, default=None, help="기대본문 폴더 (user_*/news_*.json)")
    parser.add_argument(
        "--actual-only",
        action="store_true",
        help="기대본문 평가는 생략하고 실제본문 성능만 측정",
    )
    parser.add_argument("--weights", type=str, default="saved_models/NAML_mind_2000.h5")
    parser.add_argument(
        "--tune-log",
        type=str,
        default=None,
        help="naml_tune_actual_log.json 경로. global_best_hparams 로 CNN/드롭아웃 등 그래프를 맞춤 (튜닝 가중치 로드 시 권장)",
    )
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--cnn-filters", type=int, default=None)
    parser.add_argument("--cnn-kernel-size", type=int, default=None)
    parser.add_argument("--attention-dense-dim", type=int, default=None)
    parser.add_argument("--category-emb-dim", type=int, default=None)
    parser.add_argument("--mind-dataset-subdir", type=str, default=None)
    parser.add_argument(
        "--mind-test-tsv",
        type=str,
        default=None,
        help="테스트 impression TSV (미지정이면 naml_common 기본, 예: MIND_test_(2000).tsv). "
        "예: dataset/MIND_2000/MIND_test_(2000)_final.tsv 또는 MIND_test_(2000)_final.tsv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--disable-auto-expected-preprocess-from-tune-log",
        action="store_true",
        help="기본 자동 동작(튜닝 로그의 expected-body 전처리 설정 재사용)을 비활성화",
    )
    parser.add_argument(
        "--expected-body-first-n-sentences",
        type=int,
        default=None,
        metavar="N",
        help="기대본문 평가·OOV 집계 시 앞 N문장만 사용 (0=전체). "
        "미지정이면 문장 컷 없음(전체 본문). --tune-log의 값은 자동 적용하지 않음",
    )
    args = parser.parse_args()

    if args.mind_dataset_subdir:
        os.environ["MIND_DATASET_SUBDIR"] = args.mind_dataset_subdir

    import naml_common as _naml_common
    from naml_common import (
        MIND_NEWS_FILENAME,
        MIND_TEST_FILENAME,
        SEED,
        clip_expected_body_to_first_sentences,
        get_embedding,
        mind_data_path,
        preprocess_news_file,
        preprocess_user_file,
    )
    from naml_model_builder import build_naml_models

    mind_test_tsv_override: str | None = None
    if args.mind_test_tsv and str(args.mind_test_tsv).strip():
        mt = str(args.mind_test_tsv).strip()
        if os.path.isabs(mt) and os.path.isfile(mt):
            mind_test_tsv_override = os.path.normpath(mt)
        else:
            cand = os.path.normpath(str(_ROOT / mt))
            if os.path.isfile(cand):
                mind_test_tsv_override = cand
            else:
                cand2 = mind_data_path(os.path.basename(mt))
                if os.path.isfile(cand2):
                    mind_test_tsv_override = cand2
        if mind_test_tsv_override is None:
            print(f"오류: --mind-test-tsv 파일을 찾을 수 없습니다: {args.mind_test_tsv}", file=sys.stderr)
            sys.exit(1)
        print(f"MIND 테스트 TSV (--mind-test-tsv): {mind_test_tsv_override}", flush=True)
    effective_test_tsv = mind_test_tsv_override or mind_data_path(MIND_TEST_FILENAME)
    print(f"실제 평가 test TSV: {effective_test_tsv}", flush=True)

    weights_path = _ROOT / args.weights
    if not weights_path.is_file():
        print(f"오류: 가중치 파일 없음: {weights_path}")
        sys.exit(1)

    expected_dir_abs: str | None = None
    if not args.actual_only:
        if not args.expected_dir:
            print("오류: 기대본문 평가를 하려면 --expected-dir 가 필요합니다. (또는 --actual-only 사용)")
            sys.exit(1)
        expected_dir_abs = (
            os.path.normpath(str(_ROOT / args.expected_dir)) if not os.path.isabs(args.expected_dir) else args.expected_dir
        )
        if not os.path.isdir(expected_dir_abs):
            print(f"오류: 기대본문 폴더 없음: {expected_dir_abs}")
            sys.exit(1)

    np.random.seed(SEED)
    expected_bodies: Dict[str, str] = {}
    if expected_dir_abs is not None:
        expected_bodies = load_expected_bodies_from_dir(expected_dir_abs)
        print(f"기대본문 로드: {len(expected_bodies)}개 ({expected_dir_abs})")
    else:
        print("기대본문 평가: 비활성화 (--actual-only)")

    # tune-log 기반 자동 전처리 복원 (기대본문 튜닝 가중치의 embedding 크기 불일치 방지)
    prep_expected_train = None
    prep_expected_test = None
    if args.tune_log and not args.disable_auto_expected_preprocess_from_tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            prep_cfg = _expected_prep_from_tune_log(tl)
            if bool(prep_cfg.get("use_expected_body", False)):
                tr = _resolve_dir_like_training(str(prep_cfg.get("expected_train_dir") or ""))
                te = _resolve_dir_like_training(str(prep_cfg.get("expected_test_dir") or ""))
                if tr and te:
                    prep_expected_train = load_expected_bodies_from_dir(tr)
                    prep_expected_test = load_expected_bodies_from_dir(te)
                    print(
                        "튜닝 로그 기반 자동 전처리 적용: "
                        f"use_expected_body=True, train={len(prep_expected_train)} ({tr}), "
                        f"test={len(prep_expected_test)} ({te}) "
                        "(평가 시 앞 N문장 컷은 --expected-body-first-n-sentences 로만 지정)",
                        flush=True,
                    )
                else:
                    print(
                        "경고: tune-log 에 use_expected_body=True 이지만 expected_train/test_dir 해석 실패. "
                        "전처리는 실제본문 기준으로 진행됩니다.",
                        flush=True,
                    )

    if args.expected_body_first_n_sentences is not None:
        _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = max(0, int(args.expected_body_first_n_sentences))
        print(
            (
                f"기대본문 문장 컷: 앞 {_naml_common.EXPECTED_BODY_FIRST_N_SENTENCES}문장만 사용 (CLI)"
                if _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES > 0
                else "기대본문 문장 컷: 전체 문장 (CLI에서 0 지정)"
            ),
            flush=True,
        )
    else:
        _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = 0
        print(
            "기대본문 문장 컷: 미지정 → 평가·OOV는 전체 본문 (튜닝 로그의 N은 자동 반영하지 않음)",
            flush=True,
        )

    # 가중치의 embedding 행 수 = len(word_dict)와 일치해야 함.
    # 기대본문 튜닝 가중치면 위 자동 전처리로 same word_dict 재현을 시도한다.
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=prep_expected_train,
        expected_bodies_test=prep_expected_test,
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
        expected_bodies_train=prep_expected_train,
        expected_bodies_test=prep_expected_test,
        word_dict=word_dict,
        test_file=mind_test_tsv_override,
    )

    embedding_mat = get_embedding(word_dict)

    arch: Dict[str, float | int] = dict(_DEFAULT_ARCH)
    if args.tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            loaded = _arch_from_tune_log(tl)
            arch.update(loaded)
            print(f"튜닝 로그에서 아키텍처: {tl} → {loaded or '(global_best_hparams 없음, 기본값)'}")
        else:
            print(f"경고: --tune-log 파일 없음: {tl}", flush=True)
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
    print(f"build_naml_models 아키텍처: {arch}", flush=True)

    built = build_naml_models(
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
    model = built["model"]
    model_test = built["model_test"]
    try:
        model.load_weights(str(weights_path))
    except Exception as e:
        print(
            "\n오류: 가중치 로드 실패. 가중치가 naml_tune_actual 등으로 튜닝된 모델이라면 "
            "그때의 global_best_hparams와 동일하게 그래프를 맞춰야 합니다.\n"
            "  예: --tune-log saved_models/naml_tune_actual_log.json\n"
            "또는 --cnn-filters 등으로 수동 지정.\n",
            flush=True,
        )
        raise e

    news_index_reverse = {v: k for k, v in news_index.items()}
    news_tsv_path = mind_data_path(MIND_NEWS_FILENAME)
    raw_bodies_by_news_id = load_news_id_to_body_from_tsv(news_tsv_path)

    # 실제본문 OOV: MIND_news.tsv 원문을 기대본문과 동일 규칙으로 토큰화
    actual_bodies_unique: List[str] = []
    seen_nids = set()
    actual_bodies_per_slot: List[str] = []
    for i in range(len(all_test_pn)):
        idx = int(all_test_pn[i])
        if idx == 0:
            continue
        nid = news_index_reverse.get(idx, "") or ""
        body = raw_bodies_by_news_id.get(str(nid).strip(), "") if nid else ""
        actual_bodies_per_slot.append(body)
        if nid and str(nid).strip() not in seen_nids:
            seen_nids.add(str(nid).strip())
            actual_bodies_unique.append(body)
    oov_actual_unique = aggregate_oov_from_texts(word_dict, actual_bodies_unique)
    oov_actual_slots = aggregate_oov_from_texts(word_dict, actual_bodies_per_slot)

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
    print("테스트셋 예측 중... (실제본문)")
    score_real = model_test.predict(testgen_real, steps=test_steps, verbose=0)

    metrics_real = calc_metrics_from_scores(score_real, all_test_label, all_test_index)
    metrics_exp = None
    if not args.actual_only:
        testgen_exp = generate_batch_data_test(
            word_dict, news_words, news_body, news_v, news_sv, news_index,
            all_test_pn, all_test_label, all_test_id, all_test_user_pos, bs,
            expected_bodies=expected_bodies,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse,
        )
        print("테스트셋 예측 중... (기대본문)")
        score_exp = model_test.predict(testgen_exp, steps=test_steps, verbose=0)
        metrics_exp = calc_metrics_from_scores(score_exp, all_test_label, all_test_index)

    # 커버리지
    total_slots = 0
    matched_slots = 0
    oov_all_json = None
    oov_test_slots = None
    if not args.actual_only:
        for i in range(len(all_test_pn)):
            if int(all_test_pn[i]) == 0:
                continue
            total_slots += 1
            k = _norm_expected_body_key(all_test_userid_str[i], all_test_newsid_str[i])
            if k in expected_bodies:
                matched_slots += 1

        # OOV: 기대본문을 평가와 동일하게 문장 컷 후 NAML과 동일 토큰화
        n_sent_oov = int(_naml_common.EXPECTED_BODY_FIRST_N_SENTENCES)
        exp_for_oov = [
            clip_expected_body_to_first_sentences(t, n_sent_oov) or ""
            for t in expected_bodies.values()
        ]
        oov_all_json = aggregate_oov_from_texts(word_dict, exp_for_oov)
        texts_test_matched: List[str] = []
        for i in range(len(all_test_pn)):
            if int(all_test_pn[i]) == 0:
                continue
            k = _norm_expected_body_key(all_test_userid_str[i], all_test_newsid_str[i])
            if k in expected_bodies:
                texts_test_matched.append(
                    clip_expected_body_to_first_sentences(expected_bodies[k], n_sent_oov) or ""
                )
        oov_test_slots = aggregate_oov_from_texts(word_dict, texts_test_matched)

    print("\n=== 테스트셋 성능 비교 ===")
    print(
        f"[실제본문]   MRR={metrics_real['MRR']:.6f}  "
        f"NDCG@5={metrics_real['NDCG@5']:.6f}  "
        f"Hit@1={metrics_real['Hit@1']:.6f}"
    )
    if metrics_exp is not None:
        print(
            f"[기대본문]   MRR={metrics_exp['MRR']:.6f}  "
            f"NDCG@5={metrics_exp['NDCG@5']:.6f}  "
            f"Hit@1={metrics_exp['Hit@1']:.6f}"
        )
        match_rate = (matched_slots / total_slots) if total_slots else 0.0
        print(
            f"[매칭율]     {matched_slots}/{total_slots} ({match_rate:.2%})"
        )
    oa = oov_all_json
    ot = oov_test_slots
    aru = oov_actual_unique
    ars = oov_actual_slots
    if oa is not None and ot is not None:
        print(
            f"[OOV 기대본문] JSON 항목 각 1회: {oa['oov_tokens']}/{oa['total_tokens']} "
            f"({oa['oov_token_rate']:.2%})"
        )
        print(
            f"[OOV 기대본문] 테스트에서 기대본문 쓴 슬롯만(반복 합산): "
            f"{ot['oov_tokens']}/{ot['total_tokens']} ({ot['oov_token_rate']:.2%})"
        )
    print(
        f"[OOV 실제본문] MIND TSV 테스트 후보 고유 뉴스 각 1회: "
        f"{aru['oov_tokens']}/{aru['total_tokens']} ({aru['oov_token_rate']:.2%})"
    )
    print(
        f"[OOV 실제본문] 테스트 슬롯(행마다 반복): "
        f"{ars['oov_tokens']}/{ars['total_tokens']} ({ars['oov_token_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
