# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
NAML 실제 본문만 사용: 하이퍼파라미터 탐색 후 테스트 MRR이 가장 좋았던 가중치를 저장한다.

실행 (저장소 루트에서):
python NAML/naml_tune_actual.py
python NAML/naml_tune_actual.py --trials 24 --epochs-per-trial 10 --seed 42
# 예산 절약: 24조합을 2에폭으로 걸러서 상위 5개만 10에폭 재학습
python NAML/naml_tune_actual.py --two-phase --trials 108 --screening-epochs 3 \
    --refine-top-k 10 --epochs-per-trial 10 \
    --resume-log saved_models/MIND_2000/naml_tune_actual_log.json \
    --out-log saved_models/MIND_2000/naml_tune_actual_log.json \
    --out-weights saved_models/MIND_2000/NAML_mind_2000_actual.h5 \
    --mind-dataset-subdir MIND_2000
# Adressa_2000:
python NAML/naml_tune_actual.py --two-phase --trials 108 --screening-epochs 3 \
    --refine-top-k 10 --epochs-per-trial 10 \
    --resume-log saved_models/Adressa_2000/naml_tune_actual_log.json \
    --out-log saved_models/Adressa_2000/naml_tune_actual_log.json \
    --out-weights saved_models/Adressa_2000/NAML_adressa_2000_actual.h5 \
    --mind-dataset-subdir Adressa_2000

"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from naml_common import (
    SEED,
    MAX_HISTORY_CLICKS,
    npratio,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_model_builder import build_naml_models

NCAND = 1 + npratio


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0


def hit_at_k(y_true, y_score, k=1):
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    y_score = np.array(y_score).flatten()
    y_true = np.array(y_true).flatten()
    top_k_indices = np.argsort(y_score)[::-1][:k]
    return 1.0 if np.any(y_true[top_k_indices] == 1) else 0.0


def generate_batch_data_train_actual(
    all_train_pn,
    all_label,
    all_train_id,
    all_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    batch_size,
):
    max_hist = MAX_HISTORY_CLICKS
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [
        inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))]
        for i in range(len(y) // batch_size + 1)
        if batch_size * i < len(y)
    ]
    while True:
        for batch_indices in batches:
            batch_candidate_splits = [[] for _ in range(NCAND)]
            batch_browsed_news_splits = [[] for _ in range(max_hist)]
            batch_candidate_body_splits = [[] for _ in range(NCAND)]
            batch_browsed_news_body_splits = [[] for _ in range(max_hist)]
            batch_candidate_vertical_splits = [[] for _ in range(NCAND)]
            batch_browsed_news_vertical_splits = [[] for _ in range(max_hist)]
            batch_candidate_subvertical_splits = [[] for _ in range(NCAND)]
            batch_browsed_news_subvertical_splits = [[] for _ in range(max_hist)]
            batch_labels = []
            for idx in batch_indices:
                candidate_indices = np.array(all_train_pn[idx], dtype="int32")
                candidate = news_words[candidate_indices]
                candidate_split = [np.expand_dims(candidate[k], axis=0) for k in range(candidate.shape[0])]
                candidate_body = news_body[candidate_indices]
                candidate_body_split = [
                    np.expand_dims(candidate_body[k], axis=0) for k in range(candidate_body.shape[0])
                ]
                candidate_vertical = news_v[candidate_indices]
                candidate_vertical_split = [
                    np.expand_dims(candidate_vertical[k], axis=0) for k in range(candidate_vertical.shape[0])
                ]
                candidate_subvertical = news_sv[candidate_indices]
                candidate_subvertical_split = [
                    np.expand_dims(candidate_subvertical[k], axis=0) for k in range(candidate_subvertical.shape[0])
                ]
                user_pos_indices = np.array(all_user_pos[idx], dtype="int32")
                browsed_news = news_words[user_pos_indices]
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]
                browsed_news_body_split = [
                    np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])
                ]
                browsed_news_vertical = news_v[user_pos_indices]
                browsed_news_vertical_split = [
                    np.expand_dims(browsed_news_vertical[k], axis=0) for k in range(browsed_news_vertical.shape[0])
                ]
                browsed_news_subvertical = news_sv[user_pos_indices]
                browsed_news_subvertical_split = [
                    np.expand_dims(browsed_news_subvertical[k], axis=0) for k in range(browsed_news_subvertical.shape[0])
                ]
                label = np.array(all_label[idx], dtype="float32")
                for k in range(NCAND):
                    batch_candidate_splits[k].append(candidate_split[k])
                for k in range(len(browsed_news_split)):
                    batch_browsed_news_splits[k].append(browsed_news_split[k])
                for k in range(NCAND):
                    batch_candidate_body_splits[k].append(candidate_body_split[k])
                for k in range(len(browsed_news_body_split)):
                    batch_browsed_news_body_splits[k].append(browsed_news_body_split[k])
                for k in range(NCAND):
                    batch_candidate_vertical_splits[k].append(candidate_vertical_split[k])
                for k in range(len(browsed_news_vertical_split)):
                    batch_browsed_news_vertical_splits[k].append(browsed_news_vertical_split[k])
                for k in range(NCAND):
                    batch_candidate_subvertical_splits[k].append(candidate_subvertical_split[k])
                for k in range(len(browsed_news_subvertical_split)):
                    batch_browsed_news_subvertical_splits[k].append(browsed_news_subvertical_split[k])
                batch_labels.append(label)
            batch_inputs = []
            for k in range(NCAND):
                if batch_candidate_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_splits[k], axis=0))
            for k in range(max_hist):
                if batch_browsed_news_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_splits[k], axis=0))
            for k in range(NCAND):
                if batch_candidate_body_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_body_splits[k], axis=0))
            for k in range(max_hist):
                if batch_browsed_news_body_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_body_splits[k], axis=0))
            for k in range(NCAND):
                if batch_candidate_vertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_vertical_splits[k], axis=0))
            for k in range(max_hist):
                if batch_browsed_news_vertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_vertical_splits[k], axis=0))
            for k in range(NCAND):
                if batch_candidate_subvertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_subvertical_splits[k], axis=0))
            for k in range(max_hist):
                if batch_browsed_news_subvertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_subvertical_splits[k], axis=0))
            yield (batch_inputs, np.array(batch_labels))


def generate_batch_data_test_actual(
    all_test_pn,
    all_test_label,
    all_test_id,
    all_test_user_pos,
    news_words,
    news_body,
    news_v,
    news_sv,
    batch_size,
):
    max_hist = MAX_HISTORY_CLICKS
    inputid = np.arange(len(all_test_label))
    y = all_test_label
    batches = [
        inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))]
        for i in range(len(y) // batch_size + 1)
        if batch_size * i < len(y)
    ]
    while True:
        for batch_indices in batches:
            batch_candidates = []
            batch_browsed_news = [[] for _ in range(max_hist)]
            batch_candidate_body = []
            batch_browsed_news_body = [[] for _ in range(max_hist)]
            batch_candidate_vertical = []
            batch_browsed_news_vertical = [[] for _ in range(max_hist)]
            batch_candidate_subvertical = []
            batch_browsed_news_subvertical = [[] for _ in range(max_hist)]
            batch_labels = []
            for idx in batch_indices:
                news_idx = int(all_test_pn[idx])
                candidate = np.expand_dims(news_words[news_idx], axis=0)
                candidate_body = np.expand_dims(news_body[news_idx], axis=0)
                candidate_vertical = np.expand_dims(news_v[news_idx], axis=0)
                candidate_subvertical = np.expand_dims(news_sv[news_idx], axis=0)
                user_pos_indices = np.array(all_test_user_pos[idx], dtype="int32")
                browsed_news = news_words[user_pos_indices]
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]
                browsed_news_body_split = [
                    np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])
                ]
                browsed_news_vertical = news_v[user_pos_indices]
                browsed_news_vertical_split = [
                    np.expand_dims(browsed_news_vertical[k], axis=0) for k in range(browsed_news_vertical.shape[0])
                ]
                browsed_news_subvertical = news_sv[user_pos_indices]
                browsed_news_subvertical_split = [
                    np.expand_dims(browsed_news_subvertical[k], axis=0) for k in range(browsed_news_subvertical.shape[0])
                ]
                batch_candidates.append(candidate)
                for k in range(max_hist):
                    batch_browsed_news[k].append(browsed_news_split[k])
                batch_candidate_body.append(candidate_body)
                for k in range(max_hist):
                    batch_browsed_news_body[k].append(browsed_news_body_split[k])
                batch_candidate_vertical.append(candidate_vertical)
                for k in range(max_hist):
                    batch_browsed_news_vertical[k].append(browsed_news_vertical_split[k])
                batch_candidate_subvertical.append(candidate_subvertical)
                for k in range(max_hist):
                    batch_browsed_news_subvertical[k].append(browsed_news_subvertical_split[k])
                batch_labels.append(all_test_label[idx])
            batch_inputs = [np.concatenate(batch_candidates, axis=0)]
            for k in range(max_hist):
                batch_inputs.append(np.concatenate(batch_browsed_news[k], axis=0))
            batch_inputs.append(np.concatenate(batch_candidate_body, axis=0))
            for k in range(max_hist):
                batch_inputs.append(np.concatenate(batch_browsed_news_body[k], axis=0))
            batch_inputs.append(np.concatenate(batch_candidate_vertical, axis=0))
            for k in range(max_hist):
                batch_inputs.append(np.concatenate(batch_browsed_news_vertical[k], axis=0))
            batch_inputs.append(np.concatenate(batch_candidate_subvertical, axis=0))
            for k in range(max_hist):
                batch_inputs.append(np.concatenate(batch_browsed_news_subvertical[k], axis=0))
            yield (batch_inputs, np.array(batch_labels, dtype="float32"))


def evaluate_session_metrics(
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
    batch_size,
):
    n = len(all_test_id)
    steps = (n + batch_size - 1) // batch_size
    gen = generate_batch_data_test_actual(
        all_test_pn,
        all_test_label,
        all_test_id,
        all_test_user_pos,
        news_words,
        news_body,
        news_v,
        news_sv,
        batch_size,
    )
    click_score = model_test.predict(gen, steps=steps, verbose=0)
    all_mrr, all_ndcg, all_hit1 = [], [], []
    for m in all_test_index:
        if np.sum(all_test_label[m[0] : m[1]]) == 0:
            continue
        if m[1] > len(click_score):
            continue
        session_scores = click_score[m[0] : m[1], 0]
        session_labels = all_test_label[m[0] : m[1]]
        all_mrr.append(mrr_score(session_labels, session_scores))
        all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
        all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
    if not all_mrr:
        return {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}
    return {
        "MRR": float(np.mean(all_mrr)),
        "NDCG@5": float(np.mean(all_ndcg)),
        "Hit@1": float(np.mean(all_hit1)),
    }


# 튜닝 그리드 (itertools.product 크기 = 고유 조합 개수)
HPARAM_CHOICES: dict[str, list] = {
    "learning_rate": [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    "dropout_rate": [0.2, 0.25, 0.3, 0.35, 0.4],
    "cnn_filters": [256, 300, 400, 512],
    "cnn_kernel_size": [2, 3, 4],
    "attention_dense_dim": [128, 160, 200, 256],
    "category_emb_dim": [32, 50, 64],
}

_HP_KEYS = tuple(HPARAM_CHOICES.keys())


def _hparam_grid_size() -> int:
    p = 1
    for v in HPARAM_CHOICES.values():
        p *= len(v)
    return p


def sample_hparams(rng: random.Random) -> dict:
    """단일 무작위 조합 (그리드에서 항목 하나씩 선택)."""
    return {k: rng.choice(HPARAM_CHOICES[k]) for k in _HP_KEYS}


def plan_hparam_trials(rng: random.Random, n_trials: int) -> list[dict]:
    """
    n_trials개의 하이퍼파라미터 설정을 만든다.
    고유 조합(전체 그리드)을 한 번 섞어서 앞에서부터 쓰므로, trial 수 <= 그리드 크기일 때는 중복이 없다.
    그보다 많으면 나머지는 무작위로 채우며 이때는 중복이 생길 수 있다.
    """
    vals = [HPARAM_CHOICES[k] for k in _HP_KEYS]
    combos = [dict(zip(_HP_KEYS, prod)) for prod in itertools.product(*vals)]
    rng.shuffle(combos)
    grid_n = len(combos)
    if n_trials <= grid_n:
        return combos[:n_trials]
    out = list(combos)
    for _ in range(n_trials - grid_n):
        out.append(sample_hparams(rng))
    return out


def _hp_key(hp: dict) -> tuple:
    return tuple((k, hp[k]) for k in _HP_KEYS)


def _load_seen_hparam_keys_from_log(log_path: str) -> set[tuple]:
    """이전 로그 JSON에서 이미 시도한 hparams 키를 읽는다."""
    seen: set[tuple] = set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("trials", [])
        if not isinstance(rows, list):
            return seen
        for r in rows:
            if not isinstance(r, dict):
                continue
            hp = r.get("hparams")
            if not isinstance(hp, dict):
                continue
            if all(k in hp for k in _HP_KEYS):
                seen.add(_hp_key(hp))
    except Exception as e:
        print(f"경고: resume log를 읽지 못해 skip-seen을 적용하지 않습니다: {e}")
    return seen


def _load_previous_best_from_log(log_path: str) -> tuple[float, dict | None]:
    """이전 로그 JSON에서 최고 MRR 및 해당 hparams를 읽는다."""
    best_mrr = -1.0
    best_hp: dict | None = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        gb = data.get("global_best_mrr", None)
        gh = data.get("global_best_hparams", None)
        if gb is not None:
            try:
                best_mrr = float(gb)
                if isinstance(gh, dict):
                    best_hp = gh
            except Exception:
                pass

        rows = data.get("trials", [])
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                m = r.get("best_mrr_in_trial", None)
                hp = r.get("hparams", None)
                try:
                    mv = float(m)
                except Exception:
                    continue
                if mv > best_mrr:
                    best_mrr = mv
                    best_hp = hp if isinstance(hp, dict) else best_hp
    except Exception as e:
        print(f"경고: 이전 최고 성능 로드 실패: {e}")
    return best_mrr, best_hp


def _load_json_or_none(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def run_trial(
    hp: dict,
    epochs: int,
    batch_size: int,
    word_dict,
    embedding_mat,
    category,
    subcategory,
    news_words,
    news_body,
    news_v,
    news_sv,
    all_train_pn,
    all_label,
    all_train_id,
    all_user_pos,
    all_test_pn,
    all_test_label,
    all_test_id,
    all_test_user_pos,
    all_test_index,
    trial_seed: int,
):
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    tf.random.set_seed(trial_seed)

    built = build_naml_models(
        word_dict,
        embedding_mat,
        category,
        subcategory,
        hp["learning_rate"],
        clear_session=True,
        dropout_rate=hp["dropout_rate"],
        cnn_filters=hp["cnn_filters"],
        cnn_kernel_size=hp["cnn_kernel_size"],
        attention_dense_dim=hp["attention_dense_dim"],
        category_emb_dim=hp["category_emb_dim"],
    )
    model = built["model"]
    model_test = built["model_test"]

    n_train = len(all_train_id)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size

    best_mrr = -1.0
    best_weights = None
    best_metrics = None

    for ep in range(epochs):
        traingen = generate_batch_data_train_actual(
            all_train_pn,
            all_label,
            all_train_id,
            all_user_pos,
            news_words,
            news_body,
            news_v,
            news_sv,
            batch_size,
        )
        model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=0)
        current_metrics = evaluate_session_metrics(
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
            batch_size,
        )
        mrr = current_metrics["MRR"]
        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = model.get_weights()
            best_metrics = current_metrics

    if best_weights is not None:
        model.set_weights(best_weights)
    if best_metrics is None:
        best_metrics = {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0}
    return best_mrr, best_metrics, model


def main():
    ap = argparse.ArgumentParser(description="NAML 실제 본문 하이퍼파라미터 탐색")
    ap.add_argument("--trials", type=int, default=12, help="무작위 시도 횟수")
    ap.add_argument("--epochs-per-trial", type=int, default=8, help="각 시도당 에폭 수")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--mind-dataset-subdir",
        type=str,
        default=None,
        help="dataset/ 하위 폴더 (예: MIND_2000, Adressa_2000). import 전 argv에서도 인식; 미지정 시 MIND_2000",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-weights",
        type=str,
        default=os.path.join(_ROOT, "saved_models", "NAML_mind_2000.h5"),
        help="전역 최고 테스트 MRR을 갱신할 때 저장할 가중치 파일 경로 (.h5). 상대 경로는 실행 시 작업 디렉터리 기준",
    )
    ap.add_argument(
        "--out-log",
        type=str,
        default=os.path.join(_ROOT, "saved_models", "naml_tune_actual_log.json"),
        help="튜닝 로그 JSON 경로. 미지정 시 항상 프로젝트 루트 기준 saved_models/naml_tune_actual_log.json "
        "(--resume-log 와 경로가 같지 않음; 이어붙이려면 보통 동일 파일로 --out-log 지정). 상대 경로는 실행 시 CWD 기준",
    )
    ap.add_argument(
        "--allow-duplicate-hparams",
        action="store_true",
        help="True이면 매 trial 독립 무작위 샘플(중복 가능). 기본은 그리드 셔플로 중복 없이 순회",
    )
    ap.add_argument(
        "--two-phase",
        action="store_true",
        help="1차: 모든 trial을 짧은 에폭으로 스크리닝 → 테스트 MRR 상위 k조합만 2차에서 --epochs-per-trial 로 재학습",
    )
    ap.add_argument(
        "--screening-epochs",
        type=int,
        default=2,
        help="--two-phase 일 때 1차(스크리닝) 에폭 수 (기본 2)",
    )
    ap.add_argument(
        "--refine-top-k",
        type=int,
        default=5,
        help="--two-phase 일 때 1차 이후 2차로 넘길 상위 조합 개수 (기본 5)",
    )
    ap.add_argument(
        "--resume-log",
        type=str,
        default=None,
        help="기존 튜닝 로그 JSON 경로. 지정하면 해당 로그의 hparams를 읽어 이미 시도한 조합은 제외",
    )
    ap.add_argument(
        "--append-log",
        action="store_true",
        help="out-log가 이미 있으면 기존 trials 뒤에 이번 run trials를 이어붙여 저장",
    )
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"데이터 로드 (실제 본문만, {os.environ.get('MIND_DATASET_SUBDIR', 'MIND_2000')} train/test)...")
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=None,
        expected_bodies_test=None,
        expected_bodies_vocab_extra=None,
    )
    (
        _userid_dict,
        all_train_pn,
        all_label,
        all_train_id,
        all_test_pn,
        all_test_label,
        all_test_id,
        all_user_pos,
        all_test_user_pos,
        all_test_index,
        _c1,
        _c2,
        _tr_u,
        _tr_n,
        _te_u,
        _te_n,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    embedding_mat = get_embedding(word_dict)
    print(f"  train 샘플: {len(all_train_id)}, test 행: {len(all_test_id)}, 뉴스: {len(news_index)}")

    out_dir = os.path.dirname(os.path.abspath(args.out_weights))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.dirname(os.path.abspath(args.out_log))
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    print(f"로그 저장 경로: {os.path.abspath(args.out_log)}", flush=True)
    print(f"가중치 저장 경로(전역 최고 MRR 갱신 시): {os.path.abspath(args.out_weights)}", flush=True)

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: dict | None = None
    log_trials = []

    grid_n = _hparam_grid_size()
    seen_hparam_keys: set[tuple] = set()
    resume_log_path: str | None = None
    if args.resume_log:
        resume_log_path = (
            os.path.join(_ROOT, args.resume_log) if not os.path.isabs(args.resume_log) else args.resume_log
        )
        if os.path.isfile(resume_log_path):
            seen_hparam_keys = _load_seen_hparam_keys_from_log(resume_log_path)
            print(f"resume-log 로드: {resume_log_path} (이미 시도한 조합 {len(seen_hparam_keys)}개)")
            prev_best_mrr, prev_best_hp = _load_previous_best_from_log(resume_log_path)
            if prev_best_mrr > global_best_mrr:
                global_best_mrr = prev_best_mrr
                global_best_hp = prev_best_hp
            print(
                f"resume-log 기준 이전 최고 MRR={global_best_mrr:.6f} "
                f"(이 값을 초과할 때만 {args.out_weights} 저장)"
            )
        else:
            print(f"경고: --resume-log 파일이 없어 skip-seen 생략: {resume_log_path}")

    if args.allow_duplicate_hparams:
        trial_hparams = []
        local_keys: set[tuple] = set()
        max_attempts = max(args.trials * 300, 3000)
        attempts = 0
        while len(trial_hparams) < args.trials and attempts < max_attempts:
            attempts += 1
            hp = sample_hparams(rng)
            k = _hp_key(hp)
            if k in seen_hparam_keys or k in local_keys:
                continue
            trial_hparams.append(hp)
            local_keys.add(k)
        if len(trial_hparams) < args.trials:
            print(
                f"경고: unseen 조합 부족으로 {args.trials}개 중 {len(trial_hparams)}개만 생성 "
                f"(allow-duplicate-hparams + skip-seen)"
            )
    else:
        # 전체 고유 조합 순서를 만든 뒤, 이미 시도한 조합을 제외하고 앞에서부터 사용
        all_planned = plan_hparam_trials(rng, grid_n)
        trial_hparams = [hp for hp in all_planned if _hp_key(hp) not in seen_hparam_keys][: args.trials]
        if len(trial_hparams) < args.trials:
            print(
                f"경고: unseen 고유 조합이 부족하여 요청 {args.trials}개 중 {len(trial_hparams)}개만 실행합니다."
            )
        if args.trials > grid_n:
            print(
                f"경고: 고유 그리드 조합은 {grid_n}개인데 trials={args.trials} → "
                f"처음 {grid_n}개는 중복 없음, 이후는 무작위 보충(중복 가능)."
            )
    run_trials = len(trial_hparams)
    if run_trials == 0:
        print("실행할 새 조합이 없습니다. --resume-log 를 바꾸거나 --seed/--trials 설정을 조정하세요.")
        return

    def _one_trial(
        trial_idx: int,
        total_in_phase: int,
        hp: dict,
        epochs: int,
        trial_seed: int,
        phase_label: str,
    ) -> None:
        nonlocal global_best_mrr, global_best_hp
        print(f"\n--- [{phase_label}] {trial_idx + 1}/{total_in_phase}  hparams={hp} ---")
        best_mrr, best_metrics, model = run_trial(
            hp,
            epochs,
            args.batch_size,
            word_dict,
            embedding_mat,
            category,
            subcategory,
            news_words,
            news_body,
            news_v,
            news_sv,
            all_train_pn,
            all_label,
            all_train_id,
            all_user_pos,
            all_test_pn,
            all_test_label,
            all_test_id,
            all_test_user_pos,
            all_test_index,
            trial_seed,
        )
        print(
            f"  trial best MRR (best epoch in trial): {best_mrr:.6f}  | best epoch metrics: "
            f"MRR={best_metrics['MRR']:.6f} NDCG@5={best_metrics['NDCG@5']:.6f} Hit@1={best_metrics['Hit@1']:.6f}"
        )
        log_trials.append(
            {
                "phase": phase_label,
                "hparams": hp,
                "epochs_in_phase": epochs,
                "best_mrr_in_trial": best_mrr,
                "best_epoch": best_metrics,
            }
        )
        if best_mrr > global_best_mrr:
            global_best_mrr = best_mrr
            global_best_hp = dict(hp)
            model.save_weights(args.out_weights)
            print(f"  [전역 갱신] 저장 → {args.out_weights}  MRR={global_best_mrr:.6f}")
        K.clear_session()

    if args.two_phase:
        k = min(args.refine_top_k, run_trials)
        print(
            f"\n[2-phase] 1차: trials={run_trials}, epochs={args.screening_epochs} → "
            f"상위 {k}개를 2차에서 epochs={args.epochs_per_trial} 로 재학습\n"
        )
        screening_rows: list[tuple[float, dict, dict]] = []
        for t in range(run_trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            print(f"\n--- [screening] {t + 1}/{run_trials}  hparams={hp} ---")
            best_mrr, best_metrics, model = run_trial(
                hp,
                args.screening_epochs,
                args.batch_size,
                word_dict,
                embedding_mat,
                category,
                subcategory,
                news_words,
                news_body,
                news_v,
                news_sv,
                all_train_pn,
                all_label,
                all_train_id,
                all_user_pos,
                all_test_pn,
                all_test_label,
                all_test_id,
                all_test_user_pos,
                all_test_index,
                trial_seed,
            )
            print(
                f"  trial best MRR (best epoch in trial): {best_mrr:.6f}  | best epoch metrics: "
                f"MRR={best_metrics['MRR']:.6f} NDCG@5={best_metrics['NDCG@5']:.6f} Hit@1={best_metrics['Hit@1']:.6f}"
            )
            log_trials.append(
                {
                    "phase": "screening",
                    "hparams": hp,
                    "epochs_in_phase": args.screening_epochs,
                    "best_mrr_in_trial": best_mrr,
                    "best_epoch": best_metrics,
                }
            )
            if best_mrr > global_best_mrr:
                global_best_mrr = best_mrr
                global_best_hp = dict(hp)
                model.save_weights(args.out_weights)
                print(f"  [전역 갱신] 저장 → {args.out_weights}  MRR={global_best_mrr:.6f}")
            screening_rows.append((best_mrr, dict(hp), best_metrics))
            K.clear_session()

        screening_rows.sort(key=lambda x: -x[0])
        top_hps: list[dict] = []
        seen_keys: set[tuple] = set()
        for mrr, hp, _lm in screening_rows:
            key = tuple(sorted(hp.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            top_hps.append(hp)
            if len(top_hps) >= k:
                break

        print(f"\n[2-phase] 2차(refine): 상위 {len(top_hps)}개 조합, 각 {args.epochs_per_trial} epochs\n")
        for j, hp in enumerate(top_hps):
            trial_seed = args.seed + 884422 + j * 9973
            _one_trial(j, len(top_hps), hp, args.epochs_per_trial, trial_seed, "refine")
    else:
        for t in range(run_trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            _one_trial(t, run_trials, hp, args.epochs_per_trial, trial_seed, "single")

    summary = {
        "global_best_mrr": global_best_mrr,
        "global_best_hparams": global_best_hp,
        "trials": log_trials,
        "epochs_per_trial": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hparam_grid_size": grid_n,
        "allow_duplicate_hparams": bool(args.allow_duplicate_hparams),
        "resume_log": resume_log_path,
        "num_seen_hparams_loaded": len(seen_hparam_keys),
        "num_trials_requested": int(args.trials),
        "num_trials_executed": int(run_trials),
        "two_phase": bool(args.two_phase),
        "screening_epochs": args.screening_epochs if args.two_phase else None,
        "refine_top_k": args.refine_top_k if args.two_phase else None,
    }
    append_mode = bool(args.append_log or args.resume_log)
    if append_mode and os.path.isfile(args.out_log):
        old = _load_json_or_none(args.out_log)
        if old is not None:
            old_trials = old.get("trials", [])
            if not isinstance(old_trials, list):
                old_trials = []
            merged_trials = old_trials + summary["trials"]
            old_best = old.get("global_best_mrr", -1.0)
            try:
                old_best = float(old_best)
            except Exception:
                old_best = -1.0
            if old_best > summary["global_best_mrr"]:
                summary["global_best_mrr"] = old_best
                old_hp = old.get("global_best_hparams", None)
                if isinstance(old_hp, dict):
                    summary["global_best_hparams"] = old_hp
            summary["trials"] = merged_trials
            summary["append_log"] = True
            summary["append_log_auto_by_resume"] = bool(args.resume_log and not args.append_log)
            summary["appended_from"] = args.out_log
            summary["num_trials_total_after_append"] = len(merged_trials)
    with open(args.out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n완료. 전역 최고 MRR={global_best_mrr:.6f}, 로그: {args.out_log}")
    if global_best_hp:
        print(f"최적 hparams: {global_best_hp}")


if __name__ == "__main__":
    main()
