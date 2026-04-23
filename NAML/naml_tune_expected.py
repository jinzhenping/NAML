# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
NAML 하이퍼파라미터 탐색:
- 기본: 실제 본문 사용
- 옵션: 기대본문(user_*/news_*.json) 사용 + 앞 N문장만 사용

실행 (저장소 루트에서):
python NAML/naml_tune_expected.py
python NAML/naml_tune_expected.py --trials 24 --epochs-per-trial 10 --seed 42
# 예산 절약: 24조합을 2에폭으로 걸러서 상위 5개만 10에폭 재학습
python NAML/naml_tune_expected.py --two-phase --trials 108 --screening-epochs 3 \
    --refine-top-k 10 --epochs-per-trial 10  --resume-log saved_models/naml_tune_expected_preference_log.json \
    --use-expected-body \
    --expected-train-dir user_preference/expected_body/MIND_2000/train_3cluster_11_13_8_rawtitle \
    --expected-test-dir user_preference/expected_body/MIND_2000/test_3cluster_11_13_8_rawtitle \
    --expected-body-first-n-sentences 3 \
    --out-log saved_models/naml_tune_expected_preference_log.json \
    --out-weights saved_models/NAML_mind_2000_expected_preference.h5

저장: saved_models/NAML_mind_2000.h5 (model.save_weights, build_naml_models 와 동일 구조)

기대본문 튜닝(앞 3문장):
python NAML/naml_tune_expected.py \
    --use-expected-body \
    --expected-train-dir body_generation/output/MIND_2000/train_3cluster_11_13_8 \
    --expected-test-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
    --expected-body-first-n-sentences 3

요청 조합 고정 탐색(필터×커널만):
python NAML/naml_tune_expected.py \
    --use-expected-body --expected-body-first-n-sentences 3 \
    --expected-train-dir body_generation/output/MIND_2000/train_3cluster_11_13_8 \
    --expected-test-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
    --fixed-filter-kernel-grid \
    --grid-cnn-filters 256 384 512 --grid-cnn-kernel-sizes 3 4 \
    --fixed-learning-rate 0.001 --fixed-dropout-rate 0.25 \
    --fixed-attention-dense-dim 160 --fixed-category-emb-dim 64 --trials 6 \
    --out-log saved_models/naml_tune_expected_log.json \
    --out-weights saved_models/NAML_mind_2000_expected.h5

기본적으로 이산 그리드를 한 번 섞은 뒤 순서대로 쓰므로, trials가 그리드 크기(6×5×4×3×4×3=4320) 이하면
같은 하이퍼파라미터 조합이 두 번 나오지 않는다. 예전 방식(매 trial 독립 무작위, 중복 가능)은
--allow-duplicate-hparams 로 사용한다.

4320조합을 다 돌 필요는 없다. 보통 trials=15~40 정도의 무작위(셔플) 부분집합으로도 충분한 경우가 많고,
예산을 아끼려면 --two-phase 로 1차 짧은 에폭 스크리닝 후 상위 k개만 길게 재학습한다 (multi-fidelity).
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

import naml_common as _naml_common
from naml_common import (
    SEED,
    MAX_HISTORY_CLICKS,
    npratio,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_model_builder import build_naml_models
from naml_batch_generators import (
    generate_batch_data_train as generate_batch_data_train_expected,
    generate_batch_data_test as generate_batch_data_test_expected,
)

NCAND = 1 + npratio


def _norm_expected_body_key(uid, nid):
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def _resolve_expected_body_dir(path_option: str | None) -> str | None:
    if not path_option or not str(path_option).strip():
        return None
    p = str(path_option).strip()
    if os.path.isabs(p) and os.path.isdir(p):
        return os.path.normpath(p)
    cand = os.path.normpath(os.path.join(_ROOT, p))
    if os.path.isdir(cand):
        return cand
    legacy = os.path.normpath(os.path.join(_ROOT, "body_generation", "output", p))
    if os.path.isdir(legacy):
        return legacy
    return None


def load_expected_bodies_from_dir(base_dir: str | None):
    expected_bodies = {}
    if not base_dir or not os.path.isdir(base_dir):
        return expected_bodies
    for user_folder in os.listdir(base_dir):
        user_path = os.path.join(base_dir, user_folder)
        if not os.path.isdir(user_path) or not user_folder.startswith("user_"):
            continue
        user_id = user_folder.replace("user_", "")
        for filename in os.listdir(user_path):
            if not (filename.startswith("news_") and filename.endswith(".json")):
                continue
            news_id = filename.replace("news_", "").replace(".json", "")
            file_path = os.path.join(user_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "generated_body" in data:
                    key = _norm_expected_body_key(user_id, news_id)
                    expected_bodies[key] = data["generated_body"]
            except Exception:
                continue
    return expected_bodies


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
    word_dict,
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
    *,
    use_expected_body=False,
    expected_bodies_test=None,
    all_test_userid_str=None,
    all_test_newsid_str=None,
    news_index=None,
):
    n = len(all_test_id)
    steps = (n + batch_size - 1) // batch_size
    if use_expected_body:
        gen = generate_batch_data_test_expected(
            word_dict=word_dict,
            news_words=news_words,
            news_body=news_body,
            news_v=news_v,
            news_sv=news_sv,
            news_index=news_index,
            all_test_pn=all_test_pn,
            all_label=all_test_label,
            all_test_id=all_test_id,
            all_test_user_pos=all_test_user_pos,
            batch_size=batch_size,
            expected_bodies=expected_bodies_test,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
        )
    else:
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
    *,
    use_expected_body=False,
    expected_bodies_train=None,
    expected_bodies_test=None,
    all_train_userid_str=None,
    all_train_newsid_str=None,
    all_test_userid_str=None,
    all_test_newsid_str=None,
    news_index=None,
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
        if use_expected_body:
            traingen = generate_batch_data_train_expected(
                word_dict=word_dict,
                news_words=news_words,
                news_body=news_body,
                news_v=news_v,
                news_sv=news_sv,
                news_index=news_index,
                all_train_pn=all_train_pn,
                all_label=all_label,
                all_train_id=all_train_id,
                all_user_pos=all_user_pos,
                batch_size=batch_size,
                expected_bodies=expected_bodies_train,
                all_userid_str=all_train_userid_str,
                all_train_newsid_str=all_train_newsid_str,
            )
        else:
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
            word_dict,
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
            use_expected_body=use_expected_body,
            expected_bodies_test=expected_bodies_test,
            all_test_userid_str=all_test_userid_str,
            all_test_newsid_str=all_test_newsid_str,
            news_index=news_index,
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
    ap = argparse.ArgumentParser(description="NAML 하이퍼파라미터 탐색 (actual by default, expected-body optional)")
    ap.add_argument("--trials", type=int, default=12, help="무작위 시도 횟수")
    ap.add_argument("--epochs-per-trial", type=int, default=8, help="각 시도당 에폭 수")
    ap.add_argument("--batch-size", type=int, default=16)
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
    )
    ap.add_argument(
        "--out-log",
        type=str,
        default=os.path.join(_ROOT, "saved_models", "naml_tune_actual_log.json"),
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
    ap.add_argument(
        "--use-expected-body",
        action="store_true",
        help="학습/평가에서 실제본문 대신 기대본문 JSON을 사용",
    )
    ap.add_argument(
        "--expected-train-dir",
        type=str,
        default="body_generation/output/MIND_2000/train_3cluster_11_13_8",
        help="train 기대본문 상위 폴더 (user_*/news_*.json). --use-expected-body 일 때 사용",
    )
    ap.add_argument(
        "--expected-test-dir",
        type=str,
        default="body_generation/output/MIND_2000/test_3cluster_11_13_8",
        help="test 기대본문 상위 폴더 (user_*/news_*.json). --use-expected-body 일 때 사용",
    )
    ap.add_argument(
        "--expected-body-first-n-sentences",
        type=int,
        default=3,
        help="기대본문에서 앞 N문장만 사용 (0=전체). --use-expected-body 일 때 적용",
    )
    ap.add_argument(
        "--fixed-filter-kernel-grid",
        action="store_true",
        help="cnn_filters x cnn_kernel_size 조합만 순회하고 나머지 hparams는 고정값 사용",
    )
    ap.add_argument(
        "--grid-cnn-filters",
        type=int,
        nargs="+",
        default=[256, 384, 512],
        help="--fixed-filter-kernel-grid 일 때 사용할 cnn_filters 목록",
    )
    ap.add_argument(
        "--grid-cnn-kernel-sizes",
        type=int,
        nargs="+",
        default=[3, 4],
        help="--fixed-filter-kernel-grid 일 때 사용할 cnn_kernel_size 목록",
    )
    ap.add_argument("--fixed-learning-rate", type=float, default=0.001)
    ap.add_argument("--fixed-dropout-rate", type=float, default=0.25)
    ap.add_argument("--fixed-attention-dense-dim", type=int, default=160)
    ap.add_argument("--fixed-category-emb-dim", type=int, default=64)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    use_expected_body = bool(args.use_expected_body)
    expected_bodies_train = None
    expected_bodies_test = None
    if use_expected_body:
        _train_dir = _resolve_expected_body_dir(args.expected_train_dir)
        _test_dir = _resolve_expected_body_dir(args.expected_test_dir)
        if not _train_dir or not os.path.isdir(_train_dir):
            raise FileNotFoundError(f"--expected-train-dir 경로를 찾을 수 없습니다: {args.expected_train_dir}")
        if not _test_dir or not os.path.isdir(_test_dir):
            raise FileNotFoundError(f"--expected-test-dir 경로를 찾을 수 없습니다: {args.expected_test_dir}")
        _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = max(0, int(args.expected_body_first_n_sentences))
        expected_bodies_train = load_expected_bodies_from_dir(_train_dir)
        expected_bodies_test = load_expected_bodies_from_dir(_test_dir)
        print(
            f"데이터 로드 (기대본문, 앞 {_naml_common.EXPECTED_BODY_FIRST_N_SENTENCES}문장): "
            f"train={len(expected_bodies_train)} test={len(expected_bodies_test)}"
        )
    else:
        print(f"데이터 로드 (실제 본문만, {os.environ.get('MIND_DATASET_SUBDIR', 'MIND_2000')} train/test)...")

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=expected_bodies_train if use_expected_body else None,
        expected_bodies_test=expected_bodies_test if use_expected_body else None,
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
        all_train_userid_str,
        all_train_newsid_str,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=expected_bodies_train if use_expected_body else None,
        expected_bodies_test=expected_bodies_test if use_expected_body else None,
        word_dict=word_dict,
    )

    embedding_mat = get_embedding(word_dict)
    print(f"  train 샘플: {len(all_train_id)}, test 행: {len(all_test_id)}, 뉴스: {len(news_index)}")

    out_dir = os.path.dirname(os.path.abspath(args.out_weights))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

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

    if args.fixed_filter_kernel_grid:
        filters = [int(x) for x in args.grid_cnn_filters]
        kernels = [int(x) for x in args.grid_cnn_kernel_sizes]
        fixed_base = {
            "learning_rate": float(args.fixed_learning_rate),
            "dropout_rate": float(args.fixed_dropout_rate),
            "attention_dense_dim": int(args.fixed_attention_dense_dim),
            "category_emb_dim": int(args.fixed_category_emb_dim),
        }
        all_grid = []
        for f in filters:
            for ksz in kernels:
                hp = dict(fixed_base)
                hp["cnn_filters"] = int(f)
                hp["cnn_kernel_size"] = int(ksz)
                all_grid.append(hp)
        rng.shuffle(all_grid)
        trial_hparams = [hp for hp in all_grid if _hp_key(hp) not in seen_hparam_keys][: args.trials]
        print(
            f"fixed filter-kernel grid 모드: {len(filters)}x{len(kernels)}={len(all_grid)} 조합, "
            f"고정값={fixed_base}"
        )
        if len(trial_hparams) < min(args.trials, len(all_grid)):
            print(f"경고: resume-log 제외 후 실행 가능한 조합이 {len(trial_hparams)}개입니다.")
    elif args.allow_duplicate_hparams:
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
        best_mrr, best_epoch_metrics, model = run_trial(
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
            use_expected_body=use_expected_body,
            expected_bodies_train=expected_bodies_train,
            expected_bodies_test=expected_bodies_test,
            all_train_userid_str=all_train_userid_str,
            all_train_newsid_str=all_train_newsid_str,
            all_test_userid_str=all_test_userid_str,
            all_test_newsid_str=all_test_newsid_str,
            news_index=news_index,
        )
        print(
            f"  trial best MRR (best epoch in trial): {best_mrr:.6f}  | best epoch metrics: "
            f"MRR={best_epoch_metrics['MRR']:.6f} NDCG@5={best_epoch_metrics['NDCG@5']:.6f} Hit@1={best_epoch_metrics['Hit@1']:.6f}"
        )
        log_trials.append(
            {
                "phase": phase_label,
                "hparams": hp,
                "epochs_in_phase": epochs,
                "best_mrr_in_trial": best_mrr,
                "best_epoch": best_epoch_metrics,
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
            best_mrr, best_epoch_metrics, model = run_trial(
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
                use_expected_body=use_expected_body,
                expected_bodies_train=expected_bodies_train,
                expected_bodies_test=expected_bodies_test,
                all_train_userid_str=all_train_userid_str,
                all_train_newsid_str=all_train_newsid_str,
                all_test_userid_str=all_test_userid_str,
                all_test_newsid_str=all_test_newsid_str,
                news_index=news_index,
            )
            print(
                f"  trial best MRR (best epoch in trial): {best_mrr:.6f}  | best epoch metrics: "
                f"MRR={best_epoch_metrics['MRR']:.6f} NDCG@5={best_epoch_metrics['NDCG@5']:.6f} Hit@1={best_epoch_metrics['Hit@1']:.6f}"
            )
            log_trials.append(
                {
                    "phase": "screening",
                    "hparams": hp,
                    "epochs_in_phase": args.screening_epochs,
                    "best_mrr_in_trial": best_mrr,
                    "best_epoch": best_epoch_metrics,
                }
            )
            if best_mrr > global_best_mrr:
                global_best_mrr = best_mrr
                global_best_hp = dict(hp)
                model.save_weights(args.out_weights)
                print(f"  [전역 갱신] 저장 → {args.out_weights}  MRR={global_best_mrr:.6f}")
            screening_rows.append((best_mrr, dict(hp), best_epoch_metrics))
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
        "use_expected_body": use_expected_body,
        "expected_train_dir": args.expected_train_dir if use_expected_body else None,
        "expected_test_dir": args.expected_test_dir if use_expected_body else None,
        "expected_body_first_n_sentences": (
            int(args.expected_body_first_n_sentences) if use_expected_body else None
        ),
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
        "fixed_filter_kernel_grid": bool(args.fixed_filter_kernel_grid),
        "grid_cnn_filters": [int(x) for x in args.grid_cnn_filters] if args.fixed_filter_kernel_grid else None,
        "grid_cnn_kernel_sizes": [int(x) for x in args.grid_cnn_kernel_sizes] if args.fixed_filter_kernel_grid else None,
        "fixed_learning_rate": float(args.fixed_learning_rate) if args.fixed_filter_kernel_grid else None,
        "fixed_dropout_rate": float(args.fixed_dropout_rate) if args.fixed_filter_kernel_grid else None,
        "fixed_attention_dense_dim": int(args.fixed_attention_dense_dim) if args.fixed_filter_kernel_grid else None,
        "fixed_category_emb_dim": int(args.fixed_category_emb_dim) if args.fixed_filter_kernel_grid else None,
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
