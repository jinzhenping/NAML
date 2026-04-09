# -*- coding: utf-8 -*-
"""
NAML 실제 본문만 사용: 하이퍼파라미터 탐색 후 테스트 MRR이 가장 좋았던 가중치를 저장한다.

실행 (저장소 루트에서):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/naml_tune_actual.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/naml_tune_actual.py --trials 24 --epochs-per-trial 10 --seed 42
# 예산 절약: 24조합을 2에폭으로 걸러서 상위 5개만 10에폭 재학습
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/naml_tune_actual.py --two-phase --trials 24 --screening-epochs 2 --refine-top-k 5 --epochs-per-trial 10

저장: saved_models/NAML_mind_2000.h5 (model.save_weights, build_naml_models 와 동일 구조)

기대본문·MAIN_TESTSET 경로는 사용하지 않는다.

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

os.environ["MIND_DATASET_SUBDIR"] = "MIND_2000"
os.environ["MIND_TRAIN_FILENAME"] = "MIND_train_(2000).tsv"
os.environ["MIND_TEST_FILENAME"] = "MIND_test_(2000).tsv"

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
    last_metrics = None

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
        last_metrics = evaluate_session_metrics(
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
        mrr = last_metrics["MRR"]
        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)
    return best_mrr, last_metrics, model


def main():
    ap = argparse.ArgumentParser(description="NAML 실제 본문 하이퍼파라미터 탐색")
    ap.add_argument("--trials", type=int, default=12, help="무작위 시도 횟수")
    ap.add_argument("--epochs-per-trial", type=int, default=8, help="각 시도당 에폭 수")
    ap.add_argument("--batch-size", type=int, default=16)
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
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("데이터 로드 (실제 본문만, MIND_2000 train/test)...")
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

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: dict | None = None
    log_trials = []

    grid_n = _hparam_grid_size()
    if args.allow_duplicate_hparams:
        trial_hparams = [sample_hparams(rng) for _ in range(args.trials)]
    else:
        trial_hparams = plan_hparam_trials(rng, args.trials)
        if args.trials > grid_n:
            print(
                f"경고: 고유 그리드 조합은 {grid_n}개인데 trials={args.trials} → "
                f"처음 {grid_n}개는 중복 없음, 이후는 무작위 보충(중복 가능)."
            )

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
        best_mrr, last_metrics, model = run_trial(
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
            f"  trial best MRR (best epoch in trial): {best_mrr:.6f}  | last epoch: "
            f"MRR={last_metrics['MRR']:.6f} NDCG@5={last_metrics['NDCG@5']:.6f} Hit@1={last_metrics['Hit@1']:.6f}"
        )
        log_trials.append(
            {
                "phase": phase_label,
                "hparams": hp,
                "epochs_in_phase": epochs,
                "best_mrr_in_trial": best_mrr,
                "last_epoch": last_metrics,
            }
        )
        if best_mrr > global_best_mrr:
            global_best_mrr = best_mrr
            global_best_hp = dict(hp)
            model.save_weights(args.out_weights)
            print(f"  [전역 갱신] 저장 → {args.out_weights}  MRR={global_best_mrr:.6f}")
        K.clear_session()

    if args.two_phase:
        k = min(args.refine_top_k, args.trials)
        print(
            f"\n[2-phase] 1차: trials={args.trials}, epochs={args.screening_epochs} → "
            f"상위 {k}개를 2차에서 epochs={args.epochs_per_trial} 로 재학습\n"
        )
        screening_rows: list[tuple[float, dict, dict]] = []
        for t in range(args.trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            print(f"\n--- [screening] {t + 1}/{args.trials}  hparams={hp} ---")
            best_mrr, last_metrics, model = run_trial(
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
                f"  trial best MRR (best epoch in trial): {best_mrr:.6f}  | last epoch: "
                f"MRR={last_metrics['MRR']:.6f} NDCG@5={last_metrics['NDCG@5']:.6f} Hit@1={last_metrics['Hit@1']:.6f}"
            )
            log_trials.append(
                {
                    "phase": "screening",
                    "hparams": hp,
                    "epochs_in_phase": args.screening_epochs,
                    "best_mrr_in_trial": best_mrr,
                    "last_epoch": last_metrics,
                }
            )
            if best_mrr > global_best_mrr:
                global_best_mrr = best_mrr
                global_best_hp = dict(hp)
                model.save_weights(args.out_weights)
                print(f"  [전역 갱신] 저장 → {args.out_weights}  MRR={global_best_mrr:.6f}")
            screening_rows.append((best_mrr, dict(hp), last_metrics))
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
        for t in range(args.trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            _one_trial(t, args.trials, hp, args.epochs_per_trial, trial_seed, "single")

    summary = {
        "global_best_mrr": global_best_mrr,
        "global_best_hparams": global_best_hp,
        "trials": log_trials,
        "epochs_per_trial": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hparam_grid_size": grid_n,
        "allow_duplicate_hparams": bool(args.allow_duplicate_hparams),
        "two_phase": bool(args.two_phase),
        "screening_epochs": args.screening_epochs if args.two_phase else None,
        "refine_top_k": args.refine_top_k if args.two_phase else None,
    }
    with open(args.out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n완료. 전역 최고 MRR={global_best_mrr:.6f}, 로그: {args.out_log}")
    if global_best_hp:
        print(f"최적 hparams: {global_best_hp}")


if __name__ == "__main__":
    main()
