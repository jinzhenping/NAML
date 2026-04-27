#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
NAML 지식 증류 학습 (user+news distill):
  L = L_rec + lambda_user * L_distill_user + lambda_exp * L_distill_exp

- L_rec: 기존 CE (학생 후보 본문=기대본문)
- L_distill_user: student user_rep(히스토리 기반) vs teacher user_rep 코사인 (1-cos)
- L_distill_exp: 후보마다 student newsEncoder(기대본문) vs teacher newsEncoder(실제/기대 본문 선택) 코사인 (1-cos)

python NAML/naml_kd_train_userdistill.py \
  --teacher-weights saved_models/NAML_mind_2000.h5 \
  --tune-log saved_models/naml_tune_actual_log.json \
  --expected-body-train-dir body_generation/output/MIND_2000/train_3cluster_11_13_8 \
  --expected-body-test-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
  --expected-body-first-n-sentences 3 \
  --mind-dataset-subdir MIND_2000 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --lambda-distill-user 0.2 \
  --lambda-distill-exp 0.2 \
  --epochs 10 \
  --output-weights saved_models/NAML_kd_student_userdistill.h5 --teacher-exp-use-expected-body \
  --num-runs 3

# 학습 기대본문 문장 수: --expected-body-first-n-sentences N (= --train-expected-body-first-n-sentences N, 기본 3, 0=전체)
# 에폭 평가 [기대본문] 지표: --eval-expected-body-first-n-sentences (기본 0=전체)

교사 가중치가 naml_tune_actual 로 튜닝된 경우 CNN 폭 등이 다르므로 eval_test_expected 와 동일하게:
  --tune-log saved_models/naml_tune_actual_log.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.optimizers import Adam

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from eval_cluster_batch import load_expected_bodies_from_train_dir
from naml_eval_test import (
    _DEFAULT_ARCH,
    _arch_from_tune_log,
    calc_metrics_from_scores,
    load_expected_bodies_from_dir,
)
from naml_batch_generators import generate_batch_data_test
from naml_common import (
    MAX_BODY_LENGTH,
    MAX_HISTORY_CLICKS,
    SEED,
    clip_expected_body_to_first_sentences,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
import naml_common as _naml_common
from naml_model_builder import build_naml_models


def _norm_expected_body_key(uid, nid):
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def _input_indices(H: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    cand_title = list(range(5))
    cand_body = [5 + H + k for k in range(5)]
    base_v = 5 + H + 5 + H
    cand_v = [base_v + k for k in range(5)]
    base_sv = base_v + 5 + H
    cand_sv = [base_sv + k for k in range(5)]
    return cand_title, cand_body, cand_v, cand_sv


def _history_input_indices(H: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    hist_title = [5 + k for k in range(H)]
    hist_body = [5 + H + 5 + k for k in range(H)]
    base_v = 5 + H + 5 + H
    hist_v = [base_v + 5 + k for k in range(H)]
    base_sv = base_v + 5 + H
    hist_sv = [base_sv + 5 + k for k in range(H)]
    return hist_title, hist_body, hist_v, hist_sv


def generate_batch_data_train_kd(
    word_dict: dict,
    news_words: np.ndarray,
    news_body: np.ndarray,
    news_v: np.ndarray,
    news_sv: np.ndarray,
    news_index: dict,
    all_train_pn: np.ndarray,
    all_label: np.ndarray,
    all_train_id: np.ndarray,
    all_user_pos: np.ndarray,
    batch_size: int,
    expected_bodies: Dict[Tuple[str, str], str],
    all_userid_str: Sequence,
    all_train_newsid_str: Sequence,
    news_index_reverse: Dict[int, str],
    H: int,
    shuffle: bool = True,
):
    inputid = np.arange(len(all_label))
    if shuffle:
        np.random.shuffle(inputid)
    batches = [
        inputid[range(batch_size * i, min(len(all_label), batch_size * (i + 1)))]
        for i in range(len(all_label) // batch_size + 1)
        if batch_size * i < len(all_label)
    ]

    while True:
        for batch_indices in batches:
            batch_candidate_splits = [[] for _ in range(5)]
            batch_browsed_news_splits = [[] for _ in range(H)]
            batch_candidate_body_splits = [[] for _ in range(5)]
            batch_browsed_news_body_splits = [[] for _ in range(H)]
            batch_candidate_vertical_splits = [[] for _ in range(5)]
            batch_browsed_news_vertical_splits = [[] for _ in range(H)]
            batch_candidate_subvertical_splits = [[] for _ in range(5)]
            batch_browsed_news_subvertical_splits = [[] for _ in range(H)]
            batch_labels: List[np.ndarray] = []
            batch_actual_flat: List[np.ndarray] = []

            for idx in batch_indices:
                candidate_indices = np.array(all_train_pn[idx], dtype=np.int32)
                candidate = news_words[candidate_indices]
                candidate_split = [np.expand_dims(candidate[k], axis=0) for k in range(candidate.shape[0])]

                user_id_str = all_userid_str[idx]
                news_ids_str = all_train_newsid_str[idx]
                candidate_body_list = []
                actual_rows: List[np.ndarray] = []

                for j, news_idx in enumerate(all_train_pn[idx]):
                    actual_rows.append(np.array(news_body[int(news_idx)], dtype=np.int32))
                    if news_idx == 0:
                        candidate_body_list.append(news_body[0])
                        continue
                    news_id_str = news_ids_str[j] if j < len(news_ids_str) else ""
                    key = _norm_expected_body_key(user_id_str, news_id_str)
                    if key in expected_bodies:
                        expected_body = expected_bodies[key]
                        _eb = clip_expected_body_to_first_sentences(
                            expected_body, _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES
                        )
                        body_tokens = word_tokenize(_eb.lower()) if _eb else []
                        word_id: List[int] = []
                        for w in body_tokens:
                            if w in word_dict:
                                word_id.append(word_dict[w][0])
                        word_id = word_id[:MAX_BODY_LENGTH]
                        word_id = word_id + [0] * (MAX_BODY_LENGTH - len(word_id))
                        candidate_body_list.append(np.array(word_id, dtype=np.int32))
                    else:
                        candidate_body_list.append(news_body[int(news_idx)])

                candidate_body = np.array(candidate_body_list)
                candidate_body_split = [np.expand_dims(candidate_body[k], axis=0) for k in range(5)]

                candidate_vertical = news_v[candidate_indices]
                candidate_vertical_split = [
                    np.expand_dims(candidate_vertical[k], axis=0) for k in range(candidate_vertical.shape[0])
                ]
                candidate_subvertical = news_sv[candidate_indices]
                candidate_subvertical_split = [
                    np.expand_dims(candidate_subvertical[k], axis=0) for k in range(candidate_subvertical.shape[0])
                ]

                user_pos_indices = np.array(all_user_pos[idx], dtype=np.int32)
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

                label = np.array(all_label[idx], dtype=np.float32)

                for k in range(5):
                    batch_candidate_splits[k].append(candidate_split[k])
                for k in range(len(browsed_news_split)):
                    batch_browsed_news_splits[k].append(browsed_news_split[k])
                for k in range(5):
                    batch_candidate_body_splits[k].append(candidate_body_split[k])
                for k in range(len(browsed_news_body_split)):
                    batch_browsed_news_body_splits[k].append(browsed_news_body_split[k])
                for k in range(5):
                    batch_candidate_vertical_splits[k].append(candidate_vertical_split[k])
                for k in range(len(browsed_news_vertical_split)):
                    batch_browsed_news_vertical_splits[k].append(browsed_news_vertical_split[k])
                for k in range(5):
                    batch_candidate_subvertical_splits[k].append(candidate_subvertical_split[k])
                for k in range(len(browsed_news_subvertical_split)):
                    batch_browsed_news_subvertical_splits[k].append(browsed_news_subvertical_split[k])
                batch_labels.append(label)
                batch_actual_flat.append(np.stack(actual_rows, axis=0))

            batch_inputs: List[np.ndarray] = []
            for k in range(5):
                batch_inputs.append(np.concatenate(batch_candidate_splits[k], axis=0))
            for k in range(H):
                batch_inputs.append(np.concatenate(batch_browsed_news_splits[k], axis=0))
            for k in range(5):
                batch_inputs.append(np.concatenate(batch_candidate_body_splits[k], axis=0))
            for k in range(H):
                batch_inputs.append(np.concatenate(batch_browsed_news_body_splits[k], axis=0))
            for k in range(5):
                batch_inputs.append(np.concatenate(batch_candidate_vertical_splits[k], axis=0))
            for k in range(H):
                batch_inputs.append(np.concatenate(batch_browsed_news_vertical_splits[k], axis=0))
            for k in range(5):
                batch_inputs.append(np.concatenate(batch_candidate_subvertical_splits[k], axis=0))
            for k in range(H):
                batch_inputs.append(np.concatenate(batch_browsed_news_subvertical_splits[k], axis=0))

            batch_labels_array = np.array(batch_labels, dtype=np.float32)
            candidate_actual = np.stack(batch_actual_flat, axis=0)
            yield (batch_inputs, batch_labels_array, candidate_actual)


def _to_tf_inputs(batch_inputs: Sequence[np.ndarray]) -> List[tf.Tensor]:
    return [tf.convert_to_tensor(x, dtype=tf.int32) for x in batch_inputs]


def run_test_set_eval(
    model_test,
    word_dict: dict,
    news_words: np.ndarray,
    news_body: np.ndarray,
    news_v: np.ndarray,
    news_sv: np.ndarray,
    news_index: dict,
    all_test_pn: np.ndarray,
    all_test_label: np.ndarray,
    all_test_id: np.ndarray,
    all_test_user_pos: np.ndarray,
    all_test_index: List,
    all_test_userid_str: Sequence,
    all_test_newsid_str: Sequence,
    news_index_reverse: Dict[int, str],
    expected_bodies_test: Optional[Dict[Tuple[str, str], str]],
    eval_batch_size: int,
    *,
    eval_expected_body_clip_n_sentences: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    n_test = len(all_test_id)
    if n_test == 0:
        return (
            {"MRR": 0.0, "NDCG@5": 0.0, "Hit@1": 0.0, "evaluated_sessions": 0},
            None,
        )
    bs = max(1, int(eval_batch_size))
    test_steps = (n_test + bs - 1) // bs

    testgen_real = generate_batch_data_test(
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
        candidate_news_body=None,
        expected_bodies=None,
        all_userid_str=all_test_userid_str,
        all_newsid_str=all_test_newsid_str,
        news_index_reverse=news_index_reverse,
    )
    score_real = model_test.predict(testgen_real, steps=test_steps, verbose=0)
    metrics_real = calc_metrics_from_scores(score_real, all_test_label, all_test_index)

    metrics_exp: Optional[Dict[str, Any]] = None
    if expected_bodies_test:
        _exp_kw: Dict[str, Any] = {}
        if eval_expected_body_clip_n_sentences is not None:
            _exp_kw["expected_body_clip_n_sentences"] = eval_expected_body_clip_n_sentences
        testgen_exp = generate_batch_data_test(
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
            candidate_news_body=None,
            expected_bodies=expected_bodies_test,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse,
            **_exp_kw,
        )
        score_exp = model_test.predict(testgen_exp, steps=test_steps, verbose=0)
        metrics_exp = calc_metrics_from_scores(score_exp, all_test_label, all_test_index)

    return metrics_real, metrics_exp


def _cosine_1minus(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    a_n = tf.nn.l2_normalize(a, axis=-1)
    b_n = tf.nn.l2_normalize(b, axis=-1)
    cos = tf.reduce_sum(a_n * b_n, axis=-1)
    return tf.reduce_mean(1.0 - cos)


def _distill_news_term(student_enc, title_k, body_exp_k, v_k, sv_k, teacher_emb_t):
    emb_s = student_enc([title_k, body_exp_k, v_k, sv_k])
    return _cosine_1minus(emb_s, teacher_emb_t)


def train_kd_userdistill(
    student_model,
    teacher_news_encoder,
    student_news_encoder,
    teacher_user_encoder,
    student_user_encoder,
    make_train_gen,
    steps_per_epoch: int,
    optimizer: Adam,
    lambda_user: float,
    lambda_exp: float,
    H: int,
    epochs: int,
    teacher_exp_use_expected_body: bool = False,
    eval_after_epoch: Optional[Callable[[], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = None,
) -> Tuple[Optional[List[np.ndarray]], int, float]:
    cand_title_idx, cand_body_idx, cand_v_idx, cand_sv_idx = _input_indices(H)
    hist_title_idx, hist_body_idx, hist_v_idx, hist_sv_idx = _history_input_indices(H)

    best_weights: Optional[List[np.ndarray]] = None
    best_epoch_1based = -1
    best_mrr_expected = -1.0

    for ep in range(epochs):
        print(f"\n=== KD(User+News) epoch {ep + 1}/{epochs} ===", flush=True)
        mean_rec = tf.keras.metrics.Mean()
        mean_user = tf.keras.metrics.Mean()
        mean_exp = tf.keras.metrics.Mean()
        mean_total = tf.keras.metrics.Mean()
        train_gen = make_train_gen()

        for step in range(steps_per_epoch):
            batch_inputs, y, cand_act = next(train_gen)
            x_list = _to_tf_inputs(batch_inputs)
            y_t = tf.convert_to_tensor(y, dtype=tf.float32)
            cand_act_t = tf.convert_to_tensor(cand_act, dtype=tf.int32)

            # Teacher forward는 gradient가 필요 없으므로 tape 바깥에서 계산해
            # 그래프/activation 메모리 점유를 낮춘다.
            teacher_news_targets: List[tf.Tensor] = []
            for k in range(5):
                t_k = x_list[cand_title_idx[k]]
                # 기본은 교사에 실제본문을 넣고, 옵션으로 기대본문 입력도 허용.
                if teacher_exp_use_expected_body:
                    b_t_k = x_list[cand_body_idx[k]]
                else:
                    b_t_k = cand_act_t[:, k, :]
                vk = x_list[cand_v_idx[k]]
                svk = x_list[cand_sv_idx[k]]
                teacher_news_targets.append(
                    tf.stop_gradient(teacher_news_encoder([t_k, b_t_k, vk, svk], training=False))
                )
            t_user = tf.stop_gradient(
                teacher_user_encoder(
                    [x_list[i] for i in hist_title_idx]
                    + [x_list[i] for i in hist_body_idx]
                    + [x_list[i] for i in hist_v_idx]
                    + [x_list[i] for i in hist_sv_idx],
                    training=False,
                )
            )

            with tf.GradientTape() as tape:
                logits = student_model(x_list, training=True)
                loss_rec = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_t, logits))

                loss_d_exp_parts = []
                for k in range(5):
                    t_k = x_list[cand_title_idx[k]]
                    be_k = x_list[cand_body_idx[k]]
                    vk = x_list[cand_v_idx[k]]
                    svk = x_list[cand_sv_idx[k]]
                    loss_d_exp_parts.append(
                        _distill_news_term(student_news_encoder, t_k, be_k, vk, svk, teacher_news_targets[k])
                    )
                loss_d_exp = tf.add_n(loss_d_exp_parts) / 5.0

                s_user = student_user_encoder(
                    [x_list[i] for i in hist_title_idx]
                    + [x_list[i] for i in hist_body_idx]
                    + [x_list[i] for i in hist_v_idx]
                    + [x_list[i] for i in hist_sv_idx],
                    training=True,
                )
                loss_d_user = _cosine_1minus(s_user, t_user)

                loss = loss_rec + lambda_user * loss_d_user + lambda_exp * loss_d_exp

            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

            mean_rec.update_state(loss_rec)
            mean_user.update_state(loss_d_user)
            mean_exp.update_state(loss_d_exp)
            mean_total.update_state(loss)

            if (step + 1) % max(1, steps_per_epoch // 5) == 0 or step == 0:
                print(
                    f"  step {step + 1}/{steps_per_epoch}  "
                    f"L_rec={float(mean_rec.result()):.4f}  "
                    f"L_user={float(mean_user.result()):.4f}  "
                    f"L_exp={float(mean_exp.result()):.4f}  "
                    f"total={float(mean_total.result()):.4f}",
                    flush=True,
                )

        print(
            f"epoch {ep + 1} end: L_rec={float(mean_rec.result()):.4f}  "
            f"L_user={float(mean_user.result()):.4f}  "
            f"L_exp={float(mean_exp.result()):.4f}  "
            f"total={float(mean_total.result()):.4f}",
            flush=True,
        )

        if eval_after_epoch is not None:
            print(f"  [테스트셋] 에폭 {ep + 1} 평가 중...", flush=True)
            mr, me = eval_after_epoch()
            print(
                f"  [실제본문] MRR={mr['MRR']:.6f}  NDCG@5={mr['NDCG@5']:.6f}  "
                f"Hit@1={mr['Hit@1']:.6f}  (세션 {mr.get('evaluated_sessions', 0)})",
                flush=True,
            )
            if me is not None:
                print(
                    f"  [기대본문] MRR={me['MRR']:.6f}  NDCG@5={me['NDCG@5']:.6f}  "
                    f"Hit@1={me['Hit@1']:.6f}  (세션 {me.get('evaluated_sessions', 0)})",
                    flush=True,
                )
            else:
                print(
                    "  [기대본문] 생략 (--expected-body-test-dir 미지정이거나 JSON 0개)",
                    flush=True,
                )

            if me is not None:
                mrr_e = float(me["MRR"])
                if mrr_e > best_mrr_expected:
                    best_mrr_expected = mrr_e
                    best_epoch_1based = ep + 1
                    best_weights = [np.array(w) for w in student_model.get_weights()]

    return best_weights, best_epoch_1based, best_mrr_expected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NAML KD(User+News): L_rec + lambda_user * L_distill_user + lambda_exp * L_distill_exp"
    )
    ap.add_argument("--teacher-weights", type=str, required=True, help="교사 model.save_weights 경로 (프로젝트 루트 기준)")
    ap.add_argument(
        "--tune-log",
        type=str,
        default=None,
        help="naml_tune_actual_log.json. global_best_hparams 로 학생·교사 그래프를 맞춤 (튜닝 교사 가중치 시 권장)",
    )
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--cnn-filters", type=int, default=None)
    ap.add_argument("--cnn-kernel-size", type=int, default=None)
    ap.add_argument("--attention-dense-dim", type=int, default=None)
    ap.add_argument("--category-emb-dim", type=int, default=None)
    ap.add_argument(
        "--expected-body-train-dir",
        type=str,
        required=True,
        help="기대본문 JSON 루트 (user_*/news_*.json), train 전용",
    )
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument(
        "--lambda-distill-user",
        type=float,
        default=0.5,
        metavar="LAMBDA_USER",
        help="L_total = L_rec + LAMBDA_USER * L_distill_user + LAMBDA_EXP * L_distill_exp",
    )
    ap.add_argument(
        "--lambda-distill-exp",
        type=float,
        default=0.5,
        metavar="LAMBDA_EXP",
        help="L_total = L_rec + LAMBDA_USER * L_distill_user + LAMBDA_EXP * L_distill_exp",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument(
        "--output-weights",
        type=str,
        default="saved_models/NAML_kd_student_userdistill.h5",
        help="학생 가중치 저장 (save_weights). 기대본문 테스트가 있으면 MRR 최고 에폭, 없으면 마지막 에폭",
    )
    ap.add_argument(
        "--expected-body-test-dir",
        type=str,
        default=None,
        help="테스트용 기대본문 JSON 폴더 (있으면 에폭마다 기대본문 지표도 계산)",
    )
    ap.add_argument(
        "--no-epoch-eval",
        action="store_true",
        help="매 에폭 종료 후 테스트셋 평가 생략",
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="테스트 predict 배치 크기 (기본: --batch-size 와 동일)",
    )
    ap.add_argument(
        "--teacher-exp-use-expected-body",
        action="store_true",
        help="L_distill_exp에서 teacher newsEncoder 입력 본문을 실제본문 대신 기대본문으로 사용",
    )
    ap.add_argument(
        "--expected-body-first-n-sentences",
        "--train-expected-body-first-n-sentences",
        type=int,
        default=3,
        metavar="N",
        dest="expected_body_first_n_sentences",
        help="학습 시 기대본문 앞 N문장만 사용 (0=전체). 배치 입력·L_rec·L_distill_exp(및 --teacher-exp-use-expected-body 시 교사 입력). "
        "별칭: --train-expected-body-first-n-sentences (기본 3)",
    )
    ap.add_argument(
        "--eval-expected-body-first-n-sentences",
        type=int,
        default=0,
        metavar="N",
        help="에폭마다 테스트셋 [기대본문] 지표 계산 시 앞 N문장만 사용 (0=전체, 기본 0). 학습과 독립",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="동일 설정으로 학습을 몇 번 반복할지 (기본 1). 기대본문 MRR 최고 run/epoch 가중치를 저장",
    )
    ap.add_argument(
        "--run-seed-step",
        type=int,
        default=1,
        help="반복 실행 시 run마다 seed 증가 폭 (run_seed = seed + i*step, 기본 1)",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = max(0, int(args.expected_body_first_n_sentences))
    _eval_exp_n = max(0, int(args.eval_expected_body_first_n_sentences))
    print(
        (
            f"학습 기대본문: 앞 {_naml_common.EXPECTED_BODY_FIRST_N_SENTENCES}문장"
            if _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES > 0
            else "학습 기대본문: 전체 문장"
        ),
        flush=True,
    )
    print(
        (
            f"평가 기대본문([기대본문] 지표): 앞 {_eval_exp_n}문장"
            if _eval_exp_n > 0
            else "평가 기대본문([기대본문] 지표): 전체 문장 (기본)"
        ),
        flush=True,
    )

    sub = args.mind_dataset_subdir or os.environ.get("MIND_DATASET_SUBDIR", "MIND_2000")
    os.environ["MIND_DATASET_SUBDIR"] = sub

    train_dir = str(_ROOT / args.expected_body_train_dir) if not os.path.isabs(args.expected_body_train_dir) else args.expected_body_train_dir
    if not os.path.isdir(train_dir):
        print(f"오류: 기대본문 폴더 없음: {train_dir}", file=sys.stderr)
        sys.exit(1)

    expected_bodies_train = load_expected_bodies_from_train_dir(train_dir)
    if not expected_bodies_train:
        print("경고: 기대본문이 0개입니다. L_rec은 실제본문 폴백이 많아집니다.", flush=True)
    print(f"기대본문 로드: {len(expected_bodies_train)}개 ({train_dir})")

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=None,
        expected_bodies_test=None,
    )
    print(f"word_dict 크기 (뉴스만): {len(word_dict)} — 교사 가중치와 임베딩 행 수를 맞춥니다.", flush=True)
    embedding_mat = get_embedding(word_dict)

    (
        _uid,
        all_train_pn,
        all_label,
        all_train_id,
        all_test_pn,
        all_test_label,
        all_test_id,
        all_user_pos,
        all_test_user_pos,
        all_test_index,
        _ct,
        _ce,
        all_train_userid_str,
        all_train_newsid_str,
        all_test_userid_str,
        all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    news_index_reverse = {v: k for k, v in news_index.items()}
    H = MAX_HISTORY_CLICKS

    arch: Dict[str, float | int] = dict(_DEFAULT_ARCH)
    if args.tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            loaded = _arch_from_tune_log(tl)
            arch.update(loaded)
            print(f"튜닝 로그 아키텍처: {tl} → {loaded or '(global_best_hparams 없음, 기본값)'}", flush=True)
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

    _kw = dict(
        dropout_rate=float(arch["dropout_rate"]),
        cnn_filters=int(arch["cnn_filters"]),
        cnn_kernel_size=int(arch["cnn_kernel_size"]),
        attention_dense_dim=int(arch["attention_dense_dim"]),
        category_emb_dim=int(arch["category_emb_dim"]),
    )

    tw = str(_ROOT / args.teacher_weights) if not os.path.isabs(args.teacher_weights) else args.teacher_weights
    if not os.path.isfile(tw):
        print(f"오류: 교사 가중치 없음: {tw}", file=sys.stderr)
        sys.exit(1)

    n_train = len(all_train_id)
    steps_per_epoch = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    print(
        f"샘플 수: {n_train}, batch_size={args.batch_size}, steps/epoch={steps_per_epoch}, "
        f"lambda_user={args.lambda_distill_user}, lambda_exp={args.lambda_distill_exp}, "
        f"teacher_exp_body={'expected' if args.teacher_exp_use_expected_body else 'actual'}"
    )

    expected_bodies_test: Optional[Dict[Tuple[str, str], str]] = None
    if args.expected_body_test_dir:
        test_d = str(_ROOT / args.expected_body_test_dir) if not os.path.isabs(args.expected_body_test_dir) else args.expected_body_test_dir
        if os.path.isdir(test_d):
            expected_bodies_test = load_expected_bodies_from_dir(test_d)
            print(f"테스트 기대본문: {len(expected_bodies_test)}개 ({test_d})")
        else:
            print(f"경고: --expected-body-test-dir 없음 또는 디렉터리 아님: {test_d}", flush=True)

    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    def eval_after_epoch_fn() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        return run_test_set_eval(
            model_test,
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
            all_test_index,
            all_test_userid_str,
            all_test_newsid_str,
            news_index_reverse,
            expected_bodies_test,
            eval_bs,
            eval_expected_body_clip_n_sentences=_eval_exp_n,
        )

    eval_cb: Optional[Callable[[], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = None
    if not args.no_epoch_eval:
        eval_cb = eval_after_epoch_fn

    def make_train_gen():
        return generate_batch_data_train_kd(
            word_dict,
            news_words,
            news_body,
            news_v,
            news_sv,
            news_index,
            all_train_pn,
            all_label,
            all_train_id,
            all_user_pos,
            args.batch_size,
            expected_bodies_train,
            all_train_userid_str,
            all_train_newsid_str,
            news_index_reverse,
            H,
            shuffle=True,
        )

    out_w = str(_ROOT / args.output_weights) if not os.path.isabs(args.output_weights) else args.output_weights
    os.makedirs(os.path.dirname(out_w) or ".", exist_ok=True)
    if args.num_runs < 1:
        print("오류: --num-runs 는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(2)

    best_overall_mrr = float("-inf")
    best_overall_run = 1
    best_overall_epoch = args.epochs
    best_overall_weights = None
    fallback_last_weights = None
    fallback_last_model = None

    for run_idx in range(args.num_runs):
        run_no = run_idx + 1
        run_seed = int(args.seed) + run_idx * int(args.run_seed_step)
        np.random.seed(run_seed)
        tf.random.set_seed(run_seed)
        print(f"\n=== KD run {run_no}/{args.num_runs} (seed={run_seed}) ===", flush=True)

        built_s = build_naml_models(
            word_dict, embedding_mat, category, subcategory, args.learning_rate, clear_session=True, **_kw
        )
        student_model = built_s["model"]
        student_news = built_s["newsEncoder"]
        model_test = built_s["model_test"]
        student_user = tf.keras.Model(
            built_s["browsed_news_input"] + built_s["browsed_body_input"] + built_s["browsed_v_input"] + built_s["browsed_sv_input"],
            built_s["user_rep"],
        )

        built_t = build_naml_models(
            word_dict, embedding_mat, category, subcategory, args.learning_rate, clear_session=False, **_kw
        )
        teacher_full = built_t["model"]
        teacher_news = built_t["newsEncoder"]
        teacher_user = tf.keras.Model(
            built_t["browsed_news_input"] + built_t["browsed_body_input"] + built_t["browsed_v_input"] + built_t["browsed_sv_input"],
            built_t["user_rep"],
        )
        try:
            teacher_full.load_weights(tw)
        except Exception as e:
            print(
                "\n오류: 교사 가중치 로드 실패. 튜닝된 NAML_mind_2000.h5 라면 그때의 global_best_hparams와 "
                "동일한 그래프가 필요합니다.\n"
                "  예: --tune-log saved_models/naml_tune_actual_log.json\n",
                flush=True,
            )
            raise e
        teacher_full.trainable = False
        teacher_news.trainable = False
        teacher_user.trainable = False
        print(f"교사 가중치 로드 (동결): {tw}")

        optimizer = Adam(learning_rate=args.learning_rate)
        best_w, best_ep, best_mrr = train_kd_userdistill(
            student_model,
            teacher_news,
            student_news,
            teacher_user,
            student_user,
            make_train_gen,
            steps_per_epoch,
            optimizer,
            float(args.lambda_distill_user),
            float(args.lambda_distill_exp),
            H,
            args.epochs,
            teacher_exp_use_expected_body=bool(args.teacher_exp_use_expected_body),
            eval_after_epoch=eval_cb,
        )
        fallback_last_weights = [np.array(w) for w in student_model.get_weights()]
        fallback_last_model = student_model

        if best_w is not None:
            print(
                f"[run {run_no}] 기대본문 MRR 최고 에폭 {best_ep}/{args.epochs} (MRR={best_mrr:.6f})",
                flush=True,
            )
            if best_mrr > best_overall_mrr:
                best_overall_mrr = float(best_mrr)
                best_overall_run = run_no
                best_overall_epoch = int(best_ep)
                best_overall_weights = [np.array(w) for w in best_w]
        else:
            print(f"[run {run_no}] 에폭 평가 없음/지표 없음 → run 마지막 에폭 가중치 보관", flush=True)

    if best_overall_weights is not None and fallback_last_model is not None:
        fallback_last_model.set_weights(best_overall_weights)
        print(
            f"\n저장: 전체 {args.num_runs}회 중 최고 run={best_overall_run}, epoch={best_overall_epoch}/{args.epochs} "
            f"(MRR={best_overall_mrr:.6f}) 가중치 → {out_w}",
            flush=True,
        )
        fallback_last_model.save_weights(out_w)
    else:
        if fallback_last_model is None or fallback_last_weights is None:
            print("오류: 저장할 학생 가중치가 없습니다.", file=sys.stderr)
            sys.exit(1)
        fallback_last_model.set_weights(fallback_last_weights)
        print(
            "\n저장: 기대본문 테스트 지표가 없거나 에폭 평가가 꺼져 있어 "
            "마지막 run의 마지막 에폭 가중치를 저장합니다.",
            flush=True,
        )
        fallback_last_model.save_weights(out_w)

    print(f"학생 가중치 저장 완료: {out_w}")


if __name__ == "__main__":
    main()
