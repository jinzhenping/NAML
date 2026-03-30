#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML 지식 증류 학습: L_KD = L_rec + lambda * L_distill

- L_rec: 후보·히스토리 입력에 대해 기존과 동일한 categorical_crossentropy (후보 본문=기대본문).
- L_distill: 후보마다 학생 newsEncoder(기대본문) vs 동결 교사 newsEncoder(실제본문) 임베딩 코사인,
  mean_k mean_batch (1 - cos).

교사 가중치: 사전학습된 saved_models/NAML_mind_2000.h5 등 (model.save_weights 형식, build_naml_models 와 동일 구조).

--output-weights 저장: 에폭마다 테스트셋 기대본문 MRR을 잰 경우, 그중 MRR이 가장 높은 에폭의 가중치를 저장.
(기대본문 평가가 없으면 마지막 에폭 가중치)

프로젝트 루트에서:
  set PYTHONPATH=NAML
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/naml_kd_train.py --teacher-weights saved_models/NAML_mind_2000.h5 \
    --expected-body-train-dir body_generation/output/MIND_2000/train_3cluster_11_13_8 \
    --expected-body-test-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
    --mind-dataset-subdir MIND_2000 --epochs 5 --lambda-distill 0.5 \
    --output-weights saved_models/NAML_mind_2000_kd.h5

  # 에폭마다 테스트 평가 끄기: --no-epoch-eval
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
from eval_test_expected import calc_metrics_from_scores, load_expected_bodies_from_dir
from naml_batch_generators import generate_batch_data_test
from naml_common import (
    MAX_BODY_LENGTH,
    MAX_HISTORY_CLICKS,
    MAX_SENT_LENGTH,
    SEED,
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_model_builder import build_naml_models


def _norm_expected_body_key(uid, nid):
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def _input_indices(H: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    """batch_inputs 리스트에서 후보 k별 텐서 인덱스 (NAML.py generate_batch_data_train 순서와 동일)."""
    cand_title = list(range(5))
    cand_body = [5 + H + k for k in range(5)]
    base_v = 5 + H + 5 + H
    cand_v = [base_v + k for k in range(5)]
    base_sv = base_v + 5 + H
    cand_sv = [base_sv + k for k in range(5)]
    return cand_title, cand_body, cand_v, cand_sv


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
    """
    기대본문 후보로 batch_inputs 생성 + 후보 실제본문 토큰 (B,5,300) 추가 반환.
    Yields: (batch_inputs, batch_labels, candidate_actual_bodies)  # actual: int32 (B,5,300)
    """
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
                        body_tokens = word_tokenize(expected_body.lower()) if expected_body else []
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
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """테스트셋 실제본문 / 기대본문(있으면) 각각 model_test.predict 후 MRR·NDCG@5·Hit@1."""
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
        )
        score_exp = model_test.predict(testgen_exp, steps=test_steps, verbose=0)
        metrics_exp = calc_metrics_from_scores(score_exp, all_test_label, all_test_index)

    return metrics_real, metrics_exp


def _distill_term(student_enc, teacher_enc, title_k, body_exp_k, body_act_k, v_k, sv_k):
    emb_s = student_enc([title_k, body_exp_k, v_k, sv_k])
    emb_t = tf.stop_gradient(teacher_enc([title_k, body_act_k, v_k, sv_k]))
    emb_s_n = tf.nn.l2_normalize(emb_s, axis=-1)
    emb_t_n = tf.nn.l2_normalize(emb_t, axis=-1)
    cos = tf.reduce_sum(emb_s_n * emb_t_n, axis=-1)
    return tf.reduce_mean(1.0 - cos)


def train_kd(
    student_model,
    teacher_news_encoder,
    student_news_encoder,
    make_train_gen,
    steps_per_epoch: int,
    optimizer: Adam,
    lam: float,
    H: int,
    epochs: int,
    eval_after_epoch: Optional[Callable[[], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = None,
) -> Tuple[Optional[List[np.ndarray]], int, float]:
    """
    반환: (기대본문 테스트 MRR 최고일 때 가중치 스냅샷, 해당 에폭(1-based), 그때 MRR).
    기대본문 지표를 한 번도 갱신하지 못하면 (None, -1, -1.0).
    """
    cand_title_idx, cand_body_idx, cand_v_idx, cand_sv_idx = _input_indices(H)

    best_weights: Optional[List[np.ndarray]] = None
    best_epoch_1based = -1
    best_mrr_expected = -1.0

    for ep in range(epochs):
        print(f"\n=== KD epoch {ep + 1}/{epochs} ===", flush=True)
        mean_rec = tf.keras.metrics.Mean()
        mean_dist = tf.keras.metrics.Mean()
        mean_total = tf.keras.metrics.Mean()
        train_gen = make_train_gen()

        for step in range(steps_per_epoch):
            batch_inputs, y, cand_act = next(train_gen)
            x_list = _to_tf_inputs(batch_inputs)
            y_t = tf.convert_to_tensor(y, dtype=tf.float32)
            cand_act_t = tf.convert_to_tensor(cand_act, dtype=tf.int32)

            with tf.GradientTape() as tape:
                logits = student_model(x_list, training=True)
                loss_rec = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_t, logits))

                loss_d_parts = []
                for k in range(5):
                    t_k = x_list[cand_title_idx[k]]
                    be_k = x_list[cand_body_idx[k]]
                    ba_k = cand_act_t[:, k, :]
                    vk = x_list[cand_v_idx[k]]
                    svk = x_list[cand_sv_idx[k]]
                    loss_d_parts.append(_distill_term(student_news_encoder, teacher_news_encoder, t_k, be_k, ba_k, vk, svk))
                loss_d = tf.add_n(loss_d_parts) / 5.0
                loss = loss_rec + lam * loss_d

            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

            mean_rec.update_state(loss_rec)
            mean_dist.update_state(loss_d)
            mean_total.update_state(loss)

            if (step + 1) % max(1, steps_per_epoch // 5) == 0 or step == 0:
                print(
                    f"  step {step + 1}/{steps_per_epoch}  "
                    f"L_rec={float(mean_rec.result()):.4f}  "
                    f"L_distill={float(mean_dist.result()):.4f}  "
                    f"total={float(mean_total.result()):.4f}",
                    flush=True,
                )

        print(
            f"epoch {ep + 1} end: L_rec={float(mean_rec.result()):.4f}  "
            f"L_distill={float(mean_dist.result()):.4f}  "
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
    ap = argparse.ArgumentParser(description="NAML KD: L_rec + lambda * L_distill (기대본문 학생, 실제본문 교사)")
    ap.add_argument("--teacher-weights", type=str, required=True, help="교사 model.save_weights 경로 (프로젝트 루트 기준)")
    ap.add_argument(
        "--expected-body-train-dir",
        type=str,
        required=True,
        help="기대본문 JSON 루트 (user_*/news_*.json), train 전용",
    )
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument(
        "--lambda-distill",
        type=float,
        default=0.5,
        metavar="LAMBDA",
        help="L_KD = L_rec + LAMBDA * L_distill",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument(
        "--output-weights",
        type=str,
        default="saved_models/NAML_kd_student.h5",
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
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

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
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=None,
    )
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
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    news_index_reverse = {v: k for k, v in news_index.items()}
    H = MAX_HISTORY_CLICKS

    built_s = build_naml_models(word_dict, embedding_mat, category, subcategory, args.learning_rate, clear_session=True)
    student_model = built_s["model"]
    student_news = built_s["newsEncoder"]
    model_test = built_s["model_test"]

    built_t = build_naml_models(word_dict, embedding_mat, category, subcategory, args.learning_rate, clear_session=False)
    teacher_full = built_t["model"]
    teacher_news = built_t["newsEncoder"]

    tw = str(_ROOT / args.teacher_weights) if not os.path.isabs(args.teacher_weights) else args.teacher_weights
    if not os.path.isfile(tw):
        print(f"오류: 교사 가중치 없음: {tw}", file=sys.stderr)
        sys.exit(1)
    teacher_full.load_weights(tw)
    teacher_full.trainable = False
    teacher_news.trainable = False
    print(f"교사 가중치 로드 (동결): {tw}")

    optimizer = Adam(learning_rate=args.learning_rate)

    n_train = len(all_train_id)
    steps_per_epoch = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    print(f"샘플 수: {n_train}, batch_size={args.batch_size}, steps/epoch={steps_per_epoch}, λ={args.lambda_distill}")

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

    best_w, best_ep, best_mrr = train_kd(
        student_model,
        teacher_news,
        student_news,
        make_train_gen,
        steps_per_epoch,
        optimizer,
        float(args.lambda_distill),
        H,
        args.epochs,
        eval_after_epoch=eval_cb,
    )

    out_w = str(_ROOT / args.output_weights) if not os.path.isabs(args.output_weights) else args.output_weights
    os.makedirs(os.path.dirname(out_w) or ".", exist_ok=True)

    if best_w is not None:
        student_model.set_weights(best_w)
        print(
            f"\n저장: 테스트셋 기대본문 MRR 최고 에폭 {best_ep}/{args.epochs} "
            f"(MRR={best_mrr:.6f}) 가중치 → {out_w}",
            flush=True,
        )
    else:
        print(
            "\n저장: 기대본문 테스트 지표가 없거나 에폭 평가가 꺼져 있어 마지막 에폭 가중치를 저장합니다.",
            flush=True,
        )

    student_model.save_weights(out_w)
    print(f"학생 가중치 저장 완료: {out_w}")


if __name__ == "__main__":
    main()
