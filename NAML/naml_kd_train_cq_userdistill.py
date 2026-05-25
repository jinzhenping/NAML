#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML KD (후보 쿼리 사용자 인코딩 학생 + 양성 슬롯 사용자 증류):

  L = L_rec + lambda_user * L_distill_user_pos + lambda_exp * L_distill_exp

- 교사: `build_naml_models_candidate_query_user` + `--teacher-weights` (예: `naml_tune_actual_cq_teacher.py` 로 학습한 CQ 가중치).
  `--tune-log` 에는 그 학습 시 저장한 JSON(`global_best_hparams`)을 지정해 아키텍처를 맞춘다.
- 학생: 동일 CQ 그래프(가중치는 별도 초기화 후 KD).
- L_distill_user_pos: 양성 슬롯의 학생·교사 user_rep 각각 `model_user_stack` 에서 뽑아 1-cos 정렬.

예시:

python NAML/naml_kd_train_cq_userdistill.py \
  --teacher-weights saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5 \
  --tune-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json \
  --expected-body-train-dir user_preference/expected_body/MIND_2000/train_3cluster_11_13_8_rawtitle \
  --expected-body-test-dir user_preference/expected_body/MIND_2000/test_3cluster_11_13_8_rawtitle \
  --expected-body-first-n-sentences 2 \
  --mind-dataset-subdir MIND_2000 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lambda-distill-user 0.2 \
  --lambda-distill-exp 0.1 \
  --epochs 10 \
  --output-weights saved_models/MIND_2000/NAML_kd_student_cq_userdistill.h5 --teacher-exp-use-expected-body \
  --num-runs 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()

from eval_cluster_batch import load_expected_bodies_from_train_dir
from naml_eval_test import (
    _DEFAULT_ARCH,
    _arch_from_tune_log,
)
from naml_common import SEED, preprocess_news_file, preprocess_user_file, get_embedding
import naml_common as _naml_common
from naml_model_builder import build_naml_models_candidate_query_user

from naml_kd_train_userdistill import (
    _input_indices,
    generate_batch_data_train_kd,
    _to_tf_inputs,
    _cosine_1minus,
    _distill_news_term,
    run_test_set_eval,
)


def train_kd_cq_userdistill(
    student_model,
    model_user_stack,
    teacher_news_encoder,
    student_news_encoder,
    teacher_user_stack,
    make_train_gen,
    steps_per_epoch: int,
    optimizer: Adam,
    lambda_user: float,
    lambda_exp: float,
    H: int,
    epochs: int,
    teacher_exp_use_expected_body: bool = False,
    eval_after_epoch: Optional[Callable[[], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]] = None,
) -> Tuple[Optional[List[np.ndarray]], int, float, Optional[Dict[str, Any]]]:
    cand_title_idx, cand_body_idx, cand_v_idx, cand_sv_idx = _input_indices(H)

    best_weights: Optional[List[np.ndarray]] = None
    best_epoch_1based = -1
    best_mrr_expected = -1.0
    best_metrics_expected: Optional[Dict[str, Any]] = None

    for ep in range(epochs):
        train_gen = make_train_gen()

        for step in range(steps_per_epoch):
            batch_inputs, y, cand_act = next(train_gen)
            x_list = _to_tf_inputs(batch_inputs)
            y_t = tf.convert_to_tensor(y, dtype=tf.float32)
            cand_act_t = tf.convert_to_tensor(cand_act, dtype=tf.int32)

            teacher_news_targets: List[tf.Tensor] = []
            for k in range(5):
                t_k = x_list[cand_title_idx[k]]
                if teacher_exp_use_expected_body:
                    b_t_k = x_list[cand_body_idx[k]]
                else:
                    b_t_k = cand_act_t[:, k, :]
                vk = x_list[cand_v_idx[k]]
                svk = x_list[cand_sv_idx[k]]
                teacher_news_targets.append(
                    tf.stop_gradient(teacher_news_encoder([t_k, b_t_k, vk, svk], training=False))
                )
            pos = tf.argmax(y_t, axis=-1, output_type=tf.int32)
            batch_i = tf.range(tf.shape(y_t)[0], dtype=tf.int32)
            pos_idx2 = tf.stack([batch_i, pos], axis=-1)
            t_all = tf.stop_gradient(teacher_user_stack(x_list, training=False))
            t_user = tf.gather_nd(t_all, pos_idx2)

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

                s_all = model_user_stack(x_list, training=True)
                s_pos = tf.gather_nd(s_all, pos_idx2)
                loss_d_user = _cosine_1minus(s_pos, t_user)

                loss = loss_rec + lambda_user * loss_d_user + lambda_exp * loss_d_exp

            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

        if eval_after_epoch is not None:
            print(f"  [테스트셋] 에폭 {ep + 1} 평가 중...", flush=True)
            mr, me = eval_after_epoch()
            if mr is not None:
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
                    best_metrics_expected = dict(me)

    return best_weights, best_epoch_1based, best_mrr_expected, best_metrics_expected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NAML KD: CQ 교사·CQ 학생 + 양성 슬롯 사용자 증류 (교사 가중치는 CQ 사전학습)"
    )
    ap.add_argument("--teacher-weights", type=str, required=True)
    ap.add_argument(
        "--tune-log",
        type=str,
        default=None,
        help="CQ 교사/학생 빌드용 JSON (global_best_hparams). 예: naml_tune_actual_cq_teacher_log.json",
    )
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--cnn-filters", type=int, default=None)
    ap.add_argument("--cnn-kernel-size", type=int, default=None)
    ap.add_argument("--attention-dense-dim", type=int, default=None)
    ap.add_argument("--category-emb-dim", type=int, default=None)
    ap.add_argument("--expected-body-train-dir", type=str, required=True)
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument(
        "--max-history-clicks",
        type=int,
        default=None,
        metavar="N",
    )
    ap.add_argument("--lambda-distill-user", type=float, default=0.5)
    ap.add_argument("--lambda-distill-exp", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument(
        "--output-weights",
        type=str,
        default="saved_models/NAML_kd_student_cq_userdistill.h5",
    )
    ap.add_argument("--expected-body-test-dir", type=str, default=None)
    ap.add_argument("--no-epoch-eval", action="store_true")
    ap.add_argument(
        "--eval-actual-body",
        action="store_true",
        help="에폭 평가에 실제본문 predict 포함 (기본: 기대본문만 평가)",
    )
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--teacher-exp-use-expected-body", action="store_true")
    ap.add_argument(
        "--expected-body-first-n-sentences",
        "--train-expected-body-first-n-sentences",
        type=int,
        default=3,
        metavar="N",
        dest="expected_body_first_n_sentences",
    )
    ap.add_argument("--eval-expected-body-first-n-sentences", type=int, default=0, metavar="N")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--num-runs", type=int, default=1)
    ap.add_argument("--run-seed-step", type=int, default=1)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = max(0, int(args.expected_body_first_n_sentences))
    _eval_exp_n = max(0, int(args.eval_expected_body_first_n_sentences))

    sub = args.mind_dataset_subdir or os.environ.get("MIND_DATASET_SUBDIR", "MIND_2000")
    os.environ["MIND_DATASET_SUBDIR"] = sub

    train_dir = (
        str(_ROOT / args.expected_body_train_dir)
        if not os.path.isabs(args.expected_body_train_dir)
        else args.expected_body_train_dir
    )
    if not os.path.isdir(train_dir):
        print(f"오류: 기대본문 폴더 없음: {train_dir}", file=sys.stderr)
        sys.exit(1)

    expected_bodies_train = load_expected_bodies_from_train_dir(train_dir)
    print(f"기대본문 로드: {len(expected_bodies_train)}개 ({train_dir})")

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=None,
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
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    from naml_common import MAX_HISTORY_CLICKS

    news_index_reverse = {v: k for k, v in news_index.items()}
    H = MAX_HISTORY_CLICKS

    arch: Dict[str, Any] = dict(_DEFAULT_ARCH)
    if args.tune_log:
        tl = os.path.normpath(str(_ROOT / args.tune_log)) if not os.path.isabs(args.tune_log) else args.tune_log
        if os.path.isfile(tl):
            loaded = _arch_from_tune_log(tl)
            arch.update(loaded)
            print(f"튜닝 로그 아키텍처: {tl} → {loaded or '(global_best_hparams 없음)'}", flush=True)
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

    expected_bodies_test = None
    if args.expected_body_test_dir:
        from naml_eval_test import load_expected_bodies_from_dir

        test_d = (
            str(_ROOT / args.expected_body_test_dir)
            if not os.path.isabs(args.expected_body_test_dir)
            else args.expected_body_test_dir
        )
        if os.path.isdir(test_d):
            expected_bodies_test = load_expected_bodies_from_dir(test_d)
            print(f"테스트 기대본문: {len(expected_bodies_test)}개 ({test_d})")

    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    out_w = str(_ROOT / args.output_weights) if not os.path.isabs(args.output_weights) else args.output_weights
    os.makedirs(os.path.dirname(out_w) or ".", exist_ok=True)

    best_overall_mrr = float("-inf")
    best_overall_run = 1
    best_overall_epoch = args.epochs
    best_overall_weights = None
    best_overall_metrics: Optional[Dict[str, Any]] = None
    fallback_last_weights = None
    fallback_last_model = None

    for run_idx in range(max(1, args.num_runs)):
        run_no = run_idx + 1
        run_seed = int(args.seed) + run_idx * int(args.run_seed_step)
        np.random.seed(run_seed)
        tf.random.set_seed(run_seed)
        print(f"\n=== KD CQ run {run_no}/{args.num_runs} (seed={run_seed}) ===", flush=True)

        built_s = build_naml_models_candidate_query_user(
            word_dict, embedding_mat, category, subcategory, args.learning_rate, clear_session=True, **_kw
        )
        student_model = built_s["model"]
        model_user_stack = built_s["model_user_stack"]
        model_test = built_s["model_test"]

        built_t = build_naml_models_candidate_query_user(
            word_dict, embedding_mat, category, subcategory, args.learning_rate, clear_session=False, **_kw
        )
        teacher_full = built_t["model"]
        teacher_news = built_t["newsEncoder"]
        teacher_user_stack = built_t["model_user_stack"]
        try:
            teacher_full.load_weights(tw)
        except Exception as e:
            print(
                "\n오류: 교사 가중치 로드 실패. CQ 교사는 `naml_tune_actual_cq_teacher.py` 산출물과 "
                "동일 아키텍처가 필요합니다. --tune-log 에 해당 run 의 global_best_hparams 가 있는 JSON을 지정하거나 "
                "--cnn-filters 등으로 수동 맞추세요.\n",
                flush=True,
            )
            raise e
        teacher_full.trainable = False
        teacher_news.trainable = False
        teacher_user_stack.trainable = False
        print(f"교사(CQ) 가중치 로드 (동결): {tw}")

        def eval_after_epoch_fn():
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
                eval_actual_body=bool(args.eval_actual_body),
            )

        eval_cb = None if args.no_epoch_eval else eval_after_epoch_fn

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

        optimizer = Adam(learning_rate=args.learning_rate)
        best_w, best_ep, best_mrr, best_metrics = train_kd_cq_userdistill(
            student_model,
            model_user_stack,
            teacher_news,
            built_s["newsEncoder"],
            teacher_user_stack,
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
                best_overall_metrics = dict(best_metrics) if best_metrics is not None else None
        else:
            print(f"[run {run_no}] 에폭 평가 없음/지표 없음 → run 마지막 에폭 가중치 보관", flush=True)

    if best_overall_weights is not None and fallback_last_model is not None:
        fallback_last_model.set_weights(best_overall_weights)
        print(
            f"\n저장: run={best_overall_run}, epoch={best_overall_epoch}/{args.epochs} "
            f"(MRR={best_overall_mrr:.6f}) → {out_w}",
            flush=True,
        )
        if best_overall_metrics is not None:
            print(
                "[최고 기대본문] "
                f"MRR={float(best_overall_metrics.get('MRR', 0.0)):.6f}  "
                f"NDCG@5={float(best_overall_metrics.get('NDCG@5', 0.0)):.6f}  "
                f"Hit@1={float(best_overall_metrics.get('Hit@1', 0.0)):.6f}",
                flush=True,
            )
        fallback_last_model.save_weights(out_w)
    else:
        if fallback_last_model is None or fallback_last_weights is None:
            print("오류: 저장할 학생 가중치가 없습니다.", file=sys.stderr)
            sys.exit(1)
        fallback_last_model.set_weights(fallback_last_weights)
        print("\n저장: 마지막 run 마지막 에폭 가중치", flush=True)
        fallback_last_model.save_weights(out_w)

    print(f"학생 가중치 저장 완료: {out_w}")


if __name__ == "__main__":
    main()
