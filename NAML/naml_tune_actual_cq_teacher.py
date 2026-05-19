# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
후보 뉴스를 사용자 히스토리 어텐션의 쿼리로 쓰는 NAML(`build_naml_models_candidate_query_user`)의
하이퍼파라미터 탐색. 기본은 **실제 본문**; `--use-expected-body` 시 학습은 `--expected-body-first-n-sentences`,
튜닝 중 테스트 MRR 평가는 **기대본문 전체** (문장 컷 없음).

`naml_kd_train_cq_userdistill.py` 의 학생 그래프와 동일한 사용자 경로이므로, 여기서 저장한 가중치를
동일 아키텍처·동일 `HPARAM_CHOICES` 로 KD 교사로 쓸 수 있다(교사·학생 구조 정렬).

실행 (저장소 루트에서):
  python NAML/naml_tune_actual_cq_teacher.py --trials 12 --epochs-per-trial 8 --seed 42

테스트셋 평가 (CQ 그래프·튜닝 로그와 동일 hparam):
  python NAML/naml_eval_test_cq.py --weights ... --tune-log ... --mind-dataset-subdir MIND_2000 --actual-only

하이퍼 조합:
  - 기본: 아래 `HPARAM_CHOICES` 를 직접 수정.
  - CLI: `--fixed-learning-rate`, `--fixed-dropout-rate`, `--grid-cnn-filters`, … 로 덮어쓰기.
  - `--fixed-filter-kernel-grid` 는 CNN 폭·커널을 `--grid-cnn-filters` / `--grid-cnn-kernel-sizes` 로만 탐색할 때 쓰며, 이 플래그 사용 시 두 `--grid-*` 는 필수.
  - 동일 조합을 여러 시드로: `--repeat-per-combo N` (단일 페이즈만; `--two-phase` 와 병행 불가).
  
  python NAML/naml_tune_actual_cq_teacher.py --two-phase --trials 72 --screening-epochs 3 \
  --mind-dataset-subdir MIND_2000 \
  --refine-top-k 10 --epochs-per-trial 10 \
  --resume-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json \
  --out-weights saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5 \
  --out-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json

  python NAML/naml_tune_actual_cq_teacher.py \
  --mind-dataset-subdir MIND_2000 \
  --fixed-filter-kernel-grid \
  --grid-cnn-filters 512 \
  --grid-cnn-kernel-sizes 3 \
  --fixed-learning-rate 0.001 \
  --fixed-dropout-rate 0.25 \
  --fixed-attention-dense-dim 160 \
  --fixed-category-emb-dim 64 \
  --trials 1 \
  --epochs-per-trial 10 \
  --repeat-per-combo 5 \
  --out-weights saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5 \
  --out-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json

  # CQ 교사 + 기대본문 (학습·튜닝 중 테스트 MRR 모두 기대본문):
  python NAML/naml_tune_actual_cq_teacher.py --two-phase --trials 36 --screening-epochs 3 \
  --mind-dataset-subdir MIND_2000 --refine-top-k 5 --epochs-per-trial 8 \
  --use-expected-body \
  --expected-train-dir user_preference/expected_body/MIND_2000/train_3cluster_11_13_8_rawtitle \
  --expected-test-dir user_preference/expected_body/MIND_2000/test_3cluster_11_13_8_rawtitle \
  --expected-body-first-n-sentences 3 \
  --out-weights saved_models/MIND_2000/NAML_cq_mind_2000_expected.h5 \
  --out-log saved_models/MIND_2000/naml_tune_expected_cq_log.json
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
    get_embedding,
    preprocess_news_file,
    preprocess_user_file,
)
from naml_model_builder import build_naml_models_candidate_query_user
from naml_batch_generators import generate_batch_data_train as generate_batch_data_train_expected

from naml_tune_expected import (
    _resolve_expected_body_dir,
    evaluate_session_metrics,
    generate_batch_data_train_actual,
    load_expected_bodies_from_dir,
)

# 튜닝 그리드: 필요 시 이 dict 만 수정 (itertools.product 크기 = 고유 조합 개수)
HPARAM_CHOICES: dict[str, list] = {
    "learning_rate": [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    "dropout_rate": [0.2, 0.25, 0.3, 0.35, 0.4],
    "cnn_filters": [256, 300, 400, 512],
    "cnn_kernel_size": [2, 3, 4],
    "attention_dense_dim": [128, 160, 200, 256],
    "category_emb_dim": [32, 50, 64],
}

_HPARAM_KEY_ORDER = (
    "learning_rate",
    "dropout_rate",
    "cnn_filters",
    "cnn_kernel_size",
    "attention_dense_dim",
    "category_emb_dim",
)
_HP_KEYS = _HPARAM_KEY_ORDER


def effective_hparam_choices_from_args(ns: argparse.Namespace) -> dict[str, list]:
    """CLI 고정값·부분 그리드로 HPARAM_CHOICES 를 복사·덮어쓴다."""
    out: dict[str, list] = {k: list(v) for k, v in HPARAM_CHOICES.items()}
    if ns.fixed_learning_rate is not None:
        out["learning_rate"] = [float(ns.fixed_learning_rate)]
    if ns.fixed_dropout_rate is not None:
        out["dropout_rate"] = [float(ns.fixed_dropout_rate)]
    if ns.fixed_attention_dense_dim is not None:
        out["attention_dense_dim"] = [int(ns.fixed_attention_dense_dim)]
    if ns.fixed_category_emb_dim is not None:
        out["category_emb_dim"] = [int(ns.fixed_category_emb_dim)]
    if ns.grid_cnn_filters is not None:
        out["cnn_filters"] = [int(x) for x in ns.grid_cnn_filters]
    if ns.grid_cnn_kernel_sizes is not None:
        out["cnn_kernel_size"] = [int(x) for x in ns.grid_cnn_kernel_sizes]
    return out


def _hparam_grid_size_for(choices: dict[str, list]) -> int:
    p = 1
    for k in _HPARAM_KEY_ORDER:
        p *= len(choices[k])
    return p


def sample_hparams_for(rng: random.Random, choices: dict[str, list]) -> dict:
    return {k: rng.choice(choices[k]) for k in _HPARAM_KEY_ORDER}


def plan_hparam_trials_for(rng: random.Random, n_trials: int, choices: dict[str, list]) -> list[dict]:
    vals = [choices[k] for k in _HPARAM_KEY_ORDER]
    combos = [dict(zip(_HPARAM_KEY_ORDER, prod)) for prod in itertools.product(*vals)]
    rng.shuffle(combos)
    grid_n = len(combos)
    if n_trials <= grid_n:
        return combos[:n_trials]
    out = list(combos)
    for _ in range(n_trials - grid_n):
        out.append(sample_hparams_for(rng, choices))
    return out


def _hp_key(hp: dict) -> tuple:
    return tuple((k, hp[k]) for k in _HPARAM_KEY_ORDER)


def _load_seen_hparam_keys_from_log(log_path: str) -> set[tuple]:
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


def _load_previous_best_from_log(log_path: str) -> tuple[float, dict | None, dict | None]:
    """(best_mrr, best_hparams, best_epoch_metrics). best_epoch_metrics 는 MRR/NDCG@5/Hit@1 키."""
    best_mrr = -1.0
    best_hp: dict | None = None
    best_metrics: dict | None = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"경고: 이전 최고 성능 로드 실패: {e}")
        return best_mrr, best_hp, best_metrics

    rows = data.get("trials", [])
    if isinstance(rows, list):
        for r in rows:
            m = r.get("best_mrr_in_trial", None)
            hp = r.get("hparams", None)
            try:
                mv = float(m)
            except Exception:
                continue
            if mv > best_mrr:
                best_mrr = mv
                best_hp = hp if isinstance(hp, dict) else best_hp
                be = r.get("best_epoch")
                best_metrics = dict(be) if isinstance(be, dict) else best_metrics

    gg = data.get("global_best_mrr", None)
    gh = data.get("global_best_hparams", None)
    gm = data.get("global_best_epoch_metrics", None)
    if gg is not None:
        try:
            gfv = float(gg)
        except Exception:
            gfv = -1.0
        if gfv > best_mrr and isinstance(gh, dict):
            best_mrr = gfv
            best_hp = dict(gh)
            if isinstance(gm, dict):
                best_metrics = dict(gm)

    return best_mrr, best_hp, best_metrics


def _load_json_or_none(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _metrics_from_trials_for_mrr(trials: list, target_mrr: float) -> dict | None:
    """trials 항목 중 best_mrr_in_trial 이 target_mrr 과 일치(근사)인 첫 best_epoch dict."""
    for r in reversed(trials):
        if not isinstance(r, dict):
            continue
        try:
            mv = float(r.get("best_mrr_in_trial", -1.0))
        except Exception:
            continue
        if abs(mv - target_mrr) > 1e-5:
            continue
        be = r.get("best_epoch")
        if isinstance(be, dict):
            return dict(be)
    return None


def run_trial_cq(
    hp: dict,
    epochs: int,
    trial_seed: int,
    *,
    batch_size: int,
    word_dict,
    embedding_mat,
    category,
    subcategory,
    news_words,
    news_body,
    news_v,
    news_sv,
    news_index,
    all_train_pn,
    all_label,
    all_train_id,
    all_user_pos,
    all_train_userid_str,
    all_train_newsid_str,
    all_test_pn,
    all_test_label,
    all_test_id,
    all_test_user_pos,
    all_test_index,
    all_test_userid_str,
    all_test_newsid_str,
    use_expected_body: bool = False,
    expected_bodies_train=None,
    expected_bodies_test=None,
    history_body_title_only: bool = False,
):
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    tf.random.set_seed(trial_seed)

    built = build_naml_models_candidate_query_user(
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

    for _ep in range(epochs):
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
                history_body_title_only=history_body_title_only,
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
                history_body_title_only=history_body_title_only,
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
            history_body_title_only=history_body_title_only,
            eval_expected_body_clip_n_sentences=0 if use_expected_body else None,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NAML CQ-user 교사 하이퍼파라미터 탐색 (기본 실제본문, --use-expected-body 로 기대본문)"
    )
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs-per-trial", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mind-dataset-subdir", type=str, default=None)
    ap.add_argument(
        "--max-history-clicks",
        type=int,
        default=None,
        metavar="N",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-weights",
        type=str,
        default=os.path.join(_ROOT, "saved_models", "NAML_cq_teacher_mind_2000_actual.h5"),
    )
    ap.add_argument(
        "--out-log",
        type=str,
        default=os.path.join(_ROOT, "saved_models", "naml_tune_actual_cq_teacher_log.json"),
    )
    ap.add_argument("--allow-duplicate-hparams", action="store_true")
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--screening-epochs", type=int, default=2)
    ap.add_argument("--refine-top-k", type=int, default=5)
    ap.add_argument("--resume-log", type=str, default=None)
    ap.add_argument("--append-log", action="store_true")
    ap.add_argument(
        "--repeat-per-combo",
        type=int,
        default=1,
        metavar="N",
        help="동일 hparam 조합을 서로 다른 시드로 N번 반복 (단일 페이즈만; --two-phase 와 병행 불가, 기본 1)",
    )
    ap.add_argument(
        "--fixed-filter-kernel-grid",
        action="store_true",
        help="CNN filters/kernel 을 --grid-cnn-filters / --grid-cnn-kernel-sizes 로만 지정할 때 사용(두 인자 필수)",
    )
    ap.add_argument(
        "--grid-cnn-filters",
        nargs="+",
        type=int,
        default=None,
        metavar="F",
        help="cnn_filters 후보 목록 (예: 400 512). 지정 시 파일 기본 그리드의 해당 축을 덮어씀",
    )
    ap.add_argument(
        "--grid-cnn-kernel-sizes",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="cnn_kernel_size 후보 목록 (예: 3 4)",
    )
    ap.add_argument("--fixed-learning-rate", type=float, default=None, metavar="LR")
    ap.add_argument("--fixed-dropout-rate", type=float, default=None, metavar="D")
    ap.add_argument("--fixed-attention-dense-dim", type=int, default=None, metavar="DIM")
    ap.add_argument("--fixed-category-emb-dim", type=int, default=None, metavar="DIM")
    ap.add_argument(
        "--use-expected-body",
        action="store_true",
        help="학습·튜닝 중 테스트 MRR 평가 모두 기대본문(user_*/news_*.json) 사용",
    )
    ap.add_argument(
        "--expected-train-dir",
        type=str,
        default=None,
        help="train 기대본문 상위 폴더. --use-expected-body 일 때 필수",
    )
    ap.add_argument(
        "--expected-test-dir",
        type=str,
        default=None,
        help="test 기대본문 상위 폴더. --use-expected-body 일 때 필수",
    )
    ap.add_argument(
        "--expected-body-first-n-sentences",
        type=int,
        default=3,
        help="학습 시 기대본문 앞 N문장만 (0=전체). 튜닝 중 테스트 MRR 평가는 항상 기대본문 전체",
    )
    ap.add_argument(
        "--history-body-title-only",
        action="store_true",
        help="히스토리 본문 슬롯은 제목 토큰만(0 패딩). 후보는 기대/실제 본문 규칙 그대로",
    )
    args = ap.parse_args()

    if args.fixed_filter_kernel_grid:
        if not args.grid_cnn_filters or not args.grid_cnn_kernel_sizes:
            print(
                "오류: --fixed-filter-kernel-grid 는 --grid-cnn-filters 와 --grid-cnn-kernel-sizes 가 "
                "각각 하나 이상 필요합니다.",
                file=sys.stderr,
            )
            sys.exit(2)
    if args.two_phase and args.repeat_per_combo > 1:
        print("오류: --two-phase 와 --repeat-per-combo>1 은 함께 사용할 수 없습니다.", file=sys.stderr)
        sys.exit(2)
    if args.repeat_per_combo < 1:
        print("오류: --repeat-per-combo 는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(2)

    choices = effective_hparam_choices_from_args(args)
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
        if not _train_dir:
            print(f"오류: --expected-train-dir 없음: {args.expected_train_dir}", file=sys.stderr)
            sys.exit(1)
        if not _test_dir:
            print(f"오류: --expected-test-dir 없음: {args.expected_test_dir}", file=sys.stderr)
            sys.exit(1)
        _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES = max(0, int(args.expected_body_first_n_sentences))
        expected_bodies_train = load_expected_bodies_from_dir(_train_dir)
        expected_bodies_test = load_expected_bodies_from_dir(_test_dir)
        if not expected_bodies_train:
            print(f"오류: train 기대본문 0개: {_train_dir}", file=sys.stderr)
            sys.exit(1)
        if not expected_bodies_test:
            print(f"오류: test 기대본문 0개: {_test_dir}", file=sys.stderr)
            sys.exit(1)

    body_mode = "기대본문" if use_expected_body else "실제 본문"
    print(
        f"데이터 로드 ({body_mode}, CQ-user NAML, "
        f"{os.environ.get('MIND_DATASET_SUBDIR', 'MIND_2000')} train/test)..."
    )
    if use_expected_body:
        n_train_clip = _naml_common.EXPECTED_BODY_FIRST_N_SENTENCES
        train_clip_msg = (
            f"학습 앞 {n_train_clip}문장"
            if n_train_clip > 0
            else "학습 전체 문장"
        )
        print(
            f"  기대본문 train={len(expected_bodies_train)} test={len(expected_bodies_test)} "
            f"({train_clip_msg}, 평가=전체)",
            flush=True,
        )
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
    log_dir = os.path.dirname(os.path.abspath(args.out_log))
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    print(f"로그: {os.path.abspath(args.out_log)}", flush=True)
    print(f"가중치(전역 최고 MRR 갱신 시): {os.path.abspath(args.out_weights)}", flush=True)

    print(f"유효 하이퍼 그리드(축별 후보 수): { {k: len(v) for k, v in choices.items()} }", flush=True)

    rng = random.Random(args.seed)
    global_best_mrr = -1.0
    global_best_hp: dict | None = None
    global_best_metrics: dict | None = None
    log_trials: list[dict] = []

    grid_n = _hparam_grid_size_for(choices)
    seen_hparam_keys: set[tuple] = set()
    resume_log_path: str | None = None
    if args.resume_log:
        resume_log_path = (
            os.path.join(_ROOT, args.resume_log) if not os.path.isabs(args.resume_log) else args.resume_log
        )
        if os.path.isfile(resume_log_path):
            seen_hparam_keys = _load_seen_hparam_keys_from_log(resume_log_path)
            print(f"resume-log: {resume_log_path} (이미 시도한 조합 {len(seen_hparam_keys)}개)")
            prev_best_mrr, prev_best_hp, prev_best_metrics = _load_previous_best_from_log(resume_log_path)
            if prev_best_mrr > global_best_mrr:
                global_best_mrr = prev_best_mrr
                global_best_hp = prev_best_hp
                if isinstance(prev_best_metrics, dict):
                    global_best_metrics = dict(prev_best_metrics)
            print(
                f"resume 기준 이전 최고 MRR={global_best_mrr:.6f} "
                f"(초과 시 {args.out_weights} 저장)"
            )
        else:
            print(f"경고: --resume-log 없음: {resume_log_path}")

    if args.allow_duplicate_hparams:
        trial_hparams: list[dict] = []
        local_keys: set[tuple] = set()
        max_attempts = max(args.trials * 300, 3000)
        attempts = 0
        while len(trial_hparams) < args.trials and attempts < max_attempts:
            attempts += 1
            hp = sample_hparams_for(rng, choices)
            k = _hp_key(hp)
            if k in seen_hparam_keys or k in local_keys:
                continue
            trial_hparams.append(hp)
            local_keys.add(k)
        if len(trial_hparams) < args.trials:
            print(
                f"경고: unseen 조합 부족으로 {args.trials}개 중 {len(trial_hparams)}개만 생성"
            )
    else:
        all_planned = plan_hparam_trials_for(rng, grid_n, choices)
        trial_hparams = [hp for hp in all_planned if _hp_key(hp) not in seen_hparam_keys][: args.trials]
        if len(trial_hparams) < args.trials:
            print(f"경고: unseen 조합 부족, 요청 {args.trials}개 중 {len(trial_hparams)}개만 실행")
        if args.trials > grid_n:
            print(
                f"경고: 고유 그리드 {grid_n}개인데 trials={args.trials} → 이후는 무작위 보충(중복 가능)"
            )

    run_trials = len(trial_hparams)
    if run_trials == 0:
        print("실행할 새 조합이 없습니다.")
        return

    trial_schedule: list[tuple[dict, int, int]] | None = None
    if not args.two_phase:
        trial_schedule = []
        for ci, hp in enumerate(trial_hparams):
            for r in range(max(1, int(args.repeat_per_combo))):
                trial_schedule.append((hp, ci, r))

    history_body_title_only = bool(args.history_body_title_only)
    trial_kw = dict(
        batch_size=args.batch_size,
        word_dict=word_dict,
        embedding_mat=embedding_mat,
        category=category,
        subcategory=subcategory,
        news_words=news_words,
        news_body=news_body,
        news_v=news_v,
        news_sv=news_sv,
        news_index=news_index,
        all_train_pn=all_train_pn,
        all_label=all_label,
        all_train_id=all_train_id,
        all_user_pos=all_user_pos,
        all_train_userid_str=all_train_userid_str,
        all_train_newsid_str=all_train_newsid_str,
        all_test_pn=all_test_pn,
        all_test_label=all_test_label,
        all_test_id=all_test_id,
        all_test_user_pos=all_test_user_pos,
        all_test_index=all_test_index,
        all_test_userid_str=all_test_userid_str,
        all_test_newsid_str=all_test_newsid_str,
        use_expected_body=use_expected_body,
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=expected_bodies_test,
        history_body_title_only=history_body_title_only,
    )

    def _one_trial(
        trial_idx: int,
        total_in_phase: int,
        hp: dict,
        epochs: int,
        trial_seed: int,
        phase_label: str,
        extra_log: dict | None = None,
    ) -> None:
        nonlocal global_best_mrr, global_best_hp, global_best_metrics
        print(f"\n--- [{phase_label}] {trial_idx + 1}/{total_in_phase}  hparams={hp} ---")
        best_mrr, best_metrics, model = run_trial_cq(
            hp,
            epochs,
            trial_seed,
            **trial_kw,
        )
        print(
            f"  trial best MRR: {best_mrr:.6f}  |  "
            f"MRR={best_metrics['MRR']:.6f} NDCG@5={best_metrics['NDCG@5']:.6f} Hit@1={best_metrics['Hit@1']:.6f}"
        )
        row: dict = {
            "phase": phase_label,
            "hparams": hp,
            "epochs_in_phase": epochs,
            "best_mrr_in_trial": best_mrr,
            "best_epoch": best_metrics,
        }
        if extra_log:
            row.update(extra_log)
        log_trials.append(row)
        if best_mrr > global_best_mrr:
            global_best_mrr = best_mrr
            global_best_hp = dict(hp)
            global_best_metrics = dict(best_metrics)
            model.save_weights(args.out_weights)
            print(
                f"  [전역 갱신] 저장 → {args.out_weights}  "
                f"MRR={global_best_mrr:.6f}  NDCG@5={best_metrics['NDCG@5']:.6f}  Hit@1={best_metrics['Hit@1']:.6f}"
            )
        K.clear_session()

    if args.two_phase:
        k = min(args.refine_top_k, run_trials)
        print(
            f"\n[2-phase] 1차: trials={run_trials}, epochs={args.screening_epochs} → "
            f"상위 {k}개를 2차 epochs={args.epochs_per_trial}\n"
        )
        screening_rows: list[tuple[float, dict, dict]] = []
        for t in range(run_trials):
            hp = trial_hparams[t]
            trial_seed = args.seed + t * 9973
            print(f"\n--- [screening] {t + 1}/{run_trials}  hparams={hp} ---")
            best_mrr, best_metrics, model = run_trial_cq(
                hp,
                args.screening_epochs,
                trial_seed,
                **trial_kw,
            )
            print(
                f"  trial best MRR: {best_mrr:.6f}  |  "
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
                global_best_metrics = dict(best_metrics)
                model.save_weights(args.out_weights)
                print(
                    f"  [전역 갱신] 저장 → {args.out_weights}  "
                    f"MRR={global_best_mrr:.6f}  NDCG@5={best_metrics['NDCG@5']:.6f}  Hit@1={best_metrics['Hit@1']:.6f}"
                )
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

        print(f"\n[2-phase] 2차(refine): 상위 {len(top_hps)}개, 각 {args.epochs_per_trial} epochs\n")
        for j, hp in enumerate(top_hps):
            trial_seed = args.seed + 884422 + j * 9973
            _one_trial(j, len(top_hps), hp, args.epochs_per_trial, trial_seed, "refine", extra_log=None)
    else:
        assert trial_schedule is not None
        for t, (hp, ci, r) in enumerate(trial_schedule):
            trial_seed = args.seed + t * 9973 + r * 7919
            _one_trial(
                t,
                len(trial_schedule),
                hp,
                args.epochs_per_trial,
                trial_seed,
                "single",
                extra_log={"combo_idx": ci, "repeat_idx": r},
            )

    summary = {
        "model_user_encoder": "candidate_query_cross_attention",
        "use_expected_body": use_expected_body,
        "expected_train_dir": args.expected_train_dir if use_expected_body else None,
        "expected_test_dir": args.expected_test_dir if use_expected_body else None,
        "expected_body_first_n_sentences": (
            int(args.expected_body_first_n_sentences) if use_expected_body else None
        ),
        "history_body_title_only": history_body_title_only,
        "global_best_mrr": global_best_mrr,
        "global_best_hparams": global_best_hp,
        "global_best_epoch_metrics": dict(global_best_metrics) if global_best_metrics else None,
        "trials": log_trials,
        "max_history_clicks": int(MAX_HISTORY_CLICKS),
        "epochs_per_trial": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hparam_grid_size": grid_n,
        "hparam_choices_effective": {k: list(v) for k, v in choices.items()},
        "repeat_per_combo": int(args.repeat_per_combo),
        "allow_duplicate_hparams": bool(args.allow_duplicate_hparams),
        "resume_log": resume_log_path,
        "num_seen_hparams_loaded": len(seen_hparam_keys),
        "num_trials_requested": int(args.trials),
        "num_unique_hparam_combos": len(trial_hparams),
        "num_training_runs": len(log_trials),
        "num_trials_executed": len(log_trials),
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
                old_m = old.get("global_best_epoch_metrics", None)
                if isinstance(old_m, dict):
                    summary["global_best_epoch_metrics"] = dict(old_m)
                else:
                    fb = _metrics_from_trials_for_mrr(old_trials, old_best)
                    summary["global_best_epoch_metrics"] = fb
            summary["trials"] = merged_trials
            summary["append_log"] = True
            summary["append_log_auto_by_resume"] = bool(args.resume_log and not args.append_log)
            summary["appended_from"] = args.out_log
            summary["num_trials_total_after_append"] = len(merged_trials)

    with open(args.out_log, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    bm = summary.get("global_best_epoch_metrics")
    print(f"\n완료. 로그: {args.out_log}", flush=True)
    if isinstance(bm, dict) and all(k in bm for k in ("MRR", "NDCG@5", "Hit@1")):
        print(
            f"전역 최고 (테스트 세션): MRR={float(bm['MRR']):.6f}  "
            f"NDCG@5={float(bm['NDCG@5']):.6f}  Hit@1={float(bm['Hit@1']):.6f}",
            flush=True,
        )
    else:
        print(
            f"전역 최고 MRR={float(summary.get('global_best_mrr', -1.0)):.6f}  "
            f"(NDCG@5/Hit@1 은 이 로그에 global_best_epoch_metrics 없음)",
            flush=True,
        )
    best_hp_out = summary.get("global_best_hparams")
    if isinstance(best_hp_out, dict) and best_hp_out:
        print(f"최적 hparams: {best_hp_out}", flush=True)


if __name__ == "__main__":
    main()
