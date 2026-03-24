"""
트레이닝 세션별 user_rep(히스토리 기반) → 유저 ID별 평균 벡터 → k-means.
NAML.py를 실행하지 않으며, naml_common + naml_model_builder만 사용.

실행 예 (프로젝트 루트):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/cluster_train_users_kmeans.py --k 3
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/cluster_train_users_kmeans.py --k 3 --assign-test

데이터셋 폴더는 아래 「사용자 설정」에서 지정 (PowerShell $env: 불필요).
가중치 기본: 프로젝트 루트/saved_models/NAML_mind_2000.h5
클러스터 CSV 기본: 이 스크립트와 같은 폴더 (NAML/user_kmeans_k{K}_{SUBDIR}.csv)
--assign-test: 트레이닝에서 fit한 동일 k-means로 테스트 세션 user_rep → 유저별 평균 후 predict (별도 CSV).
"""
import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Optional, Sequence

import numpy as np
import tensorflow as tf
from keras.models import Model

# ---------------------------------------------------------------------------
# 사용자 설정 — 여기만 수정해도 됩니다 ($env:MIND_DATASET_SUBDIR 불필요)
# ---------------------------------------------------------------------------
# dataset/<이 이름>/ 아래 MIND_news.tsv, MIND_train_*.tsv, MIND_test_*.tsv 사용
SCRIPT_MIND_DATASET_SUBDIR = "MIND_2000"

# True: 아래 값이 환경변수보다 우선 (권장). False: 셸에 설정한 MIND_* 환경변수 사용
USE_SCRIPT_DATASET_CONFIG = True

# 선택: TSV 파일명을 고정할 때만 문자열 지정. None이면 서브폴더 규칙/자동 탐색
SCRIPT_MIND_NEWS_FILENAME: Optional[str] = None
SCRIPT_MIND_TRAIN_FILENAME: Optional[str] = None
SCRIPT_MIND_TEST_FILENAME: Optional[str] = None

# NAML.py와 동일 학습률 (모델 compile에만 사용)
MAIN_LR = 0.0005

# 기본 가중치 (프로젝트 루트 기준 → <루트>/saved_models/NAML_mind_2000.h5)
DEFAULT_WEIGHTS_RELATIVE = os.path.join("saved_models", "NAML_mind_2000.h5")
# ---------------------------------------------------------------------------


def _apply_dataset_env(mind_subdir_cli: Optional[str]) -> None:
    """naml_common import 전에 호출. dataset 경로 관련 환경변수 설정."""
    if USE_SCRIPT_DATASET_CONFIG:
        subdir = mind_subdir_cli if mind_subdir_cli else SCRIPT_MIND_DATASET_SUBDIR
        os.environ["MIND_DATASET_SUBDIR"] = subdir
        for key, val in (
            ("MIND_NEWS_FILENAME", SCRIPT_MIND_NEWS_FILENAME),
            ("MIND_TRAIN_FILENAME", SCRIPT_MIND_TRAIN_FILENAME),
            ("MIND_TEST_FILENAME", SCRIPT_MIND_TEST_FILENAME),
        ):
            if val:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]
    elif mind_subdir_cli:
        os.environ["MIND_DATASET_SUBDIR"] = mind_subdir_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="NAML 트레이닝 유저 k-means (user_rep)")
    parser.add_argument("--k", type=int, default=3, help="클러스터 개수 (>=2)")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="학습된 가중치 .h5 (기본: 프로젝트 루트/saved_models/NAML_mind_2000.h5)",
    )
    parser.add_argument("--batch", type=int, default=64, help="user_rep 추론 배치 크기")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 CSV 경로 (기본: cluster_train_users_kmeans.py와 같은 폴더/user_kmeans_k{K}_{SUBDIR}.csv)",
    )
    parser.add_argument(
        "--mind-subdir",
        type=str,
        default=None,
        help="dataset 하위 폴더명 (기본: 스크립트 상단 SCRIPT_MIND_DATASET_SUBDIR). "
        "USE_SCRIPT_DATASET_CONFIG=True 일 때만 의미 있음.",
    )
    parser.add_argument(
        "--assign-test",
        action="store_true",
        help="트레이닝으로 fit한 k-means를 그대로 써서 테스트 세션 user_rep(유저별 평균)에 cluster 할당. 별도 CSV 저장.",
    )
    parser.add_argument(
        "--out-test",
        type=str,
        default=None,
        help="--assign-test 시 테스트 CSV 경로 (기본: user_kmeans_k{K}_{SUBDIR}_test.csv)",
    )
    args = parser.parse_args()

    if args.k < 2:
        print("오류: --k 는 2 이상이어야 합니다.")
        sys.exit(1)

    _apply_dataset_env(args.mind_subdir)

    # 환경 반영 후에만 로드 (MIND_* 경로가 여기서 확정됨)
    from naml_common import (
        MAX_BODY_LENGTH,
        MAX_SENT_LENGTH,
        SEED,
        MIND_DATASET_SUBDIR,
        get_embedding,
        preprocess_news_file,
        preprocess_user_file,
    )
    from naml_model_builder import build_naml_models

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("오류: scikit-learn 필요 (pip install scikit-learn)")
        sys.exit(1)

    _naml_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_naml_dir)
    weights_path = args.weights or os.path.join(_project_root, DEFAULT_WEIGHTS_RELATIVE)
    if not os.path.isfile(weights_path):
        print(f"오류: 가중치 파일 없음: {weights_path}")
        sys.exit(1)

    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)

    print(f"\n{'='*60}")
    print(f"cluster_train_users_kmeans: k={args.k}, weights={weights_path}")
    print(f"데이터셋: dataset/{MIND_DATASET_SUBDIR}/")
    print("(user_rep는 유저 임베딩 테이블이 아니라 히스토리로부터 계산됩니다.)")
    print(f"{'='*60}\n")

    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=None,
        expected_bodies_test=None,
    )
    (
        _userid_dict,
        all_train_pn,
        all_label,
        all_train_id,
        _all_test_pn,
        _all_test_label,
        _all_test_id,
        all_user_pos,
        all_test_user_pos,
        all_test_index,
        _cand_tr,
        _cand_te,
        all_train_userid_str,
        _all_train_newsid_str,
        all_test_userid_str,
        _all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    print(f"트레이닝 세션 수: {len(all_train_id)}")
    print(f"테스트 세션 수: {len(all_test_index)}")

    embedding_mat = get_embedding(word_dict)
    built = build_naml_models(word_dict, embedding_mat, category, subcategory, MAIN_LR)
    model = built["model"]
    user_rep = built["user_rep"]
    browsed_news_input = built["browsed_news_input"]
    browsed_body_input = built["browsed_body_input"]
    browsed_v_input = built["browsed_v_input"]
    browsed_sv_input = built["browsed_sv_input"]
    max_sents = built["MAX_SENTS"]

    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    user_encoder_model = Model(
        inputs=browsed_news_input + browsed_body_input + browsed_v_input + browsed_sv_input,
        outputs=user_rep,
    )

    def collect_user_reps(
        session_row_indices: Sequence[int],
        user_pos_arr: Any,
        userid_per_row: Sequence[str],
    ) -> defaultdict:
        """세션당 첫 행 인덱스 목록 → 유저별 user_rep 벡터 리스트."""
        utv: defaultdict = defaultdict(list)
        n_sess = len(session_row_indices)
        for start in range(0, n_sess, args.batch):
            batch_idx = list(session_row_indices[start : start + args.batch])
            B = len(batch_idx)
            in_t = [np.zeros((B, MAX_SENT_LENGTH), dtype="int32") for _ in range(max_sents)]
            in_b = [np.zeros((B, MAX_BODY_LENGTH), dtype="int32") for _ in range(max_sents)]
            in_v = [np.zeros((B, 1), dtype="int32") for _ in range(max_sents)]
            in_sv = [np.zeros((B, 1), dtype="int32") for _ in range(max_sents)]
            for bi, idx in enumerate(batch_idx):
                user_pos_indices = np.array(user_pos_arr[idx], dtype="int32")
                bn = news_words[user_pos_indices]
                bb = news_body[user_pos_indices]
                bv = news_v[user_pos_indices]
                bsv = news_sv[user_pos_indices]
                for k in range(max_sents):
                    in_t[k][bi] = bn[k]
                    in_b[k][bi] = bb[k]
                    in_v[k][bi] = bv[k]
                    in_sv[k][bi] = bsv[k]
            feed = in_t + in_b + in_v + in_sv
            reps = user_encoder_model.predict(feed, verbose=0)
            for bi, idx in enumerate(batch_idx):
                uid = userid_per_row[idx]
                utv[uid].append(reps[bi])
        return utv

    n_train = len(all_label)
    train_session_rows = list(range(n_train))
    user_to_vecs = collect_user_reps(train_session_rows, all_user_pos, all_train_userid_str)

    user_ids_sorted = sorted(user_to_vecs.keys(), key=str)
    X = np.stack([np.mean(np.stack(user_to_vecs[u], axis=0), axis=0) for u in user_ids_sorted])
    km = KMeans(n_clusters=args.k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    # 스크립트(NAML/)와 동일 폴더에 저장 (기본)
    out_csv = args.out or os.path.join(_naml_dir, f"user_kmeans_k{args.k}_{MIND_DATASET_SUBDIR}.csv")
    _out_parent = os.path.dirname(os.path.abspath(out_csv))
    if _out_parent:
        os.makedirs(_out_parent, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("user_id,cluster\n")
        for u, lab in zip(user_ids_sorted, labels):
            f.write(f"{u},{int(lab)}\n")

    print(f"저장: {out_csv}  (유저 수: {len(user_ids_sorted)})")
    for c in range(args.k):
        print(f"  클러스터 {c}: {int(np.sum(labels == c))}명")

    if args.assign_test:
        if len(all_test_index) == 0:
            print("경고: 테스트 세션이 없어 --assign-test 를 건너뜁니다.")
        else:
            test_session_rows = [all_test_index[s][0] for s in range(len(all_test_index))]
            user_to_vecs_te = collect_user_reps(test_session_rows, all_test_user_pos, all_test_userid_str)
            user_ids_te = sorted(user_to_vecs_te.keys(), key=str)
            X_te = np.stack([np.mean(np.stack(user_to_vecs_te[u], axis=0), axis=0) for u in user_ids_te])
            labels_te = km.predict(X_te)
            out_csv_te = args.out_test or os.path.join(
                _naml_dir, f"user_kmeans_k{args.k}_{MIND_DATASET_SUBDIR}_test.csv"
            )
            _ote_parent = os.path.dirname(os.path.abspath(out_csv_te))
            if _ote_parent:
                os.makedirs(_ote_parent, exist_ok=True)
            with open(out_csv_te, "w", encoding="utf-8") as f:
                f.write("user_id,cluster\n")
                for u, lab in zip(user_ids_te, labels_te):
                    f.write(f"{u},{int(lab)}\n")
            print(f"\n[테스트 할당] 트레이닝과 동일 k-means 중심으로 predict → 저장: {out_csv_te}  (유저 수: {len(user_ids_te)})")
            for c in range(args.k):
                print(f"  테스트 클러스터 {c}: {int(np.sum(labels_te == c))}명")

    print("완료.")


if __name__ == "__main__":
    main()
