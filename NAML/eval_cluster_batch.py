#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
클러스터 배치(트레인 세션)에 대해 사전학습 NAML 가중치로
실제본문 vs 기대본문 성능을 비교하고 NAML/results/resultN.txt 에 JSON 저장.

프로젝트 루트에서:
  python NAML/eval_cluster_batch.py \\
    --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv --cluster-id 0 --batch-index 0 \\
    --train-body-dir body_generation/output/MIND_2000/cluster0_batch0

기대본문은 generate_body_cluster_train_batches.py 가 저장한
  body_generation/output/<데이터셋>/cluster<C>_batch<B>/
구조의 user_*/news_*.json 을 로드합니다.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# 프로젝트 루트
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))


def _norm_expected_body_key(uid, nid):
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def load_expected_bodies_from_train_dir(train_dir: str) -> Dict[Tuple[str, str], str]:
    expected_bodies: Dict[Tuple[str, str], str] = {}
    if not train_dir or not os.path.isdir(train_dir):
        return expected_bodies
    for user_folder in os.listdir(train_dir):
        user_path = os.path.join(train_dir, user_folder)
        if not os.path.isdir(user_path):
            continue
        if not user_folder.startswith("user_"):
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
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def load_news_titles(news_tsv: str) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    if not os.path.isfile(news_tsv):
        return titles
    with open(news_tsv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            titles[parts[0]] = parts[3]
    return titles


def _norm_cluster_uid(u: str) -> str:
    try:
        return str(int(float(str(u).strip())))
    except (ValueError, TypeError):
        return str(u).strip()


def _load_cluster_users(csv_path: Path, cluster_id: int) -> Set[str]:
    users: Set[str] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            uid = str(row.get("user_id", row.get("user", ""))).strip()
            cl = row.get("cluster", row.get("Cluster", ""))
            if not uid or cl == "":
                continue
            try:
                c = int(float(cl))
            except ValueError:
                continue
            if c == cluster_id:
                users.add(_norm_cluster_uid(uid))
    return users


def flatten_train_sessions_for_test_gen(
    session_indices: List[int],
    all_train_pn: np.ndarray,
    all_label: np.ndarray,
    all_train_id: np.ndarray,
    all_user_pos: np.ndarray,
    all_train_userid_str,
    all_train_newsid_str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], np.ndarray, List[Tuple[int, int]]]:
    """트레인 세션을 model_test / generate_batch_data_test 형식(후보당 1행)으로 펼침."""
    flat_pn: List[int] = []
    flat_label: List[int] = []
    flat_tid: List[int] = []
    flat_uid: List[str] = []
    flat_nid: List[str] = []
    flat_rows: List[np.ndarray] = []
    ranges: List[Tuple[int, int]] = []
    pos = 0
    for s in session_indices:
        ranges.append((pos, pos + 5))
        for p in range(5):
            flat_pn.append(int(all_train_pn[s][p]))
            flat_label.append(int(all_label[s][p]))
            flat_tid.append(int(all_train_id[s]))
            flat_uid.append(str(all_train_userid_str[s]).strip())
            flat_nid.append(str(all_train_newsid_str[s][p]).strip() if p < len(all_train_newsid_str[s]) else "")
            flat_rows.append(np.array(all_user_pos[s], dtype=np.int32))
            pos += 1
    flat_user_pos = np.stack(flat_rows, axis=0)
    return (
        np.array(flat_pn, dtype=np.int32),
        np.array(flat_label, dtype=np.int32),
        np.array(flat_tid, dtype=np.int32),
        flat_uid,
        flat_nid,
        flat_user_pos,
        ranges,
    )


def compute_expected_body_coverage_for_batch(
    batch_sessions: List[int],
    all_train_userid_str,
    all_train_newsid_str,
    all_train_pn: np.ndarray,
    expected_bodies: Dict[Tuple[str, str], str],
) -> Dict[str, Any]:
    """
    기대본문 평가 시 실제로 매칭되는 비율 (NAML 제너레이터와 동일 규칙).
    - 후보 슬롯: 세션당 5칸 중 news 인덱스 != 0 인 것만 집계 (패딩 제외).
    - 키: _norm_expected_body_key(user_id, news_id_str).
    """
    non_padding = 0
    matched = 0
    empty_news_id = 0
    for s in batch_sessions:
        uid = all_train_userid_str[s]
        news_ids_row = all_train_newsid_str[s]
        for j in range(5):
            news_idx = int(all_train_pn[s][j])
            if news_idx == 0:
                continue
            non_padding += 1
            nid = str(news_ids_row[j]).strip() if j < len(news_ids_row) and news_ids_row[j] is not None else ""
            if not nid:
                empty_news_id += 1
                continue
            key = _norm_expected_body_key(uid, nid)
            if key in expected_bodies:
                matched += 1
    rate = (matched / non_padding) if non_padding else 0.0
    return {
        "json_entries_loaded": len(expected_bodies),
        "batch_candidate_slots_non_padding": non_padding,
        "batch_slots_with_empty_news_id": empty_news_id,
        "batch_slots_matched_expected_body": matched,
        "batch_match_rate": round(rate, 6),
    }


def next_free_result_index(results_dir: Path) -> int:
    existing: List[int] = []
    for p in glob.glob(str(results_dir / "result*.txt")):
        m = re.search(r"result(\d+)\.txt$", os.path.basename(p))
        if m:
            existing.append(int(m.group(1)))
    return max(existing, default=-1) + 1


def _resolve_mind_dataset_subdir(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return os.environ.get("MIND_DATASET_SUBDIR", "MIND_2000")


def _pair_json_path(train_body_dir: str, uid: str, news_id: str) -> str:
    """body_generation 저장 구조: <dir>/user_<uid>/news_<news_id>.json"""
    u = _norm_cluster_uid(uid)
    n = str(news_id).strip()
    return os.path.join(train_body_dir, f"user_{u}", f"news_{n}.json")


def _load_pair_json(train_body_dir: str, uid: str, news_id: str) -> Optional[Dict[str, Any]]:
    path = _pair_json_path(train_body_dir, uid, news_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_diagnostic(
    session_idx: int,
    diag_type: str,
    train_body_dir: str,
    expected_bodies: Dict[Tuple[str, str], str],
    all_train_userid_str,
    all_train_newsid_str,
    all_label,
    news_index_reverse: Dict[int, str],
    news_titles: Dict[str, str],
    all_user_pos: np.ndarray,
) -> Dict[str, Any]:
    """
    세션 메타는 기대본문 JSON( generate_body.py 가 저장한 user_history, candidate_title, generated_body )을
    우선 사용. 없거나 비어 있으면 MIND_news.tsv + NAML 전처리 히스토리로 폴백.
    """
    uid = str(all_train_userid_str[session_idx]).strip()
    labels = all_label[session_idx]
    pos_positions = np.where(labels == 1)[0]
    p = int(pos_positions[0]) if len(pos_positions) else 0
    nid = str(all_train_newsid_str[session_idx][p]).strip() if p < len(all_train_newsid_str[session_idx]) else ""
    key = _norm_expected_body_key(uid, nid)
    gen_body = expected_bodies.get(key, "")

    hist_titles: List[str] = []
    cand_title = ""

    pair = _load_pair_json(train_body_dir, uid, nid)
    if pair:
        uh = pair.get("user_history")
        if isinstance(uh, list) and uh:
            hist_titles = [str(x) for x in uh]
        ct = pair.get("candidate_title")
        if ct is not None and str(ct).strip():
            cand_title = str(ct).strip()
        gb = pair.get("generated_body")
        if gb is not None and str(gb).strip():
            gen_body = str(gb).strip()

    if not hist_titles:
        for hidx in all_user_pos[session_idx]:
            hi = int(hidx)
            if hi <= 0:
                continue
            nid_h = news_index_reverse.get(hi, "")
            hist_titles.append(news_titles.get(nid_h, f"(idx={hi})"))

    if not cand_title:
        cand_title = news_titles.get(nid, f"(news_id={nid})")

    return {
        "type": diag_type,
        "user_click_history_titles": hist_titles,
        "candidate_news_title": cand_title,
        "generated_expected_body": gen_body,
    }


def main() -> None:
    # naml_common 은 import 시 MIND_DATASET_SUBDIR 로 경로를 고정하므로, 먼저 환경 설정 후 import
    parser = argparse.ArgumentParser(description="클러스터 배치 NAML 실제본문 vs 기대본문 평가 → results/resultN.txt")
    parser.add_argument("--cluster-csv", type=str, default="NAML/user_kmeans_k3_MIND_2000.csv")
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--sessions-per-batch", type=int, default=300)
    parser.add_argument("--mind-dataset-subdir", type=str, default=None)
    parser.add_argument(
        "--train-body-dir",
        type=str,
        required=True,
        help="기대본문 JSON 루트 (예: body_generation/output/MIND_2000/cluster0_batch0)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="saved_models/NAML_mind_2000.h5",
        help="프로젝트 루트 기준 사전학습 가중치",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--result-index", type=int, default=None, help="N in NAML/results/resultN.txt (기본: 자동 증가)")
    parser.add_argument("--no-extend-word-dict", action="store_true", help="기대본문을 word_dict에 넣지 않음 (기본은 포함)")
    args = parser.parse_args()

    sub = _resolve_mind_dataset_subdir(args.mind_dataset_subdir)
    os.environ["MIND_DATASET_SUBDIR"] = sub

    from naml_batch_generators import generate_batch_data_test, generate_batch_data_train
    from naml_common import MIND_NEWS_FILENAME, SEED, get_embedding, mind_data_path, preprocess_news_file, preprocess_user_file
    from naml_model_builder import build_naml_models

    csv_path = _ROOT / args.cluster_csv
    if not csv_path.is_file():
        print(f"오류: CSV 없음: {csv_path}")
        sys.exit(1)

    train_body_dir = os.path.normpath(str(_ROOT / args.train_body_dir))
    if not os.path.isdir(train_body_dir):
        print(f"오류: train-body-dir 없음: {train_body_dir}")
        sys.exit(1)

    weights_path = _ROOT / args.weights
    if not weights_path.is_file():
        print(f"오류: 가중치 없음: {weights_path}")
        sys.exit(1)

    np.random.seed(SEED)
    expected_bodies = load_expected_bodies_from_train_dir(train_body_dir)
    if not expected_bodies:
        print(f"경고: 기대본문이 0개입니다. ({train_body_dir})")

    cluster_users = _load_cluster_users(csv_path, args.cluster_id)
    if not cluster_users:
        print(f"오류: 클러스터 {args.cluster_id}에 해당하는 유저가 없습니다.")
        sys.exit(1)

    if args.no_extend_word_dict:
        word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
            expected_bodies_train=None, expected_bodies_test=None
        )
    else:
        word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
            expected_bodies_train=expected_bodies, expected_bodies_test=None
        )

    embedding_mat = get_embedding(word_dict)
    (
        _userid_dict,
        all_train_pn,
        all_label,
        all_train_id,
        _all_test_pn,
        _all_test_label,
        _all_test_id,
        all_user_pos,
        _all_test_user_pos,
        _all_test_index,
        _cand_tr,
        _cand_te,
        all_train_userid_str,
        all_train_newsid_str,
        _all_test_userid_str,
        _all_test_newsid_str,
    ) = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    n_train = len(all_label)
    session_indices: List[int] = []
    for i in range(n_train):
        u = _norm_cluster_uid(str(all_train_userid_str[i]))
        if u in cluster_users:
            session_indices.append(i)

    total_sess = len(session_indices)
    spb = max(1, int(args.sessions_per_batch))
    start = args.batch_index * spb
    end = min(start + spb, total_sess)
    if start >= total_sess:
        print(f"오류: batch-index={args.batch_index} 에 해당하는 세션이 없습니다.")
        sys.exit(1)

    batch_sessions = session_indices[start:end]
    print(f"클러스터 {args.cluster_id}, 배치 {args.batch_index}: 세션 {len(batch_sessions)}개 (전체 클러스터 세션 {total_sess} 중 [{start},{end}))")

    coverage = compute_expected_body_coverage_for_batch(
        batch_sessions,
        all_train_userid_str,
        all_train_newsid_str,
        all_train_pn,
        expected_bodies,
    )
    _mr = float(coverage["batch_match_rate"])
    print(
        f"[기대본문 매칭] JSON 항목 수: {coverage['json_entries_loaded']} | "
        f"배치 후보 슬롯(패딩 제외): {coverage['batch_candidate_slots_non_padding']} | "
        f"매칭됨: {coverage['batch_slots_matched_expected_body']} | "
        f"비율: {_mr:.2%} | "
        f"뉴스ID 빈 슬롯: {coverage['batch_slots_with_empty_news_id']}"
    )

    # --- 서브셋 배열 (손실용)
    sub_pn = all_train_pn[batch_sessions]
    sub_label = all_label[batch_sessions]
    sub_tid = all_train_id[batch_sessions]
    sub_user_pos = all_user_pos[batch_sessions]
    sub_uid = [all_train_userid_str[i] for i in batch_sessions]
    sub_nid = [all_train_newsid_str[i] for i in batch_sessions]

    news_index_reverse = {v: k for k, v in news_index.items()}
    news_tsv = mind_data_path(MIND_NEWS_FILENAME)
    news_titles = load_news_titles(news_tsv)

    _built = build_naml_models(word_dict, embedding_mat, category, subcategory, args.learning_rate)
    model = _built["model"]
    model_test = _built["model_test"]

    try:
        model.load_weights(str(weights_path))
    except Exception as e:
        print(f"가중치 로드 실패, by_name 시도: {e}")
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)

    bs = args.batch_size
    steps_train = (len(batch_sessions) + bs - 1) // bs

    gen_real = generate_batch_data_train(
        word_dict,
        news_words,
        news_body,
        news_v,
        news_sv,
        news_index,
        sub_pn,
        sub_label,
        sub_tid,
        sub_user_pos,
        bs,
        expected_bodies=None,
        all_userid_str=sub_uid,
        all_train_newsid_str=sub_nid,
        news_index_reverse=news_index_reverse,
        shuffle=False,
    )
    gen_exp = generate_batch_data_train(
        word_dict,
        news_words,
        news_body,
        news_v,
        news_sv,
        news_index,
        sub_pn,
        sub_label,
        sub_tid,
        sub_user_pos,
        bs,
        expected_bodies=expected_bodies,
        all_userid_str=sub_uid,
        all_train_newsid_str=sub_nid,
        news_index_reverse=news_index_reverse,
        shuffle=False,
    )

    print("model.evaluate (실제본문 / 기대본문) …")
    ev_real = model.evaluate(gen_real, steps=steps_train, verbose=0)
    ev_exp = model.evaluate(gen_exp, steps=steps_train, verbose=0)
    loss_real = float(ev_real[0])
    loss_expected = float(ev_exp[0])

    # --- NDCG@5 (model_test)
    flat_pn, flat_label, flat_tid, flat_uid, flat_nid, flat_user_pos, ranges = flatten_train_sessions_for_test_gen(
        batch_sessions,
        all_train_pn,
        all_label,
        all_train_id,
        all_user_pos,
        all_train_userid_str,
        all_train_newsid_str,
    )
    n_flat = len(flat_pn)
    test_steps = (n_flat + bs - 1) // bs

    testgen_real = generate_batch_data_test(
        word_dict,
        news_words,
        news_body,
        news_v,
        news_sv,
        news_index,
        flat_pn,
        flat_label,
        flat_tid,
        flat_user_pos,
        bs,
        expected_bodies=None,
        all_userid_str=flat_uid,
        all_newsid_str=flat_nid,
        news_index_reverse=news_index_reverse,
        all_test_user_pos_override=flat_user_pos,
    )
    testgen_exp = generate_batch_data_test(
        word_dict,
        news_words,
        news_body,
        news_v,
        news_sv,
        news_index,
        flat_pn,
        flat_label,
        flat_tid,
        flat_user_pos,
        bs,
        expected_bodies=expected_bodies,
        all_userid_str=flat_uid,
        all_newsid_str=flat_nid,
        news_index_reverse=news_index_reverse,
        all_test_user_pos_override=flat_user_pos,
    )

    print("model_test.predict (실제본문) …")
    score_real = model_test.predict(testgen_real, steps=test_steps, verbose=0)
    print("model_test.predict (기대본문) …")
    score_exp = model_test.predict(testgen_exp, steps=test_steps, verbose=0)

    if len(score_real) < n_flat or len(score_exp) < n_flat:
        print(f"경고: 점수 길이 {len(score_real)} / 기대 {n_flat}")

    ndcg_reals: List[float] = []
    ndcg_exps: List[float] = []
    gaps: List[float] = []

    for ri, (a, b) in enumerate(ranges):
        if b > len(score_real) or b > len(score_exp):
            continue
        sl = flat_label[a:b].astype(np.float32)
        sr = score_real[a:b, 0]
        se = score_exp[a:b, 0]
        if np.sum(sl) == 0:
            continue
        nr = ndcg_score(sl, sr, k=5)
        ne = ndcg_score(sl, se, k=5)
        ndcg_reals.append(nr)
        ndcg_exps.append(ne)
        gaps.append(abs(float(nr) - float(ne)))

    ndcg5_real = float(np.mean(ndcg_reals)) if ndcg_reals else 0.0
    ndcg5_expected = float(np.mean(ndcg_exps)) if ndcg_exps else 0.0

    # 진단: gap 최소 = success, 최대 = failure
    diagnostic_samples: List[Dict[str, Any]] = []
    if batch_sessions and gaps:
        min_i = int(np.argmin(gaps))
        max_i = int(np.argmax(gaps))
        s_min = batch_sessions[min_i]
        s_max = batch_sessions[max_i]
        # failure: 실제 vs 기대 NDCG 차이가 가장 큰 세션, success: 차이가 가장 작은 세션
        diagnostic_samples.append(
            build_diagnostic(
                s_max,
                "failure",
                train_body_dir,
                expected_bodies,
                all_train_userid_str,
                all_train_newsid_str,
                all_label,
                news_index_reverse,
                news_titles,
                all_user_pos,
            )
        )
        diagnostic_samples.append(
            build_diagnostic(
                s_min,
                "success",
                train_body_dir,
                expected_bodies,
                all_train_userid_str,
                all_train_newsid_str,
                all_label,
                news_index_reverse,
                news_titles,
                all_user_pos,
            )
        )

    out_obj = {
        "performance_feedback": {
            "loss_expected": round(loss_expected, 6),
            "loss_real": round(loss_real, 6),
            "ndcg5_expected": round(ndcg5_expected, 6),
            "ndcg5_real": round(ndcg5_real, 6),
        },
        "expected_body_coverage": coverage,
        "diagnostic_samples": diagnostic_samples,
    }

    results_dir = _ROOT / "NAML" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    n = args.result_index if args.result_index is not None else next_free_result_index(results_dir)
    out_path = results_dir / f"result{n}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"저장: {out_path}")
    print(json.dumps(out_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
