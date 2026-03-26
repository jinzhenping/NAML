#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프리트레인 NAML 가중치로 테스트셋 평가 시 **본문 입력을 사용하지 않는** 경우(ablation).

후보 뉴스 본문 + 클릭 히스토리에 있는 각 뉴스 본문을 모두 패딩 시퀀스(news_body[0])로 넣습니다.
제목·카테고리·히스토리 제목 등은 그대로입니다.

프로젝트 루트에서:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/eval_test_no_body.py \\
    --weights saved_models/NAML_mind_2000.h5 --mind-dataset-subdir MIND_2000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_batch_generators import generate_batch_data_test

from eval_test_expected import calc_metrics_from_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="테스트셋 평가 — 본문 미사용(패딩만)")
    parser.add_argument("--weights", type=str, default="saved_models/NAML_mind_2000.h5")
    parser.add_argument("--mind-dataset-subdir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()

    if args.mind_dataset_subdir:
        os.environ["MIND_DATASET_SUBDIR"] = args.mind_dataset_subdir

    from naml_common import SEED, get_embedding, preprocess_news_file, preprocess_user_file
    from naml_model_builder import build_naml_models

    weights_path = _ROOT / args.weights
    if not weights_path.is_file():
        print(f"오류: 가중치 파일 없음: {weights_path}")
        sys.exit(1)

    np.random.seed(SEED)
    word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
        expected_bodies_train=None,
        expected_bodies_test=None,
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
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict,
    )

    embedding_mat = get_embedding(word_dict)
    built = build_naml_models(word_dict, embedding_mat, category, subcategory, args.learning_rate)
    model = built["model"]
    model_test = built["model_test"]
    model.load_weights(str(weights_path))

    news_index_reverse = {v: k for k, v in news_index.items()}
    bs = args.batch_size
    test_steps = (len(all_test_id) + bs - 1) // bs

    testgen_no_body = generate_batch_data_test(
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
        expected_bodies=None,
        all_userid_str=all_test_userid_str,
        all_newsid_str=all_test_newsid_str,
        news_index_reverse=news_index_reverse,
        omit_body=True,
    )

    print("테스트셋 예측 중... (본문 미사용 — 패딩만)")
    score = model_test.predict(testgen_no_body, steps=test_steps, verbose=0)
    metrics = calc_metrics_from_scores(score, all_test_label, all_test_index)

    print("\n=== 테스트셋 성능 (본문 미사용) ===")
    print(
        f"MRR={metrics['MRR']:.6f}  NDCG@5={metrics['NDCG@5']:.6f}  "
        f"Hit@1={metrics['Hit@1']:.6f}  (세션 수: {metrics['evaluated_sessions']})"
    )
    print(
        "\n※ 제목·카테고리·히스토리 제목은 유지하고, "
        "후보 본문·히스토리 뉴스 본문만 패딩으로 둔 결과입니다."
    )


if __name__ == "__main__":
    main()
