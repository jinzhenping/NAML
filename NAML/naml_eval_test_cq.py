#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIND_2000: --mind-dataset-subdir MIND_2000
# Adressa_2000: --mind-dataset-subdir Adressa_2000
"""
CQ NAML (`build_naml_models_candidate_query_user`) 테스트셋 평가.

`naml_eval_test.py` 와 동일한 지표(MRR, NDCG@5, Hit@1)이며, 항상 `--cq-user-encoder` 로 그래프를 맞춘다.
`naml_tune_actual_cq_teacher.py` 로 저장한 교사 가중치·`naml_tune_actual_cq_teacher_log.json` 과 함께 쓴다.

실제본문만:

  python NAML/naml_eval_test_cq.py \
    --weights saved_models/MIND_2000/NAML_cq_teacher_mind_2000_actual.h5 \
    --tune-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json \
    --mind-dataset-subdir MIND_2000 \
    --actual-only

실제본문 + 기대본문:

  python NAML/naml_eval_test_cq.py \
  --expected-dir body_generation/output/MIND_2000/test_3cluster_11_13_8 \
  --weights saved_models/MIND_2000/NAML_kd_student_cq_0.5_0.5_7.h5 \
  --tune-log saved_models/MIND_2000/naml_tune_actual_cq_teacher_log.json \
  --mind-dataset-subdir MIND_2000 \
  --expected-body-first-n-sentences 7 \
  --mind-test-tsv dataset/MIND_2000/MIND_test_2000_final.tsv

표준 NAML 평가는 `NAML/naml_eval_test.py` (--cq-user-encoder 없음).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "NAML") not in sys.path:
    sys.path.insert(0, str(_ROOT / "NAML"))

from naml_dataset_env import apply_dataset_env_from_argv

apply_dataset_env_from_argv()


def _ensure_cq_flag() -> None:
    if "--cq-user-encoder" not in sys.argv:
        sys.argv.insert(1, "--cq-user-encoder")


if __name__ == "__main__":
    _ensure_cq_flag()
    from naml_eval_test import main

    main()
