#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CQ NAML 테스트 top-1 CSV (`generate`/`real` = Category/Subcategory/Title 문자열).

  python NAML/naml_export_test_top1_cq.py \
  --weights saved_models/Adressa_2000/NAML_cq_teacher_adressa_2000_actual.h5 \
  --tune-log saved_models/Adressa_2000/naml_tune_actual_cq_teacher_log.json \
  --mind-dataset-subdir Adressa_2000 \
  --mind-test-tsv dataset/Adressa_2000/Adressa_test_2000_final.tsv \
  --actual-only \
  --out-csv NAML/export/Adressa_prediction_result_export.csv
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
    from naml_export_test_top1_csv import main

    main()
