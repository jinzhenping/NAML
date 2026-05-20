#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CROWN (및 동일 엔트리의 LIME+CROWN 등) 하이퍼파라미터 탐색.

NAML/naml_tune_actual.py 와 비슷하게: 여러 trial을 한 프로세스에서 돌리고,
모든 산출물을 tune_runs/<run_name>/ 아래에 trial별로 모은다.

실행 (반드시 CROWN 디렉터리에서, `--` 뒤에 main.py 에 넘길 인자):

cd CROWN
python crown_actual_tune.py --trials 72 --seed 42 --epochs-per-trial 16 -- --mode train --dataset mind2000 --news_encoder CROWN --content_encoder CROWN --user_encoder CROWN
python crown_actual_tune.py --trials 108 --seed 42 --epochs-per-trial 16 -- --mode train --dataset mind2000 --news_encoder LIME --content_encoder CROWN --user_encoder CROWN

  (반드시 `--` 앞은 튜닝 전용 옵션, 뒤는 main.py 와 동일한 CROWN 인자. `--` 가 없으면 사용법만 출력하고 종료한다.)

옵션:
  --tune-out-root   기본 tune_runs
  --tune-run-name   생략 시 날짜+시간 자동
  --epochs-per-trial  각 trial 학습 에폭 상한 (config.epoch 덮어씀)
  --two-phase       짧은 에폭으로 스크리닝 후 상위 k만 전체 에폭 재학습
  --screening-epochs / --refine-top-k
  --rank-metric     스크리닝/정렬 기준: mrr | ndcg5 | hit1 | auc (기본 mrr)
  --resume-log      이전 summary.json 경로 → 이미 시도한 hparam 조합 스킵
  --append-log      기존 out-log 에 trial 이어붙이기

trial 마다 저장:
  trial_XXX/hparams.json, metrics.json, checkpoint_best, dev_ranks.txt, test_ranks.txt
전역:
  run_meta.json, summary.json, leaderboard.csv, INDEX.md

학습 후보 수는 튜닝 중 항상 고정: 양성 1 + 네거티브 4 (--negative_sample_num 4).

`--` 뒤 `--news_encoder` / `--user_encoder` (및 HDC+FIM 시 `--click_predictor FIM`) 조합에 따라
탐색 그리드가 자동 전환된다 (예: CNN+LSTUR → cnn_kernel_num, long_term_masking_probability).
CROWN/LIME+CROWN 은 intent_num·intent_embedding_dim 을 탐색한다.
`--tune-profile` 로 강제 지정 가능 (auto | crown | cnn_lstur | cne_sue | naml_att | hdc_fim | generic).

`--tune-scope auto` (기본): CNN+LSTUR / CNE+SUE / NAML+ATT / HDC+FIM 은 **lr만** 탐색하고 아래를 고정한다.
  dropout_rate=0.4, cnn_kernel_num=300, cnn_window_size=2, attention_dim=128,
  category_embedding_dim=subCategory_embedding_dim=64 (CNE·HDC는 CNN 필드 제외).
  CROWN/LIME 등은 `--tune-scope full` 과 동일한 다축 그리드.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Any

_CROWN_ROOT = os.path.dirname(os.path.abspath(__file__))

# 튜닝 시 후보 수 고정: 양성 1 + 네거티브 4 (= config.negative_sample_num 4)
TUNE_NEGATIVE_SAMPLE_NUM = 4

_COMMON_LRS = [5e-5, 1e-4, 2e-4]
_COMMON_DROPOUTS = [0.15, 0.2, 0.25, 0.3]
_COMMON_WDS = [0.0, 1e-5]

_TUNE_PROFILE_CHOICES = (
    "auto",
    "crown",
    "cnn_lstur",
    "cne_sue",
    "naml_att",
    "hdc_fim",
    "generic",
)

_TUNE_SCOPE_CHOICES = ("auto", "full", "lr_only")

# 네 baseline (NAML/CNN 계열 논문 설정에 맞춘 고정값; CROWN config 키 이름)
_BASELINE_LR_ONLY_PROFILES = frozenset({"cnn_lstur", "cne_sue", "naml_att", "hdc_fim"})

_BASELINE_FIXED_COMMON: dict[str, Any] = {
    "dropout_rate": 0.4,
    "category_embedding_dim": 64,
    "subCategory_embedding_dim": 64,
}

_BASELINE_FIXED_CNN_STYLE: dict[str, Any] = {
    "cnn_kernel_num": 300,
    "cnn_window_size": 2,
    "attention_dim": 128,
}

_LR_ONLY_GRID = [1e-5, 2e-5, 5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4]


def _chdir_crown() -> None:
    os.chdir(_CROWN_ROOT)


def _hparam_key(hp: dict[str, Any]) -> str:
    s = json.dumps(hp, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


_STORE_TRUE_FLAGS = frozenset(
    {
        "no_dev",
        "no_self_connection",
        "no_adjacent_normalization",
        "no_gcn_residual",
        "gcn_layer_norm",
    }
)


def merge_crown_argv(base: list[str], overrides: dict[str, Any]) -> list[str]:
    """base argv에서 overrides에 등장하는 플래그를 제거한 뒤 trial 값을 붙인다."""
    flags_to_replace = set(overrides)
    for st in _STORE_TRUE_FLAGS:
        if st in overrides:
            flags_to_replace.add(st)

    out: list[str] = []
    i = 0
    while i < len(base):
        tok = base[i]
        if not tok.startswith("--"):
            out.append(tok)
            i += 1
            continue
        name = tok[2:].replace("-", "_")
        if name in _STORE_TRUE_FLAGS:
            if name in flags_to_replace:
                i += 1
                continue
            out.append(tok)
            i += 1
            continue
        if name in flags_to_replace:
            i += 1
            if i < len(base) and not base[i].startswith("--"):
                i += 1
            continue
        out.append(tok)
        i += 1
        if i < len(base) and not base[i].startswith("--"):
            out.append(base[i])
            i += 1

    for k, v in overrides.items():
        # Config 파서는 --negative_sample_num 처럼 언더스코어 옵션명을 쓴다. 하이픈 형태는 unrecognized 로 실패한다.
        flag = "--" + k
        if k in _STORE_TRUE_FLAGS:
            if v:
                out.append(flag)
        else:
            out.extend([flag, str(v)])
    return out


def parse_crown_argv_flags(argv: list[str]) -> dict[str, str]:
    """`--key val` / `--key=val` 형태만 추출 (store_true 플래그 값 없음)."""
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if "=" in body:
            name, val = body.split("=", 1)
            out[name.replace("-", "_")] = val
            i += 1
            continue
        name = body.replace("-", "_")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[name] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def detect_tune_profile(crown_argv: list[str]) -> str:
    """baseline argv 에서 튜닝 프로필 키를 추론."""
    f = parse_crown_argv_flags(crown_argv)
    news = f.get("news_encoder", "").upper()
    user = f.get("user_encoder", "").upper()
    click = f.get("click_predictor", "dot_product").upper()
    if news == "CNN" and user == "LSTUR":
        return "cnn_lstur"
    if news == "CNE" and user == "SUE":
        return "cne_sue"
    if news == "NAML" and user == "ATT":
        return "naml_att"
    if news == "HDC" and user == "FIM":
        return "hdc_fim"
    if news in ("CROWN", "LIME") or user == "CROWN":
        return "crown"
    return "generic"


def resolve_tune_profile(crown_argv: list[str], cli_profile: str) -> str:
    p = (cli_profile or "auto").strip().lower()
    if p not in _TUNE_PROFILE_CHOICES:
        raise ValueError(f"unknown --tune-profile {cli_profile!r}; choices: {_TUNE_PROFILE_CHOICES}")
    if p == "auto":
        return detect_tune_profile(crown_argv)
    return p


def resolve_tune_scope(profile: str, cli_scope: str) -> str:
    s = (cli_scope or "auto").strip().lower()
    if s not in _TUNE_SCOPE_CHOICES:
        raise ValueError(f"unknown --tune-scope {cli_scope!r}; choices: {_TUNE_SCOPE_CHOICES}")
    if s == "auto":
        return "lr_only" if profile in _BASELINE_LR_ONLY_PROFILES else "full"
    return s


def baseline_fixed_hparams(profile: str) -> dict[str, Any]:
    """lr_only baseline: 논문/실험 고정 하이퍼파라미터 (CROWN config 키)."""
    fixed = dict(_BASELINE_FIXED_COMMON)
    if profile in ("cnn_lstur", "naml_att"):
        fixed.update(_BASELINE_FIXED_CNN_STYLE)
    elif profile == "cne_sue":
        fixed["attention_dim"] = 128
    return fixed


def profile_tuned_param_names(profile: str, tune_scope: str = "full") -> list[str]:
    """leaderboard / 문서용: trial마다 바뀌는 config 키."""
    if tune_scope == "lr_only":
        return ["lr"]
    common = ["lr", "dropout_rate", "weight_decay", "batch_size"]
    extra: dict[str, list[str]] = {
        "crown": ["intent_num", "intent_embedding_dim"],
        "cnn_lstur": ["cnn_kernel_num", "attention_dim", "long_term_masking_probability"],
        "cne_sue": ["hidden_dim", "attention_dim", "gcn_layer_num"],
        "naml_att": ["cnn_kernel_num", "attention_dim"],
        "hdc_fim": ["HDC_filter_num", "conv3D_filter_num_first"],
        "generic": [],
    }
    return common + extra.get(profile, [])


def _grid_product(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(axes.keys())
    vals = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def default_hparam_grid(profile: str, rng: random.Random, tune_scope: str = "full") -> list[dict[str, Any]]:
    """프로필별 작은 그리드 → shuffle 후 trial 풀."""
    if tune_scope == "lr_only":
        combos = [{"lr": lr} for lr in _LR_ONLY_GRID]
        rng.shuffle(combos)
        return combos

    common = {
        "lr": _COMMON_LRS,
        "dropout_rate": _COMMON_DROPOUTS,
        "weight_decay": _COMMON_WDS,
    }
    if profile == "crown":
        axes = {
            **common,
            "intent_num": [2, 3, 4],
            "intent_embedding_dim": [300, 400],
        }
    elif profile == "cnn_lstur":
        axes = {
            **common,
            "cnn_kernel_num": [300, 400],
            "attention_dim": [300, 400],
            "long_term_masking_probability": [0.1, 0.3, 0.5],
        }
    elif profile == "cne_sue":
        axes = {
            **common,
            "hidden_dim": [300, 400],
            "attention_dim": [300, 400],
            "gcn_layer_num": [3, 4, 5],
        }
    elif profile == "naml_att":
        axes = {
            **common,
            "cnn_kernel_num": [300, 400],
            "attention_dim": [300, 400],
        }
    elif profile == "hdc_fim":
        axes = {
            **common,
            "HDC_filter_num": [100, 150, 200],
            "conv3D_filter_num_first": [24, 32],
        }
    else:  # generic
        axes = dict(common)
    combos = _grid_product(axes)
    rng.shuffle(combos)
    return combos


def random_hparam_sample(profile: str, rng: random.Random, tune_scope: str = "full") -> dict[str, Any]:
    if tune_scope == "lr_only":
        # log-uniform in [1e-5, 3e-4] for extra trials beyond grid
        log_lo, log_hi = math.log10(1e-5), math.log10(3e-4)
        lr = 10 ** rng.uniform(log_lo, log_hi)
        return {"lr": round(lr, 8)}

    hp: dict[str, Any] = {
        "lr": rng.choice([5e-5, 1e-4, 1.5e-4, 2e-4]),
        "dropout_rate": rng.choice([0.1, 0.15, 0.2, 0.25, 0.3]),
        "weight_decay": rng.choice([0.0, 1e-5, 1e-4]),
    }
    if profile == "crown":
        hp["intent_num"] = rng.choice([1, 2, 3, 4, 5])
        hp["intent_embedding_dim"] = rng.choice([200, 300, 400])
    elif profile == "cnn_lstur":
        hp["cnn_kernel_num"] = rng.choice([300, 400])
        hp["attention_dim"] = rng.choice([300, 400])
        hp["long_term_masking_probability"] = rng.choice([0.1, 0.2, 0.3, 0.5])
    elif profile == "cne_sue":
        hp["hidden_dim"] = rng.choice([300, 400])
        hp["attention_dim"] = rng.choice([300, 400])
        hp["gcn_layer_num"] = rng.choice([3, 4, 5])
    elif profile == "naml_att":
        hp["cnn_kernel_num"] = rng.choice([300, 400])
        hp["attention_dim"] = rng.choice([300, 400])
    elif profile == "hdc_fim":
        hp["HDC_filter_num"] = rng.choice([100, 150, 200])
        hp["conv3D_filter_num_first"] = rng.choice([24, 32, 40])
    if rng.random() < 0.25:
        hp["batch_size"] = rng.choice([16, 32, 64])
    return hp


def plan_trials(
    rng: random.Random,
    n_trials: int,
    seen_keys: set[str],
    allow_duplicates: bool,
    profile: str,
    tune_scope: str,
) -> list[dict[str, Any]]:
    pool = default_hparam_grid(profile, rng, tune_scope)
    chosen: list[dict[str, Any]] = []
    if not allow_duplicates:
        for hp in pool:
            key = _hparam_key(hp)
            if key in seen_keys:
                continue
            chosen.append(hp)
            seen_keys.add(key)
            if len(chosen) >= n_trials:
                break
        attempts = 0
        while len(chosen) < n_trials and attempts < n_trials * 50:
            attempts += 1
            hp = random_hparam_sample(profile, rng, tune_scope)
            key = _hparam_key(hp)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chosen.append(hp)
        return chosen[:n_trials]
    for _ in range(n_trials):
        hp = random_hparam_sample(profile, rng, tune_scope)
        # allow_duplicates 모드에서는 batch_size 를 항상 넣지 않음
        if "batch_size" in hp and rng.random() < 0.5:
            del hp["batch_size"]
        chosen.append(hp)
    return chosen


def redirect_experiment_dirs(config: Any, trial_root: str) -> None:
    """학습/평가 산출물을 trial_root 아래로만 쓰도록 경로를 덮어쓴다."""
    os.makedirs(trial_root, exist_ok=True)
    sub = ("models", "best_model", "dev_res", "test_res", "configs", "results")
    for s in sub:
        os.makedirs(os.path.join(trial_root, s), exist_ok=True)
    config.model_dir = os.path.join(trial_root, "models")
    config.best_model_dir = os.path.join(trial_root, "best_model")
    config.dev_res_dir = os.path.join(trial_root, "dev_res")
    config.test_res_dir = os.path.join(trial_root, "test_res")
    config.config_dir = os.path.join(trial_root, "configs")
    config.result_dir = os.path.join(trial_root, "results")


def corpus_signature(c: Any) -> tuple[Any, ...]:
    return (
        c.dataset,
        c.train_root,
        c.dev_root,
        c.test_root,
        c.tokenizer,
        c.word_threshold,
        c.max_title_length,
        c.max_abstract_length,
        c.max_history_num,
        c.news_encoder,
        c.user_encoder,
        c.content_encoder,
    )


def apply_corpus_derived_fields(config: Any, corpus: Any) -> None:
    """
    Corpus.__init__ 가 config 에 넣는 필드(user_num, category_num 등).
    코퍼스를 재사용하는 trial 에서는 새 Config 에 이 값이 없어 Model 초기화가 실패하므로,
    corpus.config 에서 현재 trial config 로 복사한다.
    """
    src = corpus.config
    for key in ("user_num", "category_num", "subCategory_num", "vocabulary_size"):
        if hasattr(src, key):
            val = getattr(src, key)
            setattr(config, key, val)
            config.attribute_dict[key] = val
    if hasattr(src, "entity_size"):
        config.entity_size = src.entity_size
        config.attribute_dict["entity_size"] = src.entity_size


def write_index_md(
    run_dir: str,
    run_name: str,
    tune_profile: str,
    tune_scope: str,
    baseline_fixed: dict[str, Any],
) -> None:
    path = os.path.join(run_dir, "INDEX.md")
    tuned = ", ".join(f"`{k}`" for k in profile_tuned_param_names(tune_profile, tune_scope))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# CROWN tune run: `{run_name}`\n\n")
        f.write(f"- 튜닝 프로필: `{tune_profile}` · scope: `{tune_scope}` (탐색: {tuned})\n")
        if baseline_fixed:
            fixed = ", ".join(f"`{k}={v}`" for k, v in sorted(baseline_fixed.items()))
            f.write(f"- 고정 하이퍼파라미터: {fixed}\n")
        f.write(f"- 학습 후보: 양성 1 + 네거티브 {TUNE_NEGATIVE_SAMPLE_NUM} (`negative_sample_num` 튜닝 안 함)\n")
        f.write("- `run_meta.json` — 실행 시각, 시드, baseline argv\n")
        f.write("- `summary.json` — 모든 trial 요약 (metrics, 경로)\n")
        f.write("- `leaderboard.csv` — 스프레드시트용\n")
        f.write("- `trial_XXX/` — 개별 trial\n")
        f.write("  - `hparams.json` — 해당 trial만의 하이퍼파라미터 덮어쓰기\n")
        f.write("  - `metrics.json` — dev/test AUC·MRR·NDCG@5·HIT@1\n")
        f.write("  - `checkpoint_best` — best dev 기준 체크포인트 (main과 동일 포맷 dict)\n")
        f.write("  - `dev_ranks.txt` / `test_ranks.txt` — 최종 모델 순위 파일\n")


def main() -> None:
    _chdir_crown()

    if "--" not in sys.argv:
        print(
            "사용법: python crown_tune.py [튜닝 옵션] -- [CROWN/main.py 와 동일한 인자]\n"
            "예: python crown_tune.py --trials 8 --seed 1 -- "
            "--mode train --dataset mind2000 --news_encoder LIME --content_encoder CROWN --user_encoder CROWN",
            file=sys.stderr,
        )
        sys.exit(2)

    sep = sys.argv.index("--")
    tune_argv = sys.argv[1:sep]
    crown_argv = sys.argv[sep + 1 :]

    ap = argparse.ArgumentParser(description="CROWN hyperparameter tuning")
    ap.add_argument("--trials", type=int, default=8, help="실행할 trial 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-out-root", type=str, default="tune_runs", help="모든 튜닝 실행의 상위 폴더")
    ap.add_argument("--tune-run-name", type=str, default="", help="비우면 자동 이름 (날짜_시간)")
    ap.add_argument("--epochs-per-trial", type=int, default=0, help="0이면 config.epoch 그대로")
    ap.add_argument("--rank-metric", type=str, default="mrr", choices=["mrr", "ndcg5", "hit1", "auc"])
    ap.add_argument("--two-phase", action="store_true")
    ap.add_argument("--screening-epochs", type=int, default=2)
    ap.add_argument("--refine-top-k", type=int, default=5)
    ap.add_argument("--resume-log", type=str, default="", help="이전 summary.json → 시도한 hparam 스킵")
    ap.add_argument("--append-log", action="store_true")
    ap.add_argument("--allow-duplicate-hparams", action="store_true")
    ap.add_argument(
        "--tune-profile",
        type=str,
        default="auto",
        choices=list(_TUNE_PROFILE_CHOICES),
        help="auto: -- 뒤 encoder 조합으로 그리드 선택",
    )
    ap.add_argument(
        "--tune-scope",
        type=str,
        default="auto",
        choices=list(_TUNE_SCOPE_CHOICES),
        help="auto: 네 baseline은 lr_only, CROWN 등은 full",
    )
    args = ap.parse_args(tune_argv)

    if not crown_argv:
        print("CROWN 인자가 비었습니다. `--` 뒤에 --dataset 등을 넣으세요.", file=sys.stderr)
        sys.exit(2)

    run_name = args.tune_run_name.strip() or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.tune_out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    rng = random.Random(args.seed)

    seen_hparam_keys: set[str] = set()
    existing_trials: list[dict[str, Any]] = []
    if args.resume_log:
        try:
            with open(args.resume_log, "r", encoding="utf-8") as f:
                old = json.load(f)
            for row in old.get("trials", []):
                hp = row.get("hparams", {})
                seen_hparam_keys.add(_hparam_key({k: v for k, v in hp.items() if k != "phase"}))
        except Exception as e:
            print(f"resume-log 로드 실패 (무시): {e}", flush=True)

    out_summary_path = os.path.join(run_dir, "summary.json")
    if os.path.isfile(out_summary_path) and not args.append_log and not args.resume_log:
        print(f"이미 존재: {out_summary_path}  (--append-log 또는 다른 --tune-run-name 사용)", file=sys.stderr)
        sys.exit(1)

    if args.append_log and os.path.isfile(out_summary_path):
        try:
            with open(out_summary_path, "r", encoding="utf-8") as f:
                app = json.load(f)
            for row in app.get("trials", []):
                hp = row.get("hparams", {})
                seen_hparam_keys.add(_hparam_key({k: v for k, v in hp.items() if k != "phase"}))
            existing_trials = list(app.get("trials", []))
        except Exception as e:
            print(f"append-log: 기존 summary 로드 실패: {e}", flush=True)

    tune_profile = resolve_tune_profile(crown_argv, args.tune_profile)
    tune_scope = resolve_tune_scope(tune_profile, args.tune_scope)
    baseline_fixed = baseline_fixed_hparams(tune_profile) if tune_scope == "lr_only" else {}
    trials_hp = plan_trials(
        rng,
        args.trials,
        seen_hparam_keys,
        args.allow_duplicate_hparams,
        tune_profile,
        tune_scope,
    )
    if not trials_hp:
        print("실행할 새 hparam 조합이 없습니다.", file=sys.stderr)
        sys.exit(0)

    print(
        f"[tune] profile={tune_profile}  scope={tune_scope}  "
        f"search={profile_tuned_param_names(tune_profile, tune_scope)}  "
        f"fixed={list(baseline_fixed.keys())}  trials={len(trials_hp)}",
        flush=True,
    )

    # delayed imports after chdir
    import numpy as np
    import torch

    from config import Config
    from corpus import Corpus
    from model import Model
    from trainer import Trainer
    from util import compute_scores, format_result_metrics_line, get_run_index

    meta = {
        "run_name": run_name,
        "seed": args.seed,
        "tune_profile": tune_profile,
        "tune_profile_cli": args.tune_profile,
        "tune_scope": tune_scope,
        "tune_scope_cli": args.tune_scope,
        "baseline_fixed_hparams": baseline_fixed,
        "tuned_param_names": profile_tuned_param_names(tune_profile, tune_scope),
        "negative_sample_num_fixed": TUNE_NEGATIVE_SAMPLE_NUM,
        "trials_requested": args.trials,
        "crown_argv": crown_argv,
        "two_phase": args.two_phase,
        "screening_epochs": args.screening_epochs,
        "refine_top_k": args.refine_top_k,
        "rank_metric": args.rank_metric,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    write_index_md(run_dir, run_name, tune_profile, tune_scope, baseline_fixed)

    def run_one_trial(
        trial_id: int,
        hp: dict[str, Any],
        epochs: int,
        phase_label: str,
        corpus_holder: dict[str, Any],
        *,
        wave_index: int | None = None,
        wave_total: int | None = None,
    ) -> dict[str, Any]:
        merged = merge_crown_argv(crown_argv, baseline_fixed)
        merged = merge_crown_argv(merged, hp)
        merged = merge_crown_argv(merged, {"negative_sample_num": TUNE_NEGATIVE_SAMPLE_NUM})
        if not any(merged[i] == "--mode" for i in range(len(merged))):
            merged = ["--mode", "train"] + merged
        _trial_msg = (
            f"[trial {trial_id:03d}] 시작  phase={phase_label}  epochs={epochs}"
        )
        if wave_index is not None and wave_total is not None:
            _trial_msg += f"  (단계 내 {wave_index}/{wave_total})"
        print(_trial_msg, flush=True)
        config = Config(merged)
        if config.world_size != 1:
            raise RuntimeError("crown_tune.py 는 현재 world_size=1 만 지원합니다. --world_size 1 을 넣으세요.")

        if epochs > 0:
            config.epoch = epochs
            config.attribute_dict["epoch"] = epochs

        sig = corpus_signature(config)
        if corpus_holder.get("sig") != sig or corpus_holder.get("corpus") is None:
            corpus_holder.clear()
            corpus_holder["corpus"] = Corpus(config)
            corpus_holder["sig"] = sig
            print(f"[trial {trial_id}] Corpus 로드 (signature 변경)", flush=True)
        corpus = corpus_holder["corpus"]
        apply_corpus_derived_fields(config, corpus)

        trial_dir = os.path.join(run_dir, f"trial_{trial_id:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        redirect_experiment_dirs(config, trial_dir)

        with open(os.path.join(trial_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "phase": phase_label,
                    "tune_scope": tune_scope,
                    "fixed": baseline_fixed,
                    **hp,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        torch.manual_seed(args.seed + trial_id)
        torch.cuda.manual_seed_all(args.seed + trial_id)
        np.random.seed(args.seed + trial_id)
        random.seed(args.seed + trial_id)

        model = Model(config)
        model.initialize()
        model.cuda()

        run_index = get_run_index(config.result_dir)
        trainer = Trainer(model, config, corpus, run_index)
        trainer.train()

        # Trainer 는 best_model 을 config.best_model_dir + '/#' + run_index 아래에 둔다 (trainer.py)
        best_path = os.path.join(
            config.best_model_dir,
            "#" + str(run_index),
            model.model_name,
        )
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"best checkpoint 없음: {best_path}")

        dest_ckpt = os.path.join(trial_dir, "checkpoint_best")
        shutil.copy2(best_path, dest_ckpt)

        loaded = torch.load(dest_ckpt, map_location="cuda")
        model.load_state_dict(loaded[model.model_name])
        model.eval()

        dev_path = os.path.join(trial_dir, "dev_ranks.txt")
        test_path = os.path.join(trial_dir, "test_ranks.txt")
        auc_d, mrr_d, ndcg5_d, hit1_d = compute_scores(
            model,
            corpus,
            config.batch_size * 2 // config.world_size,
            "dev",
            dev_path,
            config.dataset,
        )
        auc_t, mrr_t, ndcg5_t, hit1_t = compute_scores(
            model,
            corpus,
            config.batch_size,
            "test",
            test_path,
            config.dataset,
        )

        metrics = {
            "dev": {"auc": float(auc_d), "mrr": float(mrr_d), "ndcg5": float(ndcg5_d), "hit1": float(hit1_d)},
            "test": {"auc": float(auc_t), "mrr": float(mrr_t), "ndcg5": float(ndcg5_t), "hit1": float(hit1_t)},
            "run_index": int(run_index),
            "best_model_source": best_path,
        }
        with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        with open(os.path.join(trial_dir, "test_results_1line.txt"), "w", encoding="utf-8") as f:
            f.write(format_result_metrics_line(run_index, mrr_t, ndcg5_t, hit1_t))

        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "trial_id": trial_id,
            "phase": phase_label,
            "hparams": hp,
            "metrics": metrics,
            "trial_dir": trial_dir,
        }

    corpus_holder: dict[str, Any] = {}
    log_trials: list[dict[str, Any]] = list(existing_trials)
    base_epochs = args.epochs_per_trial

    def rank_key(row: dict[str, Any]) -> float:
        m = row["metrics"]["dev"][args.rank_metric]
        return float(m)

    if args.two_phase:
        screen_e = max(1, args.screening_epochs)
        full_e = base_epochs if base_epochs > 0 else None
        if full_e is None:
            tmp = Config(list(crown_argv))
            full_e = int(tmp.epoch)
        print(
            f"[two-phase] 1차: epochs={screen_e}, trials={len(trials_hp)} → "
            f"상위 {args.refine_top_k}개를 epochs={full_e} 로 재학습",
            flush=True,
        )
        screened: list[dict[str, Any]] = []
        for t, hp in enumerate(trials_hp):
            row = run_one_trial(
                len(log_trials),
                hp,
                screen_e,
                "screening",
                corpus_holder,
                wave_index=t + 1,
                wave_total=len(trials_hp),
            )
            log_trials.append(row)
            screened.append(row)
            _flush_summary(run_dir, meta, log_trials, args.rank_metric)
        screened.sort(key=rank_key, reverse=True)
        top = screened[: min(args.refine_top_k, len(screened))]
        for rank, srow in enumerate(top):
            hp = dict(srow["hparams"])
            if "phase" in hp:
                del hp["phase"]
            row = run_one_trial(
                len(log_trials),
                hp,
                full_e,
                f"refine_{rank}",
                corpus_holder,
                wave_index=rank + 1,
                wave_total=len(top),
            )
            log_trials.append(row)
            _flush_summary(run_dir, meta, log_trials, args.rank_metric)
    else:
        ep = base_epochs if base_epochs > 0 else 0
        for t, hp in enumerate(trials_hp):
            row = run_one_trial(
                len(log_trials),
                hp,
                ep,
                "single",
                corpus_holder,
                wave_index=t + 1,
                wave_total=len(trials_hp),
            )
            log_trials.append(row)
            _flush_summary(run_dir, meta, log_trials, args.rank_metric)

    _write_leaderboard(run_dir, log_trials, args.rank_metric, tune_profile, tune_scope)
    print(f"\n완료. 결과 디렉터리:\n  {os.path.abspath(run_dir)}\n", flush=True)


def _flush_summary(run_dir: str, meta: dict[str, Any], trials: list[dict[str, Any]], rank_metric: str) -> None:
    best = None
    best_score = float("-inf")
    for row in trials:
        if row.get("phase", "").startswith("screening"):
            continue
        sc = float(row["metrics"]["test"][rank_metric])
        if sc > best_score:
            best_score = sc
            best = row
    summary = {
        "meta": meta,
        "trials": trials,
        "best_by_test_metric": {rank_metric: best, "score": best_score},
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _write_leaderboard(
    run_dir: str,
    trials: list[dict[str, Any]],
    rank_metric: str,
    tune_profile: str,
    tune_scope: str,
) -> None:
    path = os.path.join(run_dir, "leaderboard.csv")
    rows = [r for r in trials if not str(r.get("phase", "")).startswith("screening")]
    if not rows:
        rows = list(trials)
    hp_cols = profile_tuned_param_names(tune_profile, tune_scope)
    fieldnames = [
        "trial_id",
        "phase",
        "trial_dir",
        "dev_auc",
        "dev_mrr",
        "dev_ndcg5",
        "dev_hit1",
        "test_auc",
        "test_mrr",
        "test_ndcg5",
        "test_hit1",
        *hp_cols,
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: float(x["metrics"]["test"][rank_metric]), reverse=True):
            hp = {k: r["hparams"].get(k, "") for k in fieldnames if k in r["hparams"]}
            dm = r["metrics"]["dev"]
            tm = r["metrics"]["test"]
            w.writerow(
                {
                    "trial_id": r["trial_id"],
                    "phase": r.get("phase", ""),
                    "trial_dir": r.get("trial_dir", ""),
                    "dev_auc": dm["auc"],
                    "dev_mrr": dm["mrr"],
                    "dev_ndcg5": dm["ndcg5"],
                    "dev_hit1": dm["hit1"],
                    "test_auc": tm["auc"],
                    "test_mrr": tm["mrr"],
                    "test_ndcg5": tm["ndcg5"],
                    "test_hit1": tm["hit1"],
                    **hp,
                }
            )


if __name__ == "__main__":
    main()
