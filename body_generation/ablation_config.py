# -*- coding: utf-8 -*-
"""Cluster batch pipeline ablations: one component removed per mode."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

ABLATION_CHOICES = ("full", "no_policy", "no_cluster", "no_preference")

_ROOT = Path(__file__).resolve().parent.parent


def normalize_ablation(name: Optional[str]) -> str:
    if not name or str(name).strip().lower() in ("full", "none", ""):
        return "full"
    key = str(name).strip().lower()
    if key not in ABLATION_CHOICES:
        raise ValueError(f"unknown ablation {name!r}; choices: {ABLATION_CHOICES}")
    return key


def uses_full_train(ablation: str) -> bool:
    return ablation == "no_cluster"


def body_output_dir(
    dataset_subdir: str,
    ablation: str,
    batch_index: int,
    cluster_id: int = 0,
    output_root: str = "body_generation/output",
) -> str:
    """Relative path from project root for generated expected bodies."""
    base = Path(output_root) / dataset_subdir
    if ablation != "full":
        base = base / f"ablation_{ablation}"
    if uses_full_train(ablation):
        return str(base / f"fulltrain_batch{batch_index}")
    return str(base / f"cluster{cluster_id}_batch{batch_index}")


def coordinator_output_dir(ablation: str) -> str:
    if ablation == "full":
        return "coordinator_LLM/output"
    return f"coordinator_LLM/ablations/{ablation}/output"


def naml_results_dir(ablation: str) -> str:
    if ablation == "full":
        return "NAML/results"
    return f"NAML/results/ablation_{ablation}"


def seed_coordinator_policy(ablation: str, batch_start: int) -> None:
    """Copy coordinator_LLM/output/{N}.txt into ablation output if missing."""
    if ablation == "full":
        return
    import shutil

    src_root = _ROOT / "coordinator_LLM" / "output"
    dst_root = _ROOT / Path(coordinator_output_dir(ablation))
    dst_root.mkdir(parents=True, exist_ok=True)
    for n in range(batch_start):
        src = src_root / f"{n}.txt"
        dst = dst_root / f"{n}.txt"
        if dst.is_file() or not src.is_file():
            continue
        shutil.copy2(src, dst)
        print(f"[ablation {ablation}] seed policy: {dst} <- {src}", flush=True)
