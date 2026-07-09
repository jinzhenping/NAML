#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YAML/JSON 설정에 정의된 여러 스크립트를 순서대로 실행하며 시간을 측정합니다.

프로젝트 루트에서:
  python scripts/time_scripts_batch.py scripts/timing.example.yaml
  python scripts/time_scripts_batch.py scripts/timing.example.yaml --only pipeline_smoke eval_smoke
  python scripts/time_scripts_batch.py scripts/timing.example.yaml --output timing/batch_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from timing.core import format_duration, print_result, run_timed, save_results

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML 이 필요합니다: pip install pyyaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _resolve_cwd(raw: Optional[str]) -> str:
    if not raw:
        return str(_ROOT)
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / p
    return str(p.resolve())


def _entry_command(entry: Dict[str, Any]) -> List[str]:
    if "command" in entry:
        cmd = entry["command"]
        if isinstance(cmd, str):
            raise ValueError(f"entry {entry.get('name')!r}: command must be a list, not a string")
        return [str(x) for x in cmd]
    script = entry.get("script")
    if not script:
        raise ValueError(f"entry {entry.get('name')!r}: 'command' or 'script' is required")
    cmd = [sys.executable, str(script)]
    cmd.extend(str(x) for x in entry.get("args", []))
    return cmd


def _entry_env(entry: Dict[str, Any]) -> Dict[str, str]:
    env = entry.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError(f"entry {entry.get('name')!r}: env must be a mapping")
    return {str(k): str(v) for k, v in env.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="설정 파일 기반 다중 스크립트 실행 시간 측정")
    p.add_argument("config", type=str, help="YAML 또는 JSON 설정 파일 경로")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME", help="실행할 항목 name 만 선택")
    p.add_argument("--output", type=str, default=None, help="결과 JSON 경로 (설정의 output 보다 우선)")
    p.add_argument("--append", action="store_true", help="기존 결과 파일에 추가")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="하나라도 실패하면 이후 항목 실행 중단",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 실행 예정 명령만 출력",
    )
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path
    if not config_path.exists():
        print(f"오류: 설정 파일을 찾을 수 없습니다: {config_path}", file=sys.stderr)
        sys.exit(2)

    config = _load_config(config_path)
    defaults = config.get("defaults") or {}
    entries = config.get("scripts") or config.get("entries") or []
    if not isinstance(entries, list) or not entries:
        print("오류: 설정에 scripts (또는 entries) 목록이 필요합니다.", file=sys.stderr)
        sys.exit(2)

    only = set(args.only) if args.only else None
    output_path = args.output or defaults.get("output") or config.get("output")

    all_results = []
    selected = 0

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each script entry must be a mapping")
        name = str(entry.get("name") or entry.get("label") or f"script_{selected}")
        if only is not None and name not in only:
            continue

        selected += 1
        command = _entry_command(entry)
        cwd = _resolve_cwd(entry.get("cwd") or defaults.get("cwd"))
        label = str(entry.get("label") or name)
        repeat = int(entry.get("repeat", defaults.get("repeat", 1)))
        env_overrides = _entry_env(entry)
        if defaults.get("env"):
            merged = {str(k): str(v) for k, v in defaults["env"].items()}
            merged.update(env_overrides)
            env_overrides = merged

        stdout_path = entry.get("stdout") or defaults.get("stdout")
        stderr_path = entry.get("stderr") or defaults.get("stderr")

        if args.dry_run:
            print(f"\n[{name}] (dry-run)")
            print(f"  command : {' '.join(command)}")
            print(f"  cwd     : {cwd}")
            print(f"  repeat  : {repeat}")
            continue

        entry_results = []
        for i in range(1, repeat + 1):
            result = run_timed(
                command,
                name=name,
                label=label,
                cwd=cwd,
                env_overrides=env_overrides,
                repeat_index=i if repeat > 1 else None,
                repeat_count=repeat if repeat > 1 else None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            entry_results.append(result)
            all_results.append(result)
            print_result(result)

            if args.fail_fast and not result.success:
                break

        if repeat > 1 and entry_results:
            elapsed_values = [r.elapsed_sec for r in entry_results]
            total = sum(elapsed_values)
            avg = total / len(elapsed_values)
            print(
                f"\n[{name} summary] runs={len(entry_results)}, "
                f"total={format_duration(total)}, avg={format_duration(avg)}"
            )

        if args.fail_fast and entry_results and not entry_results[-1].success:
            break

    if selected == 0:
        print("오류: 실행할 항목이 없습니다. --only 이름을 확인하세요.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print(f"\n(dry-run) {selected}개 항목")
        return

    if output_path:
        save_results(all_results, output_path, append=args.append)
        print(f"\n결과 저장: {output_path}")

    if any(not r.success for r in all_results):
        sys.exit(all_results[-1].exit_code if all_results else 1)


if __name__ == "__main__":
    main()
