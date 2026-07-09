#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
단일 명령의 실행 시간을 측정합니다.

프로젝트 루트에서:
  python scripts/time_script.py -- python scripts/run_cluster_batch_pipeline.py --start 0 --end 0
  python scripts/time_script.py --label "eval batch 0" --repeat 3 --output timing/results.json -- \\
      python NAML/eval_cluster_batch.py --batch-index 0 --train-body-dir ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from timing.core import format_duration, print_result, run_timed, save_results


def _parse_env_overrides(items: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"env override must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def _split_command(argv: List[str]) -> List[str]:
    if "--" not in argv:
        print("오류: 측정할 명령 앞에 '--' 가 필요합니다.", file=sys.stderr)
        print("예: python scripts/time_script.py -- python my_script.py --arg value", file=sys.stderr)
        sys.exit(2)
    sep = argv.index("--")
    cmd = argv[sep + 1 :]
    if not cmd:
        print("오류: '--' 뒤에 실행할 명령이 없습니다.", file=sys.stderr)
        sys.exit(2)
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="단일 스크립트/명령 실행 시간 측정")
    p.add_argument("--label", type=str, default=None, help="결과에 표시할 이름 (기본: 명령 첫 토큰)")
    p.add_argument("--name", type=str, default=None, help="결과 JSON 에 저장할 식별자 (기본: label 또는 명령)")
    p.add_argument("--cwd", type=str, default=str(_ROOT), help="작업 디렉터리 (기본: 프로젝트 루트)")
    p.add_argument("--repeat", type=int, default=1, metavar="N", help="동일 명령을 N회 반복 측정 (기본 1)")
    p.add_argument("--output", type=str, default=None, help="결과를 저장할 JSON 경로")
    p.add_argument("--append", action="store_true", help="--output 파일이 있으면 기존 결과 뒤에 추가")
    p.add_argument("--stdout", type=str, default=None, help="표준출력을 저장할 파일 경로")
    p.add_argument("--stderr", type=str, default=None, help="표준에러를 저장할 파일 경로")
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="환경 변수 오버라이드 (여러 번 지정 가능)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="비정상 종료(exit != 0) 시 반복을 중단",
    )
    p.add_argument("command", nargs=argparse.REMAINDER, help="'--' 뒤에 측정할 명령")
    args = p.parse_args()

    command = _split_command(args.command)
    if args.repeat < 1:
        print("오류: --repeat 는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(2)

    label = args.label or Path(command[0]).name
    name = args.name or label
    env_overrides = _parse_env_overrides(args.env)

    results = []
    for i in range(1, args.repeat + 1):
        result = run_timed(
            command,
            name=name,
            label=label,
            cwd=args.cwd,
            env_overrides=env_overrides,
            repeat_index=i if args.repeat > 1 else None,
            repeat_count=args.repeat if args.repeat > 1 else None,
            stdout_path=args.stdout,
            stderr_path=args.stderr,
        )
        results.append(result)
        print_result(result)

        if args.fail_fast and not result.success:
            break

    if args.repeat > 1 and results:
        elapsed_values = [r.elapsed_sec for r in results]
        total = sum(elapsed_values)
        avg = total / len(elapsed_values)
        print(f"\n[summary] runs={len(results)}, total={format_duration(total)}, avg={format_duration(avg)}")

    if args.output:
        save_results(results, args.output, append=args.append)
        print(f"\n결과 저장: {args.output}")

    if any(not r.success for r in results):
        sys.exit(results[-1].exit_code if results else 1)


if __name__ == "__main__":
    main()
