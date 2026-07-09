from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


@dataclass
class TimingResult:
    """단일 명령 실행의 시간 측정 결과."""

    name: str
    command: List[str]
    cwd: str
    exit_code: int
    elapsed_sec: float
    started_at: str
    finished_at: str
    label: Optional[str] = None
    repeat_index: Optional[int] = None
    repeat_count: Optional[int] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_duration(seconds: float) -> str:
    """초 단위 시간을 사람이 읽기 쉬운 문자열로 변환."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.3f} s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem)}m {rem % 1 * 60:.1f}s"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_env(overrides: Optional[Mapping[str, str]] = None) -> dict:
    env = os.environ.copy()
    if overrides:
        env.update(overrides)
    return env


def run_timed(
    command: Sequence[str],
    *,
    name: str,
    cwd: Union[str, Path],
    label: Optional[str] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    repeat_index: Optional[int] = None,
    repeat_count: Optional[int] = None,
    capture_output: bool = False,
    stdout_path: Optional[Union[str, Path]] = None,
    stderr_path: Optional[Union[str, Path]] = None,
) -> TimingResult:
    """명령을 실행하고 wall-clock 시간을 측정."""
    cmd = [str(part) for part in command]
    if not cmd:
        raise ValueError("command must not be empty")

    cwd_str = str(Path(cwd).resolve())
    env = _merge_env(env_overrides)

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    stdout_file = None
    stderr_file = None
    try:
        if capture_output:
            completed = subprocess.run(
                cmd,
                cwd=cwd_str,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        elif stdout_path or stderr_path:
            stdout_file = open(stdout_path, "w", encoding="utf-8") if stdout_path else None
            stderr_file = open(stderr_path, "w", encoding="utf-8") if stderr_path else None
            completed = subprocess.run(
                cmd,
                cwd=cwd_str,
                env=env,
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        else:
            completed = subprocess.run(cmd, cwd=cwd_str, env=env, check=False)
    finally:
        if stdout_file is not None:
            stdout_file.close()
        if stderr_file is not None:
            stderr_file.close()

    elapsed_sec = time.perf_counter() - t0
    finished_at = _utc_now_iso()

    if capture_output and stdout_path:
        Path(stdout_path).write_text(completed.stdout or "", encoding="utf-8")
    if capture_output and stderr_path:
        Path(stderr_path).write_text(completed.stderr or "", encoding="utf-8")

    return TimingResult(
        name=name,
        label=label,
        command=cmd,
        cwd=cwd_str,
        exit_code=completed.returncode,
        elapsed_sec=elapsed_sec,
        started_at=started_at,
        finished_at=finished_at,
        repeat_index=repeat_index,
        repeat_count=repeat_count,
        env_overrides=dict(env_overrides or {}),
        stdout_path=str(stdout_path) if stdout_path else None,
        stderr_path=str(stderr_path) if stderr_path else None,
    )


def save_results(
    results: Sequence[TimingResult],
    path: Union[str, Path],
    *,
    append: bool = False,
) -> None:
    """측정 결과를 JSON 파일로 저장."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: List[Dict[str, Any]] = [r.to_dict() for r in results]
    if append and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"existing timing file is not a JSON list: {out}")
        payload = existing + payload

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_results(path: Union[str, Path]) -> List[TimingResult]:
    """JSON 파일에서 측정 결과를 읽음."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"timing file must contain a JSON list: {path}")
    return [TimingResult(**item) for item in data]


def print_result(result: TimingResult, *, file: Any = None) -> None:
    """단일 결과를 콘솔에 출력."""
    stream = file or sys.stdout
    title = result.label or result.name
    status = "OK" if result.success else f"FAIL (exit {result.exit_code})"
    repeat = ""
    if result.repeat_count and result.repeat_index is not None:
        repeat = f" [run {result.repeat_index}/{result.repeat_count}]"

    print(f"\n[{title}]{repeat} {status}", file=stream)
    print(f"  elapsed : {format_duration(result.elapsed_sec)} ({result.elapsed_sec:.6f} s)", file=stream)
    print(f"  command : {' '.join(result.command)}", file=stream)
    print(f"  cwd     : {result.cwd}", file=stream)
    if result.stdout_path:
        print(f"  stdout  : {result.stdout_path}", file=stream)
    if result.stderr_path:
        print(f"  stderr  : {result.stderr_path}", file=stream)
