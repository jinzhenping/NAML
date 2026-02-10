"""
조율기 → 기대본문 생성(train80) → NAML 평가를 반복 실행합니다.
프로젝트 루트에서 실행: python run_cycle.py --iterations 3
"""
import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="coordinator → generate_body --train80_only → NAML 순서로 반복 실행")
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=1,
        help="반복 횟수 (기본: 1)",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="한 단계가 실패해도 다음 반복 계속 진행",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    steps = [
        ("조율기", [sys.executable, os.path.join(root, "coordinator_LLM", "coordinator.py")]),
        ("기대본문 생성(train80)", [sys.executable, os.path.join(root, "body_generation", "generate_body.py"), "--train80_only"]),
        ("NAML 평가", [sys.executable, os.path.join(root, "NAML", "NAML.py")]),
    ]

    for r in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"  반복 {r}/{args.iterations}")
        print(f"{'='*60}\n")
        for name, cmd in steps:
            print(f"[{name}] 실행: {' '.join(cmd)}\n")
            ret = subprocess.run(cmd, cwd=root)
            if ret.returncode != 0:
                print(f"\n[{name}] 종료 코드: {ret.returncode}")
                if not args.continue_on_error:
                    sys.exit(ret.returncode)
        print(f"\n반복 {r}/{args.iterations} 완료.\n")

    print("전체 반복 완료.")


if __name__ == "__main__":
    main()
