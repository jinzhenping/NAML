"""
조율기 LLM: system_prompt.yaml + (json-payload.yaml에 coordinator_LLM/output, NAML/results 최신 데이터 주입)으로
프롬프트를 구성하고, LLM 호출 후 updated_policy / updated_running_policy_summary 를 output/(N+1).txt 에 저장.
"""
import os
import re
import json
import argparse
from typing import Optional, Dict, Any, Tuple

def _dir_here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_latest_output_number(folder: str, pattern: str) -> Optional[int]:
    """
    folder 안에서 pattern에 맞는 파일명의 숫자 중 최대값 반환.
    pattern: "({})\\.txt" 형태로 숫자 그룹 하나. 예: "(\\d+)\\.txt" -> 0.txt, 1.txt
             "result({})\\.txt" -> result0.txt, result1.txt
    """
    if not os.path.isdir(folder):
        return None
    regex = re.compile(pattern)
    max_n = -1
    for name in os.listdir(folder):
        m = regex.fullmatch(name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except ValueError:
                continue
    return max_n if max_n >= 0 else None


def load_coordinator_output(output_dir: str, coord_n: Optional[int] = None) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    coordinator_LLM/output 에서 N.txt 로드.
    coord_n이 지정되면 N.txt 사용, 아니면 숫자가 가장 큰 파일 사용.
    반환: (N, data). 없으면 (None, {}) 또는 (None, 기본 policy/summary).
    """
    if coord_n is not None:
        n = coord_n
    else:
        n = get_latest_output_number(output_dir, r"(\d+)\.txt")
    if n is None:
        return None, {
            "policy": {},
            "running_policy_summary": []
        }
    path = os.path.join(output_dir, f"{n}.txt")
    if not os.path.isfile(path):
        return n, {"policy": {}, "running_policy_summary": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        data = json.loads(raw)
        # 이전 조율기 출력은 updated_policy / updated_running_policy_summary, 시드/초기 파일은 policy / running_policy_summary
        policy = data.get("updated_policy") or data.get("policy") or {}
        summary = data.get("updated_running_policy_summary") or data.get("running_policy_summary") or []
        return n, {"policy": policy, "running_policy_summary": summary}
    except Exception:
        return n, {"policy": {}, "running_policy_summary": []}


def load_naml_result(results_dir: str, result_n: Optional[int] = None) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    NAML/results 에서 resultN.txt 로드.
    result_n이 지정되면 result{result_n}.txt 사용, 아니면 숫자가 가장 큰 파일 사용.
    반환: (N, data). 없으면 (None, {}).
    """
    if result_n is not None:
        n = result_n
    else:
        n = get_latest_output_number(results_dir, r"result(\d+)\.txt")
    if n is None:
        return None, {}
    path = os.path.join(results_dir, f"result{n}.txt")
    if not os.path.isfile(path):
        return n, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return n, data
    except Exception:
        return n, {}


def load_system_prompt(system_path: str) -> str:
    with open(system_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_payload_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def fill_payload_template(
    template: str,
    coordinator_data: Dict[str, Any],
    naml_data: Dict[str, Any],
) -> str:
    """
    json-payload.yaml 템플릿의 플레이스홀더를 채움.
    - {policy} : current_policy 내부 내용 (키:값 쌍)
    - {running_policy_summary} : 배열 내부 (JSON 문자열들)
    - {loss_expected}, {loss_real}, {ndcg5_expected}, {ndcg5_real}
    - {user_click_history_titles}, {candidate_news_title}, {generated_expected_body} (첫 diagnostic_sample 기준)
    """
    out = template

    # current_policy
    policy = coordinator_data.get("policy") or {}
    if isinstance(policy, dict):
        policy_inner = ", ".join(f'"{k}": {json.dumps(v)}' for k, v in policy.items())
    else:
        policy_inner = ""
    out = out.replace("{policy}", policy_inner)

    # running_policy_summary
    summary = coordinator_data.get("running_policy_summary") or []
    if isinstance(summary, list):
        summary_inner = ", ".join(json.dumps(s) for s in summary)
    else:
        summary_inner = ""
    out = out.replace("{running_policy_summary}", summary_inner)

    # performance_feedback
    pf = naml_data.get("performance_feedback") or {}
    for key in ("loss_expected", "loss_real", "ndcg5_expected", "ndcg5_real"):
        val = pf.get(key)
        if val is None:
            val = ""
        else:
            val = str(val)
        out = out.replace("{" + key + "}", val)

    # diagnostic_samples (failure / success 모두 전달)
    samples = naml_data.get("diagnostic_samples") or []
    diagnostic_samples_str = json.dumps(samples, ensure_ascii=False)
    out = out.replace("{diagnostic_samples}", diagnostic_samples_str)

    return out


def build_prompt(system_path: str, payload_template_path: str,
                coordinator_data: Dict[str, Any], naml_data: Dict[str, Any]) -> str:
    system = load_system_prompt(system_path)
    template = load_payload_template(payload_template_path)
    payload_str = fill_payload_template(template, coordinator_data, naml_data)
    return system.strip() + "\n\n" + payload_str.strip()


def run_coordinator_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    from openai import OpenAI
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key가 필요합니다. 환경변수 OPENAI_API_KEY 또는 --api_key를 사용하세요.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


def save_output(output_dir: str, next_n: int, updated_policy: Dict, updated_summary: list) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{next_n}.txt")
    payload = {
        "updated_policy": updated_policy,
        "updated_running_policy_summary": updated_summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="조율기 LLM: 정책 업데이트")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API 키 (기본: OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="모델 이름")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="coordinator_LLM/output 경로 (기본: 이 스크립트 기준 output)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="NAML/results 경로 (기본: 프로젝트 루트 기준 NAML/results)")
    parser.add_argument("--n", type=int, default=None,
                        help="사용할 라운드 번호 (예: 2 이면 2.txt + result2.txt). 미지정 시 각각 가장 큰 번호 사용")
    args = parser.parse_args()

    base = _dir_here()
    project_root = os.path.dirname(base)
    output_dir = args.output_dir or os.path.join(base, "output")
    results_dir = args.results_dir or os.path.join(project_root, "NAML", "results")

    system_path = os.path.join(base, "system_prompt.yaml")
    payload_path = os.path.join(base, "json-payload.yaml")

    coord_n, coordinator_data = load_coordinator_output(output_dir, coord_n=args.n)
    naml_n, naml_data = load_naml_result(results_dir, result_n=args.n)

    if args.n is not None:
        print(f"참조: 라운드 N = {args.n} (coordinator {coord_n}.txt, NAML result{naml_n}.txt)")
    else:
        print(f"참조: coordinator output 최대 N = {coord_n}, NAML results 최대 N = {naml_n}")

    prompt = build_prompt(system_path, payload_path, coordinator_data, naml_data)
    print("프롬프트 구성 완료. 조율기 LLM 호출 중...")
    response_text = run_coordinator_llm(prompt, api_key=args.api_key, model=args.model)

    # 응답을 JSON으로 파싱
    try:
        # 마크다운 코드블록 제거
        if "```" in response_text:
            for block in ("```json", "```"):
                if block in response_text:
                    start = response_text.find(block) + len(block)
                    end = response_text.find("```", start)
                    if end == -1:
                        end = len(response_text)
                    response_text = response_text[start:end].strip()
                    break
        out_data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"조율기 LLM 응답 JSON 파싱 실패: {e}")
        print("응답 원문:", response_text[:500])
        return 1

    updated_policy = out_data.get("updated_policy") or {}
    updated_summary = out_data.get("updated_running_policy_summary") or []
    next_n = (coord_n if coord_n is not None else -1) + 1
    out_path = save_output(output_dir, next_n, updated_policy, updated_summary)
    print(f"저장 완료: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
