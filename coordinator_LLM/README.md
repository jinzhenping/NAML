# 조율기 LLM (Coordinator LLM)

실행 LLM의 생성 정책(tone, abstraction_level 등)을 업데이트하는 조율기입니다.  
`system_prompt.yaml`과 `json-payload.yaml`(최신 NAML 결과·이전 조율 출력으로 채움)으로 프롬프트를 만들고, 응답으로 `updated_policy` / `updated_running_policy_summary`를 `output/(N+1).txt`에 저장합니다.

## 실행 방법

**기본 실행** (API 키는 환경변수 `OPENAI_API_KEY` 사용)

```bash
# 최신 result 사용
python coordinator_LLM/coordinator.py

# result0.txt 사용
python coordinator_LLM/coordinator.py --result_n 0
```

**옵션**

| 옵션 | 설명 |
|------|------|
| `--api_key` | OpenAI API 키 (미지정 시 `OPENAI_API_KEY` 사용) |
| `--model` | 사용할 모델 (기본: `gpt-4o-mini`) |
| `--output_dir` | 조율 결과 저장 폴더 (기본: `coordinator_LLM/output`) |
| `--results_dir` | NAML 평가 결과 폴더 (기본: `NAML/results`) |

## 참조 데이터

- **coordinator_LLM/output**: 숫자가 가장 큰 `N.txt` → 현재 정책(`policy`/`updated_policy`)과 요약(`running_policy_summary`/`updated_running_policy_summary`)
- **NAML/results**: 숫자가 가장 큰 `resultN.txt` → 성능 피드백(`performance_feedback`)과 진단 샘플(`diagnostic_samples`)

위 두 경로의 최신 파일이 `json-payload.yaml` 템플릿을 채우는 데 사용됩니다.
