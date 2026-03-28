# Body Generation (LLM_E)

유저의 취향을 파악하여 후보 뉴스에 대한 기대 본문을 생성하는 실행기 LLM입니다.

## 기능

- 유저의 클릭 히스토리에서 최근 10개 뉴스의 제목 추출 (10개 이상이면 최근 10개, 적으면 전부 사용)
- 후보 뉴스 제목을 기반으로 유저가 기대할 본문 생성
- 유저당 여러 후보는 **병렬 API 호출**로 처리
- ChatGPT API를 사용한 본문 생성
- 생성 결과를 JSON 파일로 저장

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 경로

- 기본 사용 폴더명은 `generate_body.py` 상단 **`DEFAULT_MIND_DATASET_SUBDIR`** (기본 `"MIND"`). `--mind_dataset_subdir` 또는 `MIND_DATASET_SUBDIR`로 덮어쓸 수 있음.
- `dataset/<폴더>/` 안에 `MIND_news.tsv` + `MIND_train_*.tsv` 1개 + `MIND_test_*.tsv` 1개면 **자동 인식** (`MIND_1000` 등).
- `MIND_2000` 은 코드에 프리셋으로 `(2000)` 파일명이 박혀 있음.
- 그 외는 환경변수 `MIND_*_FILENAME` 또는 NAML의 `MIND_DATASET_PRESETS` 에 폴더 추가.

## 환경 설정

OpenAI API 키를 환경변수로 설정하세요:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Windows (CMD)
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

또는 코드에서 직접 API 키를 전달할 수 있습니다.

## 테스트 유저 클러스터별 정책 (단일 출력 폴더)

`NAML/user_kmeans_k*_..._test.csv` 처럼 **테스트 유저 → 클러스터** 매핑이 있을 때, 클러스터 0/1/2마다 **서로 다른 정책 JSON**을 쓰고 싶으면 `generate_body_test_cluster_policies.py`를 씁니다. 생성물은 `--output`으로 지정한 **한 폴더** 아래에만 쌓입니다 (`user_<id>/news_<id>.json`).

```bash
python body_generation/generate_body_test_cluster_policies.py \
  --cluster-csv NAML/user_kmeans_k3_MIND_2000_test.csv \
  --policy-files coordinator_LLM/output_cluster0/11.txt coordinator_LLM/output_cluster1/13.txt coordinator_LLM/output_cluster2/8.txt \
  --output body_generation/output/MIND_2000/test_3cluster_11_13_8 \
  --mind-dataset-subdir MIND_2000
```

- 정책 파일 형식은 `coordinator_LLM/output/N.txt`와 동일하게 `updated_policy` 또는 `policy` 필드를 가진 JSON이면 됩니다.
- `--policy-files`는 **클러스터 id 0, 1, 2, … 순서**와 맞춥니다. CSV에 나온 최대 클러스터 번호가 `len(policy-files)-1`을 넘으면 안 됩니다.
- `--dry-run`으로 클러스터별 쌍 개수만 확인할 수 있습니다.

## 출력 경로

- `--output` 기본값은 `body_generation/output`이며, **사용 중인 데이터셋 폴더명** 아래에 실행별 폴더가 생깁니다.
- 예: `--mind_dataset_subdir MIND_2000` (또는 `MIND_DATASET_SUBDIR=MIND_2000`) → `body_generation/output/MIND_2000/train0` / `test_0` 등.

## 사용법

### 뉴스 처리

```bash
python body_generation/generate_body.py --user_id 1 --candidate_news_id N51332

python body_generation/generate_body.py --start_user_id 962
```

### 모든 candidate_news 처리

```bash
# 트레이닝 데이터 사용 (기본 dataset/MIND → output/MIND/trainN)
python body_generation/generate_body.py


# 테스트 데이터 사용
python body_generation/generate_body.py --use_test

python body_generation/generate_body.py --policy_file 2   # 정책만 지정
```

### 클러스터별 트레이닝 세션 배치 → coordinator `N.txt` 정책

`NAML/user_kmeans_k*.csv`(컬럼 `user_id`, `cluster`)와 **NAML `preprocess_user_file`과 동일한** 트레이닝 세션 순서를 씁니다.  
지정한 클러스터에 속한 유저의 세션만 모은 뒤, **세션을 300개씩** 나누고, **배치 0 → `coordinator_LLM/output/0.txt`**, 배치 1 → `1.txt` … 로 기대본문을 생성합니다.

```bash
# PYTHONPATH에 NAML 필요 (프로젝트 루트에서)
set PYTHONPATH=NAML
python body_generation/generate_body_cluster_train_batches.py --cluster-id 0 --batch-index 0 --mind-dataset-subdir MIND_2000

# 배치 1 → 1.txt, CSV 경로 지정
python body_generation/generate_body_cluster_train_batches.py ^
  --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv --cluster-id 0 --batch-index 1 --mind-dataset-subdir MIND_2000

# 세션 수·쌍 수만 확인 (API 호출 없음)
python body_generation/generate_body_cluster_train_batches.py --cluster-id 0 --batch-index 0 --dry-run --mind-dataset-subdir MIND_2000

# 해당 클러스터를 몇 개 배치로 나눌 수 있는지(세션 수·batch-index 범위)만 출력
python body_generation/generate_body_cluster_train_batches.py --cluster-id 0 --batch-count-only --mind-dataset-subdir MIND_2000

# 트레이닝셋 전체체를 몇개 배치로 나눌 수 있는지
python body_generation/generate_body_cluster_train_batches.py --full-train --batch-count-only --sessions-per-batch 500 --mind-dataset-subdir MIND_2000

# 트레이닝셋 전체 한 번에 + 저장 폴더 직접 지정 (프로젝트 루트 기준 상대 경로)
set PYTHONPATH=NAML
python body_generation/generate_body_train_cluster_policies.py \
  --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv \
  --policy-files coordinator_LLM/output_cluster0/11.txt coordinator_LLM/output_cluster1/13.txt coordinator_LLM/output_cluster2/8.txt \
  --output body_generation/output/MIND_2000/train_3cluster_11_13_8 \
  --mind-dataset-subdir MIND_2000
```

- 출력(기본): `body_generation/output/<데이터셋>/cluster<C>_batch<B>/` (`user_<id>/news_<뉴스ID>.json`, `all_results_pairs.json`)
- `--output-dir DIR`: 위 경로 대신 **DIR에 바로** 저장 (상대 경로는 프로젝트 루트 기준)
- 정책: `coordinator_LLM/output/N.txt` 는 기본적으로 배치 번호와 동일한 `N`(다르면 `--policy-file N`). **임의 JSON 파일**은 `--policy-path path/to/policy.txt`(coordinator 출력과 동일 형식) — 지정 시 `N.txt`·`--policy-file`보다 우선
