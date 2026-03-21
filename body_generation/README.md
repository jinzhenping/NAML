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
