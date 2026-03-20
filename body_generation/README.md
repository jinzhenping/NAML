# Body Generation (LLM_E)

유저의 취향을 파악하여 후보 뉴스에 대한 기대 본문을 생성하는 실행기 LLM입니다.

## 기능

- 유저의 클릭 히스토리에서 최근 10개 뉴스의 제목 추출 (10개 이상이면 최근 10개, 적으면 전부 사용)
- 후보 뉴스 제목을 기반으로 유저가 기대할 본문 생성
- 모든 candidate_news에 대해 한 번에 하나씩 처리 가능
- ChatGPT API를 사용한 본문 생성
- 생성 결과를 JSON 파일로 저장

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 경로

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

## 사용법

### 뉴스 처리

```bash
python body_generation/generate_body.py --user_id 1 --candidate_news_id N51332

python body_generation/generate_body.py --start_user_id 962
```

### 모든 candidate_news 처리

```bash
# 트레이닝 데이터 사용
python body_generation/generate_body.py

# 테스트 데이터 사용
python body_generation/generate_body.py --use_test
```

# 트레이닝셋 앞 80% 후보에만 생성
python body_generation/generate_body.py --train80_only

# 트레이닝셋 앞 80% 중 앞 500세션만 생성 (NAML 배치 0 → train80_batch0)
python body_generation/generate_body.py --train80_only --train80_first_k 500 --policy_file 0

# 두 번째 500세션 생성 (NAML 배치 1 → train80_batch1, coordinator 1.txt 정책)
python body_generation/generate_body.py --train80_only --train80_first_k 500 --train80_batch_index 1 --policy_file 1

# 특정 유저만, 트레이닝 80% 후보만
python body_generation/generate_body.py --user_id 1 --train80_only

# 트레이닝셋 뒤 20% 후보에만 생성
# 정책을 2.txt로 고정
python body_generation/generate_body.py --train20_only --policy_file 2

python body_generation/generate_body.py --use_test --policy_file 2

# 트레이닝셋 유저별 후반 20% 후보 기대본문 생성
python body_generation/generate_body.py --train20_only --train20_per_user --train20_first_k 500 --train20_batch_index 0 --policy_file 1
