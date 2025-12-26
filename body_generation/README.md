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

### 옵션

- `--user_id`: 유저 ID (선택, 없으면 모든 유저 처리)
- `--candidate_news_id`: 후보 뉴스 ID (선택, 없으면 모든 candidate_news 처리)
- `--output`: 출력 디렉토리 경로 (기본값: body_generation/output)
- `--use_test`: 테스트 데이터 사용 (기본값: 학습 데이터 사용)
- `--api_key`: OpenAI API 키 (선택, 환경변수 사용 가능)
- `--model`: 사용할 모델명 (기본값: gpt-4o-mini)
