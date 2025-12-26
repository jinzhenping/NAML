# NAML (Neural News Recommendation with Attentive Multi-View Learning)

MIND 데이터셋을 사용한 뉴스 추천 모델 학습 및 테스트

## 설치

```bash
# 프로젝트 루트에서
pip install -r requirements.txt

# NLTK 데이터 다운로드 (처음 실행 시 자동으로 다운로드됨)
python -c "import nltk; nltk.download('punkt')"
```

## 리눅스 환경에서 실행

### 방법 1: 학습 및 테스트 (권장)

```bash
# 기본 실행 (GPU 0 사용, 3 에포크, 학습 후 테스트)
python NAML/train_naml.py

# 모델 저장하며 학습
python NAML/train_naml.py --save_model models/naml_model.h5

# 전처리된 데이터 저장 (다음 실행 시 시간 절약)
python NAML/train_naml.py --save_data data/preprocessed.pkl

# 전처리된 데이터 로드하여 학습 (더 빠름)
python NAML/train_naml.py --load_data data/preprocessed.pkl --save_model models/naml_model.h5

# GPU 지정 및 에포크 수 변경
python NAML/train_naml.py --gpu 0 --epochs 5 --batch_size 64
```

### 방법 2: 테스트만 실행

```bash
# 학습된 모델로 테스트만 실행
python NAML/train_naml.py \
    --test_only \
    --load_model models/naml_model_epoch3.h5 \
    --load_data data/preprocessed.pkl

# 전처리된 데이터 없이 테스트 (자동으로 전처리)
python NAML/train_naml.py \
    --test_only \
    --load_model models/naml_model_epoch3.h5
```

### 방법 3: 모든 옵션 지정

```bash
python NAML/train_naml.py \
    --train_file dataset/MIND/MIND_train_(1000).tsv \
    --test_file dataset/MIND/MIND_test_(1000).tsv \
    --news_file dataset/MIND/MIND_news.tsv \
    --gpu 0 \
    --epochs 3 \
    --batch_size 30 \
    --npratio 4 \
    --save_model models/naml_model.h5 \
    --save_data data/preprocessed.pkl
```

### 방법 2: Jupyter Notebook 실행

```bash
# Jupyter 설치 (없는 경우)
pip install jupyter

# Jupyter 실행
jupyter notebook NAML/NAML.ipynb

# 또는 JupyterLab
jupyter lab NAML/NAML.ipynb
```

### 방법 3: 백그라운드 실행 (nohup 사용)

```bash
# 출력을 파일로 저장하며 백그라운드 실행
nohup python NAML/train_naml.py --gpu 0 --epochs 3 > naml_training.log 2>&1 &

# 진행 상황 확인
tail -f naml_training.log
```

## 옵션 설명

### 데이터 옵션
- `--train_file`: 학습 데이터 파일 경로 (기본값: `dataset/MIND/MIND_train_(1000).tsv`)
- `--test_file`: 테스트 데이터 파일 경로 (기본값: `dataset/MIND/MIND_test_(1000).tsv`)
- `--news_file`: 뉴스 데이터 파일 경로 (기본값: `dataset/MIND/MIND_news.tsv`)
- `--load_data`: 전처리된 데이터 로드 경로 (pickle 형식, 시간 절약)
- `--save_data`: 전처리된 데이터 저장 경로 (pickle 형식)

### 모델 옵션
- `--glove_path`: GloVe 임베딩 파일 경로 (없으면 랜덤 초기화)
- `--save_model`: 모델 저장 경로 (각 에포크마다 `_epoch{N}.h5`로 저장)
- `--load_model`: 모델 로드 경로 (테스트만 실행 시 사용)
- `--test_only`: 테스트만 실행 (--load_model 필요)

### 학습 옵션
- `--gpu`: 사용할 GPU ID (기본값: `0`, 여러 GPU: `"0,1"`)
- `--epochs`: 학습 에포크 수 (기본값: `3`)
- `--batch_size`: 배치 크기 (기본값: `30`)
- `--npratio`: Negative/Positive 비율 (기본값: `4`, 후보 개수 = 5)

## 데이터 구조

### 학습 데이터 (`MIND_train_(1000).tsv`)
- 형식: `user \t clicked_news \t candidate_news \t clicked`
- `clicked_news`: 공백으로 구분된 클릭 히스토리 뉴스 ID들
- `candidate_news`: 공백으로 구분된 후보 뉴스 ID들 (5개)
- `clicked`: 공백으로 구분된 클릭 여부 (1 또는 0, 5개)

### 테스트 데이터 (`MIND_test_(1000).tsv`)
- 형식: `user \t clicked_news \t candidate_news`
- 첫 번째 후보가 정답 (positive), 나머지 4개가 negative

### 뉴스 데이터 (`MIND_news.tsv`)
- 형식: `news_id \t category \t subcategory \t title \t body`

## 출력

각 에포크마다 다음 지표가 출력됩니다:
- **AUC**: Area Under ROC Curve
- **MRR**: Mean Reciprocal Rank
- **NDCG@5**: Normalized Discounted Cumulative Gain at 5
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10

## GPU 메모리 부족 시

배치 크기를 줄이세요:
```bash
python NAML/train_naml.py --batch_size 16
```

## 사용 예시

### 1. 처음 학습 (전처리 + 학습 + 테스트)
```bash
python NAML/train_naml.py \
    --save_model models/naml_model.h5 \
    --save_data data/preprocessed.pkl \
    --epochs 3
```

### 2. 추가 학습 (이미 전처리된 데이터 사용)
```bash
python NAML/train_naml.py \
    --load_data data/preprocessed.pkl \
    --load_model models/naml_model_epoch3.h5 \
    --save_model models/naml_model.h5 \
    --epochs 2
```

### 3. 테스트만 실행
```bash
python NAML/train_naml.py \
    --test_only \
    --load_model models/naml_model_epoch3.h5 \
    --load_data data/preprocessed.pkl
```

## 주의사항

1. **NLTK 데이터**: 처음 실행 시 `punkt` 토크나이저가 자동으로 다운로드됩니다.
2. **GloVe 임베딩**: 파일이 없어도 랜덤 초기화로 실행 가능합니다.
3. **GPU 메모리**: 배치 크기를 조정하여 GPU 메모리에 맞게 설정하세요.
4. **모델 저장**: 각 에포크마다 `{model_name}_epoch{N}.h5` 형식으로 저장됩니다.
5. **전처리 데이터**: `--save_data`로 저장하면 다음 실행 시 전처리 시간을 절약할 수 있습니다.
