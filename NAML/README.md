# NAML (Neural News Recommendation with Attentive Multi-View Learning)

MIND 데이터셋을 사용한 뉴스 추천 모델 학습 및 테스트

## 설치

```bash
conda activate tf28gpu

# 프로젝트 루트에서
pip install -r requirements.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/NAML.py
python NAML/NAML.py

- `USE_EXPECTED_BODY=False` 메인 학습 시 테스트셋(실제본문) **MRR**이 최고일 때 `MAIN_TRAINING_BEST_MODEL_PATH`에 가중치 저장 (`SAVE_MAIN_BEST_BY_TEST_ACTUAL_MRR`).

## 데이터셋 폴더 (`dataset/MIND`, `MIND_1000`, `MIND_2000` 등)

- **NAML.py**: `MIND_DATASET_SUBDIR` 만 `MIND` / `MIND_1000` / `MIND_2000` 등으로 바꾸면 됨 (또는 환경변수 `MIND_DATASET_SUBDIR`).
- **자동**: `MIND_news.tsv` + `MIND_train_*.tsv` 1개 + `MIND_test_*.tsv` 1개가 있으면 그 조합을 자동 사용 (`MIND_1000` 등).
- **프리셋**: `MIND_2000` 은 `MIND_train_(2000).tsv` / `MIND_test_(2000).tsv` 가 `MIND_DATASET_PRESETS` 에 등록됨. 새 폴더 규칙이 다르면 그 딕셔너리에 한 줄 추가.
- **수동 덮어쓰기**: 환경변수 `MIND_NEWS_FILENAME`, `MIND_TRAIN_FILENAME`, `MIND_TEST_FILENAME`.
- 시작 시 `[데이터셋] dataset/...` 로 실제 사용 파일명이 출력됨.
- **body_generation**: `--mind_dataset_subdir` 또는 위와 동일 규칙.
- **train_naml.py**: 동일. 필요 시 `--train_file` 등으로 전체 경로 지정.
