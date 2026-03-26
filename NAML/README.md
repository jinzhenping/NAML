# NAML (Neural News Recommendation with Attentive Multi-View Learning)

MIND 데이터셋을 사용한 뉴스 추천 모델 학습 및 테스트

## 설치

```bash
conda activate tf28gpu

# 프로젝트 루트에서
pip install -r requirements.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/NAML.py
python NAML/NAML.py
```

## 클러스터 배치 평가 (실제본문 vs 기대본문)

`generate_body_cluster_train_batches.py`로 생성한 기대본문 폴더를 두고, 사전학습 가중치 `saved_models/NAML_mind_2000.h5`로 해당 배치의 학습 세션만 평가해 `NAML/results/resultN.txt`에 JSON을 저장합니다.

```bash
# 프로젝트 루트에서 (train-body-dir 은 실제 cluster<C>_batch<B> 경로로 맞출 것)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/eval_cluster_batch.py \
  --cluster-csv NAML/user_kmeans_k3_MIND_2000.csv --cluster-id 0 --batch-index 0 \
  --train-body-dir body_generation/output/MIND_2000/cluster0_batch0
```

- `performance_feedback`: `loss_*`는 학습용 `model`의 categorical crossentropy(배치 평균), `ndcg5_*`는 `model_test` 점수로 세션별 NDCG@5 평균입니다.
- `expected_body_coverage`: 로드된 JSON 항목 수, 배치 내 패딩 제외 후보 슬롯 수, 그중 `_norm_expected_body_key`로 매칭된 슬롯 수·비율(`batch_match_rate`), 뉴스 ID가 빈 슬롯 수. 실행 시 콘솔에도 같은 요약이 출력됩니다.
- `diagnostic_samples`: `failure` = |NDCG_real−NDCG_expected|가 가장 큰 세션, `success` = 가장 작은 세션. 정답 후보에 대응하는 `user_<id>/news_<id>.json`이 있으면 그 안의 `candidate_title`, `user_history`, `generated_body`를 우선 사용하고, 없으면 `MIND_news.tsv`·NAML 전처리 히스토리로 보완합니다.

## 테스트셋 평가 (실제본문 vs 기대본문, 3개 지표)

프리트레인 가중치(예: `saved_models/NAML_mind_2000.h5`)를 로드해 테스트셋에서
실제본문/기대본문을 각각 평가하고, NAML 기본 지표 3개(MRR, NDCG@5, Hit@1)를 비교합니다.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/eval_test_expected.py \
  --expected-dir body_generation/output/MIND_2000/test_cluster_mixed_run1 \
  --weights saved_models/NAML_mind_2000.h5 \
  --mind-dataset-subdir MIND_2000
```

- `--expected-dir`: 기대본문 폴더 (`user_*/news_*.json`)
- 결과는 **콘솔만** 출력합니다 (성능·매칭율·OOV).
- `word_dict`는 **뉴스 TSV만**으로 만들어 사전학습 가중치와 임베딩 크기를 맞춥니다. 기대본문에만 있는 단어는 토큰화 시 제외됩니다(OOV).
- **OOV 토큰** (모두 `lower` + `word_tokenize`, `word_dict` 기준): 기대본문 2줄(JSON 1회 합산 / 기대본문 매칭 슬롯 반복 합산), 실제본문 2줄(`MIND_news.tsv` 원문 — 테스트 후보에 등장하는 **고유 뉴스** 1회 합산 / **모든 후보 슬롯** 반복 합산).

## 테스트셋 본문 미사용 ablation

후보 본문·히스토리 뉴스 본문을 모두 패딩(`news_body[0]`)으로 넣고, 제목·카테고리·히스토리 제목은 유지한 채 MRR / NDCG@5 / Hit@1만 출력합니다.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python NAML/eval_test_no_body.py \
  --weights saved_models/NAML_mind_2000.h5 \
  --mind-dataset-subdir MIND_2000
```

## 클러스터 배치 자동 파이프라인 (생성 → 평가 → 조율기)

`N = start..end`마다 `generate_body_cluster_train_batches` → `eval_cluster_batch`(`--result-index N`) → `coordinator.py`(`--n N`)를 순서대로 실행합니다.

```bash
# 프로젝트 루트에서 (예: 배치 0,1,2)
CUDA_VISIBLE_DEVICES=1 python scripts/run_cluster_batch_pipeline.py --start 0 --end 2
```

- 시작 배치 `N`에 대해 **`coordinator_LLM/output/N.txt`가 이미 있어야** 합니다(예: `0.txt` 시드).
- 조율기는 `N.txt`·`resultN.txt`를 읽고 `(N+1).txt`를 쓰므로, 다음 배치 `N+1` 생성 시 정책 `N+1.txt`를 사용하게 됩니다.
- GPU 환경 변수를 쓰지 않으려면 `--no-cuda-env` 를 붙입니다.
