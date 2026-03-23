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
