# NAML (Neural News Recommendation with Attentive Multi-View Learning)

MIND 데이터셋을 사용한 뉴스 추천 모델 학습 및 테스트

## 설치

```bash
# 프로젝트 루트에서
cd ~
conda activate tf28

pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=1 python NAML/NAML.py
python NAML/NAML.py
