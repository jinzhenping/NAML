# IMRec

****

The dataset for paper "Why Do We Click: Visual Impression-aware News Recommendation", ACM MM 2021

****

**Dataset**

- The filename corresponds to the newsid in the original dataset.
- The download link for our dataset: https://drive.google.com/file/d/1gx0OzN7qSuyRlvN0cfVUjB4tmoKvvQk1/view?usp=sharing
- imageList.npy: The newsid list of successfully crawled images in the format of [newsid,...].

**Utils**

- data_generator.py：For generating IM-MIND dataset (news  text and cover images are required).
- word_feature_generator.py：For generating visual impression representations of words.
- global_feature_generator.py：For generating global impression representations.

---

## MIND_2000 학습/평가 (ours)

뉴스 카드 합성 → ResNet-101 local/global feature → **NRMS-IM** / **FIM-IM** 학습.
에폭마다 `MIND_dev_(2000).tsv`(val) MRR로 best 선택 후 `MIND_test_(2000).tsv` 평가.

```bash
conda activate clip_cu128

# NRMS-IM
python IMRec/train_eval.py --model nrms-im --mind-dataset-subdir MIND_2000 \
    --glove-path glove/glove.6B.100d.txt

# FIM-IM
python IMRec/train_eval.py --model fim-im --mind-dataset-subdir MIND_2000 \
    --glove-path glove/glove.6B.100d.txt
```

하이퍼파라미터 튜닝 (val MRR, MM_Rec과 동일 프로토콜):

```bash
python IMRec/tune.py --model nrms-im --mind-dataset-subdir MIND_2000 --two-phase \
    --trials 24 --screening-epochs 3 --refine-top-k 5 --epochs-per-trial 30 \
    --glove-path glove/glove.6B.100d.txt

python IMRec/tune.py --model fim-im --mind-dataset-subdir MIND_2000 --two-phase \
    --trials 24 --screening-epochs 3 --refine-top-k 5 --epochs-per-trial 30 \
    --glove-path glove/glove.6B.100d.txt
```

단계만:

```bash
python IMRec/train_eval.py --stage prepare --mind-dataset-subdir MIND_2000
python IMRec/train_eval.py --stage extract --mind-dataset-subdir MIND_2000
python IMRec/train_eval.py --stage train --model nrms-im --mind-dataset-subdir MIND_2000
```

썸네일: MM_Rec과 동일 — `dataset/MIND_thumbnail/{news_id}.jpg`  
(`MM_Rec/dataset_paths.py`의 `DEFAULT_THUMBNAIL_DIR`)
결과: `IMRec/saved_models/MIND_2000/{nrms_im|fim_im}/`

