# Pure CROWN on MIND-2000 — Quickstart

This bundle contains everything needed to train a **pure CROWN** model
(no LIME freshness wrapper, no remaining-lifetime weighting) on the small
`MIND_2000` dataset.

conda activate crown_bw
python main.py  --dataset=mind2000 --config_file=pure_crown.json

## Folder layout

```
lime_pure_crown_mind2000/
├── main.py                 # entry point (training + testing)
├── config.py               # arg parsing + per-dataset settings
├── corpus.py / dataset.py  # data loading
├── model.py                # NewsRec model (news + user encoders + scorer)
├── newsEncoders.py         # CROWN, LIME, NAML, CNN, ... encoders
├── userEncoders.py         # CROWN, MINER, LSTUR, ... user encoders
├── layers.py / util.py / trainer.py / evaluate.py
├── prepare_dataset.py      # imported by config.py (kept for completeness)
├── prepare_mind2000.py     # converts dataset/MIND_2000/* → dataset/MIND-2000/*
├── pure_crown.json         # forces CROWN-CROWN-CROWN and disables lifetime weighting
├── category-mind2000.json
├── topic_wise_lifetime-mind2000.json
├── README.md / LICENSE
└── dataset/
    ├── MIND_2000/                       # original raw files
    │   ├── MIND_news.tsv
    │   ├── MIND_train_(2000).tsv
    │   ├── MIND_test_(2000).tsv         # used as dev split
    │   └── MIND_test_2000_final.tsv     # used as test split
    └── MIND-2000/                       # LIME-standard format (already converted)
        ├── train/{news.tsv, behaviors.tsv}
        ├── dev/{news.tsv, behaviors.tsv}
        └── test/{news.tsv, behaviors.tsv}
```

## Python environment

```
python 3.10.16
torch 2.0.1
torchtext 0.15.2
pandas, numpy, scikit-learn, nltk, transformers, tqdm
```

(Same as the original LIME repository. A GPU with CUDA is required: `config.py`
asserts `torch.cuda.is_available()`.)

## Step 1 (only if dataset/MIND-2000/ is missing or you want to re-convert)

```bash
python prepare_mind2000.py
```

This (re)generates `dataset/MIND-2000/{train,dev,test}/{news.tsv, behaviors.tsv}`
from the four files in `dataset/MIND_2000/`. The bundle already ships the
converted output, so this step is optional.

## Step 2 — Train + test

```bash
python main.py --config_file=pure_crown.json --dataset=mind2000
```

The `pure_crown.json` overrides force the following:

| Setting | Value | Effect |
| --- | --- | --- |
| `news_encoder` | `CROWN` | LIME wrapper (freshness/lifetime fusion) is removed |
| `content_encoder` | `CROWN` | unused when `news_encoder != LIME`, kept for clarity |
| `user_encoder` | `CROWN` | CROWN user encoder |
| `use_remaining_lifetime_weighting` | `false` | dot-product score is **not** multiplied by lifetime weight |

When training starts, verify the `Experiment setting` log contains:

```
dataset : mind2000
news_encoder : CROWN
user_encoder : CROWN
content_encoder : CROWN
use_remaining_lifetime_weighting : False
train_root : dataset/MIND-2000/train
dev_root   : dataset/MIND-2000/dev
test_root  : dataset/MIND-2000/test
```

## First-run downloads

`corpus.py` builds vocabulary + word embedding caches on the first run. It
calls `torchtext.vocab.GloVe(name='840B', dim=300, cache='../../glove')`,
which downloads **GloVe.840B.300d (~2 GB)** into the folder two levels above
this project (e.g. `C:\Users\<you>\glove\` if this bundle lives in
`C:\Users\<you>\Downloads\lime_pure_crown_mind2000\`).

After the first run the following caches are kept next to the source files
and reused:

```
user_ID-mind2000.json
news_ID-mind2000.json
subCategory-mind2000.json
vocabulary-3-MIND-32-128-mind2000.json
word_embedding-3-300-MIND-32-128-mind2000.pkl
```

## Outputs

`config.preliminary_setup()` automatically creates these output folders inside
the bundle when training starts:

```
models/mind2000/CROWN-CROWN/         # per-epoch checkpoints
best_model/mind2000/CROWN-CROWN/     # best checkpoint (selected on dev)
results/mind2000/CROWN-CROWN/        # per-run dev / test metrics
dev/{ref,res}/mind2000/CROWN-CROWN/  # dev prediction files + groundtruth
test/{ref,res}/mind2000/CROWN-CROWN/ # test prediction files + groundtruth
configs/mind2000/CROWN-CROWN/
```

Final test metrics (AUC, MRR, nDCG@5, nDCG@10) are printed at the end of the
run and also written to `results/mind2000/CROWN-CROWN/#<run_index>-test`.

## Data conversion summary (already done)

- `MIND_train_(2000).tsv` carries the labels `1 0 0 0 0` in its last column,
  which are re-attached to candidate IDs as `Nxxx-1 Nyyy-0 ...`.
- `MIND_test_(2000).tsv` (dev) and `MIND_test_2000_final.tsv` (test) have no
  labels; the first candidate of each row is treated as the positive sample
  (so it becomes `Nxxx-1` and the remaining four are `-0`).
- The freshness / user-topic-lifetime fields in `behaviors.tsv` are filled
  with dummy zeros / default-lifetime values because the pure-CROWN
  configuration ignores them at both the encoder and the scorer level.
