# -*- coding: utf-8 -*-
"""
Convert the small MIND_2000 dataset under `dataset/MIND_2000/` into the
LIME-standard format under `dataset/MIND-2000/{train,dev,test}/`.

Source files
------------
- dataset/MIND_2000/MIND_news.tsv
    Per line: news_id \t category \t subCategory \t title \t abstract
- dataset/MIND_2000/MIND_train_(2000).tsv (has header)
    Per line: user \t clicked_news_ids \t candidate_news_ids \t labels (e.g. "1 0 0 0 0")
- dataset/MIND_2000/MIND_test_(2000).tsv  (no header, used as dev split)
- dataset/MIND_2000/MIND_test_2000_final.tsv (no header, used as test split)
    Per line: user \t clicked_news_ids \t candidate_news_ids   (1st candidate is positive)

Target format
-------------
news.tsv (8 columns):
    news_ID \t category \t subCategory \t title \t abstract \t publishTime \t title_entities \t abstract_entities

behaviors.tsv (6 columns):
    impression_ID \t user_ID \t freshness_lifetime_literal \t history \t impressions \t lifetime_dict_json
    - freshness_lifetime_literal = "[[fresh_for_history...], [lifetime_for_history...], [pos_freshness]]"
    - impressions = "Nxxxx-1 Nyyyy-0 ..."
    - lifetime_dict_json = "[{}, {}, default_lifetime]"

The freshness/lifetime fields are filled with dummy zeros / default lifetime since
this script targets the "pure CROWN" setup where lifetime information is unused.
"""

import os
import json
from collections import OrderedDict

SRC_DIR = os.path.join('dataset', 'MIND_2000')
DST_DIR = os.path.join('dataset', 'MIND-2000')

SPLITS = OrderedDict([
    ('train', 'MIND_train_(2000).tsv'),
    ('dev', 'MIND_test_(2000).tsv'),
    ('test', 'MIND_test_2000_final.tsv'),
])

DEFAULT_LIFETIME = 110462


def load_news(path):
    """Return OrderedDict[news_id] = (category, subCategory, title, abstract)."""
    news = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            news_id, category, sub_category, title, abstract = parts[:5]
            if news_id in news:
                continue
            news[news_id] = (category, sub_category, title, abstract)
    return news


def write_news_tsv(news, out_path):
    """Write 8-column standard news.tsv (publishTime/entity fields left empty)."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for news_id, (category, sub_category, title, abstract) in news.items():
            title = title.replace('\t', ' ').replace('\n', ' ')
            abstract = abstract.replace('\t', ' ').replace('\n', ' ')
            f.write('\t'.join([
                news_id,
                category if category else 'unknown',
                sub_category if sub_category else 'unknown',
                title,
                abstract,
                '0',     # publishTime placeholder
                '[]',    # title_entities placeholder (empty JSON list)
                '[]',    # abstract_entities placeholder
            ]) + '\n')


def parse_split(src_path, has_header, first_is_positive):
    """Yield (user_id, history_ids[list], candidates[list], labels[list of '0'/'1']).

    - has_header: train file has a header line that should be skipped.
    - first_is_positive: dev/test files have no labels and the first candidate
      is the positive sample.
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue
            if has_header and idx == 0:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            user_id = parts[0]
            history = parts[1].split() if parts[1].strip() else []
            candidates = parts[2].split() if parts[2].strip() else []

            if has_header:
                labels_field = parts[3] if len(parts) >= 4 else ''
                labels = labels_field.split() if labels_field.strip() else []
                if len(labels) != len(candidates):
                    raise ValueError(
                        f"label count {len(labels)} != candidate count {len(candidates)} at line {idx+1}"
                    )
            else:
                if first_is_positive:
                    labels = ['1' if i == 0 else '0' for i in range(len(candidates))]
                else:
                    labels = ['0'] * len(candidates)

            yield user_id, history, candidates, labels


def write_behaviors_tsv(rows, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for impression_id, user_id, history, candidates, labels in rows:
            history_str = ' '.join(history)

            impressions_str = ' '.join(f'{c}-{l}' for c, l in zip(candidates, labels))

            history_len = len(history)
            freshness_list = [0] * history_len
            user_topic_lifetime_list = [DEFAULT_LIFETIME] * history_len
            pos_freshness_list = [0]
            freshness_literal = str([freshness_list, user_topic_lifetime_list, pos_freshness_list])

            lifetime_dict_str = json.dumps([{}, {}, DEFAULT_LIFETIME])

            f.write('\t'.join([
                str(impression_id),
                str(user_id),
                freshness_literal,
                history_str,
                impressions_str,
                lifetime_dict_str,
            ]) + '\n')


def main():
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"Source folder not found: {SRC_DIR}")

    news_path = os.path.join(SRC_DIR, 'MIND_news.tsv')
    if not os.path.isfile(news_path):
        raise FileNotFoundError(f"News file not found: {news_path}")
    news = load_news(news_path)
    print(f"[news] loaded {len(news)} news records from {news_path}")

    categories = sorted({cat for cat, _, _, _ in news.values() if cat})
    print(f"[news] {len(categories)} unique categories: {categories[:8]}{' ...' if len(categories) > 8 else ''}")

    for split, fname in SPLITS.items():
        out_dir = os.path.join(DST_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        write_news_tsv(news, os.path.join(out_dir, 'news.tsv'))

        src = os.path.join(SRC_DIR, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Split source file not found: {src}")

        has_header = (split == 'train')
        first_is_positive = (split != 'train')

        rows = []
        for i, (user_id, hist, cands, labels) in enumerate(
                parse_split(src, has_header=has_header, first_is_positive=first_is_positive)):
            rows.append((i + 1, user_id, hist, cands, labels))

        write_behaviors_tsv(rows, os.path.join(out_dir, 'behaviors.tsv'))
        print(f"[{split}] wrote {len(rows)} impressions to {os.path.join(out_dir, 'behaviors.tsv')}")

    cat_path = 'category-mind2000.json'
    if not os.path.exists(cat_path):
        category_dict = {}
        for cat in categories:
            if cat not in category_dict:
                category_dict[cat] = len(category_dict)
        with open(cat_path, 'w', encoding='utf-8') as f:
            json.dump(category_dict, f)
        print(f"[meta] wrote {cat_path}")
    else:
        print(f"[meta] {cat_path} already exists, skip")

    topic_path = 'topic_wise_lifetime-mind2000.json'
    if not os.path.exists(topic_path):
        topic_lifetime = {cat: DEFAULT_LIFETIME for cat in categories}
        topic_lifetime.setdefault('Unknown', DEFAULT_LIFETIME)
        with open(topic_path, 'w', encoding='utf-8') as f:
            json.dump(topic_lifetime, f)
        print(f"[meta] wrote {topic_path}")
    else:
        print(f"[meta] {topic_path} already exists, skip")

    print('Done.')


if __name__ == '__main__':
    main()
