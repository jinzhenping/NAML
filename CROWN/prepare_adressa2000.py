# -*- coding: utf-8 -*-
"""
Convert `dataset/Adressa_2000/` (NAML-style flat TSV) into CROWN/LIME layout
`dataset/Adressa-2000/{train,dev,test}/`.

Impressions / labels come from Adressa_2000. Publish times and click timestamps
come from `dataset/Adressa_2000_timeline/Adressa_news.tsv`.

Run from CROWN/:
  python prepare_adressa2000.py

Then train, e.g.:
  python main.py --dataset adressa2000 --news_encoder LIME --content_encoder CROWN --user_encoder CROWN
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime

import numpy as np

SRC_DIR = os.path.join('dataset', 'Adressa_2000')
TIMELINE_NEWS = os.path.join('dataset', 'Adressa_2000_timeline', 'Adressa_news.tsv')
DST_DIR = os.path.join('dataset', 'Adressa-2000')

SPLITS = OrderedDict([
    ('train', 'Adressa_train_(2000).tsv'),
    ('dev', 'Adressa_test_(2000).tsv'),
    ('test', 'Adressa_test_2000_final.tsv'),
])

DEFAULT_LIFETIME = 110462
LIFETIME_QUANTILE = 0.9
DATETIME_FMT = '%Y-%m-%d %H:%M:%S'
TIMELINE_SUFFIX_RE = re.compile(
    r'\t([a-f0-9]{40})\t(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\t(.+)$'
)


def load_news_basic(path: str) -> OrderedDict:
    """id -> (category, sub_category, title, abstract) from Adressa_2000 news."""
    news = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            news_id, category, sub_category, title, abstract = parts[:5]
            if news_id not in news:
                news[news_id] = (category, sub_category, title, abstract)
    return news


def load_timeline_news(path: str) -> dict:
    """
    Parse timeline news.tsv (title/abstract may contain tabs).
    Returns news_id -> dict(category, sub_category, publish_ts, click_tss).
    """
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            m = TIMELINE_SUFFIX_RE.search(line)
            if not m:
                continue
            publish_str = m.group(2)
            click_str = m.group(3)
            prefix = line[: m.start()]
            parts = prefix.split('\t')
            if len(parts) < 5:
                continue
            news_id = parts[0]
            category = parts[1]
            sub_category = parts[2]
            try:
                publish_ts = int(datetime.strptime(publish_str, DATETIME_FMT).timestamp())
            except ValueError:
                continue
            click_tss = []
            for ts_str in click_str.split(','):
                ts_str = ts_str.strip()
                if not ts_str:
                    continue
                try:
                    click_tss.append(int(datetime.strptime(ts_str, DATETIME_FMT).timestamp()))
                except ValueError:
                    continue
            click_tss.sort()
            meta[news_id] = {
                'category': category,
                'sub_category': sub_category,
                'publish_ts': publish_ts,
                'click_tss': click_tss,
            }
    return meta


def article_lifetime_seconds(publish_ts: int, click_tss: list, quantile: float = LIFETIME_QUANTILE) -> int | None:
    """Seconds from publish to the click time at `quantile` of article clicks."""
    if not click_tss:
        return None
    idx = max(0, min(len(click_tss) - 1, int(np.ceil(quantile * len(click_tss)) - 1)))
    return max(0, click_tss[idx] - publish_ts)


def compute_topic_wise_lifetime(timeline_meta: dict) -> dict[str, int]:
    by_topic: dict[str, list[int]] = defaultdict(list)
    for info in timeline_meta.values():
        lt = article_lifetime_seconds(info['publish_ts'], info['click_tss'])
        if lt is None:
            continue
        by_topic[info['category']].append(lt)
    topic_lifetime = {}
    for topic, values in by_topic.items():
        topic_lifetime[topic] = int(np.percentile(values, LIFETIME_QUANTILE * 100))
    if not topic_lifetime:
        return {'Unknown': DEFAULT_LIFETIME}
    topic_lifetime.setdefault('Unknown', int(np.median(list(topic_lifetime.values()))))
    return topic_lifetime


def parse_split(src_path: str, has_header: bool, first_is_positive: bool):
    with open(src_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n\r')
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
                        f'label count {len(labels)} != candidate count {len(candidates)} '
                        f'at line {idx + 1} in {src_path}'
                    )
            else:
                if first_is_positive:
                    labels = ['1' if i == 0 else '0' for i in range(len(candidates))]
                else:
                    labels = ['0'] * len(candidates)

            yield user_id, history, candidates, labels


def _positive_news(candidates: list, labels: list) -> str | None:
    for cand, lab in zip(candidates, labels):
        if lab == '1':
            return cand
    return candidates[0] if candidates else None


def _impression_time(
    user_id: str,
    pos_news: str,
    timeline_meta: dict,
    user_news_click_idx: dict,
) -> int:
    """Assign next global click timestamp on pos_news to this user (sequential heuristic)."""
    info = timeline_meta.get(pos_news)
    if not info:
        return 0
    publish_ts = info['publish_ts']
    click_tss = info['click_tss']
    if not click_tss:
        return publish_ts
    key = (user_id, pos_news)
    idx = user_news_click_idx.get(key, 0)
    if idx >= len(click_tss):
        idx = len(click_tss) - 1
    user_news_click_idx[key] = idx + 1
    return click_tss[idx]


def _freshness_at(impression_ts: int, news_id: str, timeline_meta: dict) -> int:
    info = timeline_meta.get(news_id)
    if not info or impression_ts <= 0:
        return 0
    return max(0, impression_ts - info['publish_ts'])


def _user_topic_value(
    user_id: str,
    topic: str,
    user_topic_lifetime: dict,
    topic_wise_lifetime: dict,
) -> int:
    if user_id in user_topic_lifetime and topic in user_topic_lifetime[user_id]:
        return user_topic_lifetime[user_id][topic]
    return topic_wise_lifetime.get(topic, topic_wise_lifetime.get('Unknown', DEFAULT_LIFETIME))


def build_train_user_topic_lifetime(
    src_path: str,
    timeline_meta: dict,
    topic_wise_lifetime: dict,
) -> dict[str, dict[str, int]]:
    """Scan train impressions in order; estimate per-user per-topic lifetime (p90 click age)."""
    user_topic_ages: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    user_news_click_idx: dict = {}
    impressions_by_user: dict[str, list] = defaultdict(list)

    for user_id, history, candidates, labels in parse_split(src_path, has_header=True, first_is_positive=False):
        pos = _positive_news(candidates, labels)
        if not pos:
            continue
        impressions_by_user[user_id].append((history, candidates, labels, pos))

    for user_id, rows in impressions_by_user.items():
        for _history, _candidates, labels, pos in rows:
            imp_ts = _impression_time(user_id, pos, timeline_meta, user_news_click_idx)
            info = timeline_meta.get(pos)
            if not info or imp_ts <= 0:
                continue
            age = max(0, imp_ts - info['publish_ts'])
            user_topic_ages[user_id][info['category']].append(age)

    user_topic_lifetime: dict[str, dict[str, int]] = {}
    for user_id, topic_map in user_topic_ages.items():
        user_topic_lifetime[user_id] = {}
        for topic, ages in topic_map.items():
            if len(ages) == 1:
                val = ages[0]
            else:
                val = int(np.percentile(ages, LIFETIME_QUANTILE * 100))
            user_topic_lifetime[user_id][topic] = max(1, val)
    return user_topic_lifetime


def build_behavior_rows(
    rows,
    timeline_meta: dict,
    topic_wise_lifetime: dict,
    user_topic_lifetime: dict,
    *,
    update_user_state: bool,
) -> list[tuple]:
    """Return list of (impression_id, user_id, history, candidates, labels, time_str, lifetime_str)."""
    user_seen_topics: dict[str, set] = defaultdict(set)
    user_news_click_idx: dict = {}
    out = []

    impressions_by_user: dict[str, list] = defaultdict(list)
    for imp_i, user_id, history, candidates, labels in rows:
        impressions_by_user[user_id].append((imp_i, history, candidates, labels))

    global_imp_id = 0
    for user_id in sorted(impressions_by_user.keys(), key=lambda u: min(x[0] for x in impressions_by_user[u])):
        for imp_i, history, candidates, labels in sorted(impressions_by_user[user_id], key=lambda x: x[0]):
            global_imp_id += 1
            pos = _positive_news(candidates, labels)
            imp_ts = _impression_time(user_id, pos, timeline_meta, user_news_click_idx) if pos else 0

            freshness_list = [_freshness_at(imp_ts, nid, timeline_meta) for nid in history]
            user_topic_list = [
                _user_topic_value(user_id, timeline_meta[nid]['category'], user_topic_lifetime, topic_wise_lifetime)
                if nid in timeline_meta
                else topic_wise_lifetime.get('Unknown', DEFAULT_LIFETIME)
                for nid in history
            ]
            pos_freshness = _freshness_at(imp_ts, pos, timeline_meta) if pos else 0

            cand_topics = set()
            for cand in candidates:
                if cand in timeline_meta:
                    cand_topics.add(timeline_meta[cand]['category'])

            seen = user_seen_topics[user_id]
            category_lifetime_dict = {
                t: _user_topic_value(user_id, t, user_topic_lifetime, topic_wise_lifetime)
                for t in seen
            }
            unseen_dict = {
                t: topic_wise_lifetime.get(t, topic_wise_lifetime.get('Unknown', DEFAULT_LIFETIME))
                for t in cand_topics
                if t not in seen
            }
            default_lifetime = topic_wise_lifetime.get('Unknown', DEFAULT_LIFETIME)

            time_str = str([freshness_list, user_topic_list, [pos_freshness]])
            lifetime_str = json.dumps([category_lifetime_dict, unseen_dict, default_lifetime])

            out.append((global_imp_id, user_id, history, candidates, labels, time_str, lifetime_str))

            if update_user_state and pos and pos in timeline_meta:
                user_seen_topics[user_id].add(timeline_meta[pos]['category'])

    return out


def write_news_tsv(news: OrderedDict, timeline_meta: dict, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        for news_id, (category, sub_category, title, abstract) in news.items():
            title = title.replace('\t', ' ').replace('\n', ' ')
            abstract = abstract.replace('\t', ' ').replace('\n', ' ')
            publish_ts = timeline_meta.get(news_id, {}).get('publish_ts', 0)
            f.write('\t'.join([
                news_id,
                category if category else 'unknown',
                sub_category if sub_category else 'unknown',
                title,
                abstract,
                str(publish_ts),
                '[]',
                '[]',
            ]) + '\n')


def write_behaviors_tsv(behavior_rows, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        for impression_id, user_id, history, candidates, labels, time_str, lifetime_str in behavior_rows:
            history_str = ' '.join(history)
            impressions_str = ' '.join(f'{c}-{l}' for c, l in zip(candidates, labels))
            f.write('\t'.join([
                str(impression_id),
                str(user_id),
                time_str,
                history_str,
                impressions_str,
                lifetime_str,
            ]) + '\n')


def _touch_empty_vec(split_dir: str, name: str) -> None:
    path = os.path.join(split_dir, name)
    if not os.path.isfile(path):
        open(path, 'a', encoding='utf-8').close()
        print(f'[{split_dir}] created empty {name}')


def main() -> None:
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f'Source folder not found: {SRC_DIR}')
    if not os.path.isfile(TIMELINE_NEWS):
        raise FileNotFoundError(f'Timeline news not found: {TIMELINE_NEWS}')

    news_path = os.path.join(SRC_DIR, 'Adressa_news.tsv')
    news = load_news_basic(news_path)
    timeline_meta = load_timeline_news(TIMELINE_NEWS)
    print(f'[news] {len(news)} records from {news_path}')
    print(f'[timeline] {len(timeline_meta)} articles with publish/click times')

    overlap = sum(1 for nid in news if nid in timeline_meta)
    print(f'[timeline] {overlap}/{len(news)} news ids have timeline metadata')

    topic_wise_lifetime = compute_topic_wise_lifetime(timeline_meta)
    print(f'[lifetime] topic-wise stats for {len(topic_wise_lifetime)} categories')

    train_src = os.path.join(SRC_DIR, SPLITS['train'])
    user_topic_lifetime = build_train_user_topic_lifetime(
        train_src, timeline_meta, topic_wise_lifetime
    )
    print(f'[lifetime] user-topic profiles for {len(user_topic_lifetime)} train users')

    categories = sorted({cat for cat, _, _, _ in news.values() if cat})

    for split, fname in SPLITS.items():
        out_dir = os.path.join(DST_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        write_news_tsv(news, timeline_meta, os.path.join(out_dir, 'news.tsv'))
        _touch_empty_vec(out_dir, 'entity_embedding.vec')
        _touch_empty_vec(out_dir, 'context_embedding.vec')

        src = os.path.join(SRC_DIR, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f'Split source not found: {src}')

        has_header = split == 'train'
        first_is_positive = split != 'train'
        raw_rows = list(parse_split(src, has_header=has_header, first_is_positive=first_is_positive))
        indexed_rows = [(i, uid, h, c, l) for i, (uid, h, c, l) in enumerate(raw_rows)]
        behavior_rows = build_behavior_rows(
            indexed_rows,
            timeline_meta,
            topic_wise_lifetime,
            user_topic_lifetime,
            update_user_state=(split == 'train'),
        )
        write_behaviors_tsv(behavior_rows, os.path.join(out_dir, 'behaviors.tsv'))
        print(f'[{split}] wrote {len(behavior_rows)} impressions -> {out_dir}/')

    cat_path = 'category-adressa2000.json'
    with open(cat_path, 'w', encoding='utf-8') as f:
        json.dump({cat: i for i, cat in enumerate(categories)}, f, ensure_ascii=False, indent=2)
    print(f'[meta] wrote {cat_path} ({len(categories)} categories)')

    topic_path = 'topic_wise_lifetime-adressa2000.json'
    with open(topic_path, 'w', encoding='utf-8') as f:
        json.dump(topic_wise_lifetime, f, ensure_ascii=False, indent=2)
    print(f'[meta] wrote {topic_path}')

    print('Done. Use --dataset adressa2000 with LIME/CROWN encoders.')


if __name__ == '__main__':
    main()
