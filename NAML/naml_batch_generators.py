# -*- coding: utf-8 -*-
"""
NAML 학습/테스트용 배치 제너레이터 (NAML.py에서 분리).
전역 의존 없이 인자로 텐서·딕셔너리를 받습니다.
"""
from __future__ import annotations

import numpy as np
from nltk.tokenize import word_tokenize

import naml_common as _naml_common_runtime
from naml_common import MAX_HISTORY_CLICKS, clip_expected_body_to_first_sentences


def _norm_expected_body_key(uid, nid):
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ""
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ""
    n = str(nid).strip() if nid is not None else ""
    return (u, n)


def generate_batch_data_train(
    word_dict,
    news_words,
    news_body,
    news_v,
    news_sv,
    news_index,
    all_train_pn,
    all_label,
    all_train_id,
    all_user_pos,
    batch_size,
    candidate_news_body=None,
    expected_bodies=None,
    all_userid_str=None,
    all_train_newsid_str=None,
    news_index_reverse=None,
    use_expected_body_positive_only=False,
    shuffle=True,
):
    if news_index_reverse is None:
        news_index_reverse = {v: k for k, v in news_index.items()}

    inputid = np.arange(len(all_label))
    if shuffle:
        np.random.shuffle(inputid)
    y = all_label
    batches = [
        inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))]
        for i in range(len(y) // batch_size + 1)
        if batch_size * i < len(y)
    ]

    while True:
        for batch_indices in batches:
            batch_candidate_splits = [[] for _ in range(5)]
            batch_browsed_news_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_candidate_body_splits = [[] for _ in range(5)]
            batch_browsed_news_body_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_candidate_vertical_splits = [[] for _ in range(5)]
            batch_browsed_news_vertical_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_candidate_subvertical_splits = [[] for _ in range(5)]
            batch_browsed_news_subvertical_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_labels = []

            for idx in batch_indices:
                candidate_indices = np.array(all_train_pn[idx], dtype="int32")
                candidate = news_words[candidate_indices]
                candidate_split = [np.expand_dims(candidate[k], axis=0) for k in range(candidate.shape[0])]

                if expected_bodies is not None and all_userid_str is not None and all_train_newsid_str is not None:
                    user_id_str = all_userid_str[idx]
                    news_ids_str = all_train_newsid_str[idx]
                    candidate_body_list = []
                    label_vec = all_label[idx] if use_expected_body_positive_only else None

                    for j, news_idx in enumerate(all_train_pn[idx]):
                        if news_idx == 0:
                            candidate_body_list.append(news_body[0])
                        else:
                            if use_expected_body_positive_only and label_vec is not None:
                                is_positive = j < len(label_vec) and label_vec[j] == 1
                                if not is_positive:
                                    candidate_body_list.append(news_body[news_idx])
                                    continue
                            news_id_str = news_ids_str[j] if j < len(news_ids_str) else ""
                            key = _norm_expected_body_key(user_id_str, news_id_str)
                            if key in expected_bodies:
                                expected_body = expected_bodies[key]
                                _eb = clip_expected_body_to_first_sentences(
                                    expected_body, _naml_common_runtime.EXPECTED_BODY_FIRST_N_SENTENCES
                                )
                                body_tokens = word_tokenize(_eb.lower()) if _eb else []
                                word_id = []
                                for word in body_tokens:
                                    if word in word_dict:
                                        word_id.append(word_dict[word][0])
                                word_id = word_id[:300]
                                word_id = word_id + [0] * (300 - len(word_id))
                                candidate_body_list.append(np.array(word_id, dtype="int32"))
                            else:
                                candidate_body_list.append(news_body[news_idx])

                    candidate_body = np.array(candidate_body_list)
                elif candidate_news_body is not None:
                    candidate_body = candidate_news_body[candidate_indices]
                else:
                    candidate_body = news_body[candidate_indices]

                candidate_body_split = [np.expand_dims(candidate_body[k], axis=0) for k in range(candidate_body.shape[0])]

                candidate_vertical = news_v[candidate_indices]
                candidate_vertical_split = [np.expand_dims(candidate_vertical[k], axis=0) for k in range(candidate_vertical.shape[0])]
                candidate_subvertical = news_sv[candidate_indices]
                candidate_subvertical_split = [np.expand_dims(candidate_subvertical[k], axis=0) for k in range(candidate_subvertical.shape[0])]

                user_pos_indices = np.array(all_user_pos[idx], dtype="int32")
                browsed_news = news_words[user_pos_indices]
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]
                browsed_news_body_split = [np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])]
                browsed_news_vertical = news_v[user_pos_indices]
                browsed_news_vertical_split = [np.expand_dims(browsed_news_vertical[k], axis=0) for k in range(browsed_news_vertical.shape[0])]
                browsed_news_subvertical = news_sv[user_pos_indices]
                browsed_news_subvertical_split = [np.expand_dims(browsed_news_subvertical[k], axis=0) for k in range(browsed_news_subvertical.shape[0])]

                label = np.array(all_label[idx], dtype="float32")

                for k in range(5):
                    batch_candidate_splits[k].append(candidate_split[k])
                for k in range(len(browsed_news_split)):
                    batch_browsed_news_splits[k].append(browsed_news_split[k])
                for k in range(5):
                    batch_candidate_body_splits[k].append(candidate_body_split[k])
                for k in range(len(browsed_news_body_split)):
                    batch_browsed_news_body_splits[k].append(browsed_news_body_split[k])
                for k in range(5):
                    batch_candidate_vertical_splits[k].append(candidate_vertical_split[k])
                for k in range(len(browsed_news_vertical_split)):
                    batch_browsed_news_vertical_splits[k].append(browsed_news_vertical_split[k])
                for k in range(5):
                    batch_candidate_subvertical_splits[k].append(candidate_subvertical_split[k])
                for k in range(len(browsed_news_subvertical_split)):
                    batch_browsed_news_subvertical_splits[k].append(browsed_news_subvertical_split[k])
                batch_labels.append(label)

            batch_inputs = []
            for k in range(5):
                if batch_candidate_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_splits[k], axis=0))
            for k in range(MAX_HISTORY_CLICKS):
                if batch_browsed_news_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_splits[k], axis=0))
            for k in range(5):
                if batch_candidate_body_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_body_splits[k], axis=0))
            for k in range(MAX_HISTORY_CLICKS):
                if batch_browsed_news_body_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_body_splits[k], axis=0))
            for k in range(5):
                if batch_candidate_vertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_vertical_splits[k], axis=0))
            for k in range(MAX_HISTORY_CLICKS):
                if batch_browsed_news_vertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_vertical_splits[k], axis=0))
            for k in range(5):
                if batch_candidate_subvertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_candidate_subvertical_splits[k], axis=0))
            for k in range(MAX_HISTORY_CLICKS):
                if batch_browsed_news_subvertical_splits[k]:
                    batch_inputs.append(np.concatenate(batch_browsed_news_subvertical_splits[k], axis=0))

            batch_labels_array = np.array(batch_labels)
            yield (batch_inputs, batch_labels_array)


def generate_batch_data_test(
    word_dict,
    news_words,
    news_body,
    news_v,
    news_sv,
    news_index,
    all_test_pn,
    all_label,
    all_test_id,
    all_test_user_pos,
    batch_size,
    candidate_news_body=None,
    expected_bodies=None,
    all_userid_str=None,
    all_newsid_str=None,
    news_index_reverse=None,
    all_test_user_pos_override=None,
    *,
    expected_body_clip_n_sentences=None,
):
    if news_index_reverse is None:
        news_index_reverse = {v: k for k, v in news_index.items()}

    inputid = np.arange(len(all_label))
    y = all_label
    batches = [
        inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))]
        for i in range(len(y) // batch_size + 1)
        if batch_size * i < len(y)
    ]

    MAX_SENTS = MAX_HISTORY_CLICKS
    while True:
        for batch_indices in batches:
            batch_candidates = []
            batch_browsed_news = [[] for _ in range(MAX_SENTS)]
            batch_candidate_body = []
            batch_browsed_news_body = [[] for _ in range(MAX_SENTS)]
            batch_candidate_vertical = []
            batch_browsed_news_vertical = [[] for _ in range(MAX_SENTS)]
            batch_candidate_subvertical = []
            batch_browsed_news_subvertical = [[] for _ in range(MAX_SENTS)]
            batch_labels = []
            for idx in batch_indices:
                news_idx = int(all_test_pn[idx])
                candidate = news_words[news_idx]
                candidate = np.expand_dims(candidate, axis=0)
                if expected_bodies is not None and all_userid_str is not None and all_newsid_str is not None:
                    user_id_str = all_userid_str[idx]
                    news_id_str = all_newsid_str[idx]
                    if news_idx == 0:
                        candidate_body = news_body[0]
                    else:
                        key = _norm_expected_body_key(user_id_str, news_id_str)
                        if key in expected_bodies:
                            expected_body = expected_bodies[key]
                            _n = (
                                expected_body_clip_n_sentences
                                if expected_body_clip_n_sentences is not None
                                else _naml_common_runtime.EXPECTED_BODY_FIRST_N_SENTENCES
                            )
                            _eb = clip_expected_body_to_first_sentences(expected_body, _n)
                            body_tokens = word_tokenize(_eb.lower()) if _eb else []
                            word_id = []
                            for word in body_tokens:
                                if word in word_dict:
                                    word_id.append(word_dict[word][0])
                            word_id = word_id[:300]
                            word_id = word_id + [0] * (300 - len(word_id))
                            candidate_body = np.array(word_id, dtype="int32")
                        else:
                            candidate_body = news_body[news_idx]
                elif candidate_news_body is not None:
                    candidate_body = candidate_news_body[news_idx]
                else:
                    candidate_body = news_body[news_idx]
                candidate_body = np.expand_dims(candidate_body, axis=0)
                candidate_vertical = np.expand_dims(news_v[news_idx], axis=0)
                candidate_subvertical = np.expand_dims(news_sv[news_idx], axis=0)
                user_pos = all_test_user_pos_override if all_test_user_pos_override is not None else all_test_user_pos
                user_pos_indices = np.array(user_pos[idx], dtype="int32")
                browsed_news = news_words[user_pos_indices]
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]
                browsed_news_body_split = [
                    np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])
                ]
                browsed_news_vertical = news_v[user_pos_indices]
                browsed_news_vertical_split = [np.expand_dims(browsed_news_vertical[k], axis=0) for k in range(browsed_news_vertical.shape[0])]
                browsed_news_subvertical = news_sv[user_pos_indices]
                browsed_news_subvertical_split = [np.expand_dims(browsed_news_subvertical[k], axis=0) for k in range(browsed_news_subvertical.shape[0])]
                batch_candidates.append(candidate)
                for k in range(MAX_SENTS):
                    batch_browsed_news[k].append(browsed_news_split[k])
                batch_candidate_body.append(candidate_body)
                for k in range(MAX_SENTS):
                    batch_browsed_news_body[k].append(browsed_news_body_split[k])
                batch_candidate_vertical.append(candidate_vertical)
                for k in range(MAX_SENTS):
                    batch_browsed_news_vertical[k].append(browsed_news_vertical_split[k])
                batch_candidate_subvertical.append(candidate_subvertical)
                for k in range(MAX_SENTS):
                    batch_browsed_news_subvertical[k].append(browsed_news_subvertical_split[k])
                batch_labels.append(all_label[idx])
            batch_inputs = [np.concatenate(batch_candidates, axis=0)]
            for k in range(MAX_SENTS):
                batch_inputs.append(np.concatenate(batch_browsed_news[k], axis=0))
            batch_inputs.append(np.concatenate(batch_candidate_body, axis=0))
            for k in range(MAX_SENTS):
                batch_inputs.append(np.concatenate(batch_browsed_news_body[k], axis=0))
            batch_inputs.append(np.concatenate(batch_candidate_vertical, axis=0))
            for k in range(MAX_SENTS):
                batch_inputs.append(np.concatenate(batch_browsed_news_vertical[k], axis=0))
            batch_inputs.append(np.concatenate(batch_candidate_subvertical, axis=0))
            for k in range(MAX_SENTS):
                batch_inputs.append(np.concatenate(batch_browsed_news_subvertical[k], axis=0))
            batch_labels_array = np.array(batch_labels, dtype="float32")
            yield (batch_inputs, batch_labels_array)
