from typing import Optional

import random
import nltk 
from nltk.tokenize import word_tokenize
import datetime
import time
import itertools
import numpy as np
import pickle
from numpy.linalg import cholesky
import json
import os
import glob
import subprocess
import sys
import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
from nltk.tokenize import word_tokenize

from naml_common import (
    SEED,
    MAX_HISTORY_CLICKS,
    MAX_SENT_LENGTH,
    MAX_BODY_LENGTH,
    npratio,
    mind_data_path,
    MIND_DATASET_SUBDIR,
    MIND_NEWS_FILENAME,
    MIND_TRAIN_FILENAME,
    MIND_TEST_FILENAME,
    preprocess_user_file,
    preprocess_news_file,
    get_embedding,
)
from naml_model_builder import build_naml_models

USE_EXPECTED_BODY = False  # True: 학습·전처리에서 기대 본문 사용, False: 학습은 MIND 실제 본문
# USE_EXPECTED_BODY=True일 때 기대본문 JSON 위치 (프로젝트 루트 기준 상대 경로 권장)
# 기본: <EXPECTED_BODY_OUTPUT_DIR>/train/, <EXPECTED_BODY_OUTPUT_DIR>/test/ 아래 user_*/news_*.json
EXPECTED_BODY_OUTPUT_DIR = 'body_generation/output'
# None이면 위 규칙. 지정 시 해당 폴더를 직접 사용 (train/test를 서로 다른 루트에 둘 때)
EXPECTED_BODY_TRAIN_DIR = 'body_generation/output/MIND_2000/train_3cluster_11_13_8'
EXPECTED_BODY_TEST_DIR = 'body_generation/output/MIND_2000/test_3cluster_11_13_8'

MAIN_TRAINING_LEARNING_RATE = 0.0005  # 메인 학습 루프(및 동일 model.compile) Adam 학습률
# 매 에폭 테스트셋 "기대본문" MRR/NDCG용 JSON 루트 (user_*/news_*.json).
# 해석 순서: (1) 절대 경로이면 그대로 (2) 프로젝트 루트 상대 (3) body_generation/output/<이 값>
# 예: user_preference/expected_body/MIND_2000/test_3cluster_11_13_8
MAIN_TESTSET_EXPECTED_BODY_DIR = 'user_preference/expected_body/MIND_2000/test_3cluster_11_13_8'
MAIN_TESTSET_EXPECTED_BODY_DIR_2 = None  # 두 번째 기대본문 폴더 (None이면 사용 안 함)
# USE_EXPECTED_BODY=False일 때, 위 MAIN_TESTSET 기대본문 단어를 word_dict에 넣을지 (기대본문 지표에 권장)
INCLUDE_MAIN_TEST_EXPECTED_TOKENS_IN_WORD_DICT = True
MAIN_TRAINING_EPOCHS = 20  # 메인 학습 루프 에폭 수
# USE_EXPECTED_BODY=False이고 True일 때: 테스트셋 실제본문 MRR 최고 에폭에 가중치 저장
SAVE_MAIN_BEST_BY_TEST_ACTUAL_MRR = False
MAIN_TRAINING_BEST_MODEL_PATH = 'saved_models/NAML_mind_2000.h5'


def _naml_project_root() -> str:
    """NAML/NAML.py 기준 프로젝트(저장소) 루트."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_expected_body_dir(path_option: Optional[str]) -> Optional[str]:
    """
    기대본문 상위 폴더(user_*/news_*.json 직계 부모)를 실제 경로로 해석.
    None/빈 문자열 → None. 그 외: 절대 경로 → 프로젝트 루트 상대 → body_generation/output/상대 순으로 탐색.
    """
    if path_option is None:
        return None
    p = str(path_option).strip()
    if not p:
        return None
    if os.path.isabs(p) and os.path.isdir(p):
        return os.path.normpath(p)
    root = _naml_project_root()
    cand = os.path.normpath(os.path.join(root, p))
    if os.path.isdir(cand):
        return cand
    legacy = os.path.normpath(os.path.join(root, 'body_generation', 'output', p))
    if os.path.isdir(legacy):
        return legacy
    return None


def load_expected_bodies(output_dir=None, dataset_type='train'):
    """
    기대 본문 로드 (유저별로 다른 기대본문 지원)
    output_dir가 None이면 상단 EXPECTED_BODY_OUTPUT_DIR 사용.
    <output_dir>/<dataset_type>/user_{user_id}/news_{news_id}.json에서 기대 본문 로드
    반환: {(user_id, news_id): generated_body} 형태의 딕셔너리
    """
    if output_dir is None:
        output_dir = EXPECTED_BODY_OUTPUT_DIR
    expected_bodies = {}  # {(user_id, news_id): generated_body}
    base_path = os.path.join(output_dir, dataset_type)
    
    if not os.path.exists(base_path):
        print(f"경고: 기대 본문 폴더를 찾을 수 없습니다: {base_path}")
        return expected_bodies
    
    # 각 유저 폴더 탐색
    for user_folder in os.listdir(base_path):
        user_path = os.path.join(base_path, user_folder)
        if not os.path.isdir(user_path):
            continue
        
        # user_folder에서 user_id 추출 (예: "user_1" -> "1")
        if user_folder.startswith('user_'):
            user_id = user_folder.replace('user_', '')
        else:
            continue
        
        # 각 뉴스 JSON 파일 읽기
        for filename in os.listdir(user_path):
            if filename.startswith('news_') and filename.endswith('.json'):
                news_id = filename.replace('news_', '').replace('.json', '')
                file_path = os.path.join(user_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'generated_body' in data:
                            key = _norm_expected_body_key(user_id, news_id)
                            expected_bodies[key] = data['generated_body']
                except Exception:
                    continue
        
    print(f"기대 본문 로드 완료: {len(expected_bodies)}개 ({dataset_type})")
    return expected_bodies


def _norm_expected_body_key(uid, nid):
    """
    기대본문 딕셔너리 키 정규화. TSV의 user가 758.0으로 들어오거나 body_generation이 user_758로 저장해도 동일 키로 매칭.
    """
    try:
        u = str(int(float(uid))).strip() if uid is not None and str(uid).strip() else ''
    except (ValueError, TypeError):
        u = str(uid).strip() if uid is not None else ''
    n = str(nid).strip() if nid is not None else ''
    return (u, n)


def load_expected_bodies_from_train_dir(train_dir):
    """
    body_generation/output/trainN 구조에서 해당 폴더 전체의 기대 본문 로드.
    train_dir: trainN 폴더 전체 경로. user_{id}/news_{id}.json 에서 generated_body 수집.
    반환: {(user_id, news_id): generated_body} — 키는 _norm_expected_body_key로 정규화됨.
    """
    expected_bodies = {}
    if not train_dir or not os.path.isdir(train_dir):
        return expected_bodies
    for user_folder in os.listdir(train_dir):
        user_path = os.path.join(train_dir, user_folder)
        if not os.path.isdir(user_path):
            continue
        if not user_folder.startswith('user_'):
            continue
        user_id = user_folder.replace('user_', '')
        for filename in os.listdir(user_path):
            if not (filename.startswith('news_') and filename.endswith('.json')):
                continue
            news_id = filename.replace('news_', '').replace('.json', '')
            file_path = os.path.join(user_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'generated_body' in data:
                        key = _norm_expected_body_key(user_id, news_id)
                        expected_bodies[key] = data['generated_body']
            except Exception:
                continue
    return expected_bodies


def _expected_body_dir_for_train_or_test(dataset_type: str) -> str:
    """train 또는 test용 기대본문 폴더 경로 (user_*/news_*.json 상위 디렉터리)."""
    if dataset_type == 'train' and EXPECTED_BODY_TRAIN_DIR:
        return os.path.normpath(EXPECTED_BODY_TRAIN_DIR)
    if dataset_type == 'test' and EXPECTED_BODY_TEST_DIR:
        return os.path.normpath(EXPECTED_BODY_TEST_DIR)
    return os.path.normpath(os.path.join(EXPECTED_BODY_OUTPUT_DIR, dataset_type))


# 기대 본문 사용 여부는 상단에서 설정 (USE_EXPECTED_BODY)
# word_dict 생성에 기대본문을 포함하기 위해 먼저 기대본문을 로드
expected_bodies_train = None
expected_bodies_test = None
expected_bodies_word_dict_test_only = None

if USE_EXPECTED_BODY:
    # 기대 본문 로드 (경로는 상단 EXPECTED_BODY_*)
    print("\n기대 본문 로드 중...")
    _train_dir = _expected_body_dir_for_train_or_test('train')
    _test_dir = _expected_body_dir_for_train_or_test('test')
    print(f"  train 폴더: {_train_dir}")
    print(f"  test 폴더: {_test_dir}")
    expected_bodies_train = load_expected_bodies_from_train_dir(_train_dir)
    expected_bodies_test = load_expected_bodies_from_train_dir(_test_dir)
    print(f"로드된 기대 본문: train={len(expected_bodies_train)}개, test={len(expected_bodies_test)}개")
elif INCLUDE_MAIN_TEST_EXPECTED_TOKENS_IN_WORD_DICT and MAIN_TESTSET_EXPECTED_BODY_DIR:
    # 학습은 실제 본문이지만, 매 에폭 기대본문 테스트 지표에 쓰일 토큰을 word_dict에 포함
    _wd = resolve_expected_body_dir(MAIN_TESTSET_EXPECTED_BODY_DIR)
    if _wd:
        expected_bodies_word_dict_test_only = load_expected_bodies_from_train_dir(_wd)
        print(
            f"\nword_dict용 테스트 기대본문 토큰 로드: {len(expected_bodies_word_dict_test_only)}개 ({_wd})"
        )
    else:
        print(
            f"\n경고: MAIN_TESTSET_EXPECTED_BODY_DIR 을 찾을 수 없어 word_dict에 기대본문 토큰을 넣지 않습니다: "
            f"{MAIN_TESTSET_EXPECTED_BODY_DIR!r}"
        )

# 뉴스 데이터를 전처리 (기대본문도 word_dict 생성에 포함)
word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
    expected_bodies_train=expected_bodies_train,
    expected_bodies_test=expected_bodies_test
    if USE_EXPECTED_BODY
    else expected_bodies_word_dict_test_only,
)

# 뉴스 인덱스를 사용하여 유저 데이터 전처리
if USE_EXPECTED_BODY:
    userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index, candidate_news_ids_train, candidate_news_ids_test, all_train_userid_str, all_train_newsid_str, all_test_userid_str, all_test_newsid_str = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=expected_bodies_train,
        expected_bodies_test=expected_bodies_test,
        word_dict=word_dict
    )
    
    print(f"수집된 후보 뉴스 ID: train={len(candidate_news_ids_train)}개, test={len(candidate_news_ids_test)}개")
    
    # 실제로 사용 가능한 기대본문 개수 확인
    available_expected_bodies_train = 0
    available_expected_bodies_test = 0
    total_candidates_train = 0
    total_candidates_test = 0
    missing_expected_bodies_train = 0
    missing_expected_bodies_test = 0
    
    # 학습 데이터: 각 샘플의 (user_id, news_id) 조합 확인
    # 학습 데이터는 샘플당 5개 후보를 포함
    for i in range(len(all_train_userid_str)):
        user_id = all_train_userid_str[i]
        news_ids = all_train_newsid_str[i]  # 5개 후보 뉴스 ID 리스트
        for news_id in news_ids:
            if news_id:  # 빈 문자열이 아닌 경우만
                total_candidates_train += 1
                if (user_id, news_id) in expected_bodies_train:
                    available_expected_bodies_train += 1
                else:
                    missing_expected_bodies_train += 1
    
    # 테스트 데이터: 각 샘플의 (user_id, news_id) 조합 확인
    # 테스트 데이터는 각 샘플이 개별 후보 뉴스
    for i in range(len(all_test_userid_str)):
        user_id = all_test_userid_str[i]
        news_id = all_test_newsid_str[i]  # 단일 후보 뉴스 ID
        if news_id:  # 빈 문자열이 아닌 경우만
            total_candidates_test += 1
            if (user_id, news_id) in expected_bodies_test:
                available_expected_bodies_test += 1
            else:
                missing_expected_bodies_test += 1
    
    print(f"\n[기대본문 사용 통계]")
    print(f"  - 로드된 기대본문: train={len(expected_bodies_train)}개, test={len(expected_bodies_test)}개")
    print(f"  - 실제 사용 가능한 기대본문: train={available_expected_bodies_train}개, test={available_expected_bodies_test}개")
    print(f"  - 학습 샘플 수: {len(all_train_id)}개 (총 후보 뉴스 수: {total_candidates_train}개)")
    print(f"  - 테스트 샘플 수: {len(all_test_id)}개 (총 후보 뉴스 수: {total_candidates_test}개)")
    if total_candidates_train > 0:
        train_coverage = (available_expected_bodies_train / total_candidates_train) * 100
        print(f"  - 학습 데이터 기대본문 커버리지: {train_coverage:.2f}% ({available_expected_bodies_train}/{total_candidates_train})")
        if missing_expected_bodies_train > 0:
            print(f"  - 학습 데이터 기대본문 누락: {missing_expected_bodies_train}개 (원본 본문 사용)")
    if total_candidates_test > 0:
        test_coverage = (available_expected_bodies_test / total_candidates_test) * 100
        print(f"  - 테스트 데이터 기대본문 커버리지: {test_coverage:.2f}% ({available_expected_bodies_test}/{total_candidates_test})")
        if missing_expected_bodies_test > 0:
            print(f"  - 테스트 데이터 기대본문 누락: {missing_expected_bodies_test}개 (원본 본문 사용)")
else:
    # 학습·전처리에서 원본 본문 사용 (매 에폭 테스트 기대본문 지표는 아래 MAIN_TESTSET_* 로 별도)
    print("\n원본 본문 사용 모드 (학습 후보 본문 = MIND 실제 본문)")
    userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index, candidate_news_ids_train, candidate_news_ids_test, all_train_userid_str, all_train_newsid_str, all_test_userid_str, all_test_newsid_str = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict
    )


print(f"뉴스 개수: {len(news_index)}")
print(f"카테고리 개수: {len(category)}")
print(f"서브카테고리 개수: {len(subcategory)}")


# GloVe 파일이 없으면 랜덤 초기화 사용
embedding_mat = get_embedding(word_dict)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def hit_at_k(y_true, y_score, k=1):
    """
    Hit@K 계산: 상위 K개 예측 중 정답이 포함되는지 여부 (0 또는 1)
    """
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0
    # y_score를 1차원 배열로 변환 (필요한 경우)
    y_score = np.array(y_score).flatten()
    y_true = np.array(y_true).flatten()
    sorted_indices = np.argsort(y_score)[::-1]  # 점수 내림차순 정렬
    top_k_indices = sorted_indices[:k]
    return 1.0 if np.any(y_true[top_k_indices] == 1) else 0.0

os.environ['PYTHONHASHSEED'] = str(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.random.set_seed(SEED)

print(f"Seed 고정 완료: {SEED}")



def generate_batch_data_train(all_train_pn,all_label,all_train_id,batch_size, candidate_news_body=None, expected_bodies=None, all_userid_str=None, all_newsid_str=None, news_index_reverse=None, use_expected_body_positive_only=False):
    """
    candidate_news_body: 후보 뉴스의 기대 본문 배열 (None이면 원본 news_body 사용)
    expected_bodies: 유저별 기대 본문 딕셔너리 {(user_id, news_id): generated_body}
    all_userid_str: 유저 ID 문자열 배열
    all_newsid_str: 후보 뉴스 ID 문자열 배열 (각 샘플마다 5개)
    news_index_reverse: 뉴스 인덱스 -> 뉴스 ID 역매핑
    use_expected_body_positive_only: True이면 label=1인 정답 후보에만 기대본문을 사용하고, 나머지 negative 후보는 실제 본문 사용
    """
    
    # news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
    if news_index_reverse is None:
        news_index_reverse = {v: k for k, v in news_index.items()}
    
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] 
               for i in range(len(y)//batch_size+1)
               if batch_size*i < len(y)]

    while (True):
        for batch_indices in batches:
            batch_candidate_splits = [[] for _ in range(5)]  # 5개 후보 뉴스
            batch_browsed_news_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_candidate_body_splits = [[] for _ in range(5)]
            batch_browsed_news_body_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_candidate_vertical_splits = [[] for _ in range(5)]
            batch_browsed_news_vertical_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_candidate_subvertical_splits = [[] for _ in range(5)]
            batch_browsed_news_subvertical_splits = [[] for _ in range(MAX_HISTORY_CLICKS)]
            batch_labels = []
            
            for idx in batch_indices:
                # 리스트를 NumPy 배열로 변환하여 인덱싱
                candidate_indices = np.array(all_train_pn[idx], dtype='int32')
                candidate = news_words[candidate_indices]  # shape: (5, 30)
                # 각 후보 뉴스의 제목을 개별적으로 추출하고 배치 차원 추가 (5개의 (1, 30) 배열)
                candidate_split = [np.expand_dims(candidate[k], axis=0) for k in range(candidate.shape[0])]
                
                # 후보 뉴스는 유저별 기대 본문 사용
                if expected_bodies is not None and all_userid_str is not None and all_newsid_str is not None:
                    # 각 후보 뉴스에 대해 해당 유저의 기대본문 찾기
                    user_id_str = all_userid_str[idx]  # 리스트 인덱싱
                    news_ids_str = all_newsid_str[idx]  # 리스트 인덱싱, 5개 후보 뉴스 ID
                    candidate_body_list = []
                    label_vec = all_label[idx] if use_expected_body_positive_only else None
                    
                    for j, news_idx in enumerate(all_train_pn[idx]):
                        if news_idx == 0:  # 패딩
                            candidate_body_list.append(news_body[0])
                        else:
                            # 정답 후보만 기대본문을 사용하도록 설정된 경우, negative 후보는 실제 본문 사용
                            if use_expected_body_positive_only and label_vec is not None:
                                is_positive = (j < len(label_vec) and label_vec[j] == 1)
                                if not is_positive:
                                    candidate_body_list.append(news_body[news_idx])
                                    continue
                            news_id_str = news_ids_str[j] if j < len(news_ids_str) else ''
                            # 유저별 기대본문 찾기 (키 정규화로 758 vs 758.0 등 통일)
                            key = _norm_expected_body_key(user_id_str, news_id_str)
                            if key in expected_bodies:
                                # 기대본문 토큰화 및 인덱스 변환
                                expected_body = expected_bodies[key]
                                body_tokens = word_tokenize(expected_body.lower()) if expected_body else []
                                word_id = []
                                for word in body_tokens:
                                    if word in word_dict:
                                        word_id.append(word_dict[word][0])
                                word_id = word_id[:300]
                                word_id = word_id + [0] * (300 - len(word_id))
                                candidate_body_list.append(np.array(word_id, dtype='int32'))
                            else:
                                # 기대본문이 없으면 원본 본문 사용
                                candidate_body_list.append(news_body[news_idx])
                    
                    candidate_body = np.array(candidate_body_list)  # shape: (5, 300)
                elif candidate_news_body is not None:
                    # candidate_news_body 사용 (현재 사용되지 않음)
                    candidate_body = candidate_news_body[candidate_indices]  # shape: (5, 300)
                else:
                    # 원본 본문 사용 (USE_EXPECTED_BODY=False일 때 이 경로 사용)
                    candidate_body = news_body[candidate_indices]  # shape: (5, 300)
                
                # 각 후보 뉴스의 본문을 개별적으로 추출하고 배치 차원 추가 (5개의 (1, 300) 배열)
                candidate_body_split = [np.expand_dims(candidate_body[k], axis=0) for k in range(candidate_body.shape[0])]
                
                candidate_vertical = news_v[candidate_indices]  # shape: (5, 1)
                candidate_vertical_split = [np.expand_dims(candidate_vertical[k], axis=0) for k in range(candidate_vertical.shape[0])]
                candidate_subvertical = news_sv[candidate_indices]  # shape: (5, 1)
                candidate_subvertical_split = [np.expand_dims(candidate_subvertical[k], axis=0) for k in range(candidate_subvertical.shape[0])]
                
                user_pos_indices = np.array(all_user_pos[idx], dtype='int32')
                browsed_news = news_words[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 30)
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 300)
                browsed_news_body_split = [np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])]
                browsed_news_vertical = news_v[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 1)
                browsed_news_vertical_split = [np.expand_dims(browsed_news_vertical[k], axis=0) for k in range(browsed_news_vertical.shape[0])]
                browsed_news_subvertical = news_sv[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 1)
                browsed_news_subvertical_split = [np.expand_dims(browsed_news_subvertical[k], axis=0) for k in range(browsed_news_subvertical.shape[0])]
                
                label = all_label[idx]
                # label을 numpy array로 변환 (categorical_crossentropy는 one-hot 형식 필요)
                label = np.array(label, dtype='float32')  # shape: (5,)
                
                # 배치에 추가
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
            
            # 배치를 concatenate하여 yield
            batch_inputs = []
            # candidate splits (5개)
            for k in range(5):
                if len(batch_candidate_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_candidate_splits[k], axis=0))
            # browsed news splits
            for k in range(MAX_HISTORY_CLICKS):
                if len(batch_browsed_news_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_browsed_news_splits[k], axis=0))
            # candidate body splits (5개)
            for k in range(5):
                if len(batch_candidate_body_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_candidate_body_splits[k], axis=0))
            # browsed news body splits
            for k in range(MAX_HISTORY_CLICKS):
                if len(batch_browsed_news_body_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_browsed_news_body_splits[k], axis=0))
            # candidate vertical splits (5개)
            for k in range(5):
                if len(batch_candidate_vertical_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_candidate_vertical_splits[k], axis=0))
            # browsed news vertical splits
            for k in range(MAX_HISTORY_CLICKS):
                if len(batch_browsed_news_vertical_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_browsed_news_vertical_splits[k], axis=0))
            # candidate subvertical splits (5개)
            for k in range(5):
                if len(batch_candidate_subvertical_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_candidate_subvertical_splits[k], axis=0))
            # browsed news subvertical splits
            for k in range(MAX_HISTORY_CLICKS):
                if len(batch_browsed_news_subvertical_splits[k]) > 0:
                    batch_inputs.append(np.concatenate(batch_browsed_news_subvertical_splits[k], axis=0))
            
            batch_labels_array = np.array(batch_labels)  # shape: (batch_size, 5)
            
            yield (batch_inputs, batch_labels_array)



def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size, candidate_news_body=None, expected_bodies=None, all_userid_str=None, all_newsid_str=None, news_index_reverse=None, all_test_user_pos_override=None):
    """
    candidate_news_body: 후보 뉴스의 기대 본문 배열 (None이면 원본 news_body 사용)
    expected_bodies: 유저별 기대 본문 딕셔너리 {(user_id, news_id): generated_body}
    all_userid_str: 유저 ID 문자열 배열
    all_newsid_str: 후보 뉴스 ID 문자열 배열
    news_index_reverse: 뉴스 인덱스 -> 뉴스 ID 역매핑
    all_test_user_pos_override: None이면 전역 all_test_user_pos 사용, 지정 시 해당 배열로 유저 히스토리 사용 (트레이닝 80% 평가용)
    """
    
    # news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
    if news_index_reverse is None:
        news_index_reverse = {v: k for k, v in news_index.items()}
    
    inputid = np.arange(len(all_label))
    y = all_label
    # 빈 배치 방지: 마지막 배치가 비어있을 수 있으므로 필터링
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] 
               for i in range(len(y)//batch_size+1)
               if batch_size*i < len(y)]  # 빈 배치 제외

    MAX_SENTS = MAX_HISTORY_CLICKS
    while (True):
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
                            body_tokens = word_tokenize(expected_body.lower()) if expected_body else []
                            word_id = []
                            for word in body_tokens:
                                if word in word_dict:
                                    word_id.append(word_dict[word][0])
                            word_id = word_id[:300]
                            word_id = word_id + [0] * (300 - len(word_id))
                            candidate_body = np.array(word_id, dtype='int32')
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
                user_pos_indices = np.array(user_pos[idx], dtype='int32')
                browsed_news = news_words[user_pos_indices]
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]
                browsed_news_body_split = [np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])]
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
            batch_labels_array = np.array(batch_labels, dtype='float32')
            yield (batch_inputs, batch_labels_array)

results_actual = []
results_expected = []
results_expected_2 = []
_built = build_naml_models(
    word_dict, embedding_mat, category, subcategory, MAIN_TRAINING_LEARNING_RATE
)
model = _built['model']
model_test = _built['model_test']
newsEncoder = _built['newsEncoder']
user_rep = _built['user_rep']
browsed_news_input = _built['browsed_news_input']
browsed_body_input = _built['browsed_body_input']
browsed_v_input = _built['browsed_v_input']
browsed_sv_input = _built['browsed_sv_input']
MAX_SENTS = _built['MAX_SENTS']

# news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
news_index_reverse = {v: k for k, v in news_index.items()}

# ========== 메인 학습 루프 ==========
# 매 에폭 테스트셋 기대본문 평가용 (resolve_expected_body_dir 로 프로젝트 루트 또는 body_generation/output)
need_keys_test = set()
if all_test_userid_str is not None and all_test_newsid_str is not None:
    for i in range(len(all_test_userid_str)):
        u = all_test_userid_str[i]
        n = all_test_newsid_str[i] if i < len(all_test_newsid_str) else ''
        need_keys_test.add(_norm_expected_body_key(u, n))
    need_keys_test.discard(('', ''))
test_body_dir_main = None
expected_bodies_main_test = None
if MAIN_TESTSET_EXPECTED_BODY_DIR is not None:
    # None이면 기대본문 평가는 스킵(실제본문만 평가)
    test_body_dir_main = resolve_expected_body_dir(MAIN_TESTSET_EXPECTED_BODY_DIR)
    expected_bodies_main_test = (
        load_expected_bodies_from_train_dir(test_body_dir_main)
        if test_body_dir_main
        else None
    )
    if expected_bodies_main_test is not None:
        print(
            f"메인 학습: 매 에폭 테스트셋 기대본문 평가 시 {test_body_dir_main} ({len(expected_bodies_main_test)}개)"
        )
        if need_keys_test:
            matched_test = sum(1 for k in need_keys_test if k in expected_bodies_main_test)
            pct = 100.0 * matched_test / len(need_keys_test)
            print(
                f"테스트셋 기대본문 매칭: 필요 키 {len(need_keys_test)}개 중 기대본문 존재 {matched_test}개 ({pct:.1f}%)"
            )
        print()
    elif str(MAIN_TESTSET_EXPECTED_BODY_DIR).strip():
        print(
            f"경고: 기대본문 평가 폴더를 찾을 수 없습니다: {MAIN_TESTSET_EXPECTED_BODY_DIR!r} "
            f"(프로젝트 루트 상대 또는 body_generation/output 하위)\n"
        )
# 두 번째 기대본문 폴더 (MAIN_TESTSET_EXPECTED_BODY_DIR_2)
expected_bodies_main_test_2 = None
if MAIN_TESTSET_EXPECTED_BODY_DIR_2:
    test_body_dir_main_2 = resolve_expected_body_dir(MAIN_TESTSET_EXPECTED_BODY_DIR_2)
    if test_body_dir_main_2:
        expected_bodies_main_test_2 = load_expected_bodies_from_train_dir(test_body_dir_main_2)
        print(
            f"메인 학습: 매 에폭 테스트셋 기대본문(2) 평가 시 {test_body_dir_main_2} 사용 ({len(expected_bodies_main_test_2)}개)"
        )
        if need_keys_test:
            matched_test_2 = sum(1 for k in need_keys_test if k in expected_bodies_main_test_2)
            pct_2 = 100.0 * matched_test_2 / len(need_keys_test)
            print(f"테스트셋 기대본문(2) 매칭: 필요 키 {len(need_keys_test)}개 중 기대본문 존재 {matched_test_2}개 ({pct_2:.1f}%)")
        print()
    else:
        print(
            f"경고: 두 번째 기대본문 폴더 없음 ({MAIN_TESTSET_EXPECTED_BODY_DIR_2!r}), 기대본문(2) 평가 생략\n"
        )

best_main_test_mrr_actual = -1.0
best_main_test_epoch_actual = -1
for ep in range(MAIN_TRAINING_EPOCHS):
    np.random.seed(SEED + ep)
    random.seed(SEED + ep)
    
    if USE_EXPECTED_BODY:
        # 유저별 기대본문 사용
        traingen=generate_batch_data_train(
            all_train_pn, all_label, all_train_id, 16, 
            candidate_news_body=None,
            expected_bodies=expected_bodies_train,
            all_userid_str=all_train_userid_str,
            all_newsid_str=all_train_newsid_str,
            news_index_reverse=news_index_reverse
        )
    else:
        # 원본 본문 사용
        traingen=generate_batch_data_train(all_train_pn,all_label,all_train_id, 16, candidate_news_body=None)

    actual_train_samples = len(all_train_id)
    steps_per_epoch = (actual_train_samples + 15) // 16
    model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch)
    
    actual_test_samples = len(all_test_id)
    test_steps = (actual_test_samples + 15) // 16

    # [1] 테스트셋 실제본문으로 평가
    testgen_actual = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, 16, candidate_news_body=None)
    click_score = model_test.predict(testgen_actual, steps=test_steps, verbose=1)
    # print(f"[디버깅] 실제 생성된 click_score 수: {len(click_score)}")
    
    # click_score가 실제 샘플 수와 일치하는지 확인 (디버깅용, 필요 시 주석 해제)
    # if len(click_score) != actual_test_samples:
    #     print(f"[경고] click_score({len(click_score)})가 실제 샘플 수({actual_test_samples})와 일치하지 않습니다!")
    #     print(f"[경고] 차이: {actual_test_samples - len(click_score)}개 샘플이 누락되었습니다.")

    all_mrr=[]
    all_ndcg=[]
    all_hit1=[]
    
    # # click_score 디버깅 출력
    # print(f"\n[디버깅] click_score 형태: {click_score.shape}")
    # print(f"[디버깅] 전체 click_score 통계:")
    # print(f"  - 최소값: {np.min(click_score):.6f}")
    # print(f"  - 최대값: {np.max(click_score):.6f}")
    # print(f"  - 평균값: {np.mean(click_score):.6f}")
    # print(f"  - 표준편차: {np.std(click_score):.6f}")
    # print(f"[디버깅] 실제 샘플 수: {len(all_test_id)}")
    # print(f"[디버깅] all_test_index에 저장된 세션 수: {len(all_test_index)}")
    
    session_count = 0
    excluded_no_label = 0  # 정답이 없는 세션
    excluded_out_of_range = 0  # click_score 범위를 벗어난 세션
    total_sessions = len(all_test_index)

    # 범위를 벗어난 세션의 예시 출력
    out_of_range_examples = []
    
    for m in all_test_index:
        # 제외 이유 확인
        has_label = np.sum(all_test_label[m[0]:m[1]]) != 0
        in_range = m[1] <= len(click_score)
        
        if not has_label:
            excluded_no_label += 1
        if not in_range:
            excluded_out_of_range += 1
            if len(out_of_range_examples) < 3:
                out_of_range_examples.append((m[0], m[1], len(click_score)))
        
        if has_label and in_range:
            session_scores = click_score[m[0]:m[1],0]
            session_labels = all_test_label[m[0]:m[1]]
            
            # # 처음 5개 세션만 상세 출력
            # if session_count < 5:
            #     print(f"\n[디버깅] 세션 {session_count + 1}:")
            #     print(f"  - 인덱스 범위: [{m[0]}, {m[1]})")
            #     print(f"  - 점수: {session_scores}")
            #     print(f"  - 레이블: {session_labels}")
            #     print(f"  - 정답 위치: {np.where(session_labels == 1)[0]}")
            #     sorted_indices = np.argsort(session_scores)[::-1]
            #     print(f"  - 정렬된 인덱스 (내림차순): {sorted_indices}")
            #     print(f"  - 1위 인덱스: {sorted_indices[0]}, 점수: {session_scores[sorted_indices[0]]:.6f}")
            #     hit1_val = hit_at_k(session_labels, session_scores, k=1)
            #     print(f"  - Hit@1: {hit1_val}")
            
            all_mrr.append(mrr_score(session_labels, session_scores))
            all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
            all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
            session_count += 1
    
    # # 디버깅용 세션 통계 출력
    # print(f"\n[디버깅] 세션 통계:")
    # print(f"  - all_test_index에 저장된 총 세션 수: {total_sessions}")
    # print(f"  - 실제 샘플 수 (all_test_id): {len(all_test_id)}")
    # print(f"  - click_score 샘플 수: {len(click_score)}")
    # print(f"  - 평가된 세션 수: {session_count}")
    # print(f"  - 제외된 세션 수: {total_sessions - session_count}")
    # print(f"    * 정답 없음: {excluded_no_label}개")
    # print(f"    * click_score 범위 벗어남: {excluded_out_of_range}개")
    # if out_of_range_examples:
    #     print(f"\n[디버깅] 범위 벗어난 세션 예시:")
    #     for start, end, max_idx in out_of_range_examples:
    #         print(f"    - 인덱스 [{start}, {end}), click_score 길이: {max_idx}, 초과: {end - max_idx}")
    # print(f"  - Hit@1 값 분포: 0={sum(1 for x in all_hit1 if x == 0)}, 1={sum(1 for x in all_hit1 if x == 1)}")
    
    # 결과 저장
    epoch_results_actual = {
        'MRR': np.mean(all_mrr),
        'NDCG@5': np.mean(all_ndcg),
        'Hit@1': np.mean(all_hit1)
    }

    # [2] 테스트셋 기대본문(MAIN_TESTSET_EXPECTED_BODY_DIR)으로 평가
    epoch_results_expected = None
    if expected_bodies_main_test is not None:
        testgen_expected = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 16,
            candidate_news_body=None,
            expected_bodies=expected_bodies_main_test,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse
        )
        click_score_exp = model_test.predict(testgen_expected, steps=test_steps, verbose=0)
        all_mrr_exp, all_ndcg_exp, all_hit1_exp = [], [], []
        for i in range(len(all_test_index)):
            start, end = all_test_index[i]
            session_scores = click_score_exp[start:end].flatten()
            session_labels = all_test_label[start:end]
            if np.sum(session_labels) == 0:
                continue
            all_mrr_exp.append(mrr_score(session_labels, session_scores))
            all_ndcg_exp.append(ndcg_score(session_labels, session_scores, k=5))
            all_hit1_exp.append(hit_at_k(session_labels, session_scores, k=1))
        if all_mrr_exp:
            epoch_results_expected = {
                'MRR': np.mean(all_mrr_exp),
                'NDCG@5': np.mean(all_ndcg_exp),
                'Hit@1': np.mean(all_hit1_exp)
            }
    # [3] 테스트셋 기대본문(2)(MAIN_TESTSET_EXPECTED_BODY_DIR_2)으로 평가
    epoch_results_expected_2 = None
    if expected_bodies_main_test_2 is not None:
        testgen_expected_2 = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 16,
            candidate_news_body=None,
            expected_bodies=expected_bodies_main_test_2,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse
        )
        click_score_exp_2 = model_test.predict(testgen_expected_2, steps=test_steps, verbose=0)
        all_mrr_exp_2, all_ndcg_exp_2, all_hit1_exp_2 = [], [], []
        for i in range(len(all_test_index)):
            start, end = all_test_index[i]
            session_scores = click_score_exp_2[start:end].flatten()
            session_labels = all_test_label[start:end]
            if np.sum(session_labels) == 0:
                continue
            all_mrr_exp_2.append(mrr_score(session_labels, session_scores))
            all_ndcg_exp_2.append(ndcg_score(session_labels, session_scores, k=5))
            all_hit1_exp_2.append(hit_at_k(session_labels, session_scores, k=1))
        if all_mrr_exp_2:
            epoch_results_expected_2 = {
                'MRR': np.mean(all_mrr_exp_2),
                'NDCG@5': np.mean(all_ndcg_exp_2),
                'Hit@1': np.mean(all_hit1_exp_2)
            }

    results_actual.append([
        float(epoch_results_actual["MRR"]),
        float(epoch_results_actual["NDCG@5"]),
        float(epoch_results_actual["Hit@1"]),
    ])
    results_expected.append(
        None
        if epoch_results_expected is None
        else [
            float(epoch_results_expected["MRR"]),
            float(epoch_results_expected["NDCG@5"]),
            float(epoch_results_expected["Hit@1"]),
        ]
    )
    results_expected_2.append(
        None
        if epoch_results_expected_2 is None
        else [
            float(epoch_results_expected_2["MRR"]),
            float(epoch_results_expected_2["NDCG@5"]),
            float(epoch_results_expected_2["Hit@1"]),
        ]
    )

    # 테스트셋 실제본문 MRR 최고 에폭 가중치 저장
    if not USE_EXPECTED_BODY:
        if SAVE_MAIN_BEST_BY_TEST_ACTUAL_MRR and len(all_mrr) > 0:
            mrr_a = float(epoch_results_actual["MRR"])
            if mrr_a > best_main_test_mrr_actual:
                best_main_test_mrr_actual = mrr_a
                best_main_test_epoch_actual = ep + 1
                _best_dir = os.path.dirname(MAIN_TRAINING_BEST_MODEL_PATH)
                if _best_dir and not os.path.exists(_best_dir):
                    os.makedirs(_best_dir, exist_ok=True)
                model.save_weights(MAIN_TRAINING_BEST_MODEL_PATH)
                print(
                    f"  [저장] 테스트셋(실제본문) MRR 최고 갱신 Epoch {ep+1}: "
                    f"MRR={mrr_a:.6f} → {MAIN_TRAINING_BEST_MODEL_PATH}"
                )
    current_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate
    print(f"\n{'='*60}")
    print(f"Epoch {ep+1}/{MAIN_TRAINING_EPOCHS} - Test Results (LR: {current_lr:.6f})")
    print(f"{'='*60}")
    print(f"[실제본문] MRR: {epoch_results_actual['MRR']:.6f}  NDCG@5: {epoch_results_actual['NDCG@5']:.6f}  Hit@1: {epoch_results_actual['Hit@1']:.6f}")
    if epoch_results_expected is not None:
        print(f"[기대본문({MAIN_TESTSET_EXPECTED_BODY_DIR})] MRR: {epoch_results_expected['MRR']:.6f}  NDCG@5: {epoch_results_expected['NDCG@5']:.6f}  Hit@1: {epoch_results_expected['Hit@1']:.6f}")
    if epoch_results_expected_2 is not None:
        print(f"[기대본문(2)({MAIN_TESTSET_EXPECTED_BODY_DIR_2})] MRR: {epoch_results_expected_2['MRR']:.6f}  NDCG@5: {epoch_results_expected_2['NDCG@5']:.6f}  Hit@1: {epoch_results_expected_2['Hit@1']:.6f}")
    print(f"{'='*60}\n")

# 전체 결과 요약 (실제본문 / 기대본문 각각)
print(f"\n{'='*60}")
print("Final Results Summary — 실제본문 (테스트셋)")
print(f"{'='*60}")
print(f"{'Epoch':<10} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
print(f"{'-'*60}")
for i, result in enumerate(results_actual, 1):
    mrr, ndcg5, hit1 = result
    print(f"{i:<10} {mrr:<12.6f} {ndcg5:<12.6f} {hit1:<12.6f}")
print(f"{'='*72}")

if any(r is not None for r in results_expected):
    print(f"\n{'='*60}")
    print(f"Final Results Summary — 기대본문 ({MAIN_TESTSET_EXPECTED_BODY_DIR})")
    print(f"{'='*60}")
    print(f"{'Epoch':<10} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
    print(f"{'-'*60}")
    for i, result in enumerate(results_expected, 1):
        if result is None:
            print(f"{i:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            mrr, ndcg5, hit1 = result
            print(f"{i:<10} {mrr:<12.6f} {ndcg5:<12.6f} {hit1:<12.6f}")
    print(f"{'='*72}")

if any(r is not None for r in results_expected_2):
    print(f"\n{'='*60}")
    print(f"Final Results Summary — 기대본문(2) ({MAIN_TESTSET_EXPECTED_BODY_DIR_2})")
    print(f"{'='*60}")
    print(f"{'Epoch':<10} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
    print(f"{'-'*60}")
    for i, result in enumerate(results_expected_2, 1):
        if result is None:
            print(f"{i:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            mrr, ndcg5, hit1 = result
            print(f"{i:<10} {mrr:<12.6f} {ndcg5:<12.6f} {hit1:<12.6f}")
    print(f"{'='*72}")

# 최고 성능 (실제본문 기준)
best_mrr_idx = int(np.argmax([r[0] for r in results_actual]))
best_mrr_epoch = best_mrr_idx + 1
best_hit1_idx = int(np.argmax([r[2] for r in results_actual]))
best_hit1_epoch = best_hit1_idx + 1
print(f"\n[실제본문] Best MRR  : Epoch {best_mrr_epoch} - {results_actual[best_mrr_idx][0]:.6f}")
print(f"[실제본문] Best Hit@1: Epoch {best_hit1_epoch} - {results_actual[best_hit1_idx][2]:.6f}")

if any(r is not None for r in results_expected):
    _exp_valid = [(i, r) for i, r in enumerate(results_expected) if r is not None]
    if _exp_valid:
        _best_i, _best_r = max(_exp_valid, key=lambda t: t[1][0])
        print(f"[기대본문] Best MRR  : Epoch {_best_i + 1} - {_best_r[0]:.6f}")

if any(r is not None for r in results_expected_2):
    _exp2_valid = [(i, r) for i, r in enumerate(results_expected_2) if r is not None]
    if _exp2_valid:
        _best_i2, _best_r2 = max(_exp2_valid, key=lambda t: t[1][0])
        print(f"[기대본문(2)] Best MRR  : Epoch {_best_i2 + 1} - {_best_r2[0]:.6f}")

if (not USE_EXPECTED_BODY) and best_main_test_epoch_actual > 0 and SAVE_MAIN_BEST_BY_TEST_ACTUAL_MRR:
    print(
        f"저장된 최고(테스트 실제본문 MRR): Epoch {best_main_test_epoch_actual} - "
        f"{best_main_test_mrr_actual:.6f} → {MAIN_TRAINING_BEST_MODEL_PATH}"
    )
print(f"{'='*60}\n")
