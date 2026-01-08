#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import random
import nltk 
from nltk.tokenize import word_tokenize
import datetime
import time
import random
import itertools
import numpy as np
import pickle
from numpy.linalg import cholesky
# from keras.utils.np_utils import *  # 최신 Keras에서는 제거됨, 사용하지 않으므로 주석 처리

# 재현성을 위한 seed 고정 (전역 설정)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 모델 하이퍼파라미터 설정
MAX_HISTORY_CLICKS = 10  # 클릭 히스토리 개수 (한 곳에서 설정)
MAX_SENT_LENGTH = 30     # 제목 최대 단어 수
MAX_BODY_LENGTH = 300    # 본문 최대 단어 수
npratio = 4              # negative sampling 비율
USE_EXPECTED_BODY = True  # True: 기대 본문 사용, False: 원본 본문 사용

# In[ ]:


def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)


# In[ ]:


def load_expected_bodies(output_dir='body_generation/output', dataset_type='train'):
    """
    기대 본문 로드 (유저별로 다른 기대본문 지원)
    output_dir/{dataset_type}/user_{user_id}/news_{news_id}.json에서 기대 본문 로드
    반환: {(user_id, news_id): generated_body} 형태의 딕셔너리
    """
    import json
    import os
    
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
                            # (user_id, news_id) 튜플을 키로 사용
                            expected_bodies[(user_id, news_id)] = data['generated_body']
                except Exception as e:
                    # 조용히 넘어감 (파일이 없거나 형식이 다를 수 있음)
                    continue
    
    print(f"기대 본문 로드 완료: {len(expected_bodies)}개 ({dataset_type})")
    return expected_bodies


def preprocess_user_file(train_file='dataset/MIND/MIND_train_(1000).tsv', 
                         test_file='dataset/MIND/MIND_test_(1000).tsv',
                         news_index=None, npratio=4,
                         expected_bodies_train=None, expected_bodies_test=None,
                         word_dict=None):
    """
    MIND 데이터셋 형식에 맞게 전처리
    train_file: user, clicked_news, candidate_news, clicked
    test_file: user, clicked_news, candidate_news (clicked 없음)
    
    expected_bodies_train/test: 후보 뉴스의 기대 본문 딕셔너리 {news_id: generated_body}
    word_dict: 단어 사전 (기대 본문 토큰화에 필요)
    """
    userid_dict = {}
    
    # 학습 데이터 로드
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = f.readlines()[1:]  # 헤더 제거
    
    # 테스트 데이터 로드
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    
    # 유저 ID 딕셔너리 생성
    for line in train_data:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            userid = parts[0]
            if userid not in userid_dict:
                userid_dict[userid] = len(userid_dict)
    
    for line in test_data:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            userid = parts[0]
            if userid not in userid_dict:
                userid_dict[userid] = len(userid_dict)
    
    all_train_id = []
    all_train_pn = []
    all_label = []
    all_train_userid_str = []  # 유저 ID 문자열 저장 (기대본문 매칭용)
    all_train_newsid_str = []  # 후보 뉴스 ID 문자열 저장 (기대본문 매칭용)
    
    all_test_id = []
    all_test_pn = []
    all_test_label = []
    all_test_index = []
    all_test_userid_str = []  # 유저 ID 문자열 저장 (기대본문 매칭용)
    all_test_newsid_str = []  # 후보 뉴스 ID 문자열 저장 (기대본문 매칭용)
    
    all_user_pos = []
    all_test_user_pos = []
    
    # 후보 뉴스 ID 수집 (기대 본문 로드용)
    candidate_news_ids_train = set()
    candidate_news_ids_test = set()
    
    # 학습 데이터 처리
    skip_stats = {
        'invalid_format': 0,  # 컬럼이 4개 미만
        'no_clicked_history': 0,  # 클릭 히스토리가 없음
        'insufficient_candidates': 0,  # 후보가 2개 미만
        'no_positive': 0,  # positive가 없음
    }
    
    for line in train_data:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            skip_stats['invalid_format'] += 1
            continue
        
        userid = parts[0]
        clicked_news = parts[1].split()  # 클릭 히스토리
        candidate_news = parts[2].split()  # 후보 뉴스들
        clicked = parts[3].split()  # 클릭 여부 (1 또는 0)
        
        # clicked_news를 news_index로 변환 (순서 유지)
        clicked_news_ids = []
        for news_id in clicked_news:
            if news_id in news_index:
                clicked_news_ids.append(news_index[news_id])
        
        if len(clicked_news_ids) == 0:
            skip_stats['no_clicked_history'] += 1
            continue
        
        # 후보 뉴스들을 news_index로 변환
        candidate_indices = []
        candidate_labels = []
        candidate_news_filtered = []  # 필터링된 candidate_news (인덱스 대응 보장)
        for i, cand_id in enumerate(candidate_news):
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
                candidate_news_filtered.append(cand_id)  # 필터링된 뉴스 ID 저장
                candidate_news_ids_train.add(cand_id)  # 후보 뉴스 ID 수집
                is_clicked = int(clicked[i]) if i < len(clicked) else 0
                candidate_labels.append(is_clicked)
        
        # 후보가 2개 미만이거나 positive가 없으면 스킵
        if len(candidate_indices) < 2:
            skip_stats['insufficient_candidates'] += 1
            continue
        if sum(candidate_labels) == 0:
            skip_stats['no_positive'] += 1
            continue
        
        # 정확히 5개 후보로 맞추기 (npratio=4이므로 1+4=5)
        # 5개보다 많으면 처음 5개만 사용, 적으면 패딩
        target_size = 1 + npratio  # 5개
        
        if len(candidate_indices) > target_size:
            # 처음 target_size개만 사용
            candidate_indices = candidate_indices[:target_size]
            candidate_labels = candidate_labels[:target_size]
            candidate_news_filtered = candidate_news_filtered[:target_size]
        elif len(candidate_indices) < target_size:
            # 부족한 만큼 패딩 (0으로 채움, label도 0)
            padding_size = target_size - len(candidate_indices)
            candidate_indices += [0] * padding_size
            candidate_labels += [0] * padding_size
            candidate_news_filtered += [''] * padding_size  # 패딩에 대응하는 빈 문자열
        
        # 5개 후보 중 1개 positive, 나머지 negative
        # 순서를 섞기
        combined = list(zip(candidate_indices, candidate_labels, candidate_news_filtered))
        random.shuffle(combined)
        shuffle_indices, shuffle_labels, shuffle_news_ids = zip(*combined)
        
        # 유저 히스토리 (최근 MAX_HISTORY_CLICKS개 사용)
        # 후보 뉴스를 제외한 최근 클릭 기록 사용
        candidate_set = set([idx for idx in shuffle_indices if idx != 0])
        filtered_history = [idx for idx in clicked_news_ids if idx not in candidate_set]
        # 최근 MAX_HISTORY_CLICKS개 선택 (순서 유지)
        recent_history = filtered_history[-MAX_HISTORY_CLICKS:] if len(filtered_history) >= MAX_HISTORY_CLICKS else filtered_history
        allpos = [int(p) for p in recent_history]
        allpos += [0] * (MAX_HISTORY_CLICKS - len(allpos))
        
        all_train_pn.append(list(shuffle_indices))
        all_label.append(list(shuffle_labels))
        all_train_id.append(userid_dict[userid])
        all_train_userid_str.append(userid)  # 유저 ID 문자열 저장
        
        # 후보 뉴스 ID 문자열 저장 (shuffle된 순서에 맞춰)
        # candidate_indices와 candidate_news의 매핑은 이미 shuffle_news_ids에 포함됨
        all_train_newsid_str.append(list(shuffle_news_ids))
        all_user_pos.append(allpos)
    
    # 스킵 통계 출력 (디버깅용, 필요 시 주석 해제)
    # total_train_lines = len(train_data)
    # total_skipped = sum(skip_stats.values())
    # total_processed = total_train_lines - total_skipped
    # print(f"\n[학습 데이터 전처리 통계]")
    # print(f"  - 총 라인 수: {total_train_lines}")
    # print(f"  - 처리된 라인 수: {total_processed}")
    # print(f"  - 제외된 라인 수: {total_skipped}")
    # print(f"    * 컬럼 부족 (4개 미만): {skip_stats['invalid_format']}개")
    # print(f"    * 클릭 히스토리 없음: {skip_stats['no_clicked_history']}개")
    # print(f"    * 후보 부족 (2개 미만): {skip_stats['insufficient_candidates']}개")
    # print(f"    * positive 없음: {skip_stats['no_positive']}개")
    
    # 테스트 데이터 처리
    for line in test_data:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        
        userid = parts[0]
        clicked_news = parts[1].split()
        candidate_news = parts[2].split()
        
        # clicked_news를 news_index로 변환 (순서 유지)
        clicked_news_ids = []
        for news_id in clicked_news:
            if news_id in news_index:
                clicked_news_ids.append(news_index[news_id])
        
        if len(clicked_news_ids) == 0 or len(candidate_news) == 0:
            continue
        
        # 세션 인덱스 시작
        sess_index = [len(all_test_pn)]
        
        # 후보 뉴스들을 news_index로 변환 (히스토리 필터링 전에 먼저 처리)
        candidate_indices = []
        candidate_news_filtered = []  # 필터링된 candidate_news (인덱스 대응 보장)
        for cand_id in candidate_news:
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
                candidate_news_filtered.append(cand_id)  # 필터링된 뉴스 ID 저장
                candidate_news_ids_test.add(cand_id)  # 후보 뉴스 ID 수집
        
        if len(candidate_indices) < 2:
            continue
        
        # 유저 히스토리 (최근 MAX_HISTORY_CLICKS개 사용)
        # 후보 뉴스를 제외한 최근 클릭 기록 사용 (데이터 누수 방지)
        candidate_set = set([idx for idx in candidate_indices if idx != 0])
        filtered_history = [idx for idx in clicked_news_ids if idx not in candidate_set]
        # 최근 MAX_HISTORY_CLICKS개 선택 (순서 유지)
        recent_history = filtered_history[-MAX_HISTORY_CLICKS:] if len(filtered_history) >= MAX_HISTORY_CLICKS else filtered_history
        allpos = [int(p) for p in recent_history]
        allpos += [0] * (MAX_HISTORY_CLICKS - len(allpos))
        
        # 5개 후보 중 첫 번째가 positive, 나머지가 negative
        # 테스트에서도 순서를 섞어야 모델이 순서 패턴을 학습하지 않음
        candidate_labels = [1 if i == 0 else 0 for i in range(len(candidate_indices))]
        combined = list(zip(candidate_indices, candidate_labels, candidate_news_filtered))
        random.shuffle(combined)
        shuffle_indices, shuffle_labels, shuffle_news_ids = zip(*combined)
        shuffle_news_ids = list(shuffle_news_ids)  # 튜플을 리스트로 변환
        
        for cand_idx, label, news_id_str in zip(shuffle_indices, shuffle_labels, shuffle_news_ids):
            all_test_pn.append(int(cand_idx))
            all_test_label.append(label)
            all_test_id.append(userid_dict[userid])
            all_test_userid_str.append(userid)  # 유저 ID 문자열 저장
            all_test_newsid_str.append(news_id_str)  # 뉴스 ID 문자열 저장
            all_test_user_pos.append(allpos)
        
        sess_index.append(len(all_test_pn))
        all_test_index.append(sess_index)
    
    all_train_pn = np.array(all_train_pn, dtype='int32')
    all_label = np.array(all_label, dtype='int32')
    all_train_id = np.array(all_train_id, dtype='int32')
    all_test_pn = np.array(all_test_pn, dtype='int32')
    all_test_label = np.array(all_test_label, dtype='int32')
    all_test_id = np.array(all_test_id, dtype='int32')
    all_user_pos = np.array(all_user_pos, dtype='int32')
    all_test_user_pos = np.array(all_test_user_pos, dtype='int32')
    
    # 후보 뉴스 ID 반환 (기대 본문 처리용)
    return userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index, candidate_news_ids_train, candidate_news_ids_test, all_train_userid_str, all_train_newsid_str, all_test_userid_str, all_test_newsid_str


# In[ ]:


def preprocess_news_file(file='dataset/MIND/MIND_news.tsv', expected_bodies_train=None, expected_bodies_test=None):
    """
    MIND 뉴스 데이터 전처리
    형식: news_id, category, subcategory, title, body
    expected_bodies_train, expected_bodies_test: 기대본문 딕셔너리 (word_dict 생성에 포함)
    """
    with open(file, 'r', encoding='utf-8') as f:
        newsdata = f.readlines()
    
    news = {}
    category = {'None': 0}
    subcategory = {'None': 0}
    
    for newsline in newsdata:
        line = newsline.strip().split('\t')
        if len(line) < 5:
            continue
        news_id = line[0]
        cat = line[1] if line[1] else 'None'
        subcat = line[2] if line[2] else 'None'
        title = line[3] if len(line) > 3 else ''
        body = line[4] if len(line) > 4 else ''
        
        # 토큰화
        title_tokens = word_tokenize(title.lower()) if title else []
        body_tokens = word_tokenize(body.lower()) if body else []
        
        news[news_id] = [cat, subcat, title_tokens, body_tokens]
        
        if cat not in category:
            category[cat] = len(category)
        if subcat not in subcategory:
            subcategory[subcat] = len(subcategory)
    
    # 단어 사전 생성
    word_dict_raw = {'PADDING': [0, 999999]}
    
    # 원본 본문으로 word_dict_raw 생성
    for docid in news:
        for word in news[docid][2]:  # title
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                word_dict_raw[word] = [len(word_dict_raw), 1]
        for word in news[docid][3]:  # body
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                word_dict_raw[word] = [len(word_dict_raw), 1]
    
    # 기대본문도 word_dict_raw에 추가
    if expected_bodies_train is not None or expected_bodies_test is not None:
        expected_body_count = 0
        for expected_bodies in [expected_bodies_train, expected_bodies_test]:
            if expected_bodies is None:
                continue
            for (user_id, news_id), generated_body in expected_bodies.items():
                if generated_body:
                    body_tokens = word_tokenize(generated_body.lower())
                    for word in body_tokens:
                        if word in word_dict_raw:
                            word_dict_raw[word][1] += 1
                        else:
                            word_dict_raw[word] = [len(word_dict_raw), 1]
                    expected_body_count += 1
        if expected_body_count > 0:
            print(f"기대본문 {expected_body_count}개를 word_dict 생성에 포함했습니다.")
    
    # 최소 빈도 3 이상만 사용
    word_dict = {}
    for i in word_dict_raw:
        if word_dict_raw[i][1] >= 3:
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
    
    print(f"단어 사전 크기: {len(word_dict)} (전체: {len(word_dict_raw)})")
    
    # 뉴스 제목 인덱싱 (최대 30단어)
    news_words = [[0] * 30]
    news_index = {'0': 0}
    
    for newsid in news:
        word_id = []
        news_index[newsid] = len(news_index)
        for word in news[newsid][2]:  # title
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id = word_id[:30]
        news_words.append(word_id + [0] * (30 - len(word_id)))
    
    news_words = np.array(news_words, dtype='int32')
    
    # 뉴스 본문 인덱싱 (최대 300단어)
    news_body = [[0] * 300]
    for newsid in news:
        word_id = []
        for word in news[newsid][3]:  # body
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id = word_id[:300]
        news_body.append(word_id + [0] * (300 - len(word_id)))
    
    news_body = np.array(news_body, dtype='int32')
    
    # 카테고리 인덱싱
    news_v = [[0]]
    news_sv = [[0]]
    for newsid in news:
        news_v.append([category[news[newsid][0]]])
    for newsid in news:
        news_sv.append([subcategory[news[newsid][1]]])
    
    news_v = np.array(news_v, dtype='int32')
    news_sv = np.array(news_sv, dtype='int32')
    
    return word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index


def create_candidate_news_body(news_index, news_body, candidate_news_ids, expected_bodies, word_dict):
    """
    후보 뉴스의 본문을 기대 본문으로 대체한 배열 생성
    news_body: 원본 본문 배열
    candidate_news_ids: 후보 뉴스 ID 집합
    expected_bodies: 기대 본문 딕셔너리 {news_id: generated_body}
    word_dict: 단어 사전
    """
    # news_body를 깊은 복사 (NumPy 배열이므로 .copy()로 충분하지만 명시적으로)
    import numpy as np
    candidate_news_body = np.array(news_body, copy=True)
    
    # 후보 뉴스의 본문을 기대 본문으로 대체
    replaced_count = 0
    not_in_index = 0
    not_in_expected = 0
    empty_body = 0
    
    for news_id in candidate_news_ids:
        if news_id not in news_index:
            not_in_index += 1
            continue
        if news_id not in expected_bodies:
            not_in_expected += 1
            continue
            
        news_idx = news_index[news_id]
        if news_idx >= len(candidate_news_body):
            continue
            
        # 기대 본문 토큰화
        expected_body = expected_bodies[news_id]
        if not expected_body or len(expected_body.strip()) == 0:
            empty_body += 1
            continue
            
        body_tokens = word_tokenize(expected_body.lower()) if expected_body else []
        
        # 단어 인덱스로 변환
        word_id = []
        for word in body_tokens:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id = word_id[:300]
        word_id = word_id + [0] * (300 - len(word_id))
        
        # 본문 대체 (NumPy 배열이므로 직접 할당)
        candidate_news_body[news_idx] = np.array(word_id, dtype='int32')
        replaced_count += 1
    
    print(f"후보 뉴스 본문 대체 완료: {replaced_count}/{len(candidate_news_ids)}개")
    if not_in_index > 0:
        print(f"  - news_index에 없음: {not_in_index}개")
    if not_in_expected > 0:
        print(f"  - 기대 본문에 없음: {not_in_expected}개")
    if empty_body > 0:
        print(f"  - 빈 본문: {empty_body}개")
    
    return candidate_news_body


# In[ ]:


def get_embedding(word_dict, glove_path='glove.840B.300d.txt'):
    """
    GloVe 임베딩 로드
    glove_path가 없으면 랜덤 초기화 사용
    """
    embedding_dict = {}
    cnt = 0
    
    try:
        with open(glove_path, 'rb') as f:
            linenb = 0
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                if len(line) < 301:
                    continue
                word = line[0].decode('utf-8', errors='ignore')
                linenb += 1
                if len(word) != 0:
                    vec = [float(x) for x in line[1:]]
                    if word in word_dict:
                        embedding_dict[word] = vec
                        if cnt % 1000 == 0:
                            print(cnt, linenb, word)
                        cnt += 1
    except FileNotFoundError:
        print(f"GloVe 파일을 찾을 수 없습니다: {glove_path}")
        print("랜덤 초기화를 사용합니다.")
    
    embedding_matrix = [0] * len(word_dict)
    cand = []
    
    for i in embedding_dict:
        embedding_matrix[word_dict[i][0]] = np.array(embedding_dict[i], dtype='float32')
        cand.append(embedding_matrix[word_dict[i][0]])
    
    if len(cand) > 0:
        cand = np.array(cand, dtype='float32')
        mu = np.mean(cand, axis=0)
        Sigma = np.cov(cand.T)
        # 안정성을 위해 대각 행렬 추가
        if Sigma.shape[0] == 300:
            norm = np.random.multivariate_normal(mu, Sigma + np.eye(300) * 0.01, 1)
        else:
            norm = np.random.normal(mu, 0.1, (1, 300))
    else:
        # 임베딩이 없으면 평균 0, 표준편차 0.1로 초기화
        norm = np.random.normal(0, 0.1, (1, 300))
    
    for i in range(len(embedding_matrix)):
        if type(embedding_matrix[i]) == int:
            embedding_matrix[i] = np.reshape(norm, 300)
    
    embedding_matrix[0] = np.zeros(300, dtype='float32')
    embedding_matrix = np.array(embedding_matrix, dtype='float32')
    print(f"임베딩 행렬 shape: {embedding_matrix.shape}")
    return embedding_matrix


# In[ ]:


# 기대 본문 사용 여부는 상단에서 설정 (USE_EXPECTED_BODY)
# word_dict 생성에 기대본문을 포함하기 위해 먼저 기대본문을 로드
expected_bodies_train = None
expected_bodies_test = None

if USE_EXPECTED_BODY:
    # 기대 본문 로드
    print("\n기대 본문 로드 중...")
    expected_bodies_train = load_expected_bodies(output_dir='body_generation/output', dataset_type='train')
    expected_bodies_test = load_expected_bodies(output_dir='body_generation/output', dataset_type='test')
    
    print(f"로드된 기대 본문: train={len(expected_bodies_train)}개, test={len(expected_bodies_test)}개")

# 뉴스 데이터를 전처리 (기대본문도 word_dict 생성에 포함)
word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(
    expected_bodies_train=expected_bodies_train,
    expected_bodies_test=expected_bodies_test
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
    
    # 유저별 기대본문을 사용하므로 create_candidate_news_body는 사용하지 않음
    # 배치 생성 시 유저 ID와 뉴스 ID를 사용하여 해당 유저의 기대본문을 찾아서 사용
    candidate_news_body_train = None  # 배치 생성 시 동적으로 처리
    candidate_news_body_test = None  # 배치 생성 시 동적으로 처리
else:
    # 원본 본문 사용
    print("\n원본 본문 사용 모드")
    userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index, candidate_news_ids_train, candidate_news_ids_test, all_train_userid_str, all_train_newsid_str, all_test_userid_str, all_test_newsid_str = preprocess_user_file(
        news_index=news_index,
        expected_bodies_train=None,
        expected_bodies_test=None,
        word_dict=word_dict
    )
    # 원본 본문 사용 (None 전달 시 자동으로 원본 사용)
    candidate_news_body_train = None
    candidate_news_body_test = None


# In[ ]:


# 이미 위에서 처리했으므로 주석 처리
# word_dict,category,subcategory,news_words,news_body,news_v,news_sv,news_index=preprocess_news_file()
print(f"뉴스 개수: {len(news_index)}")
print(f"카테고리 개수: {len(category)}")
print(f"서브카테고리 개수: {len(subcategory)}")


# In[ ]:


# GloVe 파일 경로를 지정하거나 없으면 랜덤 초기화 사용
# NAML 폴더 안에 GloVe 파일이 있으면 아래 주석 해제
# embedding_mat = get_embedding(word_dict, glove_path='NAML/glove.840B.300d.txt')
embedding_mat = get_embedding(word_dict)  # 랜덤 초기화 사용


# In[ ]:


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


# In[ ]:


import os

# 재현성을 위한 seed 고정
SEED = 42

# Python random seed
random.seed(SEED)

# NumPy random seed
np.random.seed(SEED)

# Python hash seed (선택적, 딕셔너리 순서 고정)
os.environ['PYTHONHASHSEED'] = str(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
import tensorflow as tf

# TensorFlow/Keras random seed
tf.random.set_seed(SEED)

from keras.layers import *
from keras.models import Model
from keras import backend as K
# TensorFlow 2.8.0에서는 tensorflow.keras.optimizers 사용
from tensorflow.keras.optimizers import Adam

print(f"Seed 고정 완료: {SEED}")


# In[ ]:


def generate_batch_data_train(all_train_pn,all_label,all_train_id,batch_size, candidate_news_body=None, expected_bodies=None, all_userid_str=None, all_newsid_str=None, news_index_reverse=None):
    """
    candidate_news_body: 후보 뉴스의 기대 본문 배열 (None이면 원본 news_body 사용)
    expected_bodies: 유저별 기대 본문 딕셔너리 {(user_id, news_id): generated_body}
    all_userid_str: 유저 ID 문자열 배열
    all_newsid_str: 후보 뉴스 ID 문자열 배열 (각 샘플마다 5개)
    news_index_reverse: 뉴스 인덱스 -> 뉴스 ID 역매핑
    """
    import numpy as np
    from nltk.tokenize import word_tokenize
    
    # news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
    if news_index_reverse is None:
        news_index_reverse = {v: k for k, v in news_index.items()}
    
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for batch_indices in batches:
            # 배치 내 모든 샘플을 모아서 한 번에 yield
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
                    
                    for j, news_idx in enumerate(all_train_pn[idx]):
                        if news_idx == 0:  # 패딩
                            candidate_body_list.append(news_body[0])
                        else:
                            news_id_str = news_ids_str[j] if j < len(news_ids_str) else ''
                            # 유저별 기대본문 찾기
                            key = (user_id_str, news_id_str)
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


# In[ ]:


def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size, candidate_news_body=None, expected_bodies=None, all_userid_str=None, all_newsid_str=None, news_index_reverse=None):
    """
    candidate_news_body: 후보 뉴스의 기대 본문 배열 (None이면 원본 news_body 사용)
    expected_bodies: 유저별 기대 본문 딕셔너리 {(user_id, news_id): generated_body}
    all_userid_str: 유저 ID 문자열 배열
    all_newsid_str: 후보 뉴스 ID 문자열 배열
    news_index_reverse: 뉴스 인덱스 -> 뉴스 ID 역매핑
    """
    import numpy as np
    from nltk.tokenize import word_tokenize
    
    # news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
    if news_index_reverse is None:
        news_index_reverse = {v: k for k, v in news_index.items()}
    
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for batch_indices in batches:
            # batch_indices는 배치 내 샘플 인덱스 배열
            # 각 샘플에 대해 개별적으로 yield (원래 구조 유지)
            for idx in batch_indices:
                # all_test_pn[idx]는 단일 정수 (각 후보 뉴스가 개별 샘플)
                news_idx = int(all_test_pn[idx])
                candidate = news_words[news_idx]  # shape: (30,)
                candidate = np.expand_dims(candidate, axis=0)  # shape: (1, 30)
                
                # 후보 뉴스는 유저별 기대 본문 사용
                if expected_bodies is not None and all_userid_str is not None and all_newsid_str is not None:
                    # 해당 유저의 기대본문 찾기
                    user_id_str = all_userid_str[idx]  # 리스트 인덱싱
                    news_id_str = all_newsid_str[idx]  # 리스트 인덱싱, 단일 후보 뉴스 ID
                    
                    if news_idx == 0:  # 패딩
                        candidate_body = news_body[0]
                    else:
                        # 유저별 기대본문 찾기
                        key = (user_id_str, news_id_str)
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
                            candidate_body = np.array(word_id, dtype='int32')
                        else:
                            # 기대본문이 없으면 원본 본문 사용
                            candidate_body = news_body[news_idx]
                elif candidate_news_body is not None:
                    # candidate_news_body 사용 (현재 사용되지 않음)
                    candidate_body = candidate_news_body[news_idx]
                else:
                    # 원본 본문 사용 (USE_EXPECTED_BODY=False일 때 이 경로 사용)
                    candidate_body = news_body[news_idx]
                
                candidate_body = np.expand_dims(candidate_body, axis=0)  # shape: (1, 300)
                candidate_vertical = np.expand_dims(news_v[news_idx], axis=0)  # shape: (1, 1)
                candidate_subvertical = np.expand_dims(news_sv[news_idx], axis=0)  # shape: (1, 1)

                user_pos_indices = np.array(all_test_user_pos[idx], dtype='int32')
                browsed_news = news_words[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 30)
                browsed_news_split = [np.expand_dims(browsed_news[k], axis=0) for k in range(browsed_news.shape[0])]
                browsed_news_body = news_body[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 300)
                browsed_news_body_split = [np.expand_dims(browsed_news_body[k], axis=0) for k in range(browsed_news_body.shape[0])]
                browsed_news_vertical = news_v[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 1)
                browsed_news_vertical_split = [np.expand_dims(browsed_news_vertical[k], axis=0) for k in range(browsed_news_vertical.shape[0])]
                browsed_news_subvertical = news_sv[user_pos_indices]  # shape: (MAX_HISTORY_CLICKS, 1)
                browsed_news_subvertical_split = [np.expand_dims(browsed_news_subvertical[k], axis=0) for k in range(browsed_news_subvertical.shape[0])]
                
                label = all_label[idx]
                yield ([candidate] + browsed_news_split + [candidate_body] + browsed_news_body_split + [candidate_vertical]
                       + browsed_news_vertical_split + [candidate_subvertical] + browsed_news_subvertical_split, [label])


# In[ ]:


import itertools
import keras
import random
results=[]
keras.backend.clear_session()

# 모델 파라미터 (상단에서 정의한 전역 변수 사용)
MAX_SENTS = MAX_HISTORY_CLICKS  # 히스토리 클릭 개수
title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

body_input = Input(shape=(MAX_BODY_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat],trainable=True)

embedded_sequences_title = embedding_layer(title_input)
embedded_sequences_title=Dropout(0.3)(embedded_sequences_title)  # 0.2 -> 0.3으로 증가

embedded_sequences_body = embedding_layer(body_input)
embedded_sequences_body=Dropout(0.3)(embedded_sequences_body)  # 0.2 -> 0.3으로 증가

title_cnn = Conv1D(filters=400, kernel_size=3, padding='same', activation='relu', strides=1)(embedded_sequences_title)
title_cnn=Dropout(0.3)(title_cnn)  # 0.2 -> 0.3으로 증가

attention = Dense(200,activation='tanh')(title_cnn)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
title_rep=keras.layers.Dot((1, 1))([title_cnn, attention_weight])

body_cnn = Conv1D(filters=400, kernel_size=3, padding='same', activation='relu', strides=1)(embedded_sequences_body)
body_cnn=Dropout(0.3)(body_cnn)  # 0.2 -> 0.3으로 증가

attention_body = Dense(200,activation='tanh')(body_cnn)
attention_body = Flatten()(Dense(1)(attention_body))
attention_weight_body = Activation('softmax')(attention_body)
body_rep=keras.layers.Dot((1, 1))([body_cnn, attention_weight_body])

vinput=Input((1,), dtype='int32') 
svinput=Input((1,), dtype='int32') 
v_embedding_layer = Embedding(len(category)+1, 50,trainable=True)
sv_embedding_layer = Embedding(len(subcategory)+1, 50,trainable=True)
v_embedding=Dense(400,activation='relu')(Flatten()(v_embedding_layer(vinput)))
sv_embedding=Dense(400,activation='relu')(Flatten()(sv_embedding_layer(svinput)))

all_channel=[title_rep,body_rep,v_embedding,sv_embedding]
    
# Lambda 대신 Reshape 사용 (최신 Keras 호환)
views=concatenate([Reshape((1, -1))(channel) for channel in all_channel],axis=1)

attentionv = Dense(200,activation='tanh')(views)

attention_weightv = Reshape((-1,))(Dense(1)(attentionv))
attention_weightv =Activation('softmax')(attention_weightv)

newsrep=keras.layers.Dot((1, 1))([views, attention_weightv])

newsEncoder = Model([title_input,body_input,vinput,svinput],newsrep)

browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_body_input = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]

browsed_v_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_sv_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]

browsednews = [newsEncoder([browsed_news_input[_],browsed_body_input[_],browsed_v_input[_],browsed_sv_input[_] ]) for _ in range(MAX_SENTS)]
browsednewsrep =concatenate([Reshape((1, -1))(news) for news in browsednews],axis=1)    

attentionn = Dense(200,activation='tanh')(browsednewsrep)
attentionn =Flatten()(Dense(1)(attentionn))
attention_weightn = Activation('softmax')(attentionn)
user_rep=keras.layers.Dot((1, 1))([browsednewsrep, attention_weightn])

candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1+npratio)]

candidates_body = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(1+npratio)]

candidates_v = [keras.Input((1,), dtype='int32') for _ in range(1+npratio)]

candidates_sv = [keras.Input((1,), dtype='int32') for _ in range(1+npratio)]
candidate_vecs = [newsEncoder([candidates_title[_],candidates_body[_],candidates_v[_],candidates_sv[_]]) for _ in range(1+npratio)]

logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))


model = Model(candidates_title+browsed_news_input+candidates_body+browsed_body_input+
              candidates_v+browsed_v_input+candidates_sv+browsed_sv_input, logits)


candidate_one_title = keras.Input((MAX_SENT_LENGTH,))

candidate_one_body = keras.Input((MAX_BODY_LENGTH,))

candidate_one_v = keras.Input((1,))

candidate_one_sv = keras.Input((1,))

candidate_one_vec=newsEncoder([candidate_one_title,candidate_one_body,candidate_one_v,candidate_one_sv])

score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
model_test = keras.Model([candidate_one_title]+browsed_news_input+[candidate_one_body] +browsed_body_input
                         +[candidate_one_v]+browsed_v_input+[candidate_one_sv]+browsed_sv_input, score)


# Learning rate를 약간 낮춰서 더 안정적인 학습
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['acc'])

# Best AUC 추적 (최종 요약용)
best_auc = 0.0

# news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
news_index_reverse = {v: k for k, v in news_index.items()}

for ep in range(10):
    # 매 에폭마다 다른 순서로 셔플링하기 위해 seed에 에폭 번호 추가
    np.random.seed(SEED + ep)
    random.seed(SEED + ep)
    
    if USE_EXPECTED_BODY:
        # 유저별 기대본문 사용
        traingen=generate_batch_data_train(
            all_train_pn, all_label, all_train_id, 30, 
            candidate_news_body=None,
            expected_bodies=expected_bodies_train,
            all_userid_str=all_train_userid_str,
            all_newsid_str=all_train_newsid_str,
            news_index_reverse=news_index_reverse
        )
    else:
        # 원본 본문 사용
        traingen=generate_batch_data_train(all_train_pn,all_label,all_train_id, 30, candidate_news_body=None)
    # 나머지 샘플도 처리하기 위해 올림 계산
    actual_train_samples = len(all_train_id)
    steps_per_epoch = (actual_train_samples + 29) // 30  # 올림 계산 (배치 수)
    # print(f"[디버깅] 학습 샘플 수: {actual_train_samples}개")
    # print(f"[디버깅] steps_per_epoch 계산: {actual_train_samples}개 샘플 / 30 = {steps_per_epoch} steps (예상 처리 샘플 수: {steps_per_epoch * 30})")
    # print(f"[디버깅] generate_batch_data_train은 배치 단위로 yield하므로 steps_per_epoch={steps_per_epoch}이 올바릅니다.")
    model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch)
    
    if USE_EXPECTED_BODY:
        # 유저별 기대본문 사용
        testgen=generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies_test,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse
        )
    else:
        # 원본 본문 사용
        testgen=generate_batch_data_test(all_test_pn,all_test_label,all_test_id, 30, candidate_news_body=None)
    # 나머지 샘플도 처리하기 위해 올림 계산
    # 실제 샘플 수를 기준으로 정확하게 계산
    actual_test_samples = len(all_test_id)
    # generate_batch_data_test는 각 샘플을 개별적으로 yield하므로, 
    # steps는 실제 샘플 수와 같아야 합니다 (배치 크기와 무관)
    test_steps = actual_test_samples
    # print(f"[디버깅] test_steps 계산: {actual_test_samples}개 샘플 (각 샘플을 개별 yield하므로 steps=샘플 수)")
    click_score = model_test.predict(testgen, steps=test_steps, verbose=1)
    # print(f"[디버깅] 실제 생성된 click_score 수: {len(click_score)}")
    
    # click_score가 실제 샘플 수와 일치하는지 확인 (디버깅용, 필요 시 주석 해제)
    # if len(click_score) != actual_test_samples:
    #     print(f"[경고] click_score({len(click_score)})가 실제 샘플 수({actual_test_samples})와 일치하지 않습니다!")
    #     print(f"[경고] 차이: {actual_test_samples - len(click_score)}개 샘플이 누락되었습니다.")
    from sklearn.metrics import roc_auc_score
    all_auc=[]
    all_mrr=[]
    all_ndcg=[]
    all_hit1=[]
    
    # # click_score 디버깅 출력 (필요 시 주석 해제)
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
            
            # # 처음 5개 세션만 상세 출력 (필요 시 주석 해제)
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
            
            all_auc.append(roc_auc_score(session_labels, session_scores))
            all_mrr.append(mrr_score(session_labels, session_scores))
            all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
            all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
            session_count += 1
    
    # # 디버깅용 세션 통계 출력 (필요 시 주석 해제)
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
    epoch_results = {
        'AUC': np.mean(all_auc),
        'MRR': np.mean(all_mrr),
        'NDCG@5': np.mean(all_ndcg),
        'Hit@1': np.mean(all_hit1)
    }
    results.append([epoch_results['AUC'], epoch_results['MRR'], epoch_results['NDCG@5'], epoch_results['Hit@1']])
    
    # Best AUC 업데이트
    if epoch_results['AUC'] > best_auc:
        best_auc = epoch_results['AUC']
    
    # 보기 좋게 출력
    current_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate
    print(f"\n{'='*60}")
    print(f"Epoch {ep+1}/10 - Test Results (LR: {current_lr:.6f})")
    print(f"{'='*60}")
    print(f"AUC      : {epoch_results['AUC']:.6f} (Best: {best_auc:.6f})")
    print(f"MRR      : {epoch_results['MRR']:.6f}")
    print(f"NDCG@5   : {epoch_results['NDCG@5']:.6f}")
    print(f"Hit@1    : {epoch_results['Hit@1']:.6f}")
    print(f"{'='*60}\n")

# 전체 결과 요약
print(f"\n{'='*60}")
print("Final Results Summary (All Epochs)")
print(f"{'='*60}")
print(f"{'Epoch':<10} {'AUC':<12} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
print(f"{'-'*60}")
for i, result in enumerate(results, 1):
    auc, mrr, ndcg5, hit1 = result
    print(f"{i:<10} {auc:<12.6f} {mrr:<12.6f} {ndcg5:<12.6f} {hit1:<12.6f}")
print(f"{'='*72}")

# 최고 성능 찾기
best_auc_idx = np.argmax([r[0] for r in results])
best_auc_epoch = best_auc_idx + 1
best_hit1_idx = np.argmax([r[3] for r in results])
best_hit1_epoch = best_hit1_idx + 1
print(f"\nBest AUC  : Epoch {best_auc_epoch} - {results[best_auc_idx][0]:.6f}")
print(f"Best Hit@1: Epoch {best_hit1_epoch} - {results[best_hit1_idx][3]:.6f}")
print(f"{'='*60}\n")
# In[ ]:




