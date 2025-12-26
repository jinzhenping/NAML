#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAML 모델 학습 및 테스트 스크립트
리눅스 환경에서 실행 가능한 버전
"""

import csv
import random
import nltk
from nltk.tokenize import word_tokenize
import datetime
import time
import itertools
import numpy as np
import pickle
from numpy.linalg import cholesky
import os
import argparse

# Keras imports (모듈 레벨에서 import)
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *
from sklearn.metrics import roc_auc_score

# NLTK 데이터 다운로드 (처음 실행 시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK punkt 데이터를 다운로드합니다...")
    nltk.download('punkt', quiet=True)


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio)
    else:
        return random.sample(nnn, ratio)


def preprocess_news_file(file='dataset/MIND/MIND_news.tsv'):
    """
    MIND 뉴스 데이터 전처리
    형식: news_id, category, subcategory, title, body
    """
    print("뉴스 데이터 전처리 중...")
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


def preprocess_user_file(train_file='dataset/MIND/MIND_train_(1000).tsv', 
                         test_file='dataset/MIND/MIND_test_(1000).tsv',
                         news_index=None, npratio=4):
    """
    MIND 데이터셋 형식에 맞게 전처리
    train_file: user, clicked_news, candidate_news, clicked
    test_file: user, clicked_news, candidate_news (clicked 없음)
    """
    print("유저 데이터 전처리 중...")
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
    
    all_test_id = []
    all_test_pn = []
    all_test_label = []
    all_test_index = []
    
    all_user_pos = []
    all_test_user_pos = []
    
    # 학습 데이터 처리
    for line in train_data:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue
        
        userid = parts[0]
        clicked_news = parts[1].split()  # 클릭 히스토리
        candidate_news = parts[2].split()  # 후보 뉴스들
        clicked = parts[3].split()  # 클릭 여부 (1 또는 0)
        
        # clicked_news를 news_index로 변환
        clicked_news_ids = []
        for news_id in clicked_news:
            if news_id in news_index:
                clicked_news_ids.append(news_index[news_id])
        
        if len(clicked_news_ids) == 0:
            continue
        
        # 후보 뉴스들을 news_index로 변환
        candidate_indices = []
        candidate_labels = []
        for i, cand_id in enumerate(candidate_news):
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
                is_clicked = int(clicked[i]) if i < len(clicked) else 0
                candidate_labels.append(is_clicked)
        
        # 후보가 5개가 아니거나 positive가 없으면 스킵
        if len(candidate_indices) < 2 or sum(candidate_labels) == 0:
            continue
        
        # 5개 후보 중 1개 positive, 나머지 negative
        # 순서를 섞기
        combined = list(zip(candidate_indices, candidate_labels))
        random.shuffle(combined)
        shuffle_indices, shuffle_labels = zip(*combined)
        
        # 유저 히스토리 (최대 50개)
        posset = list(set(clicked_news_ids) - set(candidate_indices))
        allpos = [int(p) for p in random.sample(posset, min(50, len(posset)))[:50]]
        allpos += [0] * (50 - len(allpos))
        
        all_train_pn.append(list(shuffle_indices))
        all_label.append(list(shuffle_labels))
        all_train_id.append(userid_dict[userid])
        all_user_pos.append(allpos)
    
    # 테스트 데이터 처리
    for line in test_data:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        
        userid = parts[0]
        clicked_news = parts[1].split()
        candidate_news = parts[2].split()
        
        # clicked_news를 news_index로 변환
        clicked_news_ids = []
        for news_id in clicked_news:
            if news_id in news_index:
                clicked_news_ids.append(news_index[news_id])
        
        if len(clicked_news_ids) == 0 or len(candidate_news) == 0:
            continue
        
        # 세션 인덱스 시작
        sess_index = [len(all_test_pn)]
        
        # 유저 히스토리
        posset = list(set(clicked_news_ids))
        allpos = [int(p) for p in random.sample(posset, min(50, len(posset)))[:50]]
        allpos += [0] * (50 - len(allpos))
        
        # 후보 뉴스들을 news_index로 변환
        candidate_indices = []
        for cand_id in candidate_news:
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
        
        if len(candidate_indices) < 2:
            continue
        
        # 5개 후보 중 첫 번째가 positive, 나머지가 negative
        # 테스트에서는 순서를 섞지 않고 그대로 사용 (첫 번째가 정답)
        for i, cand_idx in enumerate(candidate_indices):
            all_test_pn.append(int(cand_idx))
            # 첫 번째 후보가 positive (정답)
            all_test_label.append(1 if i == 0 else 0)
            all_test_id.append(userid_dict[userid])
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
    
    print(f"학습 샘플: {len(all_train_id)}개")
    print(f"테스트 샘플: {len(all_test_id)}개")
    
    return userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index


def get_embedding(word_dict, glove_path='glove.840B.300d.txt'):
    """
    GloVe 임베딩 로드
    glove_path가 없으면 랜덤 초기화 사용
    """
    print("임베딩 로드 중...")
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


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0


def generate_batch_data_train(all_train_pn, all_label, all_train_id, batch_size, 
                               news_words, news_body, news_v, news_sv, all_user_pos):
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            candidate = news_words[all_train_pn[i]]
            candidate_split = [candidate[:, k, :] for k in range(candidate.shape[1])]
            candidate_body = news_body[all_train_pn[i]]
            candidate_body_split = [candidate_body[:, k, :] for k in range(candidate_body.shape[1])]
            candidate_vertical = news_v[all_train_pn[i]]
            candidate_vertical_split = [candidate_vertical[:, k, :] for k in range(candidate_vertical.shape[1])]
            candidate_subvertical = news_sv[all_train_pn[i]]
            candidate_subvertical_split = [candidate_subvertical[:, k, :] for k in range(candidate_subvertical.shape[1])]
            
            browsed_news = news_words[all_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            browsed_news_body = news_body[all_user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            browsed_news_vertical = news_v[all_user_pos[i]]
            browsed_news_vertical_split = [browsed_news_vertical[:, k, :] for k in range(browsed_news_vertical.shape[1])]
            browsed_news_subvertical = news_sv[all_user_pos[i]]
            browsed_news_subvertical_split = [browsed_news_subvertical[:, k, :] for k in range(browsed_news_subvertical.shape[1])]
            
            label = all_label[i]

            yield (candidate_split + browsed_news_split + candidate_body_split + browsed_news_body_split
                   + candidate_vertical_split + browsed_news_vertical_split + candidate_subvertical_split + browsed_news_subvertical_split, [label])


def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size,
                              news_words, news_body, news_v, news_sv, all_test_user_pos):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            candidate = news_words[all_test_pn[i]]
            candidate_body = news_body[all_test_pn[i]]
            candidate_vertical = news_v[all_test_pn[i]]
            candidate_subvertical = news_sv[all_test_pn[i]]

            browsed_news = news_words[all_test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            browsed_news_body = news_body[all_test_user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            browsed_news_vertical = news_v[all_test_user_pos[i]]
            browsed_news_vertical_split = [browsed_news_vertical[:, k, :] for k in range(browsed_news_vertical.shape[1])]
            browsed_news_subvertical = news_sv[all_test_user_pos[i]]
            browsed_news_subvertical_split = [browsed_news_subvertical[:, k, :] for k in range(browsed_news_subvertical.shape[1])]
            
            label = all_label[i]
            yield ([candidate] + browsed_news_split + [candidate_body] + browsed_news_body_split + [candidate_vertical]
                   + browsed_news_vertical_split + [candidate_subvertical] + browsed_news_subvertical_split, [label])


def main():
    parser = argparse.ArgumentParser(description='NAML 모델 학습 및 테스트')
    parser.add_argument('--train_file', type=str, default='dataset/MIND/MIND_train_(1000).tsv',
                        help='학습 데이터 파일 경로')
    parser.add_argument('--test_file', type=str, default='dataset/MIND/MIND_test_(1000).tsv',
                        help='테스트 데이터 파일 경로')
    parser.add_argument('--news_file', type=str, default='dataset/MIND/MIND_news.tsv',
                        help='뉴스 데이터 파일 경로')
    parser.add_argument('--glove_path', type=str, default=None,
                        help='GloVe 임베딩 파일 경로 (없으면 랜덤 초기화)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='사용할 GPU ID (예: 0, 1, 또는 "0,1")')
    parser.add_argument('--epochs', type=int, default=3,
                        help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='배치 크기')
    parser.add_argument('--npratio', type=int, default=4,
                        help='Negative/Positive 비율 (후보 개수 = 1 + npratio)')
    parser.add_argument('--save_model', type=str, default=None,
                        help='모델 저장 경로 (예: models/naml_model.h5)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='모델 로드 경로 (테스트만 실행 시 사용)')
    parser.add_argument('--test_only', action='store_true',
                        help='테스트만 실행 (--load_model 필요)')
    parser.add_argument('--save_data', type=str, default=None,
                        help='전처리된 데이터 저장 경로 (pickle 형식)')
    parser.add_argument('--load_data', type=str, default=None,
                        help='전처리된 데이터 로드 경로 (pickle 형식)')
    
    args = parser.parse_args()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 데이터 전처리 또는 로드
    if args.load_data:
        print(f"전처리된 데이터 로드 중: {args.load_data}")
        with open(args.load_data, 'rb') as f:
            data = pickle.load(f)
            word_dict = data['word_dict']
            category = data['category']
            subcategory = data['subcategory']
            news_words = data['news_words']
            news_body = data['news_body']
            news_v = data['news_v']
            news_sv = data['news_sv']
            news_index = data['news_index']
            userid_dict = data['userid_dict']
            all_train_pn = data['all_train_pn']
            all_label = data['all_label']
            all_train_id = data['all_train_id']
            all_test_pn = data['all_test_pn']
            all_test_label = data['all_test_label']
            all_test_id = data['all_test_id']
            all_user_pos = data['all_user_pos']
            all_test_user_pos = data['all_test_user_pos']
            all_test_index = data['all_test_index']
        print("데이터 로드 완료!")
    else:
        # 데이터 전처리
        word_dict, category, subcategory, news_words, news_body, news_v, news_sv, news_index = preprocess_news_file(args.news_file)
        
        userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index = preprocess_user_file(
            train_file=args.train_file,
            test_file=args.test_file,
            news_index=news_index,
            npratio=args.npratio
        )
        
        # 전처리된 데이터 저장
        if args.save_data:
            print(f"전처리된 데이터 저장 중: {args.save_data}")
            os.makedirs(os.path.dirname(args.save_data) if os.path.dirname(args.save_data) else '.', exist_ok=True)
            with open(args.save_data, 'wb') as f:
                pickle.dump({
                    'word_dict': word_dict,
                    'category': category,
                    'subcategory': subcategory,
                    'news_words': news_words,
                    'news_body': news_body,
                    'news_v': news_v,
                    'news_sv': news_sv,
                    'news_index': news_index,
                    'userid_dict': userid_dict,
                    'all_train_pn': all_train_pn,
                    'all_label': all_label,
                    'all_train_id': all_train_id,
                    'all_test_pn': all_test_pn,
                    'all_test_label': all_test_label,
                    'all_test_id': all_test_id,
                    'all_user_pos': all_user_pos,
                    'all_test_user_pos': all_test_user_pos,
                    'all_test_index': all_test_index
                }, f)
            print("데이터 저장 완료!")
    
    print(f"뉴스 개수: {len(news_index)}")
    print(f"카테고리 개수: {len(category)}")
    print(f"서브카테고리 개수: {len(subcategory)}")
    
    # 임베딩 로드
    if args.glove_path:
        embedding_mat = get_embedding(word_dict, glove_path=args.glove_path)
    else:
        embedding_mat = get_embedding(word_dict)
    
    # 모델 구축
    print("\n모델 구축 중...")
    keras.backend.clear_session()
    
    MAX_SENT_LENGTH = 30
    MAX_SENTS = 50
    MAX_BODY_LENGTH = 300
    npratio = args.npratio
    
    title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    body_input = Input(shape=(MAX_BODY_LENGTH,), dtype='int32')
    embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)
    
    embedded_sequences_title = embedding_layer(title_input)
    embedded_sequences_title = Dropout(0.2)(embedded_sequences_title)
    
    embedded_sequences_body = embedding_layer(body_input)
    embedded_sequences_body = Dropout(0.2)(embedded_sequences_body)
    
    title_cnn = Convolution1D(nb_filter=400, filter_length=3, padding='same', activation='relu', strides=1)(embedded_sequences_title)
    title_cnn = Dropout(0.2)(title_cnn)
    
    attention = Dense(200, activation='tanh')(title_cnn)
    attention = Flatten()(Dense(1)(attention))
    attention_weight = Activation('softmax')(attention)
    title_rep = keras.layers.Dot((1, 1))([title_cnn, attention_weight])
    
    body_cnn = Convolution1D(nb_filter=400, filter_length=3, padding='same', activation='relu', strides=1)(embedded_sequences_body)
    body_cnn = Dropout(0.2)(body_cnn)
    
    attention_body = Dense(200, activation='tanh')(body_cnn)
    attention_body = Flatten()(Dense(1)(attention_body))
    attention_weight_body = Activation('softmax')(attention_body)
    body_rep = keras.layers.Dot((1, 1))([body_cnn, attention_weight_body])
    
    vinput = Input((1,), dtype='int32')
    svinput = Input((1,), dtype='int32')
    v_embedding_layer = Embedding(len(category) + 1, 50, trainable=True)
    sv_embedding_layer = Embedding(len(subcategory) + 1, 50, trainable=True)
    v_embedding = Dense(400, activation='relu')(Flatten()(v_embedding_layer(vinput)))
    sv_embedding = Dense(400, activation='relu')(Flatten()(sv_embedding_layer(svinput)))
    
    all_channel = [title_rep, body_rep, v_embedding, sv_embedding]
    
    views = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(channel) for channel in all_channel], axis=1)
    
    attentionv = Dense(200, activation='tanh')(views)
    
    attention_weightv = Lambda(lambda x: K.squeeze(x, axis=-1))(Dense(1)(attentionv))
    attention_weightv = Activation('softmax')(attention_weightv)
    
    newsrep = keras.layers.Dot((1, 1))([views, attention_weightv])
    
    newsEncoder = Model([title_input, body_input, vinput, svinput], newsrep)
    
    browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
    browsed_body_input = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
    
    browsed_v_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]
    browsed_sv_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]
    
    browsednews = [newsEncoder([browsed_news_input[_], browsed_body_input[_], browsed_v_input[_], browsed_sv_input[_]]) for _ in range(MAX_SENTS)]
    browsednewsrep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsednews], axis=1)
    
    attentionn = Dense(200, activation='tanh')(browsednewsrep)
    attentionn = Flatten()(Dense(1)(attentionn))
    attention_weightn = Activation('softmax')(attentionn)
    user_rep = keras.layers.Dot((1, 1))([browsednewsrep, attention_weightn])
    
    candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1 + npratio)]
    candidates_body = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(1 + npratio)]
    candidates_v = [keras.Input((1,), dtype='int32') for _ in range(1 + npratio)]
    candidates_sv = [keras.Input((1,), dtype='int32') for _ in range(1 + npratio)]
    candidate_vecs = [newsEncoder([candidates_title[_], candidates_body[_], candidates_v[_], candidates_sv[_]]) for _ in range(1 + npratio)]
    
    logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
    logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))
    
    model = Model(candidates_title + browsed_news_input + candidates_body + browsed_body_input +
                  candidates_v + browsed_v_input + candidates_sv + browsed_sv_input, logits)
    
    candidate_one_title = keras.Input((MAX_SENT_LENGTH,))
    candidate_one_body = keras.Input((MAX_BODY_LENGTH,))
    candidate_one_v = keras.Input((1,))
    candidate_one_sv = keras.Input((1,))
    
    candidate_one_vec = newsEncoder([candidate_one_title, candidate_one_body, candidate_one_v, candidate_one_sv])
    
    score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
    model_test = keras.Model([candidate_one_title] + browsed_news_input + [candidate_one_body] + browsed_body_input
                             + [candidate_one_v] + browsed_v_input + [candidate_one_sv] + browsed_sv_input, score)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
    
    print("모델 구축 완료!")
    
    # 모델 로드 (테스트만 실행 시)
    if args.load_model:
        print(f"모델 로드 중: {args.load_model}")
        model.load_weights(args.load_model)
        print("모델 로드 완료!")
    
    # 테스트만 실행
    if args.test_only:
        if not args.load_model:
            raise ValueError("테스트만 실행하려면 --load_model 옵션이 필요합니다.")
        
        print("\n테스트 실행 중...")
        testgen = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, args.batch_size,
                                           news_words, news_body, news_v, news_sv, all_test_user_pos)
        click_score = model_test.predict_generator(testgen, steps=len(all_test_id) // args.batch_size, verbose=1)
        
        all_auc = []
        all_mrr = []
        all_ndcg = []
        all_ndcg2 = []
        for m in all_test_index:
            if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
                all_auc.append(roc_auc_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_mrr.append(mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_ndcg.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
                all_ndcg2.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
        
        print(f"\n{'='*60}")
        print("테스트 결과:")
        print(f"{'='*60}")
        print(f"  AUC: {np.mean(all_auc):.4f}")
        print(f"  MRR: {np.mean(all_mrr):.4f}")
        print(f"  NDCG@5: {np.mean(all_ndcg):.4f}")
        print(f"  NDCG@10: {np.mean(all_ndcg2):.4f}")
        return
    
    print(f"학습 시작: {args.epochs} 에포크, 배치 크기: {args.batch_size}")
    
    results = []
    for ep in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"에포크 {ep + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        traingen = generate_batch_data_train(all_train_pn, all_label, all_train_id, args.batch_size,
                                             news_words, news_body, news_v, news_sv, all_user_pos)
        model.fit_generator(traingen, epochs=1, steps_per_epoch=len(all_train_id) // args.batch_size, verbose=1)
        
        # 모델 저장 (각 에포크마다)
        if args.save_model:
            save_path = args.save_model.replace('.h5', f'_epoch{ep+1}.h5')
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            model.save_weights(save_path)
            print(f"모델 저장: {save_path}")
        
        # 테스트 실행
        testgen = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, args.batch_size,
                                           news_words, news_body, news_v, news_sv, all_test_user_pos)
        click_score = model_test.predict_generator(testgen, steps=len(all_test_id) // args.batch_size, verbose=1)
        
        all_auc = []
        all_mrr = []
        all_ndcg = []
        all_ndcg2 = []
        for m in all_test_index:
            if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
                all_auc.append(roc_auc_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_mrr.append(mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_ndcg.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
                all_ndcg2.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
        
        result = [np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2)]
        results.append(result)
        print(f"\n에포크 {ep + 1} 결과:")
        print(f"  AUC: {result[0]:.4f}")
        print(f"  MRR: {result[1]:.4f}")
        print(f"  NDCG@5: {result[2]:.4f}")
        print(f"  NDCG@10: {result[3]:.4f}")
    
    # 최종 모델 저장
    if args.save_model:
        final_save_path = args.save_model.replace('.h5', '_final.h5')
        model.save_weights(final_save_path)
        print(f"\n최종 모델 저장: {final_save_path}")
    
    print(f"\n{'='*60}")
    print("최종 결과:")
    print(f"{'='*60}")
    for i, result in enumerate(results, 1):
        print(f"에포크 {i}: AUC={result[0]:.4f}, MRR={result[1]:.4f}, NDCG@5={result[2]:.4f}, NDCG@10={result[3]:.4f}")


if __name__ == "__main__":
    main()

