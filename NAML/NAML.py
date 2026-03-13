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


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 모델 하이퍼파라미터 설정
MAX_HISTORY_CLICKS = 50  # 클릭 히스토리 개수
MAX_SENT_LENGTH = 30     # 제목 최대 단어 수
MAX_BODY_LENGTH = 300    # 본문 최대 단어 수
npratio = 4              # negative sampling 비율
USE_EXPECTED_BODY = True  # True: 기대 본문 사용, False: 원본 본문 사용

DO_PRETRAINING = False   # True: 트레이닝셋 80%로 pretraining 수행, False: pretraining 건너뛰기
PRETRAINING_EPOCHS = 20    # Pretraining 에폭 수
PRETRAINING_SAVE_PATH = 'saved_models/pretrained_naml_model.h5'  # Pretraining 모델 저장 경로

DO_PRETRAINING_ON_TRAIN20 = False  # True: 트레이닝셋 유저별 후반 20%로 pretraining (구조는 DO_PRETRAINING과 동일, 원본 본문 사용)
PRETRAINING_ON_TRAIN20_EPOCHS = 20  # 유저별 후반 20% 프리트레이닝 에폭 수
PRETRAINING_ON_TRAIN20_SAVE_PATH = 'saved_models/pretrained_naml_model_train20.h5'  # 저장 경로

EVAL_PRETRAINED_ON_TRAIN80 = False  # True: 저장된 프리트레이닝 모델 로드 후 트레이닝 80%에 대해 실제/기대 본문 각각 테스트

EVAL_PRETRAINED_ON_TRAIN80_FIRST_BATCH = False  # True: 트레이닝 80% 지정 배치만 평가, 기대본문은 train80_batch{N} 로드, result{N}.txt 저장
EVAL_TRAIN80_BATCH_SIZE = 500  # 배치당 세션 수
EVAL_TRAIN80_BATCH_INDEX = 0   # 배치 번호 (0=첫 500세션→result0.txt, 1=다음 500세션→result1.txt, ...)

EVAL_PRETRAINED_ON_TRAIN20_FIRST_BATCH = False  # True: 유저별 후반 20% 지정 배치만 평가, 기대본문은 train20_batch{N} 로드, result{N}.txt 저장
EVAL_TRAIN20_BATCH_SIZE = 500  # 유저별 후반 20% 배치당 세션 수 (body_generation --train20_first_k와 동일하게 맞출 것)
EVAL_TRAIN20_BATCH_INDEX = 0   # 배치 번호 (0=첫 K세션→result0.txt, 1=다음 K세션→result1.txt, ...)

EVAL_PRETRAINED_ON_TRAIN20 = False  # True: 프리트레이닝 모델 로드 후 트레이닝 후반 20%에 대해 실제/기대 본문 각각 평가
EVAL_TRAIN20_EXPECTED_BODY_DIR = 'train_20'  # 트레이닝 후반 20% 평가 시 기대본문 폴더 (예: 'train20_0'). None이면 train20_N 중 가장 큰 N 사용

EVAL_PRETRAINED_ON_TESTSET = False  # True: 저장된 프리트레이닝 모델 로드 후 테스트셋에 대해 실제/기대 본문 각각 테스트 (NDCG@5, MRR, Hit@1, Loss)
EVAL_TESTSET_EXPECTED_BODY_DIR = 'test_0'  # 테스트셋 기대본문 폴더 (body_generation/output 아래). 예: 'test', 'test_0', 'test_1'
EVAL_TESTSET_EXPECTED_BODY_DIR_2 = None  # 두 번째 기대본문 폴더 (None이면 사용 안 함). 설정 시 매 에폭 두 버전 모두 평가
PRETRAINED_MODEL_PATH = 'saved_models/pretrained_naml_model_train20.h5'  # 위 두 평가 모드에서 로드할 모델 경로
MAIN_TRAINING_EPOCHS = 20  # 메인 학습 루프 에폭 수

# 트레이닝 후반 20%만 사용해 처음부터 학습 후 테스트셋 실제/기대본문 각각 평가 (메인 학습 루프 대신 실행)
TRAIN_ON_TRAIN20_FROM_SCRATCH = False  # True: 트레이닝 후반 20%만 사용, 처음부터 학습
TRAIN_ON_TRAIN20_USE_EXPECTED_BODY = False  # True: 학습 시 기대본문 사용, False: 실제본문 사용
TRAIN_ON_TRAIN20_GENERATE_EXPECTED_BODY = False  # True: 기대본문 폴더 없을 때 body_generation으로 유저별 후반 20% 기대본문 자동 생성 후 학습
TRAIN_ON_TRAIN20_EXPECTED_BODY_DIR = 'train_last20'  # 기대본문으로 학습 시 폴더 (body_generation/output 아래)
TRAIN_ON_TRAIN20_EXPECTED_BODY_POSITIVE_ONLY = True  # True: 세션당 positive 1개만 기대본문 생성/사용, False: 5개 후보 모두
TRAIN_ON_TRAIN20_EPOCHS = 20  # 학습 에폭 수
TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR = 'test_0'  # 매 에폭 테스트셋 기대본문 평가용 폴더
TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR_2 = None  # 두 번째 기대본문 폴더 (None이면 사용 안 함). 설정 시 매 에폭 두 버전 모두 평가

FINETUNE_USER_ENCODER = False  # True: 프리트레이닝 모델 로드 후 유저 인코더만 파인튜닝 (뉴스 인코더 고정)
FINETUNE_NEWS_ENCODER = False  # True: 프리트레이닝 모델 로드 후 뉴스 인코더만 파인튜닝 (유저 쪽 attention 등 고정)
FINETUNE_FULL_MODEL = False  # True: 프리트레이닝 모델 로드 후 모델 전체 파인튜닝 (유저+뉴스 인코더 모두 학습)
FINETUNE_EXPECTED_BODY_DIR = 'train_last20'  # 기대본문 폴더 (body_generation/output/train20_0)
FINETUNE_TESTSET_EXPECTED_BODY_DIR = 'test_0'  # 파인튜닝 시 매 에폭 테스트셋 평가용 기대본문 폴더 (body_generation/output 아래)
FINETUNE_EPOCHS = 10  # 파인튜닝 에폭 수
FINETUNE_LR = 0.0005  # 파인튜닝 학습률
FINETUNE_SAVE_PATH = 'saved_models/finetuned_user_encoder.h5'  # 유저인코더만 파인튜닝 시 기대본문 NDCG@5 최고 모델 저장 경로 (None이면 저장 안 함)
FINETUNE_NEWS_ENCODER_SAVE_PATH = 'saved_models/finetuned_news_encoder.h5'  # 뉴스 인코더만 파인튜닝 시 기대본문 NDCG@5 최고 모델 저장 경로 (None이면 저장 안 함)
FINETUNE_FULL_SAVE_PATH = 'saved_models/finetuned_full_model.h5'  # 모델 전체 파인튜닝 시 기대본문 NDCG@5 최고 모델 저장 경로 (None이면 저장 안 함)


def load_expected_bodies(output_dir='body_generation/output', dataset_type='train'):
    """
    기대 본문 로드 (유저별로 다른 기대본문 지원)
    output_dir/{dataset_type}/user_{user_id}/news_{news_id}.json에서 기대 본문 로드
    반환: {(user_id, news_id): generated_body} 형태의 딕셔너리
    """
    
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
                except Exception as e:
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


def get_latest_train_folder(base_output_dir):
    """
    base_output_dir 아래에서 train0, train1, ... 중 숫자가 가장 큰 폴더 경로 반환.
    없으면 None 반환.
    """
    if not os.path.isdir(base_output_dir):
        return None
    max_num = -1
    for name in os.listdir(base_output_dir):
        path = os.path.join(base_output_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("train") and len(name) > 5:
            try:
                n = int(name[5:])
                if n > max_num:
                    max_num = n
            except ValueError:
                continue
    if max_num < 0:
        return None
    return os.path.join(base_output_dir, f"train{max_num}")


def get_latest_train20_folder(base_output_dir):
    """
    base_output_dir 아래에서 train20_0, train20_1, ... 중 숫자가 가장 큰 폴더 경로 반환.
    없으면 None 반환.
    """
    if not os.path.isdir(base_output_dir):
        return None
    prefix = "train20_"
    max_num = -1
    for name in os.listdir(base_output_dir):
        path = os.path.join(base_output_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith(prefix):
            try:
                n = int(name[len(prefix):])
                if n > max_num:
                    max_num = n
            except ValueError:
                continue
    if max_num < 0:
        return None
    return os.path.join(base_output_dir, f"train20_{max_num}")


def load_expected_body_from_train_dir(train_dir, user_id_str, news_id_str):
    """
    body_generation/output/trainN 구조에서 한 (user, news)에 대한 기대 본문과 유저 클릭 히스토리 로드.
    train_dir: trainN 폴더 전체 경로. user_{id}/news_{id}.json 에서 generated_body, user_history 반환.
    반환: (generated_body, user_history) — user_history는 리스트(최대 10개) 또는 None(파일 없음/오류 시).
    """
    if not train_dir or not user_id_str or not news_id_str or user_id_str == '?' or news_id_str == '?':
        return '', None
    path = os.path.join(train_dir, f"user_{user_id_str}", f"news_{news_id_str}.json")
    if not os.path.isfile(path):
        return '', None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            body = data.get('generated_body', '') or ''
            history = data.get('user_history')
            if isinstance(history, list):
                history = history[-10:]
            else:
                history = None
            return body, history
    except Exception:
        return '', None


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
        
        # 정답(첫 번째 후보)이 news_index에 있는지 먼저 확인
        if len(candidate_news) == 0 or candidate_news[0] not in news_index:
            continue  # 정답이 news_index에 없으면 세션 스킵
        
        # 후보 뉴스들을 news_index로 변환 (히스토리 필터링 전에 먼저 처리)
        candidate_indices = []
        candidate_news_filtered = []  # 필터링된 candidate_news (인덱스 대응 보장)
        for cand_id in candidate_news:
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
                candidate_news_filtered.append(cand_id)  # 필터링된 뉴스 ID 저장
                candidate_news_ids_test.add(cand_id)  # 후보 뉴스 ID 수집
        
        # 정답이 필터링된 후보에 포함되어 있는지 확인 (이미 위에서 확인했지만 안전장치)
        if candidate_news[0] not in candidate_news_filtered:
            continue  # 정답이 필터링된 후보에 없으면 세션 스킵
        
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
        # 정답(첫 번째 후보)이 candidate_news_filtered에서의 인덱스 찾기
        positive_index_in_filtered = candidate_news_filtered.index(candidate_news[0])
        candidate_labels = [1 if i == positive_index_in_filtered else 0 for i in range(len(candidate_indices))]
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
    
    # 최소 빈도 2 이상만 사용
    word_dict = {}
    for i in word_dict_raw:
        if word_dict_raw[i][1] >= 2:
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
    
    print(f"단어 사전 크기: {len(word_dict)} (전체: {len(word_dict_raw)})")
    
    # 뉴스 제목 인덱싱
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
    
    # 뉴스 본문 인덱싱
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
    # 원본 본문 사용
    print("\n원본 본문 사용 모드")
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

                user_pos = all_test_user_pos_override if all_test_user_pos_override is not None else all_test_user_pos
                user_pos_indices = np.array(user_pos[idx], dtype='int32')
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

results=[]
keras.backend.clear_session()

MAX_SENTS = MAX_HISTORY_CLICKS  # 히스토리 클릭 개수
title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

body_input = Input(shape=(MAX_BODY_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat],trainable=True)

embedded_sequences_title = embedding_layer(title_input)
embedded_sequences_title=Dropout(0.3)(embedded_sequences_title)

embedded_sequences_body = embedding_layer(body_input)
embedded_sequences_body=Dropout(0.3)(embedded_sequences_body)

title_cnn = Conv1D(filters=400, kernel_size=3, padding='same', activation='relu', strides=1)(embedded_sequences_title)
title_cnn=Dropout(0.3)(title_cnn)

attention = Dense(200,activation='tanh')(title_cnn)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
title_rep=keras.layers.Dot((1, 1))([title_cnn, attention_weight])

body_cnn = Conv1D(filters=400, kernel_size=3, padding='same', activation='relu', strides=1)(embedded_sequences_body)
body_cnn=Dropout(0.3)(body_cnn)

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


model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['acc'])

# news_index 역매핑 생성 (인덱스 -> 뉴스 ID)
news_index_reverse = {v: k for k, v in news_index.items()}

# ========== Pretraining 섹션 ==========
if DO_PRETRAINING:
    print(f"\n{'='*60}")
    print("Pretraining 시작: 트레이닝셋 전반부 80% 데이터로 원본 본문 사용")
    print(f"{'='*60}")
    
    # 학습 데이터를 전반부 80%로 슬라이싱
    pretrain_size = int(len(all_train_id) * 0.8)
    print(f"전체 학습 샘플 수: {len(all_train_id)}개")
    print(f"Pretraining에 사용할 샘플 수: {pretrain_size}개 (80%)")
    
    # 모든 관련 리스트를 동일한 비율로 슬라이싱
    pretrain_id = all_train_id[:pretrain_size]
    pretrain_pn = all_train_pn[:pretrain_size]
    pretrain_label = all_label[:pretrain_size]
    pretrain_userid_str = all_train_userid_str[:pretrain_size] if USE_EXPECTED_BODY else None
    pretrain_newsid_str = all_train_newsid_str[:pretrain_size] if USE_EXPECTED_BODY else None
    
    print(f"Pretraining 데이터 준비 완료")
    print(f"Pretraining 에폭 수: {PRETRAINING_EPOCHS}")
    print(f"{'='*60}\n")
    
    # Pretraining 루프 (원본 본문 사용)
    pretrain_results = []  # Pretraining 에폭별 결과 저장
    best_mrr = -1.0  # 최고 MRR 추적
    best_mrr_epoch = -1  # 최고 MRR 에폭
    for pretrain_ep in range(PRETRAINING_EPOCHS):
        np.random.seed(SEED + pretrain_ep)
        random.seed(SEED + pretrain_ep)
        
        # 원본 본문 사용 (USE_EXPECTED_BODY=False)
        pretrain_gen = generate_batch_data_train(
            pretrain_pn, pretrain_label, pretrain_id, 30, 
            candidate_news_body=None
        )
        
        actual_pretrain_samples = len(pretrain_id)
        pretrain_steps_per_epoch = (actual_pretrain_samples + 29) // 30
        
        print(f"\nPretraining Epoch {pretrain_ep+1}/{PRETRAINING_EPOCHS} - 샘플 수: {actual_pretrain_samples}개, Steps: {pretrain_steps_per_epoch}")
        model.fit(pretrain_gen, epochs=1, steps_per_epoch=pretrain_steps_per_epoch, verbose=1)
        
        # ========== Pretraining 에폭별 테스트 평가 (원본 본문 사용) ==========
        print(f"\n[Pretraining Epoch {pretrain_ep+1}] 테스트셋 평가 중... (원본 본문 사용)")
        
        # 테스트 데이터 생성 (원본 본문 사용)
        pretrain_testgen = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30, 
            candidate_news_body=None
        )
        
        actual_test_samples = len(all_test_id)
        test_steps = actual_test_samples
        click_score = model_test.predict(pretrain_testgen, steps=test_steps, verbose=0)
        
        # 평가 지표 계산
        pretrain_all_mrr = []
        pretrain_all_ndcg = []
        pretrain_all_hit1 = []
        
        session_count = 0
        for m in all_test_index:
            has_label = np.sum(all_test_label[m[0]:m[1]]) != 0
            in_range = m[1] <= len(click_score)
            
            if has_label and in_range:
                session_scores = click_score[m[0]:m[1], 0]
                session_labels = all_test_label[m[0]:m[1]]
                
                pretrain_all_mrr.append(mrr_score(session_labels, session_scores))
                pretrain_all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
                pretrain_all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
                session_count += 1
        
        # 결과 저장 및 출력
        if len(pretrain_all_mrr) > 0:
            pretrain_epoch_results = {
                'MRR': np.mean(pretrain_all_mrr),
                'NDCG@5': np.mean(pretrain_all_ndcg),
                'Hit@1': np.mean(pretrain_all_hit1)
            }
            pretrain_results.append(pretrain_epoch_results)
            
            current_mrr = pretrain_epoch_results['MRR']
            print(f"[Pretraining Epoch {pretrain_ep+1}] 테스트 결과:")
            print(f"  평가된 세션 수: {session_count}개")
            print(f"  MRR      : {current_mrr:.6f}")
            print(f"  NDCG@5   : {pretrain_epoch_results['NDCG@5']:.6f}")
            print(f"  Hit@1    : {pretrain_epoch_results['Hit@1']:.6f}")
            
            # 최고 MRR 갱신 시 모델 저장
            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_mrr_epoch = pretrain_ep + 1
                
                # 모델 저장 디렉토리 생성
                save_dir = os.path.dirname(PRETRAINING_SAVE_PATH)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # 최고 성능 모델 저장
                model.save_weights(PRETRAINING_SAVE_PATH)
                print(f"  → 최고 MRR 갱신! 모델 저장: {PRETRAINING_SAVE_PATH} (MRR: {best_mrr:.6f})")
        else:
            print(f"[Pretraining Epoch {pretrain_ep+1}] 평가 가능한 세션이 없습니다.")
    
    # Pretraining 완료 후 결과 요약 및 모델 저장
    print(f"\n{'='*60}")
    print("Pretraining 완료! 결과 요약")
    print(f"{'='*60}")
    
    if len(pretrain_results) > 0:
        print(f"{'Epoch':<10} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
        print(f"{'-'*60}")
        for i, result in enumerate(pretrain_results, 1):
            print(f"{i:<10} {result['MRR']:<12.6f} {result['NDCG@5']:<12.6f} {result['Hit@1']:<12.6f}")
        
        # 최고 성능 요약 (이미 저장된 모델 정보 출력)
        best_hit1_idx = np.argmax([r['Hit@1'] for r in pretrain_results])
        best_hit1_epoch = best_hit1_idx + 1
        print(f"{'='*60}")
        print(f"Best MRR  : Epoch {best_mrr_epoch} - {best_mrr:.6f} (모델 저장됨)")
        print(f"Best Hit@1: Epoch {best_hit1_epoch} - {pretrain_results[best_hit1_idx]['Hit@1']:.6f}")
    
    print(f"\n{'='*60}")
    print(f"최고 성능 모델 저장 완료: {PRETRAINING_SAVE_PATH}")
    print(f"  - 저장된 모델: Epoch {best_mrr_epoch} (MRR: {best_mrr:.6f})")
    print(f"{'='*60}\n")
    
    # Pretraining 완료 후 프로그램 종료
    print(f"{'='*60}")
    print("Pretraining 완료 후 프로그램을 종료합니다.")
    print(f"{'='*60}")
    import sys
    sys.exit(0)

# ========== Pretraining on 트레이닝셋 유저별 후반 20% (DO_PRETRAINING과 동일 구조, 원본 본문 사용) ==========
elif DO_PRETRAINING_ON_TRAIN20:
    print(f"\n{'='*60}")
    print("Pretraining (유저별 후반 20%): 트레이닝셋 유저별 후반 20% 데이터로 원본 본문 사용")
    print(f"{'='*60}")
    from collections import defaultdict
    user_to_indices = defaultdict(list)
    for i in range(len(all_train_id)):
        uid = all_train_userid_str[i] if all_train_userid_str is not None else i
        user_to_indices[uid].append(i)
    train20_indices = []
    for uid, indices in user_to_indices.items():
        n = len(indices)
        take_count = max(1, int(np.ceil(0.2 * n)))
        train20_indices.extend(indices[-take_count:])
    train20_indices = sorted(train20_indices)
    last20_size = len(train20_indices)
    pretrain20_pn = [all_train_pn[i] for i in train20_indices]
    pretrain20_label = [all_label[i] for i in train20_indices]
    pretrain20_id = [all_train_id[i] for i in train20_indices]
    print(f"전체 학습 샘플 수: {len(all_train_id)}개")
    print(f"Pretraining(유저별 후반 20%%)에 사용할 샘플 수: {last20_size}개, 유저 수: {len(user_to_indices)}명")
    print(f"Pretraining 에폭 수: {PRETRAINING_ON_TRAIN20_EPOCHS}")
    print(f"{'='*60}\n")
    pretrain20_results = []
    best_mrr_t20 = -1.0
    best_mrr_epoch_t20 = -1
    for pretrain_ep in range(PRETRAINING_ON_TRAIN20_EPOCHS):
        np.random.seed(SEED + pretrain_ep)
        random.seed(SEED + pretrain_ep)
        pretrain20_gen = generate_batch_data_train(
            pretrain20_pn, pretrain20_label, pretrain20_id, 30,
            candidate_news_body=None
        )
        steps_per_epoch_t20 = (last20_size + 29) // 30
        print(f"\nPretraining(유저별 후반 20%%) Epoch {pretrain_ep+1}/{PRETRAINING_ON_TRAIN20_EPOCHS} - 샘플 수: {last20_size}개, Steps: {steps_per_epoch_t20}")
        model.fit(pretrain20_gen, epochs=1, steps_per_epoch=steps_per_epoch_t20, verbose=1)
        print(f"\n[Pretraining(유저별 후반 20%%) Epoch {pretrain_ep+1}] 테스트셋 평가 중... (원본 본문 사용)")
        pretrain20_testgen = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30,
            candidate_news_body=None
        )
        test_steps_t20 = len(all_test_id)
        click_score_t20 = model_test.predict(pretrain20_testgen, steps=test_steps_t20, verbose=0)
        pretrain20_all_mrr, pretrain20_all_ndcg, pretrain20_all_hit1 = [], [], []
        session_count_t20 = 0
        for m in all_test_index:
            has_label = np.sum(all_test_label[m[0]:m[1]]) != 0
            in_range = m[1] <= len(click_score_t20)
            if has_label and in_range:
                session_scores = click_score_t20[m[0]:m[1], 0]
                session_labels = all_test_label[m[0]:m[1]]
                pretrain20_all_mrr.append(mrr_score(session_labels, session_scores))
                pretrain20_all_ndcg.append(ndcg_score(session_labels, session_scores, k=5))
                pretrain20_all_hit1.append(hit_at_k(session_labels, session_scores, k=1))
                session_count_t20 += 1
        if len(pretrain20_all_mrr) > 0:
            epoch_res = {
                'MRR': np.mean(pretrain20_all_mrr),
                'NDCG@5': np.mean(pretrain20_all_ndcg),
                'Hit@1': np.mean(pretrain20_all_hit1)
            }
            pretrain20_results.append(epoch_res)
            current_mrr = epoch_res['MRR']
            print(f"[Pretraining(유저별 후반 20%%) Epoch {pretrain_ep+1}] 테스트 결과:")
            print(f"  평가된 세션 수: {session_count_t20}개")
            print(f"  MRR      : {current_mrr:.6f}")
            print(f"  NDCG@5   : {epoch_res['NDCG@5']:.6f}")
            print(f"  Hit@1    : {epoch_res['Hit@1']:.6f}")
            if current_mrr > best_mrr_t20:
                best_mrr_t20 = current_mrr
                best_mrr_epoch_t20 = pretrain_ep + 1
                save_dir_t20 = os.path.dirname(PRETRAINING_ON_TRAIN20_SAVE_PATH)
                if save_dir_t20 and not os.path.exists(save_dir_t20):
                    os.makedirs(save_dir_t20, exist_ok=True)
                model.save_weights(PRETRAINING_ON_TRAIN20_SAVE_PATH)
                print(f"  → 최고 MRR 갱신! 모델 저장: {PRETRAINING_ON_TRAIN20_SAVE_PATH} (MRR: {best_mrr_t20:.6f})")
        else:
            print(f"[Pretraining(유저별 후반 20%%) Epoch {pretrain_ep+1}] 평가 가능한 세션이 없습니다.")
    print(f"\n{'='*60}")
    print("Pretraining(유저별 후반 20%%) 완료! 결과 요약")
    print(f"{'='*60}")
    if len(pretrain20_results) > 0:
        print(f"{'Epoch':<10} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
        print(f"{'-'*60}")
        for i, r in enumerate(pretrain20_results, 1):
            print(f"{i:<10} {r['MRR']:<12.6f} {r['NDCG@5']:<12.6f} {r['Hit@1']:<12.6f}")
        best_hit1_idx_t20 = np.argmax([r['Hit@1'] for r in pretrain20_results])
        print(f"{'='*60}")
        print(f"Best MRR  : Epoch {best_mrr_epoch_t20} - {best_mrr_t20:.6f} (모델 저장됨)")
        print(f"Best Hit@1: Epoch {best_hit1_idx_t20 + 1} - {pretrain20_results[best_hit1_idx_t20]['Hit@1']:.6f}")
    print(f"\n최고 성능 모델 저장 완료: {PRETRAINING_ON_TRAIN20_SAVE_PATH}")
    print(f"{'='*60}\n")
    print("Pretraining(유저별 후반 20%%) 완료 후 프로그램을 종료합니다.")
    import sys
    sys.exit(0)

# ========== 유저 인코더만 파인튜닝 (뉴스 인코더 고정, 트레이닝 뒤 20% + train20_0 기대본문) ==========
if FINETUNE_USER_ENCODER:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    finetune_body_dir = os.path.join(body_gen_output, FINETUNE_EXPECTED_BODY_DIR)
    if not os.path.isdir(finetune_body_dir):
        print(f"오류: 기대본문 폴더를 찾을 수 없습니다: {finetune_body_dir}")
        sys.exit(1)
    print(f"\n{'='*60}")
    print("유저 인코더만 파인튜닝 (뉴스 인코더 고정)")
    print(f"{'='*60}")
    print(f"모델 로드: {PRETRAINED_MODEL_PATH}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    print("뉴스 인코더 고정 (newsEncoder.trainable = False)")
    newsEncoder.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=FINETUNE_LR), metrics=['acc'])
    n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"파인튜닝 학습률: {FINETUNE_LR} (Adam learning_rate), 학습 가능 파라미터: {n_trainable}개")
    print("기대본문 로드:", finetune_body_dir)
    expected_bodies_finetune = load_expected_bodies_from_train_dir(finetune_body_dir)
    print(f"로드된 기대 본문: {len(expected_bodies_finetune)}개")
    pretrain_size = int(len(all_train_id) * 0.8)
    last20_size = len(all_train_id) - pretrain_size
    train20_pn = all_train_pn[pretrain_size:]
    train20_label = all_label[pretrain_size:]
    train20_id = all_train_id[pretrain_size:]
    train20_user_pos = all_user_pos[pretrain_size:]
    train20_userid_str = all_train_userid_str[pretrain_size:] if all_train_userid_str is not None else None
    train20_newsid_str = all_train_newsid_str[pretrain_size:] if all_train_newsid_str is not None else None
    print(f"트레이닝 뒤 20% 샘플 수: {last20_size}개")
    # 파인튜닝 구간 (user, news) 키 중 기대본문 매칭 개수
    if expected_bodies_finetune and train20_userid_str is not None and train20_newsid_str is not None:
        need_keys = set()
        for i in range(len(train20_userid_str)):
            u = train20_userid_str[i]
            ids = train20_newsid_str[i]
            if hasattr(ids, '__len__') and not isinstance(ids, str):
                for j in range(len(ids)):
                    need_keys.add(_norm_expected_body_key(u, ids[j] if j < len(ids) else ''))
            else:
                need_keys.add(_norm_expected_body_key(u, ids))
        need_keys.discard(('', ''))
        matched = sum(1 for k in need_keys if k in expected_bodies_finetune)
        print(f"파인튜닝 매칭: 필요 (user,news) 키 {len(need_keys)}개 중 기대본문 존재 {matched}개 ({100.0*matched/len(need_keys):.1f}%)" if need_keys else "파인튜닝 매칭: 필요 키 없음")
    finetune_gen = generate_batch_data_train(
        train20_pn, train20_label, train20_id, 30,
        candidate_news_body=None,
        expected_bodies=expected_bodies_finetune,
        all_userid_str=train20_userid_str,
        all_newsid_str=train20_newsid_str,
        news_index_reverse=news_index_reverse,
        use_expected_body_positive_only=True  # 유저 인코더 파인튜닝: 정답 후보만 기대본문 사용
    )
    steps_per_epoch = (last20_size + 29) // 30
    # 테스트셋 기대본문 로드 (매 에폭 기대본문 NDCG 평가용)
    test_body_dir = os.path.join(body_gen_output, FINETUNE_TESTSET_EXPECTED_BODY_DIR)
    expected_bodies_test_finetune = None
    if os.path.isdir(test_body_dir):
        expected_bodies_test_finetune = load_expected_bodies_from_train_dir(test_body_dir)
        print(f"테스트셋 기대본문 로드: {FINETUNE_TESTSET_EXPECTED_BODY_DIR} ({len(expected_bodies_test_finetune)}개)")
    else:
        print(f"경고: 테스트셋 기대본문 폴더 없음 ({test_body_dir}). 기대본문 NDCG 기준 저장 불가, 마지막 에폭만 저장합니다.")
    
    def _eval_testset(use_expected_body, expected_bodies=None):
        if use_expected_body and (expected_bodies is None or all_test_userid_str is None or all_test_newsid_str is None):
            return None
        testgen = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse
        )
        click_score = model_test.predict(testgen, steps=len(all_test_id), verbose=0)
        eps = 1e-7
        all_ndcg, all_mrr, all_hit1, all_loss = [], [], [], []
        for m in all_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                continue
            labels = all_test_label[s:e]
            if np.sum(labels) == 0:
                continue
            scores = click_score[s:e, 0]
            all_ndcg.append(ndcg_score(labels, scores, k=5))
            all_mrr.append(mrr_score(labels, scores))
            all_hit1.append(hit_at_k(labels, scores, k=1))
            labels_f = labels.astype(np.float32)
            scores_clip = np.clip(scores, eps, 1 - eps)
            all_loss.append(-np.mean(labels_f * np.log(scores_clip) + (1 - labels_f) * np.log(1 - scores_clip)))
        if not all_ndcg:
            return None
        return {'NDCG@5': np.mean(all_ndcg), 'MRR': np.mean(all_mrr), 'Hit@1': np.mean(all_hit1), 'Loss': np.mean(all_loss)}
    
    print(f"파인튜닝 에폭 수: {FINETUNE_EPOCHS}, steps_per_epoch: {steps_per_epoch}")
    print("매 에폭 테스트셋 평가(실제/기대 본문) 후 기대본문 NDCG@5 최고 모델 저장")
    print(f"{'='*60}\n")
    best_ndcg_expected = -1.0
    best_epoch = -1
    for epoch in range(FINETUNE_EPOCHS):
        model.fit(finetune_gen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
        res_actual = _eval_testset(use_expected_body=False)
        res_expected = _eval_testset(use_expected_body=True, expected_bodies=expected_bodies_test_finetune) if expected_bodies_test_finetune else None
        print(f"\n[에폭 {epoch+1}/{FINETUNE_EPOCHS}] 테스트셋 — 실제본문: NDCG@5={res_actual['NDCG@5']:.4f}, MRR={res_actual['MRR']:.4f}, Hit@1={res_actual['Hit@1']:.4f}, Loss={res_actual['Loss']:.4f}")
        if res_expected is not None:
            print(f"                테스트셋 — 기대본문: NDCG@5={res_expected['NDCG@5']:.4f}, MRR={res_expected['MRR']:.4f}, Hit@1={res_expected['Hit@1']:.4f}, Loss={res_expected['Loss']:.4f}")
            if res_expected['NDCG@5'] > best_ndcg_expected and FINETUNE_SAVE_PATH:
                best_ndcg_expected = res_expected['NDCG@5']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(FINETUNE_SAVE_PATH) or '.', exist_ok=True)
                model.save_weights(FINETUNE_SAVE_PATH)
                print(f"                → 기대본문 NDCG@5 최고 갱신 ({best_ndcg_expected:.4f}), 모델 저장: {FINETUNE_SAVE_PATH}")
        else:
            if res_actual['NDCG@5'] > best_ndcg_expected and FINETUNE_SAVE_PATH and expected_bodies_test_finetune is None:
                best_ndcg_expected = res_actual['NDCG@5']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(FINETUNE_SAVE_PATH) or '.', exist_ok=True)
                model.save_weights(FINETUNE_SAVE_PATH)
                print(f"                → (기대본문 없음) 실제본문 NDCG@5 기준 저장")
    if FINETUNE_SAVE_PATH and best_epoch >= 0:
        print(f"\n기대본문 NDCG@5 최고 모델: 에폭 {best_epoch}, NDCG@5={best_ndcg_expected:.4f} — {FINETUNE_SAVE_PATH}")
    print("유저 인코더 파인튜닝 완료. 프로그램을 종료합니다.")
    sys.exit(0)

# ========== 뉴스 인코더만 파인튜닝 (유저 쪽 attention 등 고정, 트레이닝 뒤 20% + 기대본문) ==========
if FINETUNE_NEWS_ENCODER:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    finetune_body_dir = os.path.join(body_gen_output, FINETUNE_EXPECTED_BODY_DIR)
    if not os.path.isdir(finetune_body_dir):
        print(f"오류: 기대본문 폴더를 찾을 수 없습니다: {finetune_body_dir}")
        sys.exit(1)
    print(f"\n{'='*60}")
    print("뉴스 인코더만 파인튜닝 (유저 쪽 attention 등 고정)")
    print(f"{'='*60}")
    print(f"모델 로드: {PRETRAINED_MODEL_PATH}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    # 유저 쪽 고정: 전체 레이어 비학습 후 newsEncoder 레이어만 학습 (서브모델도 trainable=True로 설정해야 실제 갱신됨)
    for layer in model.layers:
        layer.trainable = False
    for layer in newsEncoder.layers:
        layer.trainable = True
    newsEncoder.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=FINETUNE_LR), metrics=['acc'])
    n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"파인튜닝 학습률: {FINETUNE_LR} (Adam learning_rate), 학습 가능 파라미터: {n_trainable}개")
    print("기대본문 로드:", finetune_body_dir)
    expected_bodies_finetune = load_expected_bodies_from_train_dir(finetune_body_dir)
    print(f"로드된 기대 본문: {len(expected_bodies_finetune)}개")
    pretrain_size = int(len(all_train_id) * 0.8)
    last20_size = len(all_train_id) - pretrain_size
    train20_pn = all_train_pn[pretrain_size:]
    train20_label = all_label[pretrain_size:]
    train20_id = all_train_id[pretrain_size:]
    train20_user_pos = all_user_pos[pretrain_size:]
    train20_userid_str = all_train_userid_str[pretrain_size:] if all_train_userid_str is not None else None
    train20_newsid_str = all_train_newsid_str[pretrain_size:] if all_train_newsid_str is not None else None
    print(f"트레이닝 뒤 20% 샘플 수: {last20_size}개")
    if expected_bodies_finetune and train20_userid_str is not None and train20_newsid_str is not None:
        need_keys = set()
        for i in range(len(train20_userid_str)):
            u = train20_userid_str[i]
            ids = train20_newsid_str[i]
            if hasattr(ids, '__len__') and not isinstance(ids, str):
                for j in range(len(ids)):
                    need_keys.add(_norm_expected_body_key(u, ids[j] if j < len(ids) else ''))
            else:
                need_keys.add(_norm_expected_body_key(u, ids))
        need_keys.discard(('', ''))
        matched = sum(1 for k in need_keys if k in expected_bodies_finetune)
        print(f"파인튜닝 매칭: 필요 (user,news) 키 {len(need_keys)}개 중 기대본문 존재 {matched}개 ({100.0*matched/len(need_keys):.1f}%)" if need_keys else "파인튜닝 매칭: 필요 키 없음")
    finetune_gen = generate_batch_data_train(
        train20_pn, train20_label, train20_id, 30,
        candidate_news_body=None,
        expected_bodies=expected_bodies_finetune,
        all_userid_str=train20_userid_str,
        all_newsid_str=train20_newsid_str,
        news_index_reverse=news_index_reverse
    )
    steps_per_epoch = (last20_size + 29) // 30
    test_body_dir = os.path.join(body_gen_output, FINETUNE_TESTSET_EXPECTED_BODY_DIR)
    expected_bodies_test_finetune = None
    if os.path.isdir(test_body_dir):
        expected_bodies_test_finetune = load_expected_bodies_from_train_dir(test_body_dir)
        print(f"테스트셋 기대본문 로드: {FINETUNE_TESTSET_EXPECTED_BODY_DIR} ({len(expected_bodies_test_finetune)}개)")
    else:
        print(f"경고: 테스트셋 기대본문 폴더 없음 ({test_body_dir}). 기대본문 NDCG 기준 저장 불가, 마지막 에폭만 저장합니다.")

    def _eval_testset_news(use_expected_body, expected_bodies=None):
        if use_expected_body and (expected_bodies is None or all_test_userid_str is None or all_test_newsid_str is None):
            return None
        testgen = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse
        )
        click_score = model_test.predict(testgen, steps=len(all_test_id), verbose=0)
        eps = 1e-7
        all_ndcg, all_mrr, all_hit1, all_loss = [], [], [], []
        for m in all_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                continue
            labels = all_test_label[s:e]
            if np.sum(labels) == 0:
                continue
            scores = click_score[s:e, 0]
            all_ndcg.append(ndcg_score(labels, scores, k=5))
            all_mrr.append(mrr_score(labels, scores))
            all_hit1.append(hit_at_k(labels, scores, k=1))
            labels_f = labels.astype(np.float32)
            scores_clip = np.clip(scores, eps, 1 - eps)
            all_loss.append(-np.mean(labels_f * np.log(scores_clip) + (1 - labels_f) * np.log(1 - scores_clip)))
        if not all_ndcg:
            return None
        return {'NDCG@5': np.mean(all_ndcg), 'MRR': np.mean(all_mrr), 'Hit@1': np.mean(all_hit1), 'Loss': np.mean(all_loss)}

    print(f"파인튜닝 에폭 수: {FINETUNE_EPOCHS}, steps_per_epoch: {steps_per_epoch}")
    print("매 에폭 테스트셋 평가(실제/기대 본문) 후 기대본문 NDCG@5 최고 모델 저장")
    print(f"{'='*60}\n")
    best_ndcg_expected = -1.0
    best_epoch = -1
    for epoch in range(FINETUNE_EPOCHS):
        model.fit(finetune_gen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
        res_actual = _eval_testset_news(use_expected_body=False)
        res_expected = _eval_testset_news(use_expected_body=True, expected_bodies=expected_bodies_test_finetune) if expected_bodies_test_finetune else None
        print(f"\n[에폭 {epoch+1}/{FINETUNE_EPOCHS}] 테스트셋 — 실제본문: NDCG@5={res_actual['NDCG@5']:.4f}, MRR={res_actual['MRR']:.4f}, Hit@1={res_actual['Hit@1']:.4f}, Loss={res_actual['Loss']:.4f}")
        if res_expected is not None:
            print(f"                테스트셋 — 기대본문: NDCG@5={res_expected['NDCG@5']:.4f}, MRR={res_expected['MRR']:.4f}, Hit@1={res_expected['Hit@1']:.4f}, Loss={res_expected['Loss']:.4f}")
            if res_expected['NDCG@5'] > best_ndcg_expected and FINETUNE_NEWS_ENCODER_SAVE_PATH:
                best_ndcg_expected = res_expected['NDCG@5']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(FINETUNE_NEWS_ENCODER_SAVE_PATH) or '.', exist_ok=True)
                model.save_weights(FINETUNE_NEWS_ENCODER_SAVE_PATH)
                print(f"                → 기대본문 NDCG@5 최고 갱신 ({best_ndcg_expected:.4f}), 모델 저장: {FINETUNE_NEWS_ENCODER_SAVE_PATH}")
        else:
            if res_actual['NDCG@5'] > best_ndcg_expected and FINETUNE_NEWS_ENCODER_SAVE_PATH and expected_bodies_test_finetune is None:
                best_ndcg_expected = res_actual['NDCG@5']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(FINETUNE_NEWS_ENCODER_SAVE_PATH) or '.', exist_ok=True)
                model.save_weights(FINETUNE_NEWS_ENCODER_SAVE_PATH)
                print(f"                → (기대본문 없음) 실제본문 NDCG@5 기준 저장")
    if FINETUNE_NEWS_ENCODER_SAVE_PATH and best_epoch >= 0:
        print(f"\n기대본문 NDCG@5 최고 모델: 에폭 {best_epoch}, NDCG@5={best_ndcg_expected:.4f} — {FINETUNE_NEWS_ENCODER_SAVE_PATH}")
    print("뉴스 인코더 파인튜닝 완료. 프로그램을 종료합니다.")
    sys.exit(0)

# ========== 모델 전체 파인튜닝 (유저+뉴스 인코더 모두 학습, 트레이닝 뒤 20% + 기대본문) ==========
if FINETUNE_FULL_MODEL:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    finetune_body_dir = os.path.join(body_gen_output, FINETUNE_EXPECTED_BODY_DIR)
    if not os.path.isdir(finetune_body_dir):
        print(f"오류: 기대본문 폴더를 찾을 수 없습니다: {finetune_body_dir}")
        sys.exit(1)
    print(f"\n{'='*60}")
    print("모델 전체 파인튜닝 (유저 인코더 + 뉴스 인코더 모두 학습)")
    print(f"{'='*60}")
    print(f"모델 로드: {PRETRAINED_MODEL_PATH}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=FINETUNE_LR), metrics=['acc'])
    n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"파인튜닝 학습률: {FINETUNE_LR} (Adam learning_rate), 학습 가능 파라미터: {n_trainable}개")
    print("기대본문 로드:", finetune_body_dir)
    expected_bodies_finetune = load_expected_bodies_from_train_dir(finetune_body_dir)
    print(f"로드된 기대 본문: {len(expected_bodies_finetune)}개")
    pretrain_size = int(len(all_train_id) * 0.8)
    last20_size = len(all_train_id) - pretrain_size
    train20_pn = all_train_pn[pretrain_size:]
    train20_label = all_label[pretrain_size:]
    train20_id = all_train_id[pretrain_size:]
    train20_user_pos = all_user_pos[pretrain_size:]
    train20_userid_str = all_train_userid_str[pretrain_size:] if all_train_userid_str is not None else None
    train20_newsid_str = all_train_newsid_str[pretrain_size:] if all_train_newsid_str is not None else None
    print(f"트레이닝 뒤 20% 샘플 수: {last20_size}개")
    if expected_bodies_finetune and train20_userid_str is not None and train20_newsid_str is not None:
        need_keys = set()
        for i in range(len(train20_userid_str)):
            u = train20_userid_str[i]
            ids = train20_newsid_str[i]
            if hasattr(ids, '__len__') and not isinstance(ids, str):
                for j in range(len(ids)):
                    need_keys.add(_norm_expected_body_key(u, ids[j] if j < len(ids) else ''))
            else:
                need_keys.add(_norm_expected_body_key(u, ids))
        need_keys.discard(('', ''))
        matched = sum(1 for k in need_keys if k in expected_bodies_finetune)
        print(f"파인튜닝 매칭: 필요 (user,news) 키 {len(need_keys)}개 중 기대본문 존재 {matched}개 ({100.0*matched/len(need_keys):.1f}%)" if need_keys else "파인튜닝 매칭: 필요 키 없음")
    finetune_gen = generate_batch_data_train(
        train20_pn, train20_label, train20_id, 30,
        candidate_news_body=None,
        expected_bodies=expected_bodies_finetune,
        all_userid_str=train20_userid_str,
        all_newsid_str=train20_newsid_str,
        news_index_reverse=news_index_reverse
    )
    steps_per_epoch = (last20_size + 29) // 30
    test_body_dir = os.path.join(body_gen_output, FINETUNE_TESTSET_EXPECTED_BODY_DIR)
    expected_bodies_test_finetune = None
    if os.path.isdir(test_body_dir):
        expected_bodies_test_finetune = load_expected_bodies_from_train_dir(test_body_dir)
        print(f"테스트셋 기대본문 로드: {FINETUNE_TESTSET_EXPECTED_BODY_DIR} ({len(expected_bodies_test_finetune)}개)")
    else:
        print(f"경고: 테스트셋 기대본문 폴더 없음 ({test_body_dir}). 기대본문 NDCG 기준 저장 불가, 마지막 에폭만 저장합니다.")

    def _eval_testset_full(use_expected_body, expected_bodies=None):
        if use_expected_body and (expected_bodies is None or all_test_userid_str is None or all_test_newsid_str is None):
            return None
        testgen = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_test_userid_str,
            all_newsid_str=all_test_newsid_str,
            news_index_reverse=news_index_reverse
        )
        click_score = model_test.predict(testgen, steps=len(all_test_id), verbose=0)
        eps = 1e-7
        all_ndcg, all_mrr, all_hit1, all_loss = [], [], [], []
        for m in all_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                continue
            labels = all_test_label[s:e]
            if np.sum(labels) == 0:
                continue
            scores = click_score[s:e, 0]
            all_ndcg.append(ndcg_score(labels, scores, k=5))
            all_mrr.append(mrr_score(labels, scores))
            all_hit1.append(hit_at_k(labels, scores, k=1))
            labels_f = labels.astype(np.float32)
            scores_clip = np.clip(scores, eps, 1 - eps)
            all_loss.append(-np.mean(labels_f * np.log(scores_clip) + (1 - labels_f) * np.log(1 - scores_clip)))
        if not all_ndcg:
            return None
        return {'NDCG@5': np.mean(all_ndcg), 'MRR': np.mean(all_mrr), 'Hit@1': np.mean(all_hit1), 'Loss': np.mean(all_loss)}

    print(f"파인튜닝 에폭 수: {FINETUNE_EPOCHS}, steps_per_epoch: {steps_per_epoch}")
    print("매 에폭 테스트셋 평가(실제/기대 본문) 후 기대본문 NDCG@5 최고 모델 저장")
    print(f"{'='*60}\n")
    best_ndcg_expected = -1.0
    best_epoch = -1
    for epoch in range(FINETUNE_EPOCHS):
        model.fit(finetune_gen, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
        res_actual = _eval_testset_full(use_expected_body=False)
        res_expected = _eval_testset_full(use_expected_body=True, expected_bodies=expected_bodies_test_finetune) if expected_bodies_test_finetune else None
        print(f"\n[에폭 {epoch+1}/{FINETUNE_EPOCHS}] 테스트셋 — 실제본문: NDCG@5={res_actual['NDCG@5']:.4f}, MRR={res_actual['MRR']:.4f}, Hit@1={res_actual['Hit@1']:.4f}, Loss={res_actual['Loss']:.4f}")
        if res_expected is not None:
            print(f"                테스트셋 — 기대본문: NDCG@5={res_expected['NDCG@5']:.4f}, MRR={res_expected['MRR']:.4f}, Hit@1={res_expected['Hit@1']:.4f}, Loss={res_expected['Loss']:.4f}")
            if res_expected['NDCG@5'] > best_ndcg_expected and FINETUNE_FULL_SAVE_PATH:
                best_ndcg_expected = res_expected['NDCG@5']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(FINETUNE_FULL_SAVE_PATH) or '.', exist_ok=True)
                model.save_weights(FINETUNE_FULL_SAVE_PATH)
                print(f"                → 기대본문 NDCG@5 최고 갱신 ({best_ndcg_expected:.4f}), 모델 저장: {FINETUNE_FULL_SAVE_PATH}")
        else:
            if res_actual['NDCG@5'] > best_ndcg_expected and FINETUNE_FULL_SAVE_PATH and expected_bodies_test_finetune is None:
                best_ndcg_expected = res_actual['NDCG@5']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(FINETUNE_FULL_SAVE_PATH) or '.', exist_ok=True)
                model.save_weights(FINETUNE_FULL_SAVE_PATH)
                print(f"                → (기대본문 없음) 실제본문 NDCG@5 기준 저장")
    if FINETUNE_FULL_SAVE_PATH and best_epoch >= 0:
        print(f"\n기대본문 NDCG@5 최고 모델: 에폭 {best_epoch}, NDCG@5={best_ndcg_expected:.4f} — {FINETUNE_FULL_SAVE_PATH}")
    print("모델 전체 파인튜닝 완료. 프로그램을 종료합니다.")
    sys.exit(0)

# ========== 프리트레이닝 모델 로드 후 트레이닝 80% 지정 배치만 평가 → result{N}.txt ==========
if EVAL_PRETRAINED_ON_TRAIN80_FIRST_BATCH:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    batch_dir_name = f"train80_batch{EVAL_TRAIN80_BATCH_INDEX}"
    batch_dir = os.path.join(body_gen_output, batch_dir_name)
    if not os.path.isdir(batch_dir):
        print(f"오류: 배치 기대본문 폴더가 없습니다: {batch_dir}")
        print(f"  coordinator_LLM/output/{EVAL_TRAIN80_BATCH_INDEX}.txt 정책으로 train80 배치 {EVAL_TRAIN80_BATCH_INDEX} 기대본문 생성 후 다시 실행하세요.")
        sys.exit(1)
    print(f"\n{'='*60}")
    print(f"프리트레이닝 모델 로드: {PRETRAINED_MODEL_PATH}")
    print(f"{'='*60}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    print("모델 로드 완료.")
    pretrain_size = int(len(all_train_id) * 0.8)
    batch_start = EVAL_TRAIN80_BATCH_INDEX * EVAL_TRAIN80_BATCH_SIZE
    if batch_start >= pretrain_size:
        print(f"오류: 배치 인덱스 {EVAL_TRAIN80_BATCH_INDEX}가 범위 초과 (train80 세션 수: {pretrain_size})")
        sys.exit(1)
    n_sessions = min(EVAL_TRAIN80_BATCH_SIZE, pretrain_size - batch_start)
    print(f"트레이닝 80% 배치 {EVAL_TRAIN80_BATCH_INDEX}: 세션 {batch_start}~{batch_start + n_sessions - 1} ({n_sessions}개), 기대본문: {batch_dir_name}")
    train80_test_pn = []
    train80_test_label = []
    train80_test_id = []
    train80_test_user_pos = []
    train80_test_index = []
    train80_test_userid_str = []
    train80_test_newsid_str = []
    for i in range(batch_start, batch_start + n_sessions):
        start = len(train80_test_pn)
        pn_row = all_train_pn[i]
        label_row = all_label[i]
        n_cand = len(pn_row) if hasattr(pn_row, '__len__') else 5
        for j in range(n_cand):
            train80_test_pn.append(int(pn_row[j]) if hasattr(pn_row[j], '__int__') else pn_row[j])
            train80_test_label.append(int(label_row[j]) if hasattr(label_row[j], '__int__') else label_row[j])
            train80_test_id.append(all_train_id[i])
            train80_test_user_pos.append(all_user_pos[i])
        if all_train_userid_str is not None and all_train_newsid_str is not None:
            for j in range(n_cand):
                train80_test_userid_str.append(all_train_userid_str[i])
                train80_test_newsid_str.append(all_train_newsid_str[i][j] if j < len(all_train_newsid_str[i]) else '')
        train80_test_index.append([start, len(train80_test_pn)])
    train80_test_pn = np.array(train80_test_pn, dtype='int32')
    train80_test_label = np.array(train80_test_label, dtype='int32')
    train80_test_id = np.array(train80_test_id, dtype='int32')
    train80_test_user_pos = np.array(train80_test_user_pos, dtype='int32')
    expected_bodies_batch = load_expected_bodies_from_train_dir(batch_dir)
    print(f"로드된 기대 본문: {len(expected_bodies_batch)}개")

    def eval_train80_batch_run(use_expected_body, expected_bodies=None, all_userid_str=None, all_newsid_str=None):
        if use_expected_body and (expected_bodies is None or all_userid_str is None or all_newsid_str is None):
            return None, None, None
        testgen = generate_batch_data_test(
            train80_test_pn, train80_test_label, train80_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_userid_str,
            all_newsid_str=all_newsid_str,
            news_index_reverse=news_index_reverse,
            all_test_user_pos_override=train80_test_user_pos
        )
        steps = len(train80_test_id)
        click_score = model_test.predict(testgen, steps=steps, verbose=0)
        eps = 1e-7
        all_session_loss = []
        all_session_ndcg = []
        for m in train80_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                continue
            labels = train80_test_label[s:e].astype(np.float32)
            if np.sum(labels) == 0:
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                continue
            scores = np.clip(click_score[s:e, 0], eps, 1 - eps)
            session_bce = -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
            all_session_loss.append(float(session_bce))
            all_session_ndcg.append(ndcg_score(train80_test_label[s:e], click_score[s:e, 0], k=5))
        valid_loss = [x for x in all_session_loss if x is not None]
        if not valid_loss:
            return None, None, None
        return all_session_loss, all_session_ndcg, click_score

    lines = []
    def add(s=""):
        lines.append(s)
        print(s)
    add(f"\n트레이닝셋 전반부 80% 배치 {EVAL_TRAIN80_BATCH_INDEX} 평가 (샘플 수: {n_sessions}, 테스트 행: {len(train80_test_id)})")
    add(f"모델 경로: {PRETRAINED_MODEL_PATH}, 기대본문: {batch_dir_name}")
    add("\n[1] 실제 본문 사용:")
    loss_actual_list, ndcg_actual_list, click_actual = eval_train80_batch_run(use_expected_body=False)
    if loss_actual_list is not None:
        valid = [x for x in loss_actual_list if x is not None]
        add(f"  Loss     : {np.mean(valid):.6f}")
        add(f"  NDCG@5   : {np.mean([x for x in ndcg_actual_list if x is not None]):.6f}")
    else:
        add("  평가 가능한 세션 없음")
    add("\n[2] 기대 본문 사용:")
    loss_expected_list, ndcg_expected_list, click_expected = eval_train80_batch_run(
        use_expected_body=True,
        expected_bodies=expected_bodies_batch,
        all_userid_str=train80_test_userid_str,
        all_newsid_str=train80_test_newsid_str
    )
    if loss_expected_list is not None:
        valid = [x for x in loss_expected_list if x is not None]
        add(f"  Loss     : {np.mean(valid):.6f}")
        add(f"  NDCG@5   : {np.mean([x for x in ndcg_expected_list if x is not None]):.6f}")
    else:
        add("  [건너뜀] 기대본문 데이터 없음")

    best_sess_idx = None
    success_sess_idx = None
    max_diff = -np.inf
    min_diff = np.inf
    if loss_actual_list is not None and loss_expected_list is not None:
        candidates_max = []
        candidates_min = []
        for i in range(len(train80_test_index)):
            la, le = loss_actual_list[i], loss_expected_list[i]
            if la is None or le is None:
                continue
            diff = le - la
            if diff > max_diff:
                max_diff = diff
                candidates_max = [(diff, i)]
            elif diff == max_diff:
                candidates_max.append((diff, i))
            if diff < min_diff:
                min_diff = diff
                candidates_min = [(diff, i)]
            elif diff == min_diff:
                candidates_min.append((diff, i))
        if candidates_max:
            best_sess_idx = random.choice([c[1] for c in candidates_max])
            add(f"\n최대 diff 세션 수: {len(candidates_max)}개 (failure 1개 선택)")
        if candidates_min:
            success_sess_idx = random.choice([c[1] for c in candidates_min])
            add(f"최소 diff 세션 수: {len(candidates_min)}개 (success 1개 선택)")

    loss_real = np.mean([x for x in loss_actual_list if x is not None]) if loss_actual_list else None
    loss_expected = np.mean([x for x in loss_expected_list if x is not None]) if loss_expected_list else None
    ndcg5_real = np.mean([x for x in ndcg_actual_list if x is not None]) if ndcg_actual_list else None
    ndcg5_expected = np.mean([x for x in ndcg_expected_list if x is not None]) if ndcg_expected_list else None

    diagnostic_samples = []
    if loss_actual_list is not None and loss_expected_list is not None and (best_sess_idx is not None or success_sess_idx is not None):
        news_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'MIND', 'MIND_news.tsv')
        news_titles = {}
        if os.path.exists(news_file):
            with open(news_file, 'r', encoding='utf-8') as nf:
                for nline in nf:
                    parts = nline.strip().split('\t')
                    if len(parts) >= 4:
                        news_titles[parts[0]] = parts[3]
        for sess_idx, type_str in [(best_sess_idx, "failure"), (success_sess_idx, "success")]:
            if sess_idx is None:
                continue
            global_idx = batch_start + sess_idx
            sess_news_ids = all_train_newsid_str[global_idx] if global_idx < len(all_train_newsid_str) else ['?'] * 5
            sess_user_id_str = all_train_userid_str[global_idx] if all_train_userid_str and global_idx < len(all_train_userid_str) else '?'
            s, e = train80_test_index[sess_idx][0], train80_test_index[sess_idx][1]
            sess_labels = train80_test_label[s:e]
            pos_pos = int(np.where(sess_labels == 1)[0][0])
            positive_news_id = sess_news_ids[pos_pos] if pos_pos < len(sess_news_ids) else '?'
            user_pos_indices = all_user_pos[global_idx]
            fallback_click_titles = [news_titles.get(news_index_reverse.get(int(i), ''), '') for i in user_pos_indices if int(i) != 0]
            fallback_click_titles = fallback_click_titles[-10:]
            candidate_news_title = news_titles.get(positive_news_id, '')
            generated_expected_body, json_user_history = load_expected_body_from_train_dir(batch_dir, sess_user_id_str, positive_news_id)
            user_click_history_titles = json_user_history if json_user_history is not None else fallback_click_titles
            if generated_expected_body == '' and expected_bodies_batch:
                generated_expected_body = expected_bodies_batch.get((sess_user_id_str, positive_news_id), '') or ''
            diagnostic_samples.append({
                "type": type_str,
                "user_click_history_titles": user_click_history_titles,
                "candidate_news_title": candidate_news_title,
                "generated_expected_body": generated_expected_body
            })

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'result{EVAL_TRAIN80_BATCH_INDEX}.txt')
    payload = {
        "performance_feedback": {
            "loss_expected": float(loss_expected) if loss_expected is not None else None,
            "loss_real": float(loss_real) if loss_real is not None else None,
            "ndcg5_expected": float(ndcg5_expected) if ndcg5_expected is not None else None,
            "ndcg5_real": float(ndcg5_real) if ndcg5_real is not None else None
        },
        "diagnostic_samples": diagnostic_samples
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
    add(f"\n프리트레이닝 모델 - 트레이닝 80% 배치 {EVAL_TRAIN80_BATCH_INDEX} 평가 완료. 프로그램을 종료합니다.")
    sys.exit(0)

# ========== 프리트레이닝 모델 로드 후 유저별 후반 20% 지정 배치만 평가 → result{N}.txt ==========
if EVAL_PRETRAINED_ON_TRAIN20_FIRST_BATCH:
    import sys
    from collections import defaultdict
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    body_gen_output_t20 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    batch_dir_name_t20 = f"train20_batch{EVAL_TRAIN20_BATCH_INDEX}"
    batch_dir_t20 = os.path.join(body_gen_output_t20, batch_dir_name_t20)
    if not os.path.isdir(batch_dir_t20):
        print(f"오류: 배치 기대본문 폴더가 없습니다: {batch_dir_t20}")
        print(f"  body_generation으로 --train20_only --train20_per_user --train20_first_k {EVAL_TRAIN20_BATCH_SIZE} --train20_batch_index {EVAL_TRAIN20_BATCH_INDEX} 실행 후 다시 시도하세요.")
        sys.exit(1)
    print(f"\n{'='*60}")
    print(f"프리트레이닝 모델 로드: {PRETRAINED_MODEL_PATH}")
    print(f"{'='*60}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    print("모델 로드 완료.")
    # 유저별 후반 20% 세션 인덱스 (NAML/body_generation과 동일)
    user_to_indices_t20 = defaultdict(list)
    for i in range(len(all_train_id)):
        uid = all_train_userid_str[i] if all_train_userid_str is not None else i
        user_to_indices_t20[uid].append(i)
    train20_indices_eval = []
    for uid, indices in user_to_indices_t20.items():
        n = len(indices)
        take_count = max(1, int(np.ceil(0.2 * n)))
        train20_indices_eval.extend(indices[-take_count:])
    train20_indices_eval = sorted(train20_indices_eval)
    total_t20 = len(train20_indices_eval)
    batch_start_t20 = EVAL_TRAIN20_BATCH_INDEX * EVAL_TRAIN20_BATCH_SIZE
    if batch_start_t20 >= total_t20:
        print(f"오류: 배치 인덱스 {EVAL_TRAIN20_BATCH_INDEX}가 범위 초과 (유저별 후반 20%% 세션 수: {total_t20})")
        sys.exit(1)
    n_sessions_t20 = min(EVAL_TRAIN20_BATCH_SIZE, total_t20 - batch_start_t20)
    batch_indices_t20 = train20_indices_eval[batch_start_t20:batch_start_t20 + n_sessions_t20]
    print(f"유저별 후반 20%% 배치 {EVAL_TRAIN20_BATCH_INDEX}: 세션 {batch_start_t20}~{batch_start_t20 + n_sessions_t20 - 1} ({n_sessions_t20}개), 기대본문: {batch_dir_name_t20}")
    train20_test_pn = []
    train20_test_label = []
    train20_test_id = []
    train20_test_user_pos = []
    train20_test_index = []
    train20_test_userid_str = []
    train20_test_newsid_str = []
    for idx in batch_indices_t20:
        start = len(train20_test_pn)
        pn_row = all_train_pn[idx]
        label_row = all_label[idx]
        n_cand = len(pn_row) if hasattr(pn_row, '__len__') else 5
        for j in range(n_cand):
            train20_test_pn.append(int(pn_row[j]) if hasattr(pn_row[j], '__int__') else pn_row[j])
            train20_test_label.append(int(label_row[j]) if hasattr(label_row[j], '__int__') else label_row[j])
            train20_test_id.append(all_train_id[idx])
            train20_test_user_pos.append(all_user_pos[idx])
        if all_train_userid_str is not None and all_train_newsid_str is not None:
            for j in range(n_cand):
                train20_test_userid_str.append(all_train_userid_str[idx])
                train20_test_newsid_str.append(all_train_newsid_str[idx][j] if j < len(all_train_newsid_str[idx]) else '')
        train20_test_index.append([start, len(train20_test_pn)])
    train20_test_pn = np.array(train20_test_pn, dtype='int32')
    train20_test_label = np.array(train20_test_label, dtype='int32')
    train20_test_id = np.array(train20_test_id, dtype='int32')
    train20_test_user_pos = np.array(train20_test_user_pos, dtype='int32')
    expected_bodies_batch_t20 = load_expected_bodies_from_train_dir(batch_dir_t20)
    print(f"로드된 기대 본문: {len(expected_bodies_batch_t20)}개")

    def eval_train20_batch_run(use_expected_body, expected_bodies=None, all_userid_str=None, all_newsid_str=None):
        if use_expected_body and (expected_bodies is None or all_userid_str is None or all_newsid_str is None):
            return None, None, None
        testgen = generate_batch_data_test(
            train20_test_pn, train20_test_label, train20_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_userid_str,
            all_newsid_str=all_newsid_str,
            news_index_reverse=news_index_reverse,
            all_test_user_pos_override=train20_test_user_pos
        )
        steps = len(train20_test_id)
        click_score = model_test.predict(testgen, steps=steps, verbose=0)
        eps = 1e-7
        all_session_loss = []
        all_session_ndcg = []
        for m in train20_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                continue
            labels = train20_test_label[s:e].astype(np.float32)
            if np.sum(labels) == 0:
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                continue
            scores = np.clip(click_score[s:e, 0], eps, 1 - eps)
            session_bce = -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
            all_session_loss.append(float(session_bce))
            all_session_ndcg.append(ndcg_score(train20_test_label[s:e], click_score[s:e, 0], k=5))
        valid_loss = [x for x in all_session_loss if x is not None]
        if not valid_loss:
            return None, None, None
        return all_session_loss, all_session_ndcg, click_score

    lines_t20 = []
    def add_t20(s=""):
        lines_t20.append(s)
        print(s)
    add_t20(f"\n트레이닝셋 유저별 후반 20% 배치 {EVAL_TRAIN20_BATCH_INDEX} 평가 (샘플 수: {n_sessions_t20}, 테스트 행: {len(train20_test_id)})")
    add_t20(f"모델 경로: {PRETRAINED_MODEL_PATH}, 기대본문: {batch_dir_name_t20}")
    add_t20("\n[1] 실제 본문 사용:")
    loss_actual_t20, ndcg_actual_t20, click_actual_t20 = eval_train20_batch_run(use_expected_body=False)
    if loss_actual_t20 is not None:
        valid = [x for x in loss_actual_t20 if x is not None]
        add_t20(f"  Loss     : {np.mean(valid):.6f}")
        add_t20(f"  NDCG@5   : {np.mean([x for x in ndcg_actual_t20 if x is not None]):.6f}")
    else:
        add_t20("  평가 가능한 세션 없음")
    add_t20("\n[2] 기대 본문 사용:")
    loss_expected_t20, ndcg_expected_t20, click_expected_t20 = eval_train20_batch_run(
        use_expected_body=True,
        expected_bodies=expected_bodies_batch_t20,
        all_userid_str=train20_test_userid_str,
        all_newsid_str=train20_test_newsid_str
    )
    if loss_expected_t20 is not None:
        valid = [x for x in loss_expected_t20 if x is not None]
        add_t20(f"  Loss     : {np.mean(valid):.6f}")
        add_t20(f"  NDCG@5   : {np.mean([x for x in ndcg_expected_t20 if x is not None]):.6f}")
    else:
        add_t20("  [건너뜀] 기대본문 데이터 없음")

    best_sess_idx_t20 = None
    success_sess_idx_t20 = None
    max_diff_t20 = -np.inf
    min_diff_t20 = np.inf
    if loss_actual_t20 is not None and loss_expected_t20 is not None:
        candidates_max_t20 = []
        candidates_min_t20 = []
        for i in range(len(train20_test_index)):
            la, le = loss_actual_t20[i], loss_expected_t20[i]
            if la is None or le is None:
                continue
            diff = le - la
            if diff > max_diff_t20:
                max_diff_t20 = diff
                candidates_max_t20 = [(diff, i)]
            elif diff == max_diff_t20:
                candidates_max_t20.append((diff, i))
            if diff < min_diff_t20:
                min_diff_t20 = diff
                candidates_min_t20 = [(diff, i)]
            elif diff == min_diff_t20:
                candidates_min_t20.append((diff, i))
        if candidates_max_t20:
            best_sess_idx_t20 = random.choice([c[1] for c in candidates_max_t20])
            add_t20(f"\n최대 diff 세션 수: {len(candidates_max_t20)}개 (failure 1개 선택)")
        if candidates_min_t20:
            success_sess_idx_t20 = random.choice([c[1] for c in candidates_min_t20])
            add_t20(f"최소 diff 세션 수: {len(candidates_min_t20)}개 (success 1개 선택)")

    loss_real_t20 = np.mean([x for x in loss_actual_t20 if x is not None]) if loss_actual_t20 else None
    loss_expected_t20 = np.mean([x for x in loss_expected_t20 if x is not None]) if loss_expected_t20 else None
    ndcg5_real_t20 = np.mean([x for x in ndcg_actual_t20 if x is not None]) if ndcg_actual_t20 else None
    ndcg5_expected_t20 = np.mean([x for x in ndcg_expected_t20 if x is not None]) if ndcg_expected_t20 else None

    diagnostic_samples_t20 = []
    if loss_actual_t20 is not None and loss_expected_t20 is not None and (best_sess_idx_t20 is not None or success_sess_idx_t20 is not None):
        news_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'MIND', 'MIND_news.tsv')
        news_titles = {}
        if os.path.exists(news_file):
            with open(news_file, 'r', encoding='utf-8') as nf:
                for nline in nf:
                    parts = nline.strip().split('\t')
                    if len(parts) >= 4:
                        news_titles[parts[0]] = parts[3]
        for sess_idx, type_str in [(best_sess_idx_t20, "failure"), (success_sess_idx_t20, "success")]:
            if sess_idx is None:
                continue
            global_idx = batch_indices_t20[sess_idx]
            sess_news_ids = all_train_newsid_str[global_idx] if global_idx < len(all_train_newsid_str) else ['?'] * 5
            sess_user_id_str = all_train_userid_str[global_idx] if all_train_userid_str and global_idx < len(all_train_userid_str) else '?'
            s, e = train20_test_index[sess_idx][0], train20_test_index[sess_idx][1]
            sess_labels = train20_test_label[s:e]
            pos_pos = int(np.where(sess_labels == 1)[0][0])
            positive_news_id = sess_news_ids[pos_pos] if pos_pos < len(sess_news_ids) else '?'
            user_pos_indices = all_user_pos[global_idx]
            fallback_click_titles = [news_titles.get(news_index_reverse.get(int(i), ''), '') for i in user_pos_indices if int(i) != 0]
            fallback_click_titles = fallback_click_titles[-10:]
            candidate_news_title = news_titles.get(positive_news_id, '')
            generated_expected_body, json_user_history = load_expected_body_from_train_dir(batch_dir_t20, sess_user_id_str, positive_news_id)
            user_click_history_titles = json_user_history if json_user_history is not None else fallback_click_titles
            if generated_expected_body == '' and expected_bodies_batch_t20:
                generated_expected_body = expected_bodies_batch_t20.get(_norm_expected_body_key(sess_user_id_str, positive_news_id), '') or ''
            diagnostic_samples_t20.append({
                "type": type_str,
                "user_click_history_titles": user_click_history_titles,
                "candidate_news_title": candidate_news_title,
                "generated_expected_body": generated_expected_body
            })

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    import re
    max_n = -1
    for f in os.listdir(results_dir):
        m = re.match(r'result(\d+)\.txt$', f)
        if m:
            max_n = max(max_n, int(m.group(1)))
    next_n = max_n + 1
    out_path_t20 = os.path.join(results_dir, f'result{next_n}.txt')
    payload_t20 = {
        "performance_feedback": {
            "loss_expected": float(loss_expected_t20) if loss_expected_t20 is not None else None,
            "loss_real": float(loss_real_t20) if loss_real_t20 is not None else None,
            "ndcg5_expected": float(ndcg5_expected_t20) if ndcg5_expected_t20 is not None else None,
            "ndcg5_real": float(ndcg5_real_t20) if ndcg5_real_t20 is not None else None
        },
        "diagnostic_samples": diagnostic_samples_t20
    }
    with open(out_path_t20, 'w', encoding='utf-8') as f:
        json.dump(payload_t20, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path_t20}")
    add_t20(f"\n프리트레이닝 모델 - 유저별 후반 20% 배치 {EVAL_TRAIN20_BATCH_INDEX} 평가 완료. 프로그램을 종료합니다.")
    sys.exit(0)

# ========== 프리트레이닝 모델 로드 후 트레이닝 80% 평가 (실제본문 / 기대본문 각각) ==========
if EVAL_PRETRAINED_ON_TRAIN80:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"프리트레이닝 모델 로드: {PRETRAINED_MODEL_PATH}")
    print(f"{'='*60}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    print("모델 로드 완료. (데이터/어휘 크기가 저장 시와 다르면 embedding 등 일부 레이어는 로드되지 않을 수 있음)")
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    latest_train_dir = get_latest_train_folder(body_gen_output)
    ref_folder_name = os.path.basename(latest_train_dir) if latest_train_dir else '(trainN 없음)'
    print(f"모델 경로: {PRETRAINED_MODEL_PATH}")
    print(f"기대 본문 참조 폴더: {ref_folder_name}")
    
    # 트레이닝셋 전반부 80%를 테스트 형식으로 구성 (샘플당 5개 후보 → 5개 테스트 행)
    pretrain_size = int(len(all_train_id) * 0.8)
    train80_test_pn = []
    train80_test_label = []
    train80_test_id = []
    train80_test_user_pos = []
    train80_test_index = []
    train80_test_userid_str = []
    train80_test_newsid_str = []
    
    for i in range(pretrain_size):
        start = len(train80_test_pn)
        pn_row = all_train_pn[i]
        label_row = all_label[i]
        n_cand = len(pn_row) if hasattr(pn_row, '__len__') else 5
        for j in range(n_cand):
            train80_test_pn.append(int(pn_row[j]) if hasattr(pn_row[j], '__int__') else pn_row[j])
            train80_test_label.append(int(label_row[j]) if hasattr(label_row[j], '__int__') else label_row[j])
            train80_test_id.append(all_train_id[i])
            train80_test_user_pos.append(all_user_pos[i])
        if all_train_userid_str is not None and all_train_newsid_str is not None:
            for j in range(n_cand):
                train80_test_userid_str.append(all_train_userid_str[i])
                train80_test_newsid_str.append(all_train_newsid_str[i][j] if j < len(all_train_newsid_str[i]) else '')
        train80_test_index.append([start, len(train80_test_pn)])
    
    train80_test_pn = np.array(train80_test_pn, dtype='int32')
    train80_test_label = np.array(train80_test_label, dtype='int32')
    train80_test_id = np.array(train80_test_id, dtype='int32')
    train80_test_user_pos = np.array(train80_test_user_pos, dtype='int32')
    
    def eval_train80_run(use_expected_body, expected_bodies=None, all_userid_str=None, all_newsid_str=None):
        """한 번의 평가 실행. 세션별 loss/ndcg와 click_score 반환 (세션 인덱스와 1:1 대응)."""
        if use_expected_body and (expected_bodies is None or all_userid_str is None or all_newsid_str is None):
            return None, None, None
        testgen = generate_batch_data_test(
            train80_test_pn, train80_test_label, train80_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_userid_str,
            all_newsid_str=all_newsid_str,
            news_index_reverse=news_index_reverse,
            all_test_user_pos_override=train80_test_user_pos
        )
        steps = len(train80_test_id)
        click_score = model_test.predict(testgen, steps=steps, verbose=0)
        
        eps = 1e-7
        # train80_test_index와 동일한 길이, 유효한 세션만 값 저장 (나머지 None)
        all_session_loss = []
        all_session_ndcg = []
        for m in train80_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                continue
            labels = train80_test_label[s:e].astype(np.float32)
            if np.sum(labels) == 0:
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                continue
            scores = np.clip(click_score[s:e, 0], eps, 1 - eps)
            session_bce = -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
            all_session_loss.append(float(session_bce))
            all_session_ndcg.append(ndcg_score(train80_test_label[s:e], click_score[s:e, 0], k=5))
        
        valid_loss = [x for x in all_session_loss if x is not None]
        if not valid_loss:
            return None, None, None
        return all_session_loss, all_session_ndcg, click_score
    
    # 결과 수집 (콘솔 출력 + 파일 저장용)
    lines = []
    def add(s=""):
        lines.append(s)
        print(s)
    
    add(f"\n트레이닝셋 전반부 80% 평가 (샘플 수: {pretrain_size}, 테스트 행: {len(train80_test_id)})")
    add(f"{'='*60}")
    add(f"모델 경로: {PRETRAINED_MODEL_PATH}")
    add(f"기대 본문 참조 폴더: {ref_folder_name}")
    
    add("\n[1] 실제 본문 사용:")
    loss_actual_list, ndcg_actual_list, click_actual = eval_train80_run(use_expected_body=False)
    if loss_actual_list is not None:
        valid = [x for x in loss_actual_list if x is not None]
        add(f"  Loss     : {np.mean(valid):.6f}  (유저당 BCE 평균)")
        add(f"  NDCG@5   : {np.mean([x for x in ndcg_actual_list if x is not None]):.6f}")
    else:
        add("  평가 가능한 세션 없음")
    
    add("\n[2] 기대 본문 사용:")
    if latest_train_dir:
        add(f"  기대 본문 로드: {os.path.basename(latest_train_dir)} 폴더 참조")
        expected_bodies_train_eval = load_expected_bodies_from_train_dir(latest_train_dir)
        add(f"  로드된 기대 본문: {len(expected_bodies_train_eval)}개")
    else:
        expected_bodies_train_eval = expected_bodies_train
        if expected_bodies_train_eval is None:
            add("  기대 본문 로드 중 (train)...")
            expected_bodies_train_eval = load_expected_bodies(output_dir='body_generation/output', dataset_type='train')
            add(f"  로드된 기대 본문: {len(expected_bodies_train_eval)}개")
    loss_expected_list, ndcg_expected_list, click_expected = eval_train80_run(
        use_expected_body=True,
        expected_bodies=expected_bodies_train_eval,
        all_userid_str=train80_test_userid_str,
        all_newsid_str=train80_test_newsid_str
    )
    if loss_expected_list is not None:
        valid = [x for x in loss_expected_list if x is not None]
        add(f"  Loss     : {np.mean(valid):.6f}  (유저당 BCE 평균)")
        add(f"  NDCG@5   : {np.mean([x for x in ndcg_expected_list if x is not None]):.6f}")
    else:
        add("  [건너뜀] 기대본문 데이터 없음")
    
    # 두 loss 차이가 가장 큰 세션(failure) / 가장 작은 세션(success) 찾기
    # diff = 기대본문 loss - 실제본문 loss. 최대 = 기대본문이 상대적으로 가장 못한 세션, 최소 = 가장 잘 맞춘 세션
    best_sess_idx = None  # failure: diff 최대
    success_sess_idx = None  # success: diff 최소
    max_diff = -np.inf
    min_diff = np.inf
    if loss_actual_list is not None and loss_expected_list is not None:
        candidates_max = []  # (diff, i) 최대 diff
        candidates_min = []  # (diff, i) 최소 diff
        for i in range(len(train80_test_index)):
            la, le = loss_actual_list[i], loss_expected_list[i]
            if la is None or le is None:
                continue
            diff = le - la  # 양수면 해당 세션에서 기대본문이 실제본문보다 성능 못함
            if diff > max_diff:
                max_diff = diff
                candidates_max = [(diff, i)]
            elif diff == max_diff:
                candidates_max.append((diff, i))
            if diff < min_diff:
                min_diff = diff
                candidates_min = [(diff, i)]
            elif diff == min_diff:
                candidates_min.append((diff, i))
        if candidates_max:
            best_sess_idx = random.choice([c[1] for c in candidates_max])
            add(f"\n최대 diff를 가진 세션 수: {len(candidates_max)}개 (이 중 1개를 failure로 선택)")
        if candidates_min:
            success_sess_idx = random.choice([c[1] for c in candidates_min])
            add(f"최소 diff를 가진 세션 수: {len(candidates_min)}개 (이 중 1개를 success로 선택)")
        
        # diff 분포 출력 (기대 - 실제, 세션별)
        all_diffs = []
        for i in range(len(train80_test_index)):
            la, le = loss_actual_list[i], loss_expected_list[i]
            if la is None or le is None:
                continue
            all_diffs.append(le - la)
        if all_diffs:
            all_diffs = np.array(all_diffs)
            add(f"\n[Loss 차이(기대-실제) 분포] 세션 수: {len(all_diffs)}")
            add(f"  min    : {float(np.min(all_diffs)):.6f}")
            add(f"  max    : {float(np.max(all_diffs)):.6f}")
            add(f"  mean   : {float(np.mean(all_diffs)):.6f}")
            add(f"  std    : {float(np.std(all_diffs)):.6f}")
            add(f"  25%    : {float(np.percentile(all_diffs, 25)):.6f}")
            add(f"  50%    : {float(np.percentile(all_diffs, 50)):.6f}")
            add(f"  75%    : {float(np.percentile(all_diffs, 75)):.6f}")
            # 간단한 히스토그램 (10구간)
            hist, bin_edges = np.histogram(all_diffs, bins=10)
            add("  히스토그램 (10구간):")
            for j in range(len(hist)):
                add(f"    [{bin_edges[j]:.4f}, {bin_edges[j+1]:.4f}) : {int(hist[j])}개")
            add("")
        
        if best_sess_idx is not None:
            s, e = train80_test_index[best_sess_idx][0], train80_test_index[best_sess_idx][1]
            labels = train80_test_label[s:e]
            scores_actual = click_actual[s:e, 0]
            scores_expected = click_expected[s:e, 0]
            # 후보 뉴스 ID (트레이닝 80%의 원본 샘플 인덱스 = best_sess_idx)
            news_ids = all_train_newsid_str[best_sess_idx] if best_sess_idx < len(all_train_newsid_str) else ['?'] * 5
            user_id_str = all_train_userid_str[best_sess_idx] if all_train_userid_str and best_sess_idx < len(all_train_userid_str) else '?'
            
            rank_actual = np.argsort(scores_actual)[::-1]
            rank_expected = np.argsort(scores_expected)[::-1]
            positive_pos_actual = np.where(labels == 1)[0][0]
            positive_pos_expected = np.where(labels == 1)[0][0]
            
            add(f"\n{'='*60}")
            add("성능 차이가 가장 큰 세션 (기대본문 Loss - 실제본문 Loss 최대 = 기대본문이 상대적으로 가장 못한 세션)")
            add(f"{'='*60}")
            add(f"  세션 인덱스(트레이닝 80% 내) : {best_sess_idx}")
            add(f"  유저 ID                    : {user_id_str}")
            add(f"  후보 뉴스 ID (5개)         : {list(news_ids)}")
            add(f"  정답 레이블 (1=클릭)        : {list(labels)}")
            add(f"  실제본문 Loss (해당 세션)   : {loss_actual_list[best_sess_idx]:.6f}")
            add(f"  기대본문 Loss (해당 세션)   : {loss_expected_list[best_sess_idx]:.6f}")
            add(f"  Loss 차이 (기대 - 실제)     : {max_diff:.6f}")
            add(f"  실제본문 점수 (5개 후보)    : {[round(float(x), 6) for x in scores_actual]}")
            add(f"  기대본문 점수 (5개 후보)    : {[round(float(x), 6) for x in scores_expected]}")
            add(f"  실제본문 랭킹(점수순)      : {rank_actual.tolist()} (정답 위치: {int(positive_pos_actual)})")
            add(f"  기대본문 랭킹(점수순)      : {rank_expected.tolist()} (정답 위치: {int(positive_pos_expected)})")
            add(f"{'='*60}\n")
        
        # 로스 차이가 가장 작은 세션 (success) 출력
        if success_sess_idx is not None:
            s, e = train80_test_index[success_sess_idx][0], train80_test_index[success_sess_idx][1]
            labels_s = train80_test_label[s:e]
            scores_actual_s = click_actual[s:e, 0]
            scores_expected_s = click_expected[s:e, 0]
            news_ids_s = all_train_newsid_str[success_sess_idx] if success_sess_idx < len(all_train_newsid_str) else ['?'] * 5
            user_id_str_s = all_train_userid_str[success_sess_idx] if all_train_userid_str and success_sess_idx < len(all_train_userid_str) else '?'
            rank_actual_s = np.argsort(scores_actual_s)[::-1]
            rank_expected_s = np.argsort(scores_expected_s)[::-1]
            positive_pos_actual_s = np.where(labels_s == 1)[0][0]
            positive_pos_expected_s = np.where(labels_s == 1)[0][0]
            add(f"\n{'='*60}")
            add("성능 차이가 가장 작은 세션 (기대본문 Loss - 실제본문 Loss 최소 = 기대본문이 상대적으로 잘 맞춘 세션)")
            add(f"{'='*60}")
            add(f"  세션 인덱스(트레이닝 80% 내) : {success_sess_idx}")
            add(f"  유저 ID                    : {user_id_str_s}")
            add(f"  후보 뉴스 ID (5개)         : {list(news_ids_s)}")
            add(f"  정답 레이블 (1=클릭)        : {list(labels_s)}")
            add(f"  실제본문 Loss (해당 세션)   : {loss_actual_list[success_sess_idx]:.6f}")
            add(f"  기대본문 Loss (해당 세션)   : {loss_expected_list[success_sess_idx]:.6f}")
            add(f"  Loss 차이 (기대 - 실제)     : {min_diff:.6f}")
            add(f"  실제본문 점수 (5개 후보)    : {[round(float(x), 6) for x in scores_actual_s]}")
            add(f"  기대본문 점수 (5개 후보)    : {[round(float(x), 6) for x in scores_expected_s]}")
            add(f"  실제본문 랭킹(점수순)      : {rank_actual_s.tolist()} (정답 위치: {int(positive_pos_actual_s)})")
            add(f"  기대본문 랭킹(점수순)      : {rank_expected_s.tolist()} (정답 위치: {int(positive_pos_expected_s)})")
            add(f"{'='*60}\n")
    
    add(f"\n{'='*60}")
    add("프리트레이닝 모델 - 트레이닝 80% 평가 완료. 프로그램을 종료합니다.")
    add(f"{'='*60}\n")
    
    # NAML/results 에 JSON 저장 (performance_feedback + diagnostic_samples, result0.txt부터 순번)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    existing = [f for f in os.listdir(results_dir) if f.startswith('result') and f.endswith('.txt')]
    next_num = 0
    for f in existing:
        try:
            n = int(f.replace('result', '').replace('.txt', ''))
            if n >= next_num:
                next_num = n + 1
        except ValueError:
            pass
    out_path = os.path.join(results_dir, f'result{next_num}.txt')
    
    # performance_feedback: 실제=real, 기대=expected
    loss_real = np.mean([x for x in loss_actual_list if x is not None]) if loss_actual_list else None
    loss_expected = np.mean([x for x in loss_expected_list if x is not None]) if loss_expected_list else None
    ndcg5_real = np.mean([x for x in ndcg_actual_list if x is not None]) if ndcg_actual_list else None
    ndcg5_expected = np.mean([x for x in ndcg_expected_list if x is not None]) if ndcg_expected_list else None
    
    # diagnostic_samples: 성능 차이 최대(failure) / 최소(success) 세션 정보 (뉴스 제목·기대본문 로드)
    diagnostic_samples = []
    if loss_actual_list is not None and loss_expected_list is not None and (best_sess_idx is not None or success_sess_idx is not None):
        news_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'MIND', 'MIND_news.tsv')
        news_titles = {}
        if os.path.exists(news_file):
            with open(news_file, 'r', encoding='utf-8') as nf:
                for nline in nf:
                    parts = nline.strip().split('\t')
                    if len(parts) >= 4:
                        news_titles[parts[0]] = parts[3]
        body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
        latest_train_dir = get_latest_train_folder(body_gen_output)
        for sess_idx, type_str in [(best_sess_idx, "failure"), (success_sess_idx, "success")]:
            if sess_idx is None:
                continue
            sess_news_ids = all_train_newsid_str[sess_idx] if sess_idx < len(all_train_newsid_str) else ['?'] * 5
            sess_user_id_str = all_train_userid_str[sess_idx] if all_train_userid_str and sess_idx < len(all_train_userid_str) else '?'
            s, e = train80_test_index[sess_idx][0], train80_test_index[sess_idx][1]
            sess_labels = train80_test_label[s:e]
            pos_pos = int(np.where(sess_labels == 1)[0][0])
            positive_news_id = sess_news_ids[pos_pos] if pos_pos < len(sess_news_ids) else '?'
            user_pos_indices = all_user_pos[sess_idx]
            fallback_click_titles = [news_titles.get(news_index_reverse.get(int(i), ''), '') for i in user_pos_indices if int(i) != 0]
            fallback_click_titles = fallback_click_titles[-10:]
            candidate_news_title = news_titles.get(positive_news_id, '')
            if latest_train_dir:
                generated_expected_body, json_user_history = load_expected_body_from_train_dir(latest_train_dir, sess_user_id_str, positive_news_id)
                user_click_history_titles = json_user_history if json_user_history is not None else fallback_click_titles
                if generated_expected_body == '' and expected_bodies_train_eval:
                    generated_expected_body = expected_bodies_train_eval.get((sess_user_id_str, positive_news_id), '') or ''
            else:
                generated_expected_body = (expected_bodies_train_eval.get((sess_user_id_str, positive_news_id), '') or '') if expected_bodies_train_eval else ''
                user_click_history_titles = fallback_click_titles
            diagnostic_samples.append({
                "type": type_str,
                "user_click_history_titles": user_click_history_titles,
                "candidate_news_title": candidate_news_title,
                "generated_expected_body": generated_expected_body
            })
    
    payload = {
        "performance_feedback": {
            "loss_expected": float(loss_expected) if loss_expected is not None else None,
            "loss_real": float(loss_real) if loss_real is not None else None,
            "ndcg5_expected": float(ndcg5_expected) if ndcg5_expected is not None else None,
            "ndcg5_real": float(ndcg5_real) if ndcg5_real is not None else None
        },
        "diagnostic_samples": diagnostic_samples
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
    
    sys.exit(0)

# ========== 프리트레이닝 모델 로드 후 트레이닝 후반 20% 평가 (실제본문 / 기대본문 각각) ==========
if EVAL_PRETRAINED_ON_TRAIN20:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    print(f"\n{'='*60}")
    print(f"프리트레이닝 모델 로드: {PRETRAINED_MODEL_PATH}")
    print(f"{'='*60}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    print("모델 로드 완료.")
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    if EVAL_TRAIN20_EXPECTED_BODY_DIR:
        latest_train20_dir = os.path.join(body_gen_output, EVAL_TRAIN20_EXPECTED_BODY_DIR)
        if not os.path.isdir(latest_train20_dir):
            print(f"오류: 기대본문 폴더가 없습니다: {latest_train20_dir}")
            sys.exit(1)
    else:
        latest_train20_dir = get_latest_train20_folder(body_gen_output)
    ref_folder_name = os.path.basename(latest_train20_dir) if latest_train20_dir else '(train20_N 없음)'
    print(f"모델 경로: {PRETRAINED_MODEL_PATH}")
    print(f"기대 본문 참조 폴더: {ref_folder_name}")

    pretrain_size = int(len(all_train_id) * 0.8)
    last20_size = len(all_train_id) - pretrain_size
    train20_test_pn = []
    train20_test_label = []
    train20_test_id = []
    train20_test_user_pos = []
    train20_test_index = []
    train20_test_userid_str = []
    train20_test_newsid_str = []
    for i in range(pretrain_size, len(all_train_id)):
        start = len(train20_test_pn)
        pn_row = all_train_pn[i]
        label_row = all_label[i]
        n_cand = len(pn_row) if hasattr(pn_row, '__len__') else 5
        for j in range(n_cand):
            train20_test_pn.append(int(pn_row[j]) if hasattr(pn_row[j], '__int__') else pn_row[j])
            train20_test_label.append(int(label_row[j]) if hasattr(label_row[j], '__int__') else label_row[j])
            train20_test_id.append(all_train_id[i])
            train20_test_user_pos.append(all_user_pos[i])
        if all_train_userid_str is not None and all_train_newsid_str is not None:
            for j in range(n_cand):
                train20_test_userid_str.append(all_train_userid_str[i])
                train20_test_newsid_str.append(all_train_newsid_str[i][j] if j < len(all_train_newsid_str[i]) else '')
        train20_test_index.append([start, len(train20_test_pn)])
    train20_test_pn = np.array(train20_test_pn, dtype='int32')
    train20_test_label = np.array(train20_test_label, dtype='int32')
    train20_test_id = np.array(train20_test_id, dtype='int32')
    train20_test_user_pos = np.array(train20_test_user_pos, dtype='int32')

    def eval_train20_run(use_expected_body, expected_bodies=None, all_userid_str=None, all_newsid_str=None):
        if use_expected_body and (expected_bodies is None or all_userid_str is None or all_newsid_str is None):
            return None, None, None, None, None
        testgen = generate_batch_data_test(
            train20_test_pn, train20_test_label, train20_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_userid_str,
            all_newsid_str=all_newsid_str,
            news_index_reverse=news_index_reverse,
            all_test_user_pos_override=train20_test_user_pos
        )
        steps = len(train20_test_id)
        click_score = model_test.predict(testgen, steps=steps, verbose=0)
        eps = 1e-7
        all_session_loss = []
        all_session_ndcg = []
        all_session_mrr = []
        all_session_hit1 = []
        for m in train20_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                all_session_mrr.append(None)
                all_session_hit1.append(None)
                continue
            labels = train20_test_label[s:e]
            if np.sum(labels) == 0:
                all_session_loss.append(None)
                all_session_ndcg.append(None)
                all_session_mrr.append(None)
                all_session_hit1.append(None)
                continue
            scores = np.clip(click_score[s:e, 0], eps, 1 - eps)
            labels_f = labels.astype(np.float32)
            session_bce = -np.mean(labels_f * np.log(scores) + (1 - labels_f) * np.log(1 - scores))
            all_session_loss.append(float(session_bce))
            all_session_ndcg.append(ndcg_score(labels, click_score[s:e, 0], k=5))
            all_session_mrr.append(mrr_score(labels, click_score[s:e, 0]))
            all_session_hit1.append(hit_at_k(labels, click_score[s:e, 0], k=1))
        valid_loss = [x for x in all_session_loss if x is not None]
        if not valid_loss:
            return None, None, None, None, None
        return all_session_loss, all_session_ndcg, all_session_mrr, all_session_hit1, click_score

    lines = []
    def add(s=""):
        lines.append(s)
        print(s)
    add(f"\n트레이닝셋 후반 20% 평가 (샘플 수: {last20_size}, 테스트 행: {len(train20_test_id)})")
    add(f"{'='*60}")
    add(f"모델 경로: {PRETRAINED_MODEL_PATH}")
    add(f"기대 본문 참조 폴더: {ref_folder_name}")
    add("\n[1] 실제 본문 사용:")
    loss_actual_list, ndcg_actual_list, mrr_actual_list, hit1_actual_list, click_actual = eval_train20_run(use_expected_body=False)
    if loss_actual_list is not None:
        valid = [x for x in loss_actual_list if x is not None]
        add(f"  Loss     : {np.mean(valid):.6f}  (유저당 BCE 평균)")
        add(f"  NDCG@5   : {np.mean([x for x in ndcg_actual_list if x is not None]):.6f}")
        add(f"  MRR      : {np.mean([x for x in mrr_actual_list if x is not None]):.6f}")
        add(f"  Hit@1    : {np.mean([x for x in hit1_actual_list if x is not None]):.6f}")
    else:
        add("  평가 가능한 세션 없음")
    add("\n[2] 기대 본문 사용:")
    if latest_train20_dir:
        add(f"  기대 본문 로드: {os.path.basename(latest_train20_dir)} 폴더 참조")
        abs_train20_dir = os.path.abspath(latest_train20_dir)
        add(f"  절대 경로: {abs_train20_dir}")
        if os.path.isdir(abs_train20_dir):
            user_dirs = [d for d in os.listdir(abs_train20_dir) if os.path.isdir(os.path.join(abs_train20_dir, d)) and d.startswith('user_')]
            add(f"  user_* 하위 폴더 수: {len(user_dirs)}개")
        expected_bodies_train20_eval = load_expected_bodies_from_train_dir(latest_train20_dir)
        add(f"  로드된 기대 본문: {len(expected_bodies_train20_eval)}개")
        # 트레이닝 후반 20% 매칭 진단: NAML이 필요로 하는 (user, news) 키 중 기대본문에 있는 비율
        if expected_bodies_train20_eval and train20_test_userid_str and train20_test_newsid_str:
            need_keys = set()
            for i in range(len(train20_test_userid_str)):
                u = train20_test_userid_str[i] if i < len(train20_test_userid_str) else ''
                n = train20_test_newsid_str[i] if i < len(train20_test_newsid_str) else ''
                need_keys.add(_norm_expected_body_key(u, n))
            matched = sum(1 for k in need_keys if k in expected_bodies_train20_eval)
            add(f"  트레이닝 후반 20% 매칭: 필요 키 {len(need_keys)}개 중 기대본문 존재 {matched}개 ({100.0*matched/len(need_keys):.1f}%)" if need_keys else "  트레이닝 후반 20% 매칭: 필요 키 없음")
            if expected_bodies_train20_eval and need_keys:
                sample_loaded = list(expected_bodies_train20_eval.keys())[:3]
                sample_need = list(need_keys)[:3]
                add(f"  (참고) 로드 키 샘플: {sample_loaded}")
                add(f"  (참고) 필요 키 샘플: {sample_need}")
    else:
        expected_bodies_train20_eval = None
        add("  기대 본문 폴더 없음 (train20_0, train20_1, ... 중 하나 생성 후 다시 실행)")
    if latest_train20_dir and expected_bodies_train20_eval is not None:
        loss_expected_list, ndcg_expected_list, mrr_expected_list, hit1_expected_list, click_expected = eval_train20_run(
            use_expected_body=True,
            expected_bodies=expected_bodies_train20_eval,
            all_userid_str=train20_test_userid_str,
            all_newsid_str=train20_test_newsid_str
        )
    else:
        loss_expected_list, ndcg_expected_list, mrr_expected_list, hit1_expected_list = None, None, None, None
    if loss_expected_list is not None:
        valid = [x for x in loss_expected_list if x is not None]
        add(f"  Loss     : {np.mean(valid):.6f}  (유저당 BCE 평균)")
        add(f"  NDCG@5   : {np.mean([x for x in ndcg_expected_list if x is not None]):.6f}")
        add(f"  MRR      : {np.mean([x for x in mrr_expected_list if x is not None]):.6f}")
        add(f"  Hit@1    : {np.mean([x for x in hit1_expected_list if x is not None]):.6f}")
    else:
        add("  [건너뜀] 기대본문 데이터 없음")
    add(f"\n{'='*60}")
    add("프리트레이닝 모델 - 트레이닝 후반 20% 평가 완료. 프로그램을 종료합니다.")
    add(f"{'='*60}\n")
    sys.exit(0)

# ========== 프리트레이닝 모델 로드 후 테스트셋만 평가 (실제본문 / 기대본문 각각) ==========
if EVAL_PRETRAINED_ON_TESTSET:
    import sys
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"오류: 프리트레이닝 모델을 찾을 수 없습니다: {PRETRAINED_MODEL_PATH}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"프리트레이닝 모델 로드: {PRETRAINED_MODEL_PATH}")
    print(f"{'='*60}")
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    print("모델 로드 완료. (데이터/어휘 크기가 저장 시와 다르면 embedding 등 일부 레이어는 로드되지 않을 수 있음)")

    print(f"\n{'='*60}")
    print("프리트레이닝 모델 - 테스트셋 평가")
    print(f"{'='*60}")
    
    def eval_testset_run(use_expected_body, expected_bodies=None, all_userid_str=None, all_newsid_str=None):
        """테스트셋 한 번 평가. NDCG@5, MRR, Hit@1, Loss(유저당 BCE 평균) 반환."""
        if use_expected_body and (expected_bodies is None or all_userid_str is None or all_newsid_str is None):
            return None
        testgen = generate_batch_data_test(
            all_test_pn, all_test_label, all_test_id, 30,
            candidate_news_body=None,
            expected_bodies=expected_bodies,
            all_userid_str=all_userid_str,
            all_newsid_str=all_newsid_str,
            news_index_reverse=news_index_reverse
        )
        test_steps = len(all_test_id)
        click_score = model_test.predict(testgen, steps=test_steps, verbose=0)
        
        eps = 1e-7
        all_ndcg, all_mrr, all_hit1, all_session_loss = [], [], [], []
        for m in all_test_index:
            s, e = m[0], m[1]
            if e > len(click_score):
                continue
            labels = all_test_label[s:e]
            if np.sum(labels) == 0:
                continue
            scores = click_score[s:e, 0]
            all_ndcg.append(ndcg_score(labels, scores, k=5))
            all_mrr.append(mrr_score(labels, scores))
            all_hit1.append(hit_at_k(labels, scores, k=1))
            labels_f = labels.astype(np.float32)
            scores_clip = np.clip(scores, eps, 1 - eps)
            session_bce = -np.mean(labels_f * np.log(scores_clip) + (1 - labels_f) * np.log(1 - scores_clip))
            all_session_loss.append(float(session_bce))
        
        if not all_ndcg:
            return None
        return {
            'NDCG@5': np.mean(all_ndcg),
            'MRR': np.mean(all_mrr),
            'Hit@1': np.mean(all_hit1),
            'Loss': np.mean(all_session_loss),
        }
    
    print("\n[1] 실제 본문 사용:")
    res_actual = eval_testset_run(use_expected_body=False)
    if res_actual is not None:
        print(f"  NDCG@5   : {res_actual['NDCG@5']:.6f}")
        print(f"  MRR      : {res_actual['MRR']:.6f}")
        print(f"  Hit@1    : {res_actual['Hit@1']:.6f}")
        print(f"  Loss     : {res_actual['Loss']:.6f}  (유저당 BCE 평균)")
    else:
        print("  평가 가능한 세션 없음")
    
    print("\n[2] 기대 본문 사용:")
    body_gen_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
    test_body_dir = os.path.join(body_gen_output, EVAL_TESTSET_EXPECTED_BODY_DIR)
    if os.path.isdir(test_body_dir):
        print(f"  기대 본문 로드: {EVAL_TESTSET_EXPECTED_BODY_DIR} 폴더 참조")
        expected_bodies_test_eval = load_expected_bodies_from_train_dir(test_body_dir)
        print(f"  로드된 기대 본문: {len(expected_bodies_test_eval)}개")
    else:
        expected_bodies_test_eval = expected_bodies_test
        if expected_bodies_test_eval is None:
            print("  기대 본문 로드 중 (output/test)...")
            expected_bodies_test_eval = load_expected_bodies(output_dir='body_generation/output', dataset_type='test')
            print(f"  로드된 기대 본문: {len(expected_bodies_test_eval)}개")
    res_expected = eval_testset_run(
        use_expected_body=True,
        expected_bodies=expected_bodies_test_eval,
        all_userid_str=all_test_userid_str,
        all_newsid_str=all_test_newsid_str
    )
    if res_expected is not None:
        print(f"  NDCG@5   : {res_expected['NDCG@5']:.6f}")
        print(f"  MRR      : {res_expected['MRR']:.6f}")
        print(f"  Hit@1    : {res_expected['Hit@1']:.6f}")
        print(f"  Loss     : {res_expected['Loss']:.6f}  (유저당 BCE 평균)")
    else:
        print("  [건너뜀] 기대본문 데이터 없음")
    
    print(f"\n{'='*60}")
    print("프리트레이닝 모델 - 테스트셋 평가 완료. 프로그램을 종료합니다.")
    print(f"{'='*60}\n")
    sys.exit(0)

# ========== 메인 학습 루프 ==========
# 매 에폭 테스트셋 기대본문 평가용 (EVAL_TESTSET_EXPECTED_BODY_DIR 폴더)
body_gen_output_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'body_generation', 'output')
need_keys_test = set()
if all_test_userid_str is not None and all_test_newsid_str is not None:
    for i in range(len(all_test_userid_str)):
        u = all_test_userid_str[i]
        n = all_test_newsid_str[i] if i < len(all_test_newsid_str) else ''
        need_keys_test.add(_norm_expected_body_key(u, n))
    need_keys_test.discard(('', ''))
test_body_dir_main = os.path.join(body_gen_output_main, EVAL_TESTSET_EXPECTED_BODY_DIR)
expected_bodies_main_test = load_expected_bodies_from_train_dir(test_body_dir_main) if os.path.isdir(test_body_dir_main) else None
if expected_bodies_main_test is not None:
    print(f"메인 학습: 매 에폭 테스트셋 기대본문 평가 시 {EVAL_TESTSET_EXPECTED_BODY_DIR} 사용 ({len(expected_bodies_main_test)}개)")
    if need_keys_test:
        matched_test = sum(1 for k in need_keys_test if k in expected_bodies_main_test)
        pct = 100.0 * matched_test / len(need_keys_test)
        print(f"테스트셋 기대본문 매칭: 필요 키 {len(need_keys_test)}개 중 기대본문 존재 {matched_test}개 ({pct:.1f}%)")
    print()
# 두 번째 기대본문 폴더 (EVAL_TESTSET_EXPECTED_BODY_DIR_2)
expected_bodies_main_test_2 = None
if EVAL_TESTSET_EXPECTED_BODY_DIR_2:
    test_body_dir_main_2 = os.path.join(body_gen_output_main, EVAL_TESTSET_EXPECTED_BODY_DIR_2)
    if os.path.isdir(test_body_dir_main_2):
        expected_bodies_main_test_2 = load_expected_bodies_from_train_dir(test_body_dir_main_2)
        print(f"메인 학습: 매 에폭 테스트셋 기대본문(2) 평가 시 {EVAL_TESTSET_EXPECTED_BODY_DIR_2} 사용 ({len(expected_bodies_main_test_2)}개)")
        if need_keys_test:
            matched_test_2 = sum(1 for k in need_keys_test if k in expected_bodies_main_test_2)
            pct_2 = 100.0 * matched_test_2 / len(need_keys_test)
            print(f"테스트셋 기대본문(2) 매칭: 필요 키 {len(need_keys_test)}개 중 기대본문 존재 {matched_test_2}개 ({pct_2:.1f}%)")
        print()
    else:
        print(f"경고: 두 번째 기대본문 폴더 없음 ({test_body_dir_main_2}), 기대본문(2) 평가 생략\n")

# ========== 트레이닝 후반 20%만 사용해 처음부터 학습 (실제본문 또는 기대본문) 후 테스트셋 실제/기대본문 각각 평가 ==========
if TRAIN_ON_TRAIN20_FROM_SCRATCH:
    print(f"\n{'='*60}")
    print("트레이닝 후반 20%만 사용, 처음부터 학습 (실제본문 또는 기대본문)")
    print(f"{'='*60}")
    # 유저별로 후반 20% 세션 선택 (유저당 최소 1세션 포함되도록)
    from collections import defaultdict
    user_to_indices = defaultdict(list)
    for i in range(len(all_train_id)):
        uid = all_train_userid_str[i] if all_train_userid_str is not None else i
        user_to_indices[uid].append(i)
    train20_indices = []
    for uid, indices in user_to_indices.items():
        n = len(indices)
        take_count = max(1, int(np.ceil(0.2 * n)))
        train20_indices.extend(indices[-take_count:])
    train20_indices = sorted(train20_indices)
    last20_size = len(train20_indices)
    train20_pn = [all_train_pn[i] for i in train20_indices]
    train20_label = [all_label[i] for i in train20_indices]
    train20_id = [all_train_id[i] for i in train20_indices]
    train20_user_pos = [all_user_pos[i] for i in train20_indices]
    train20_userid_str = [all_train_userid_str[i] for i in train20_indices] if all_train_userid_str is not None else None
    train20_newsid_str = [all_train_newsid_str[i] for i in train20_indices] if all_train_newsid_str is not None else None
    print(f"학습 데이터: 유저별 후반 20%% (유저당 최소 1세션) 샘플 수 {last20_size}개, 유저 수 {len(user_to_indices)}명")
    expected_bodies_train20 = None
    if TRAIN_ON_TRAIN20_USE_EXPECTED_BODY and TRAIN_ON_TRAIN20_EXPECTED_BODY_DIR:
        train20_body_dir = os.path.join(body_gen_output_main, TRAIN_ON_TRAIN20_EXPECTED_BODY_DIR)
        need_generate = TRAIN_ON_TRAIN20_GENERATE_EXPECTED_BODY and (
            not os.path.isdir(train20_body_dir) or not os.listdir(train20_body_dir)
        )
        if need_generate:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_path = os.path.join(project_root, 'body_generation', 'generate_body.py')
            out_dir = os.path.join(project_root, 'body_generation', 'output')
            print(f"기대본문 자동 생성: 유저별 후반 20%% (body_generation) 실행 중... 출력: {TRAIN_ON_TRAIN20_EXPECTED_BODY_DIR}")
            cmd = [sys.executable, script_path, '--train20_only', '--train20_per_user',
                   '--output_subdir', TRAIN_ON_TRAIN20_EXPECTED_BODY_DIR, '--output', out_dir]
            if TRAIN_ON_TRAIN20_EXPECTED_BODY_POSITIVE_ONLY:
                cmd.insert(cmd.index('--output_subdir'), '--train20_positive_only')
            ret = subprocess.run(cmd, cwd=project_root, capture_output=False)
            if ret.returncode != 0:
                print(f"경고: 기대본문 생성 종료 코드 {ret.returncode}, 실제본문으로 학습합니다.")
        if os.path.isdir(train20_body_dir):
            expected_bodies_train20 = load_expected_bodies_from_train_dir(train20_body_dir)
            print(f"학습 본문: 기대본문 사용 ({TRAIN_ON_TRAIN20_EXPECTED_BODY_DIR}, {len(expected_bodies_train20)}개)")
        elif not need_generate:
            print(f"경고: 기대본문 폴더 없음 ({train20_body_dir}), 실제본문으로 학습합니다.")
    if expected_bodies_train20 is None:
        print("학습 본문: 실제본문 사용")
    traingen_t20 = generate_batch_data_train(
        train20_pn, train20_label, train20_id, 30,
        candidate_news_body=None,
        expected_bodies=expected_bodies_train20,
        all_userid_str=train20_userid_str,
        all_newsid_str=train20_newsid_str,
        news_index_reverse=news_index_reverse,
        use_expected_body_positive_only=TRAIN_ON_TRAIN20_EXPECTED_BODY_POSITIVE_ONLY
    )
    steps_t20 = (last20_size + 29) // 30
    test_steps_t20 = len(all_test_id)
    test_exp_dir_t20 = os.path.join(body_gen_output_main, TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR)
    expected_bodies_test_t20 = None
    if os.path.isdir(test_exp_dir_t20):
        expected_bodies_test_t20 = load_expected_bodies_from_train_dir(test_exp_dir_t20)
        print(f"매 에폭 테스트셋 기대본문 평가: {TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR} ({len(expected_bodies_test_t20)}개)")
    expected_bodies_test_t20_2 = None
    if TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR_2:
        test_exp_dir_t20_2 = os.path.join(body_gen_output_main, TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR_2)
        if os.path.isdir(test_exp_dir_t20_2):
            expected_bodies_test_t20_2 = load_expected_bodies_from_train_dir(test_exp_dir_t20_2)
            print(f"매 에폭 테스트셋 기대본문(2) 평가: {TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR_2} ({len(expected_bodies_test_t20_2)}개)")
    print()
    for ep in range(TRAIN_ON_TRAIN20_EPOCHS):
        np.random.seed(SEED + ep)
        random.seed(SEED + ep)
        model.fit(traingen_t20, epochs=1, steps_per_epoch=steps_t20, verbose=1)
        # 매 에폭 테스트셋 실제본문 평가
        testgen_actual_t20 = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, 30, candidate_news_body=None)
        click_actual_t20 = model_test.predict(testgen_actual_t20, steps=test_steps_t20, verbose=0)
        all_mrr_a, all_ndcg_a, all_hit1_a = [], [], []
        for m in all_test_index:
            if m[1] <= len(click_actual_t20) and np.sum(all_test_label[m[0]:m[1]]) != 0:
                session_scores = click_actual_t20[m[0]:m[1], 0]
                session_labels = all_test_label[m[0]:m[1]]
                all_mrr_a.append(mrr_score(session_labels, session_scores))
                all_ndcg_a.append(ndcg_score(session_labels, session_scores, k=5))
                all_hit1_a.append(hit_at_k(session_labels, session_scores, k=1))
        print(f"\n{'='*60}")
        print(f"Epoch {ep+1}/{TRAIN_ON_TRAIN20_EPOCHS} - 테스트셋 평가")
        print(f"{'='*60}")
        print(f"[실제본문] MRR: {np.mean(all_mrr_a):.6f}  NDCG@5: {np.mean(all_ndcg_a):.6f}  Hit@1: {np.mean(all_hit1_a):.6f}")
        # 매 에폭 테스트셋 기대본문 평가
        if expected_bodies_test_t20 is not None:
            testgen_exp_t20 = generate_batch_data_test(
                all_test_pn, all_test_label, all_test_id, 30,
                candidate_news_body=None,
                expected_bodies=expected_bodies_test_t20,
                all_userid_str=all_test_userid_str,
                all_newsid_str=all_test_newsid_str,
                news_index_reverse=news_index_reverse
            )
            click_exp_t20 = model_test.predict(testgen_exp_t20, steps=test_steps_t20, verbose=0)
            all_mrr_e, all_ndcg_e, all_hit1_e = [], [], []
            for m in all_test_index:
                if m[1] <= len(click_exp_t20) and np.sum(all_test_label[m[0]:m[1]]) != 0:
                    session_scores = click_exp_t20[m[0]:m[1], 0]
                    session_labels = all_test_label[m[0]:m[1]]
                    all_mrr_e.append(mrr_score(session_labels, session_scores))
                    all_ndcg_e.append(ndcg_score(session_labels, session_scores, k=5))
                    all_hit1_e.append(hit_at_k(session_labels, session_scores, k=1))
            if all_mrr_e:
                print(f"[기대본문({TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR})] MRR: {np.mean(all_mrr_e):.6f}  NDCG@5: {np.mean(all_ndcg_e):.6f}  Hit@1: {np.mean(all_hit1_e):.6f}")
        else:
            print(f"[기대본문] 폴더 없음 ({test_exp_dir_t20}), 평가 생략")
        # 매 에폭 테스트셋 기대본문(2) 평가
        if expected_bodies_test_t20_2 is not None:
            testgen_exp_t20_2 = generate_batch_data_test(
                all_test_pn, all_test_label, all_test_id, 30,
                candidate_news_body=None,
                expected_bodies=expected_bodies_test_t20_2,
                all_userid_str=all_test_userid_str,
                all_newsid_str=all_test_newsid_str,
                news_index_reverse=news_index_reverse
            )
            click_exp_t20_2 = model_test.predict(testgen_exp_t20_2, steps=test_steps_t20, verbose=0)
            all_mrr_e2, all_ndcg_e2, all_hit1_e2 = [], [], []
            for m in all_test_index:
                if m[1] <= len(click_exp_t20_2) and np.sum(all_test_label[m[0]:m[1]]) != 0:
                    session_scores = click_exp_t20_2[m[0]:m[1], 0]
                    session_labels = all_test_label[m[0]:m[1]]
                    all_mrr_e2.append(mrr_score(session_labels, session_scores))
                    all_ndcg_e2.append(ndcg_score(session_labels, session_scores, k=5))
                    all_hit1_e2.append(hit_at_k(session_labels, session_scores, k=1))
            if all_mrr_e2:
                print(f"[기대본문(2)({TRAIN_ON_TRAIN20_TESTSET_EXPECTED_BODY_DIR_2})] MRR: {np.mean(all_mrr_e2):.6f}  NDCG@5: {np.mean(all_ndcg_e2):.6f}  Hit@1: {np.mean(all_hit1_e2):.6f}")
        print(f"{'='*60}\n")

else:
    for ep in range(MAIN_TRAINING_EPOCHS):
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

        actual_train_samples = len(all_train_id)
        steps_per_epoch = (actual_train_samples + 29) // 30
        model.fit(traingen, epochs=1, steps_per_epoch=steps_per_epoch)
        
        actual_test_samples = len(all_test_id)
        test_steps = actual_test_samples

        # [1] 테스트셋 실제본문으로 평가
        testgen_actual = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, 30, candidate_news_body=None)
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
        epoch_results = {
            'MRR': np.mean(all_mrr),
            'NDCG@5': np.mean(all_ndcg),
            'Hit@1': np.mean(all_hit1)
        }
        results.append([epoch_results['MRR'], epoch_results['NDCG@5'], epoch_results['Hit@1']])
        
        # [2] 테스트셋 기대본문(EVAL_TESTSET_EXPECTED_BODY_DIR)으로 평가
        epoch_results_expected = None
        if expected_bodies_main_test is not None:
            testgen_expected = generate_batch_data_test(
                all_test_pn, all_test_label, all_test_id, 30,
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
        # [3] 테스트셋 기대본문(2)(EVAL_TESTSET_EXPECTED_BODY_DIR_2)으로 평가
        epoch_results_expected_2 = None
        if expected_bodies_main_test_2 is not None:
            testgen_expected_2 = generate_batch_data_test(
                all_test_pn, all_test_label, all_test_id, 30,
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
        current_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else model.optimizer.learning_rate
        print(f"\n{'='*60}")
        print(f"Epoch {ep+1}/{MAIN_TRAINING_EPOCHS} - Test Results (LR: {current_lr:.6f})")
        print(f"{'='*60}")
        print(f"[실제본문] MRR: {epoch_results['MRR']:.6f}  NDCG@5: {epoch_results['NDCG@5']:.6f}  Hit@1: {epoch_results['Hit@1']:.6f}")
        if epoch_results_expected is not None:
            print(f"[기대본문({EVAL_TESTSET_EXPECTED_BODY_DIR})] MRR: {epoch_results_expected['MRR']:.6f}  NDCG@5: {epoch_results_expected['NDCG@5']:.6f}  Hit@1: {epoch_results_expected['Hit@1']:.6f}")
        if epoch_results_expected_2 is not None:
            print(f"[기대본문(2)({EVAL_TESTSET_EXPECTED_BODY_DIR_2})] MRR: {epoch_results_expected_2['MRR']:.6f}  NDCG@5: {epoch_results_expected_2['NDCG@5']:.6f}  Hit@1: {epoch_results_expected_2['Hit@1']:.6f}")
        print(f"{'='*60}\n")

    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("Final Results Summary (All Epochs)")
    print(f"{'='*60}")
    print(f"{'Epoch':<10} {'MRR':<12} {'NDCG@5':<12} {'Hit@1':<12}")
    print(f"{'-'*60}")
    for i, result in enumerate(results, 1):
        mrr, ndcg5, hit1 = result
        print(f"{i:<10} {mrr:<12.6f} {ndcg5:<12.6f} {hit1:<12.6f}")
    print(f"{'='*72}")

    # 최고 성능 찾기
    best_mrr_idx = np.argmax([r[0] for r in results])
    best_mrr_epoch = best_mrr_idx + 1
    best_hit1_idx = np.argmax([r[2] for r in results])
    best_hit1_epoch = best_hit1_idx + 1
    print(f"\nBest MRR  : Epoch {best_mrr_epoch} - {results[best_mrr_idx][0]:.6f}")
    print(f"Best Hit@1: Epoch {best_hit1_epoch} - {results[best_hit1_idx][2]:.6f}")
    print(f"{'='*60}\n")
