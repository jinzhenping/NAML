"""
NAML과 공유하는 MIND 경로·전처리·GloVe 임베딩.
cluster_train_users_kmeans.py 등에서 NAML.py 전체를 import하지 않고 사용.
"""
import glob
import os
import random

import numpy as np
from nltk.tokenize import word_tokenize

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 모델/데이터 공통 하이퍼파라미터 (NAML.py와 동일)
MAX_HISTORY_CLICKS = 50
MAX_SENT_LENGTH = 30
MAX_BODY_LENGTH = 300
npratio = 4

MIND_DATASET_SUBDIR = os.environ.get('MIND_DATASET_SUBDIR', 'MIND_2000')
_PROJECT_ROOT_NAML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIND_DATASET_PRESETS = {
    'MIND_2000': ('MIND_news.tsv', 'MIND_train_(2000).tsv', 'MIND_test_(2000).tsv'),
}
_FALLBACK_NEWS = 'MIND_news.tsv'
_FALLBACK_TRAIN = 'MIND_train_(1000).tsv'
_FALLBACK_TEST = 'MIND_test_(1000).tsv'


def _discover_mind_tsv_in_folder(subdir: str):
    """dataset/<subdir>/ 안에서 MIND_news.tsv + MIND_train_*.tsv 1개 + MIND_test_*.tsv 1개 자동 선택."""
    base = os.path.join(_PROJECT_ROOT_NAML, 'dataset', subdir)
    if not os.path.isdir(base):
        return None
    news_path = os.path.join(base, 'MIND_news.tsv')
    if os.path.isfile(news_path):
        news_name = 'MIND_news.tsv'
    else:
        cand = sorted(glob.glob(os.path.join(base, '*news*.tsv')))
        if len(cand) == 1:
            news_name = os.path.basename(cand[0])
        else:
            return None
    trains = sorted(glob.glob(os.path.join(base, 'MIND_train_*.tsv')))
    tests = sorted(glob.glob(os.path.join(base, 'MIND_test_*.tsv')))
    if len(trains) != 1 or len(tests) != 1:
        return None
    return news_name, os.path.basename(trains[0]), os.path.basename(tests[0])


def _resolve_mind_filenames():
    """환경변수 > 프리셋 > 자동탐색 > FALLBACK 순."""
    sub = MIND_DATASET_SUBDIR
    if sub in MIND_DATASET_PRESETS:
        n, tr, te = MIND_DATASET_PRESETS[sub]
    else:
        disc = _discover_mind_tsv_in_folder(sub)
        if disc:
            n, tr, te = disc
        else:
            n, tr, te = _FALLBACK_NEWS, _FALLBACK_TRAIN, _FALLBACK_TEST
    if 'MIND_NEWS_FILENAME' in os.environ:
        n = os.environ['MIND_NEWS_FILENAME']
    if 'MIND_TRAIN_FILENAME' in os.environ:
        tr = os.environ['MIND_TRAIN_FILENAME']
    if 'MIND_TEST_FILENAME' in os.environ:
        te = os.environ['MIND_TEST_FILENAME']
    return n, tr, te


MIND_NEWS_FILENAME, MIND_TRAIN_FILENAME, MIND_TEST_FILENAME = _resolve_mind_filenames()


def mind_data_path(filename: str) -> str:
    """프로젝트 루트 기준 dataset/<MIND_DATASET_SUBDIR>/<filename>"""
    return os.path.join(_PROJECT_ROOT_NAML, 'dataset', MIND_DATASET_SUBDIR, filename)


print(
    f"[데이터셋] dataset/{MIND_DATASET_SUBDIR}/ → "
    f"news={MIND_NEWS_FILENAME}, train={MIND_TRAIN_FILENAME}, test={MIND_TEST_FILENAME}"
)


def preprocess_user_file(
    train_file=None,
    test_file=None,
    news_index=None,
    npratio=4,
    expected_bodies_train=None,
    expected_bodies_test=None,
    word_dict=None,
):
    """
    MIND 데이터셋 형식에 맞게 전처리
    train_file: user, clicked_news, candidate_news, clicked (None이면 MIND_DATASET_SUBDIR 기준 기본 경로)
    test_file: user, clicked_news, candidate_news (clicked 없음)
    """
    if train_file is None:
        train_file = mind_data_path(MIND_TRAIN_FILENAME)
    if test_file is None:
        test_file = mind_data_path(MIND_TEST_FILENAME)
    userid_dict = {}

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = f.readlines()[1:]

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = f.readlines()

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
    all_train_userid_str = []
    all_train_newsid_str = []

    all_test_id = []
    all_test_pn = []
    all_test_label = []
    all_test_index = []
    all_test_userid_str = []
    all_test_newsid_str = []

    all_user_pos = []
    all_test_user_pos = []

    candidate_news_ids_train = set()
    candidate_news_ids_test = set()

    for line in train_data:
        parts = line.strip().split('\t')
        if len(parts) < 4:
            continue

        userid = parts[0]
        clicked_news = parts[1].split()
        candidate_news = parts[2].split()
        clicked = parts[3].split()

        clicked_news_ids = []
        for news_id in clicked_news:
            if news_id in news_index:
                clicked_news_ids.append(news_index[news_id])

        if len(clicked_news_ids) == 0:
            continue

        candidate_indices = []
        candidate_labels = []
        candidate_news_filtered = []
        for i, cand_id in enumerate(candidate_news):
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
                candidate_news_filtered.append(cand_id)
                candidate_news_ids_train.add(cand_id)
                is_clicked = int(clicked[i]) if i < len(clicked) else 0
                candidate_labels.append(is_clicked)

        if len(candidate_indices) < 2:
            continue
        if sum(candidate_labels) == 0:
            continue

        target_size = 1 + npratio

        if len(candidate_indices) > target_size:
            candidate_indices = candidate_indices[:target_size]
            candidate_labels = candidate_labels[:target_size]
            candidate_news_filtered = candidate_news_filtered[:target_size]
        elif len(candidate_indices) < target_size:
            padding_size = target_size - len(candidate_indices)
            candidate_indices += [0] * padding_size
            candidate_labels += [0] * padding_size
            candidate_news_filtered += [''] * padding_size

        combined = list(zip(candidate_indices, candidate_labels, candidate_news_filtered))
        random.shuffle(combined)
        shuffle_indices, shuffle_labels, shuffle_news_ids = zip(*combined)

        candidate_set = set([idx for idx in shuffle_indices if idx != 0])
        filtered_history = [idx for idx in clicked_news_ids if idx not in candidate_set]
        recent_history = (
            filtered_history[-MAX_HISTORY_CLICKS:]
            if len(filtered_history) >= MAX_HISTORY_CLICKS
            else filtered_history
        )
        allpos = [int(p) for p in recent_history]
        allpos += [0] * (MAX_HISTORY_CLICKS - len(allpos))

        all_train_pn.append(list(shuffle_indices))
        all_label.append(list(shuffle_labels))
        all_train_id.append(userid_dict[userid])
        all_train_userid_str.append(userid)
        all_train_newsid_str.append(list(shuffle_news_ids))
        all_user_pos.append(allpos)

    for line in test_data:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue

        userid = parts[0]
        clicked_news = parts[1].split()
        candidate_news = parts[2].split()

        clicked_news_ids = []
        for news_id in clicked_news:
            if news_id in news_index:
                clicked_news_ids.append(news_index[news_id])

        if len(clicked_news_ids) == 0 or len(candidate_news) == 0:
            continue

        sess_index = [len(all_test_pn)]

        if len(candidate_news) == 0 or candidate_news[0] not in news_index:
            continue

        candidate_indices = []
        candidate_news_filtered = []
        for cand_id in candidate_news:
            if cand_id in news_index:
                candidate_indices.append(news_index[cand_id])
                candidate_news_filtered.append(cand_id)
                candidate_news_ids_test.add(cand_id)

        if candidate_news[0] not in candidate_news_filtered:
            continue

        if len(candidate_indices) < 2:
            continue

        candidate_set = set([idx for idx in candidate_indices if idx != 0])
        filtered_history = [idx for idx in clicked_news_ids if idx not in candidate_set]
        recent_history = (
            filtered_history[-MAX_HISTORY_CLICKS:]
            if len(filtered_history) >= MAX_HISTORY_CLICKS
            else filtered_history
        )
        allpos = [int(p) for p in recent_history]
        allpos += [0] * (MAX_HISTORY_CLICKS - len(allpos))

        positive_index_in_filtered = candidate_news_filtered.index(candidate_news[0])
        candidate_labels = [
            1 if i == positive_index_in_filtered else 0 for i in range(len(candidate_indices))
        ]
        combined = list(zip(candidate_indices, candidate_labels, candidate_news_filtered))
        random.shuffle(combined)
        shuffle_indices, shuffle_labels, shuffle_news_ids = zip(*combined)
        shuffle_news_ids = list(shuffle_news_ids)

        for cand_idx, label, news_id_str in zip(shuffle_indices, shuffle_labels, shuffle_news_ids):
            all_test_pn.append(int(cand_idx))
            all_test_label.append(label)
            all_test_id.append(userid_dict[userid])
            all_test_userid_str.append(userid)
            all_test_newsid_str.append(news_id_str)
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

    return (
        userid_dict,
        all_train_pn,
        all_label,
        all_train_id,
        all_test_pn,
        all_test_label,
        all_test_id,
        all_user_pos,
        all_test_user_pos,
        all_test_index,
        candidate_news_ids_train,
        candidate_news_ids_test,
        all_train_userid_str,
        all_train_newsid_str,
        all_test_userid_str,
        all_test_newsid_str,
    )


def preprocess_news_file(file=None, expected_bodies_train=None, expected_bodies_test=None):
    """MIND 뉴스 데이터 전처리"""
    if file is None:
        file = mind_data_path(MIND_NEWS_FILENAME)
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

        title_tokens = word_tokenize(title.lower()) if title else []
        body_tokens = word_tokenize(body.lower()) if body else []

        news[news_id] = [cat, subcat, title_tokens, body_tokens]

        if cat not in category:
            category[cat] = len(category)
        if subcat not in subcategory:
            subcategory[subcat] = len(subcategory)

    word_dict_raw = {'PADDING': [0, 999999]}

    for docid in news:
        for word in news[docid][2]:
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                word_dict_raw[word] = [len(word_dict_raw), 1]
        for word in news[docid][3]:
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                word_dict_raw[word] = [len(word_dict_raw), 1]

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

    word_dict = {}
    for i in word_dict_raw:
        if word_dict_raw[i][1] >= 2:
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]

    print(f"단어 사전 크기: {len(word_dict)} (전체: {len(word_dict_raw)})")

    news_words = [[0] * 30]
    news_index = {'0': 0}

    for newsid in news:
        word_id = []
        news_index[newsid] = len(news_index)
        for word in news[newsid][2]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id = word_id[:30]
        news_words.append(word_id + [0] * (30 - len(word_id)))

    news_words = np.array(news_words, dtype='int32')

    news_body = [[0] * 300]
    for newsid in news:
        word_id = []
        for word in news[newsid][3]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id = word_id[:300]
        news_body.append(word_id + [0] * (300 - len(word_id)))

    news_body = np.array(news_body, dtype='int32')

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
    """GloVe 임베딩 로드 (없으면 랜덤 초기화)"""
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
        if Sigma.shape[0] == 300:
            norm = np.random.multivariate_normal(mu, Sigma + np.eye(300) * 0.01, 1)
        else:
            norm = np.random.normal(mu, 0.1, (1, 300))
    else:
        norm = np.random.normal(0, 0.1, (1, 300))

    for i in range(len(embedding_matrix)):
        if type(embedding_matrix[i]) == int:
            embedding_matrix[i] = np.reshape(norm, 300)

    embedding_matrix[0] = np.zeros(300, dtype='float32')
    embedding_matrix = np.array(embedding_matrix, dtype='float32')
    print(f"임베딩 행렬 shape: {embedding_matrix.shape}")
    return embedding_matrix
