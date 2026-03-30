"""
NAML 그래프 구축 (NAML.py / cluster_train_users_kmeans.py 공용).
"""
import keras
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from naml_common import MAX_BODY_LENGTH, MAX_HISTORY_CLICKS, MAX_SENT_LENGTH, npratio


def build_naml_models(word_dict, embedding_mat, category, subcategory, learning_rate, clear_session=True):
    """
    NAML 학습용 model, 평가용 model_test 및 user_rep 서브그래프 구성요소 반환.
    clear_session: False이면 이전 그래프 유지 (동일 프로세스에서 교사+학생 이중 빌드 등).
    """
    if clear_session:
        keras.backend.clear_session()

    MAX_SENTS = MAX_HISTORY_CLICKS
    title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

    body_input = Input(shape=(MAX_BODY_LENGTH,), dtype='int32')
    embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)

    embedded_sequences_title = embedding_layer(title_input)
    embedded_sequences_title = Dropout(0.3)(embedded_sequences_title)

    embedded_sequences_body = embedding_layer(body_input)
    embedded_sequences_body = Dropout(0.3)(embedded_sequences_body)

    title_cnn = Conv1D(
        filters=400, kernel_size=3, padding='same', activation='relu', strides=1
    )(embedded_sequences_title)
    title_cnn = Dropout(0.3)(title_cnn)

    attention = Dense(200, activation='tanh')(title_cnn)
    attention = Flatten()(Dense(1)(attention))
    attention_weight = Activation('softmax')(attention)
    title_rep = keras.layers.Dot((1, 1))([title_cnn, attention_weight])

    body_cnn = Conv1D(
        filters=400, kernel_size=3, padding='same', activation='relu', strides=1
    )(embedded_sequences_body)
    body_cnn = Dropout(0.3)(body_cnn)

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

    views = concatenate([Reshape((1, -1))(channel) for channel in all_channel], axis=1)

    attentionv = Dense(200, activation='tanh')(views)

    attention_weightv = Reshape((-1,))(Dense(1)(attentionv))
    attention_weightv = Activation('softmax')(attention_weightv)

    newsrep = keras.layers.Dot((1, 1))([views, attention_weightv])

    newsEncoder = Model([title_input, body_input, vinput, svinput], newsrep)

    browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
    browsed_body_input = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]

    browsed_v_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]
    browsed_sv_input = [keras.Input((1,), dtype='int32') for _ in range(MAX_SENTS)]

    browsednews = [
        newsEncoder(
            [
                browsed_news_input[_],
                browsed_body_input[_],
                browsed_v_input[_],
                browsed_sv_input[_],
            ]
        )
        for _ in range(MAX_SENTS)
    ]
    browsednewsrep = concatenate([Reshape((1, -1))(news) for news in browsednews], axis=1)

    attentionn = Dense(200, activation='tanh')(browsednewsrep)
    attentionn = Flatten()(Dense(1)(attentionn))
    attention_weightn = Activation('softmax')(attentionn)
    user_rep = keras.layers.Dot((1, 1))([browsednewsrep, attention_weightn])

    candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1 + npratio)]

    candidates_body = [keras.Input((MAX_BODY_LENGTH,), dtype='int32') for _ in range(1 + npratio)]

    candidates_v = [keras.Input((1,), dtype='int32') for _ in range(1 + npratio)]

    candidates_sv = [keras.Input((1,), dtype='int32') for _ in range(1 + npratio)]
    candidate_vecs = [
        newsEncoder([candidates_title[_], candidates_body[_], candidates_v[_], candidates_sv[_]])
        for _ in range(1 + npratio)
    ]

    logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
    logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))

    model = Model(
        candidates_title
        + browsed_news_input
        + candidates_body
        + browsed_body_input
        + candidates_v
        + browsed_v_input
        + candidates_sv
        + browsed_sv_input,
        logits,
    )

    candidate_one_title = keras.Input((MAX_SENT_LENGTH,))

    candidate_one_body = keras.Input((MAX_BODY_LENGTH,))

    candidate_one_v = keras.Input((1,))

    candidate_one_sv = keras.Input((1,))

    candidate_one_vec = newsEncoder([candidate_one_title, candidate_one_body, candidate_one_v, candidate_one_sv])

    score = keras.layers.Activation(keras.activations.sigmoid)(
        keras.layers.dot([user_rep, candidate_one_vec], axes=-1)
    )
    model_test = keras.Model(
        [candidate_one_title]
        + browsed_news_input
        + [candidate_one_body]
        + browsed_body_input
        + [candidate_one_v]
        + browsed_v_input
        + [candidate_one_sv]
        + browsed_sv_input,
        score,
    )

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

    return {
        'model': model,
        'model_test': model_test,
        'newsEncoder': newsEncoder,
        'user_rep': user_rep,
        'browsed_news_input': browsed_news_input,
        'browsed_body_input': browsed_body_input,
        'browsed_v_input': browsed_v_input,
        'browsed_sv_input': browsed_sv_input,
        'MAX_SENTS': MAX_SENTS,
    }
