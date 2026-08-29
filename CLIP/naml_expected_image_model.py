# -*- coding: utf-8 -*-
"""
히스토리: title + 실제본문 + cat + subcat (이미지 뷰 없음)
후보: title + cat + subcat + 기대이미지 (본문 없음)

단어 임베딩 / title CNN / cat·subcat 임베딩은 공유하고,
view-attention 만 히스토리와 후보가 따로 쓴다.
"""
import keras
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam

import naml_common
from naml_common import MAX_BODY_LENGTH, MAX_SENT_LENGTH, npratio
from naml_image_model import _title_rep, _user_rep_from_history


def _cat_rep(vinput, embedding_layer, nf):
    return Dense(nf, activation="relu")(Flatten()(embedding_layer(vinput)))


def build_naml_models_expected_image(
    word_dict,
    embedding_mat,
    category,
    subcategory,
    learning_rate,
    clip_dim,
    clear_session=True,
    *,
    dropout_rate=0.3,
    cnn_filters=400,
    cnn_kernel_size=3,
    attention_dense_dim=200,
    category_emb_dim=50,
):
    if clear_session:
        keras.backend.clear_session()

    d = float(dropout_rate)
    nf = int(cnn_filters)
    nk = int(cnn_kernel_size)
    ad = int(attention_dense_dim)
    cem = int(category_emb_dim)
    clip_dim = int(clip_dim)
    MAX_SENTS = int(naml_common.MAX_HISTORY_CLICKS)

    embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)
    v_embedding_layer = Embedding(len(category) + 1, cem, trainable=True)
    sv_embedding_layer = Embedding(len(subcategory) + 1, cem, trainable=True)

    title_in = Input(shape=(MAX_SENT_LENGTH,), dtype="int32")
    titleEncoder = Model(
        title_in,
        _title_rep(title_in, embedding_layer, d, nf, nk, ad),
        name="titleEncoder",
    )

    body_in = Input(shape=(MAX_BODY_LENGTH,), dtype="int32")
    embedded_body = Dropout(d)(embedding_layer(body_in))
    body_cnn = Conv1D(filters=nf, kernel_size=nk, padding="same", activation="relu", strides=1)(
        embedded_body
    )
    body_cnn = Dropout(d)(body_cnn)
    attention_body = Dense(ad, activation="tanh")(body_cnn)
    attention_body = Flatten()(Dense(1)(attention_body))
    attention_weight_body = Activation("softmax")(attention_body)
    body_rep = keras.layers.Dot((1, 1))([body_cnn, attention_weight_body])
    bodyEncoder = Model(body_in, body_rep, name="bodyEncoder")

    v_in = Input((1,), dtype="int32")
    catEncoder = Model(v_in, _cat_rep(v_in, v_embedding_layer, nf), name="catEncoder")
    sv_in = Input((1,), dtype="int32")
    subcatEncoder = Model(sv_in, _cat_rep(sv_in, sv_embedding_layer, nf), name="subcatEncoder")

    image_in = Input(shape=(clip_dim,), dtype="float32", name="image_input")
    imageEncoder = Model(
        image_in,
        Dense(nf, activation="relu", name="image_proj")(image_in),
        name="imageEncoder",
    )

    ht = Input(shape=(MAX_SENT_LENGTH,), dtype="int32")
    hb = Input(shape=(MAX_BODY_LENGTH,), dtype="int32")
    hv = Input((1,), dtype="int32")
    hsv = Input((1,), dtype="int32")
    hist_views = concatenate(
        [
            Reshape((1, -1))(titleEncoder(ht)),
            Reshape((1, -1))(bodyEncoder(hb)),
            Reshape((1, -1))(catEncoder(hv)),
            Reshape((1, -1))(subcatEncoder(hsv)),
        ],
        axis=1,
    )
    hist_att = Dense(ad, activation="tanh")(hist_views)
    hist_w = Activation("softmax")(Reshape((-1,))(Dense(1)(hist_att)))
    hist_newsrep = keras.layers.Dot((1, 1))([hist_views, hist_w])
    histEncoder = Model([ht, hb, hv, hsv], hist_newsrep, name="histEncoder")

    ct = Input(shape=(MAX_SENT_LENGTH,), dtype="int32")
    cv = Input((1,), dtype="int32")
    csv = Input((1,), dtype="int32")
    cimg = Input(shape=(clip_dim,), dtype="float32")
    cand_views = concatenate(
        [
            Reshape((1, -1))(titleEncoder(ct)),
            Reshape((1, -1))(catEncoder(cv)),
            Reshape((1, -1))(subcatEncoder(csv)),
            Reshape((1, -1))(imageEncoder(cimg)),
        ],
        axis=1,
    )
    cand_att = Dense(ad, activation="tanh")(cand_views)
    cand_w = Activation("softmax")(Reshape((-1,))(Dense(1)(cand_att)))
    cand_newsrep = keras.layers.Dot((1, 1))([cand_views, cand_w])
    candEncoder = Model([ct, cv, csv, cimg], cand_newsrep, name="candEncoder")

    browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_body_input = [keras.Input((MAX_BODY_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_v_input = [keras.Input((1,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_sv_input = [keras.Input((1,), dtype="int32") for _ in range(MAX_SENTS)]
    browsednews = [
        histEncoder(
            [
                browsed_news_input[_],
                browsed_body_input[_],
                browsed_v_input[_],
                browsed_sv_input[_],
            ]
        )
        for _ in range(MAX_SENTS)
    ]
    user_rep = _user_rep_from_history(browsednews, ad)

    n_cand = 1 + npratio
    candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(n_cand)]
    candidates_v = [keras.Input((1,), dtype="int32") for _ in range(n_cand)]
    candidates_sv = [keras.Input((1,), dtype="int32") for _ in range(n_cand)]
    candidates_image = [keras.Input((clip_dim,), dtype="float32") for _ in range(n_cand)]
    candidate_vecs = [
        candEncoder(
            [
                candidates_title[_],
                candidates_v[_],
                candidates_sv[_],
                candidates_image[_],
            ]
        )
        for _ in range(n_cand)
    ]
    logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
    logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))
    model = Model(
        candidates_title
        + browsed_news_input
        + browsed_body_input
        + candidates_v
        + browsed_v_input
        + candidates_sv
        + browsed_sv_input
        + candidates_image,
        logits,
    )

    candidate_one_title = keras.Input((MAX_SENT_LENGTH,))
    candidate_one_v = keras.Input((1,))
    candidate_one_sv = keras.Input((1,))
    candidate_one_image = keras.Input((clip_dim,), dtype="float32")
    candidate_one_vec = candEncoder(
        [candidate_one_title, candidate_one_v, candidate_one_sv, candidate_one_image]
    )
    score = keras.layers.Activation(keras.activations.sigmoid)(
        keras.layers.dot([user_rep, candidate_one_vec], axes=-1)
    )
    model_test = keras.Model(
        [candidate_one_title]
        + browsed_news_input
        + browsed_body_input
        + [candidate_one_v]
        + browsed_v_input
        + [candidate_one_sv]
        + browsed_sv_input
        + [candidate_one_image],
        score,
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["acc"],
    )
    return {
        "model": model,
        "model_test": model_test,
        "histEncoder": histEncoder,
        "candEncoder": candEncoder,
        "user_rep": user_rep,
        "MAX_SENTS": MAX_SENTS,
        "clip_dim": clip_dim,
    }
