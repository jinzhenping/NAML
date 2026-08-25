# -*- coding: utf-8 -*-
"""
S1: NAML 뉴스 인코더
S2: S1 + CLIP 이미지 뷰 (view-attention)

title_only 이면 텍스트는 title CNN만 사용 (body/cat/subcat 제외).
"""
import keras
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from naml_common import MAX_BODY_LENGTH, MAX_HISTORY_CLICKS, MAX_SENT_LENGTH, npratio


def _title_rep(title_input, embedding_layer, d, nf, nk, ad):
    embedded = Dropout(d)(embedding_layer(title_input))
    title_cnn = Conv1D(filters=nf, kernel_size=nk, padding="same", activation="relu", strides=1)(embedded)
    title_cnn = Dropout(d)(title_cnn)
    attention = Dense(ad, activation="tanh")(title_cnn)
    attention = Flatten()(Dense(1)(attention))
    attention_weight = Activation("softmax")(attention)
    return keras.layers.Dot((1, 1))([title_cnn, attention_weight])


def _user_rep_from_history(browsednews, ad):
    browsednewsrep = concatenate([Reshape((1, -1))(news) for news in browsednews], axis=1)
    attentionn = Dense(ad, activation="tanh")(browsednewsrep)
    attentionn = Flatten()(Dense(1)(attentionn))
    attention_weightn = Activation("softmax")(attentionn)
    return keras.layers.Dot((1, 1))([browsednewsrep, attention_weightn])


def build_naml_models_title_only(
    word_dict,
    embedding_mat,
    learning_rate,
    clear_session=True,
    *,
    dropout_rate=0.3,
    cnn_filters=400,
    cnn_kernel_size=3,
    attention_dense_dim=200,
    category_emb_dim=50,
):
    """S1 title-only: 뉴스 표현 = title CNN + word attention."""
    del category_emb_dim
    if clear_session:
        keras.backend.clear_session()

    d = float(dropout_rate)
    nf = int(cnn_filters)
    nk = int(cnn_kernel_size)
    ad = int(attention_dense_dim)
    MAX_SENTS = MAX_HISTORY_CLICKS

    title_input = Input(shape=(MAX_SENT_LENGTH,), dtype="int32")
    embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)
    title_rep = _title_rep(title_input, embedding_layer, d, nf, nk, ad)
    newsEncoder = Model([title_input], title_rep, name="newsEncoder")

    browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsednews = [newsEncoder(browsed_news_input[_]) for _ in range(MAX_SENTS)]
    user_rep = _user_rep_from_history(browsednews, ad)

    n_cand = 1 + npratio
    candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(n_cand)]
    candidate_vecs = [newsEncoder(candidates_title[_]) for _ in range(n_cand)]
    logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
    logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))
    model = Model(candidates_title + browsed_news_input, logits)

    candidate_one_title = keras.Input((MAX_SENT_LENGTH,))
    candidate_one_vec = newsEncoder(candidate_one_title)
    score = keras.layers.Activation(keras.activations.sigmoid)(
        keras.layers.dot([user_rep, candidate_one_vec], axes=-1)
    )
    model_test = keras.Model([candidate_one_title] + browsed_news_input, score)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["acc"],
    )
    return {
        "model": model,
        "model_test": model_test,
        "newsEncoder": newsEncoder,
        "user_rep": user_rep,
        "MAX_SENTS": MAX_SENTS,
        "use_image": False,
        "title_only": True,
    }


def build_naml_models_title_image(
    word_dict,
    embedding_mat,
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
    """S2 title-only: title 뷰 + CLIP 이미지 뷰, view-attention."""
    del category_emb_dim
    if clear_session:
        keras.backend.clear_session()

    d = float(dropout_rate)
    nf = int(cnn_filters)
    nk = int(cnn_kernel_size)
    ad = int(attention_dense_dim)
    clip_dim = int(clip_dim)
    MAX_SENTS = MAX_HISTORY_CLICKS

    title_input = Input(shape=(MAX_SENT_LENGTH,), dtype="int32")
    embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)
    title_rep = _title_rep(title_input, embedding_layer, d, nf, nk, ad)

    image_input = Input(shape=(clip_dim,), dtype="float32", name="image_input")
    image_rep = Dense(nf, activation="relu", name="image_proj")(image_input)

    views = concatenate([Reshape((1, -1))(ch) for ch in [title_rep, image_rep]], axis=1)
    attentionv = Dense(ad, activation="tanh")(views)
    attention_weightv = Reshape((-1,))(Dense(1)(attentionv))
    attention_weightv = Activation("softmax")(attention_weightv)
    newsrep = keras.layers.Dot((1, 1))([views, attention_weightv])
    newsEncoder = Model([title_input, image_input], newsrep, name="newsEncoder")

    browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_image_input = [keras.Input((clip_dim,), dtype="float32") for _ in range(MAX_SENTS)]
    browsednews = [
        newsEncoder([browsed_news_input[_], browsed_image_input[_]]) for _ in range(MAX_SENTS)
    ]
    user_rep = _user_rep_from_history(browsednews, ad)

    n_cand = 1 + npratio
    candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(n_cand)]
    candidates_image = [keras.Input((clip_dim,), dtype="float32") for _ in range(n_cand)]
    candidate_vecs = [
        newsEncoder([candidates_title[_], candidates_image[_]]) for _ in range(n_cand)
    ]
    logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
    logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))
    model = Model(
        candidates_title + browsed_news_input + candidates_image + browsed_image_input,
        logits,
    )

    candidate_one_title = keras.Input((MAX_SENT_LENGTH,))
    candidate_one_image = keras.Input((clip_dim,), dtype="float32")
    candidate_one_vec = newsEncoder([candidate_one_title, candidate_one_image])
    score = keras.layers.Activation(keras.activations.sigmoid)(
        keras.layers.dot([user_rep, candidate_one_vec], axes=-1)
    )
    model_test = keras.Model(
        [candidate_one_title] + browsed_news_input + [candidate_one_image] + browsed_image_input,
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
        "newsEncoder": newsEncoder,
        "user_rep": user_rep,
        "MAX_SENTS": MAX_SENTS,
        "use_image": True,
        "clip_dim": clip_dim,
        "title_only": True,
    }


def build_naml_models_with_image(
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

    MAX_SENTS = MAX_HISTORY_CLICKS
    title_input = Input(shape=(MAX_SENT_LENGTH,), dtype="int32")
    body_input = Input(shape=(MAX_BODY_LENGTH,), dtype="int32")
    embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)

    title_rep = _title_rep(title_input, embedding_layer, d, nf, nk, ad)

    embedded_sequences_body = Dropout(d)(embedding_layer(body_input))
    body_cnn = Conv1D(filters=nf, kernel_size=nk, padding="same", activation="relu", strides=1)(
        embedded_sequences_body
    )
    body_cnn = Dropout(d)(body_cnn)
    attention_body = Dense(ad, activation="tanh")(body_cnn)
    attention_body = Flatten()(Dense(1)(attention_body))
    attention_weight_body = Activation("softmax")(attention_body)
    body_rep = keras.layers.Dot((1, 1))([body_cnn, attention_weight_body])

    vinput = Input((1,), dtype="int32")
    svinput = Input((1,), dtype="int32")
    v_embedding_layer = Embedding(len(category) + 1, cem, trainable=True)
    sv_embedding_layer = Embedding(len(subcategory) + 1, cem, trainable=True)
    v_embedding = Dense(nf, activation="relu")(Flatten()(v_embedding_layer(vinput)))
    sv_embedding = Dense(nf, activation="relu")(Flatten()(sv_embedding_layer(svinput)))

    image_input = Input(shape=(clip_dim,), dtype="float32", name="image_input")
    image_rep = Dense(nf, activation="relu", name="image_proj")(image_input)

    all_channel = [title_rep, body_rep, v_embedding, sv_embedding, image_rep]
    views = concatenate([Reshape((1, -1))(channel) for channel in all_channel], axis=1)
    attentionv = Dense(ad, activation="tanh")(views)
    attention_weightv = Reshape((-1,))(Dense(1)(attentionv))
    attention_weightv = Activation("softmax")(attention_weightv)
    newsrep = keras.layers.Dot((1, 1))([views, attention_weightv])

    newsEncoder = Model(
        [title_input, body_input, vinput, svinput, image_input],
        newsrep,
        name="newsEncoder",
    )

    browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_body_input = [keras.Input((MAX_BODY_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_v_input = [keras.Input((1,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_sv_input = [keras.Input((1,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_image_input = [keras.Input((clip_dim,), dtype="float32") for _ in range(MAX_SENTS)]

    browsednews = [
        newsEncoder(
            [
                browsed_news_input[_],
                browsed_body_input[_],
                browsed_v_input[_],
                browsed_sv_input[_],
                browsed_image_input[_],
            ]
        )
        for _ in range(MAX_SENTS)
    ]
    user_rep = _user_rep_from_history(browsednews, ad)

    n_cand = 1 + npratio
    candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype="int32") for _ in range(n_cand)]
    candidates_body = [keras.Input((MAX_BODY_LENGTH,), dtype="int32") for _ in range(n_cand)]
    candidates_v = [keras.Input((1,), dtype="int32") for _ in range(n_cand)]
    candidates_sv = [keras.Input((1,), dtype="int32") for _ in range(n_cand)]
    candidates_image = [keras.Input((clip_dim,), dtype="float32") for _ in range(n_cand)]
    candidate_vecs = [
        newsEncoder(
            [
                candidates_title[_],
                candidates_body[_],
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
        + candidates_body
        + browsed_body_input
        + candidates_v
        + browsed_v_input
        + candidates_sv
        + browsed_sv_input
        + candidates_image
        + browsed_image_input,
        logits,
    )

    candidate_one_title = keras.Input((MAX_SENT_LENGTH,))
    candidate_one_body = keras.Input((MAX_BODY_LENGTH,))
    candidate_one_v = keras.Input((1,))
    candidate_one_sv = keras.Input((1,))
    candidate_one_image = keras.Input((clip_dim,), dtype="float32")
    candidate_one_vec = newsEncoder(
        [
            candidate_one_title,
            candidate_one_body,
            candidate_one_v,
            candidate_one_sv,
            candidate_one_image,
        ]
    )
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
        + browsed_sv_input
        + [candidate_one_image]
        + browsed_image_input,
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
        "newsEncoder": newsEncoder,
        "user_rep": user_rep,
        "MAX_SENTS": MAX_SENTS,
        "use_image": True,
        "clip_dim": clip_dim,
        "title_only": False,
    }
