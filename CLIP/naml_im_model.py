# -*- coding: utf-8 -*-
"""
NAML + IMRec visual impression (5번째 뷰).

텍스트 4뷰는 build_naml_models_with_image 와 동일.
5번째 뷰: title word embedding + local/global impression (NRMS-IM 뉴스 인코더).
"""
from __future__ import annotations

import math

import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from im_features import IM_GLOBAL_DIM, IM_LOCAL_DIM, IM_N_COVER
from naml_common import MAX_BODY_LENGTH, MAX_HISTORY_CLICKS, MAX_SENT_LENGTH, npratio
from naml_image_model import _title_rep, _user_rep_from_history


def _sqrt_dim(d: int):
    return math.sqrt(float(d))


def _memory_impression(emb, word_mask, cues, cue_mask, dropout_rate):
    """emb [B,L,D], cues [B,M,C], masks float 0/1."""
    d = int(emb.shape[-1])
    q = Dense(d, name="im_mem_q")(emb)
    k = Dense(d, name="im_mem_k")(cues)
    v = Dense(d, name="im_mem_v")(cues)
    scores = tf.matmul(q, k, transpose_b=True) / _sqrt_dim(d)
    cue_mask = tf.cast(cue_mask, emb.dtype)
    word_mask = tf.cast(word_mask, emb.dtype)
    scores = scores + (1.0 - tf.expand_dims(cue_mask, 1)) * -1e9
    attn = tf.nn.softmax(scores, axis=-1)
    attn = attn * tf.expand_dims(word_mask, -1)
    ctx = tf.matmul(attn, v)
    out = ctx + Dense(d, name="im_mem_vword")(emb)
    return Dropout(dropout_rate)(out)


def _self_attn_enh(x, mask, dropout_rate):
    d = int(x.shape[-1])
    q = Dense(d, name="im_sa_q")(x)
    k = Dense(d, name="im_sa_k")(x)
    scores = tf.matmul(q, k, transpose_b=True) / _sqrt_dim(d)
    mask = tf.cast(mask, x.dtype)
    scores = scores + (1.0 - mask[:, None, :]) * -1e9
    attn = tf.nn.softmax(scores, axis=-1)
    out = tf.matmul(attn, x)
    return Dropout(dropout_rate)(out)


def _additive_pool(x, mask, attn_hidden):
    mask = tf.cast(mask, x.dtype)
    a = Dense(attn_hidden, activation="tanh")(x)
    a = tf.squeeze(Dense(1)(a), axis=-1)
    a = a + (1.0 - mask) * -1e9
    empty = tf.reduce_sum(mask, axis=-1, keepdims=True) <= 0
    w = tf.nn.softmax(a, axis=-1)
    w = tf.where(empty, tf.zeros_like(w), w)
    return tf.reduce_sum(tf.expand_dims(w, -1) * x, axis=1)


def _global_gate(e, g, dim):
    o = Dense(dim, name="im_glob_proj")(g)
    gate = Dense(1, activation="sigmoid", name="im_glob_gate")(Concatenate()([e, o]))
    return gate * e + (1.0 - gate) * o


def _build_impression_rep(
    title_input,
    word_vis_in,
    word_vis_mask_in,
    cover_in,
    category_vis_in,
    global_in,
    embedding_layer,
    dropout_rate,
    attn_hidden,
):
    """Returns [B, emb_dim] before projection to nf."""
    emb = Dropout(dropout_rate)(embedding_layer(title_input))
    word_mask = tf.cast(word_vis_mask_in, emb.dtype)
    cat = tf.expand_dims(category_vis_in, 1)
    cues = Concatenate(axis=1)([word_vis_in, cover_in, cat])
    cues = Dense(IM_LOCAL_DIM, name="im_cue_proj")(cues)
    b = tf.shape(cues)[0]
    cover_ones = tf.ones((b, IM_N_COVER), dtype=emb.dtype)
    cat_ones = tf.ones((b, 1), dtype=emb.dtype)
    cue_mask = Concatenate(axis=1)([word_mask, cover_ones, cat_ones])
    hat = _memory_impression(emb, word_mask, cues, cue_mask, dropout_rate)
    star = _self_attn_enh(hat, word_mask, dropout_rate)
    e = _additive_pool(star, word_mask, attn_hidden)
    emb_dim = int(emb.shape[-1])
    return _global_gate(e, global_in, emb_dim)


def build_naml_models_im(
    word_dict,
    embedding_mat,
    category,
    subcategory,
    learning_rate,
    clear_session=True,
    *,
    dropout_rate=0.3,
    cnn_filters=400,
    cnn_kernel_size=3,
    attention_dense_dim=200,
    category_emb_dim=50,
):
    """Full-text NAML + impression 5th view."""
    if clear_session:
        keras.backend.clear_session()

    d = float(dropout_rate)
    nf = int(cnn_filters)
    nk = int(cnn_kernel_size)
    ad = int(attention_dense_dim)
    cem = int(category_emb_dim)
    L = MAX_SENT_LENGTH

    title_input = Input(shape=(L,), dtype="int32", name="title_input")
    body_input = Input(shape=(MAX_BODY_LENGTH,), dtype="int32")
    vinput = Input((1,), dtype="int32")
    svinput = Input((1,), dtype="int32")
    im_word_vis = Input(shape=(L, IM_LOCAL_DIM), dtype="float32", name="im_word_vis")
    im_word_mask = Input(shape=(L,), dtype="float32", name="im_word_mask")
    im_cover = Input(shape=(IM_N_COVER, IM_LOCAL_DIM), dtype="float32", name="im_cover")
    im_cat_vis = Input(shape=(IM_LOCAL_DIM,), dtype="float32", name="im_cat_vis")
    im_global = Input(shape=(IM_GLOBAL_DIM,), dtype="float32", name="im_global")

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

    v_embedding_layer = Embedding(len(category) + 1, cem, trainable=True)
    sv_embedding_layer = Embedding(len(subcategory) + 1, cem, trainable=True)
    v_embedding = Dense(nf, activation="relu")(Flatten()(v_embedding_layer(vinput)))
    sv_embedding = Dense(nf, activation="relu")(Flatten()(sv_embedding_layer(svinput)))

    imp_vec = _build_impression_rep(
        title_input,
        im_word_vis,
        im_word_mask,
        im_cover,
        im_cat_vis,
        im_global,
        embedding_layer,
        d,
        ad,
    )
    impression_rep = Dense(nf, activation="relu", name="impression_proj")(imp_vec)

    all_channel = [title_rep, body_rep, v_embedding, sv_embedding, impression_rep]
    views = concatenate([Reshape((1, -1))(channel) for channel in all_channel], axis=1)
    attentionv = Dense(ad, activation="tanh")(views)
    attention_weightv = Reshape((-1,))(Dense(1)(attentionv))
    attention_weightv = Activation("softmax")(attention_weightv)
    newsrep = keras.layers.Dot((1, 1))([views, attention_weightv])

    news_inputs = [
        title_input,
        body_input,
        vinput,
        svinput,
        im_word_vis,
        im_word_mask,
        im_cover,
        im_cat_vis,
        im_global,
    ]
    newsEncoder = Model(news_inputs, newsrep, name="newsEncoder")

    MAX_SENTS = MAX_HISTORY_CLICKS
    browsed_news_input = [keras.Input((L,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_body_input = [keras.Input((MAX_BODY_LENGTH,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_v_input = [keras.Input((1,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_sv_input = [keras.Input((1,), dtype="int32") for _ in range(MAX_SENTS)]
    browsed_im_wv = [keras.Input((L, IM_LOCAL_DIM), dtype="float32") for _ in range(MAX_SENTS)]
    browsed_im_wm = [keras.Input((L,), dtype="float32") for _ in range(MAX_SENTS)]
    browsed_im_cover = [keras.Input((IM_N_COVER, IM_LOCAL_DIM), dtype="float32") for _ in range(MAX_SENTS)]
    browsed_im_cat = [keras.Input((IM_LOCAL_DIM,), dtype="float32") for _ in range(MAX_SENTS)]
    browsed_im_glob = [keras.Input((IM_GLOBAL_DIM,), dtype="float32") for _ in range(MAX_SENTS)]

    def _enc_hist(i):
        return newsEncoder(
            [
                browsed_news_input[i],
                browsed_body_input[i],
                browsed_v_input[i],
                browsed_sv_input[i],
                browsed_im_wv[i],
                browsed_im_wm[i],
                browsed_im_cover[i],
                browsed_im_cat[i],
                browsed_im_glob[i],
            ]
        )

    browsednews = [_enc_hist(_) for _ in range(MAX_SENTS)]
    user_rep = _user_rep_from_history(browsednews, ad)

    n_cand = 1 + npratio
    candidates_title = [keras.Input((L,), dtype="int32") for _ in range(n_cand)]
    candidates_body = [keras.Input((MAX_BODY_LENGTH,), dtype="int32") for _ in range(n_cand)]
    candidates_v = [keras.Input((1,), dtype="int32") for _ in range(n_cand)]
    candidates_sv = [keras.Input((1,), dtype="int32") for _ in range(n_cand)]
    candidates_im_wv = [keras.Input((L, IM_LOCAL_DIM), dtype="float32") for _ in range(n_cand)]
    candidates_im_wm = [keras.Input((L,), dtype="float32") for _ in range(n_cand)]
    candidates_im_cover = [keras.Input((IM_N_COVER, IM_LOCAL_DIM), dtype="float32") for _ in range(n_cand)]
    candidates_im_cat = [keras.Input((IM_LOCAL_DIM,), dtype="float32") for _ in range(n_cand)]
    candidates_im_glob = [keras.Input((IM_GLOBAL_DIM,), dtype="float32") for _ in range(n_cand)]

    def _enc_cand(i):
        return newsEncoder(
            [
                candidates_title[i],
                candidates_body[i],
                candidates_v[i],
                candidates_sv[i],
                candidates_im_wv[i],
                candidates_im_wm[i],
                candidates_im_cover[i],
                candidates_im_cat[i],
                candidates_im_glob[i],
            ]
        )

    candidate_vecs = [_enc_cand(_) for _ in range(n_cand)]
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
        + candidates_im_wv
        + browsed_im_wv
        + candidates_im_wm
        + browsed_im_wm
        + candidates_im_cover
        + browsed_im_cover
        + candidates_im_cat
        + browsed_im_cat
        + candidates_im_glob
        + browsed_im_glob,
        logits,
    )

    candidate_one_title = keras.Input((L,))
    candidate_one_body = keras.Input((MAX_BODY_LENGTH,))
    candidate_one_v = keras.Input((1,))
    candidate_one_sv = keras.Input((1,))
    candidate_one_im_wv = keras.Input((L, IM_LOCAL_DIM), dtype="float32")
    candidate_one_im_wm = keras.Input((L,), dtype="float32")
    candidate_one_im_cover = keras.Input((IM_N_COVER, IM_LOCAL_DIM), dtype="float32")
    candidate_one_im_cat = keras.Input((IM_LOCAL_DIM,), dtype="float32")
    candidate_one_im_glob = keras.Input((IM_GLOBAL_DIM,), dtype="float32")
    candidate_one_vec = newsEncoder(
        [
            candidate_one_title,
            candidate_one_body,
            candidate_one_v,
            candidate_one_sv,
            candidate_one_im_wv,
            candidate_one_im_wm,
            candidate_one_im_cover,
            candidate_one_im_cat,
            candidate_one_im_glob,
        ]
    )
    score = keras.layers.Activation(keras.activations.sigmoid)(
        keras.layers.dot([user_rep, candidate_one_vec], axes=-1)
    )
    model_test = Model(
        [candidate_one_title]
        + browsed_news_input
        + [candidate_one_body]
        + browsed_body_input
        + [candidate_one_v]
        + browsed_v_input
        + [candidate_one_sv]
        + browsed_sv_input
        + [candidate_one_im_wv]
        + browsed_im_wv
        + [candidate_one_im_wm]
        + browsed_im_wm
        + [candidate_one_im_cover]
        + browsed_im_cover
        + [candidate_one_im_cat]
        + browsed_im_cat
        + [candidate_one_im_glob]
        + browsed_im_glob,
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
        "use_impression": True,
        "title_only": False,
    }
