# -*- coding: utf-8 -*-
"""
히스토리: title + 실제본문 + cat + subcat (이미지 뷰 없음)
후보: title + cat + subcat + 기대이미지 (본문 없음)

단어 임베딩 / title CNN / cat·subcat 임베딩은 공유하고,
view-attention 만 히스토리와 후보가 따로 쓴다.
"""
import os

import keras
from keras.layers import *
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import naml_common
from naml_common import MAX_BODY_LENGTH, MAX_SENT_LENGTH, npratio
from naml_image_model import _title_rep, _user_rep_from_history

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


def _h5_decode(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf8")
    return str(x)


def load_h5_weights_by_name(model, filepath: str) -> int:
    """
    Keras HDF5 load_weights 는 중첩 Model + 공유 레이어에서
    Conv1D Keras1 transpose 를 잘못 적용한다 (axes don't match array).
    저장된 그룹을 레이어 이름으로 찾아 preprocess 없이 넣는다.
    """
    import h5py
    import numpy as np

    filepath = os.path.abspath(filepath)
    n_set = 0
    with h5py.File(filepath, "r") as f:
        root = f
        if "layer_names" not in root.attrs and "model_weights" in root:
            root = root["model_weights"]
        if "layer_names" not in root.attrs:
            raise ValueError(f"HDF5에 layer_names가 없습니다: {filepath}")

        layers = {}
        stack = [model]
        seen = set()
        while stack:
            cur = stack.pop()
            for layer in getattr(cur, "layers", []):
                if id(layer) in seen:
                    continue
                seen.add(id(layer))
                layers.setdefault(layer.name, layer)
                if hasattr(layer, "layers"):
                    stack.append(layer)

        layer_names = [_h5_decode(n) for n in root.attrs["layer_names"]]
        missing = []
        shape_mismatch = []
        for name in layer_names:
            if name not in root:
                continue
            g = root[name]
            raw_names = list(g.attrs.get("weight_names", []))
            if not raw_names:
                continue
            values = [np.asarray(g[_h5_decode(wn)]) for wn in raw_names]
            layer = layers.get(name)
            if layer is None:
                missing.append(name)
                continue
            symbolic = list(layer.weights)
            if len(symbolic) != len(values):
                raise ValueError(
                    f"{name}: 파일 {len(values)}개 vs 모델 {len(symbolic)}개 가중치. "
                    "cnn_filters 등 hparams가 튜닝 로그와 다른지 확인하세요."
                )
            layer_bad = []
            pairs = []
            for i, (var, val) in enumerate(zip(symbolic, values)):
                expected = tuple(int(x) for x in var.shape)
                got = tuple(int(x) for x in val.shape)
                if expected != got:
                    layer_bad.append(
                        f"{name}[{i}] {getattr(var, 'name', i)} expect {expected} got {got}"
                    )
                else:
                    pairs.append((var, val))
            if layer_bad:
                shape_mismatch.extend(layer_bad)
                continue
            K.batch_set_value(pairs)
            n_set += 1

        if shape_mismatch:
            raise ValueError(
                "가중치 shape 불일치 (tune-log hparams와 가중치 파일이 다를 수 있음):\n  "
                + "\n  ".join(shape_mismatch[:20])
            )
        if missing:
            print(f"[weights] h5 그룹 중 모델에 없는 레이어 skip: {missing[:10]}", flush=True)
    print(f"[weights] loaded by name from {filepath}  layers={n_set}", flush=True)
    return n_set
