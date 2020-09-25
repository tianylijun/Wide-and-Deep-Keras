import csv

import numpy as np
import pandas as pd
import os
import argparse
import datetime
import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2


def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


def wide(df_train, df_test, wide_cols, x_cols, target):
    df_wide = df_test

    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(
        df_wide.select_dtypes(include=['object']).columns)

    wide_cols += list(crossed_columns_d.keys())

    for k, v in crossed_columns_d.items():
        df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x), axis=1)

    df_wide = df_wide[wide_cols]

    dummy_cols = [
        c for c in wide_cols if c in categorical_columns + list(crossed_columns_d.keys())]
    df_wide = pd.get_dummies(df_wide, columns=[x for x in dummy_cols])

    test = df_wide

    cols = [c for c in test.columns if c != target]
    X_test = test[cols].values
    y_test = test[target].values.reshape(-1, 1)
    return X_test, y_test


if __name__ == '__main__':
    print(tf.__version__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb_file", type=str, default="model_1_13/saved_model.h5")
    args = vars(ap.parse_args())
    pb_file = args["pb_file"]

    # inp = 1
    # em = Embedding(9, 8, input_length=1, embeddings_regularizer=l2(1e-3))
    # l = len(em.get_weights())
    # out = em(0)
    # out = em(1)
    # out = em(2)
    # out = em(3)
    # out = em(4)
    # out = em(5)
    # out = em(6)
    # out = em(7)
    # out = em(8)
    # K.set_learning_phase(0)
    # print(tf.keras.backend.learning_phase())
    # model = tf.keras.models.load_model(pb_file)
    model = tf.keras.models.load_model(pb_file, compile=False)
    # model.summary()
    inp = model.input
    print(inp)
    output = model.output
    print(output)

    wide = np.loadtxt("data/wide.txt")
    wide = wide.astype(dtype=np.int32).reshape([1, len(wide)])
    deep = np.loadtxt("data/deep.txt")
    workclass_inp = deep[0].astype(dtype=np.int32).reshape([1, 1])
    education_inp = deep[1].astype(dtype=np.int32).reshape([1, 1])
    marital_status_inp = deep[2].astype(dtype=np.int32).reshape([1, 1])
    occupation_inp = deep[3].astype(dtype=np.int32).reshape([1, 1])
    relationship_inp = deep[4].astype(dtype=np.int32).reshape([1, 1])
    race_inp = deep[5].astype(dtype=np.int32).reshape([1, 1])
    gender_inp = deep[6].astype(dtype=np.int32).reshape([1, 1])
    native_country_inp = deep[7].astype(dtype=np.int32).reshape([1, 1])
    age_in = deep[8].astype(dtype=np.float).reshape([1, 1])
    capital_gain_in = deep[9].astype(dtype=np.float).reshape([1, 1])
    capital_loss_in = deep[10].astype(dtype=np.float).reshape([1, 1])
    hours_per_week_in = deep[11].astype(dtype=np.float).reshape([1, 1])

    inputs = [wide, workclass_inp, education_inp, marital_status_inp, occupation_inp, relationship_inp, race_inp,
              gender_inp, native_country_inp, age_in, capital_gain_in, capital_loss_in, hours_per_week_in]

    output = model.predict(inputs)  # n : 0.02082383
    print(output)
