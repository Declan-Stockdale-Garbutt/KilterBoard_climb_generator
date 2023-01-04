import streamlit as st
import pandas as pd
import kilter_utils
import time
import os
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
import sqlite3
import pandasql as ps
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
from collections import Counter
import sklearn
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm     as cm
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import xgboost as xgb
from numpy import loadtxt
from sklearn.model_selection import train_test_split

# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import tabgan
from tabgan.sampler import GANGenerator
import kilter_utils


st.subheader("Train models - Optional")
st.write("This page is where the model are trained for grade prediction based on angle, quality and sequence")
st.write(' pretrained models are included in this app. This page allows for  straighforward approach to retrain the models based on new data')


data_path = f"{os.getcwd()}/data/"
model_path = f"{os.getcwd()}/models/"

with st.spinner(text="Loading in data - Required for training and predicting"):
    # get data
    cnx = kilter_utils.read_in_sqllite_file(data_path)
    df_climbing = pd.read_sql_query("SELECT board_angle, frames, quality_average, v_grade FROM (  SELECT board_angle, grade, boulder_name, TRIM(substr(boulder_name, instr(boulder_name,'/')+2)) AS v_grade, frames, name, ascensionist_count ,quality_average  FROM ( SELECT climb_stats.angle AS board_angle, round(difficulty_average,0) AS grade, frames, name, ascensionist_count, quality_average       FROM climbs  INNER JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid  WHERE is_listed = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140 and frames NOT LIKE '%r2%' and frames NOT LIKE '%r3%') t1 INNER JOIN difficulty_grades ON difficulty_grades.difficulty = t1.grade ORDER BY board_angle ASC, grade ASC) WHERE ascensionist_count >=5  ", cnx)


    # split frames into list
    df_climbing = kilter_utils.split_frames(df_climbing)
    df_climbing = kilter_utils.raw_holds_to_basic(df_climbing)

    X_df_base, _ = kilter_utils.perform_index_based_tokenization(data_path, df_climbing, base = True)
    df_climbing_base = pd.concat([df_climbing, X_df_base], axis=1)
    df_climbing_base  = df_climbing_base.sample(frac = 1)
    # split data into X and y
    features, target = kilter_utils.split_df_into_features_and_target(df_climbing_base)
    X_train, X_test, y_train, y_test = kilter_utils.split_data(features, target)


    X_df_raw,_ = kilter_utils.perform_index_based_tokenization(data_path, df_climbing, base=False)
    df_climbing_raw = pd.concat([df_climbing, X_df_raw], axis=1)
    df_climbing_raw = df_climbing_raw.sample(frac = 1)
    # split data into X and y
    features_raw, target_raw = kilter_utils.split_df_into_features_and_target(df_climbing_raw)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = kilter_utils.split_data(features_raw, target_raw)

    X_train.columns = X_train.columns.astype(str)
    X_train_raw.columns = X_train_raw.columns.astype(str)



    X_test_raw.columns = X_test_raw.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

# try to load in models
try:
    st.write(model_path)
    st.write(os.listdir(model_path))
    # Create df
    model_accuracy_df = pd.DataFrame(columns=['Model','Exact Accuracy','One off Accuracy'])

    st.write('xgb_model_stripped loading in')
    xgb_model_stripped = pickle.load(open(f"{model_path}xgb_model_stripped.pickle", "rb"))
    xgb_model_raw = pickle.load(open(f"{model_path}xgb_model_raw.pickle", "rb"))

    exact, one_off = kilter_utils.make_predictions(xgb_model_stripped,X_test,y_test)
    dict = {'Model' : 'xgb_model_stripped','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    exact, one_off = kilter_utils.make_predictions(xgb_model_raw,X_test_raw,y_test_raw)
    dict = {'Model' : 'xgb_model_raw','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    st.write('rf models loading in')
    rf_model_stripped = pickle.load(open(f"{model_path}rf_model_stripped.pickle", "rb"))
    rf_model_raw = pickle.load(open(f"{model_path}rf_model_raw.pickle", "rb"))
    
    st.write('rf_model_stripped loading in')
    exact, one_off = kilter_utils.make_predictions(rf_model_stripped,X_test,y_test)
    st.write('break?')
    dict = {'Model' : 'rf_model_stripped','Exact Accuracy': exact,'One off Accuracy':one_off}
    st.write('break here?')
    
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    st.write('rf_model_raw loading in')
    exact, one_off = kilter_utils.make_predictions(rf_model_raw,X_test_raw,y_test_raw)
    dict = {'Model' : 'rf_model_raw','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    #st.write('lstm models loading in')
    #lstm_model_stripped = pickle.load(open(f"{model_path}lstm_model_stripped.pickle", "rb"))
    #lstm_model_raw = pickle.load(open(f"{model_path}lstm_model_raw.pickle", "rb"))

    #exact, one_off = kilter_utils.make_predictions(lstm_model_stripped,X_test,y_test)
    #dict = {'Model' : 'lstm_model_stripped','Exact Accuracy': exact,'One off Accuracy':one_off}
    #model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)


    #exact, one_off = kilter_utils.make_predictions(lstm_model_raw,X_test_raw,y_test_raw)
    #dict = {'Model' : 'lstm_model_raw','Exact Accuracy': exact,'One off Accuracy':one_off}
    #model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)


    st.write('Accuracy of models ')
    st.dataframe(model_accuracy_df)

except:
    st.write('No models found')
    pass



if 'train_models_button' not in st.session_state:
    st.session_state.train_models_button = False

train_models_button = st.button("Train models")

if train_models_button:

    st.session_state.train_models_button = True
    st.write('Note on model meaning')
    st.write('The sequences are generated with relevant hold type e.g. hand, foot etc. Models with stripped have this value removed so only the hold id is uses. For models with raw have the hold type encoded in the sequence')

    model_accuracy_df = pd.DataFrame(columns=['Model','Exact Accuracy','One off Accuracy'])


    #train models
    st.write('Training xgboost on stripped hold info')
    xgb_model_stripped = xgb.XGBClassifier()
    xgb_model_stripped.fit(X_train, y_train)
    pickle.dump(xgb_model_stripped, open(f"{model_path}xgb_model_stripped.pickle", "wb"))

    exact, one_off = kilter_utils.make_predictions(xgb_model_stripped,X_test,y_test)
    dict = {'Model' : 'xgb_model_stripped','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    st.write('Training xgboost on raw hold info')
    xgb_model_raw = xgb.XGBClassifier()
    xgb_model_raw.fit(X_train_raw, y_train_raw)
    pickle.dump(xgb_model_raw, open(f"{model_path}xgb_model_raw.pickle", "wb"))

    exact, one_off = kilter_utils.make_predictions(xgb_model_raw,X_test,y_test)
    dict = {'Model' : 'xgb_model_raw','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    st.write('Training random forest on stripped hold info')
    rf_model_stripped = RandomForestClassifier(n_estimators=50, max_depth=20,max_features="auto", random_state=44)
    rf_model_stripped.fit(X_train, y_train)
    pickle.dump(rf_model_stripped, open(f"{model_path}rf_model_stripped.pickle", "wb"))

    exact, one_off = kilter_utils.make_predictions(rf_model_stripped,X_test,y_test)
    dict = {'Model' : 'rf_model_stripped','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    st.write('Training random forest on raw info')
    rf_model_raw = RandomForestClassifier(n_estimators=50,  max_depth=20, max_features="auto", random_state=44)
    rf_model_raw.fit(X_train_raw, y_train_raw)
    pickle.dump(rf_model_raw, open(f"{model_path}rf_model_raw.pickle", "wb"))

    exact, one_off = kilter_utils.make_predictions(rf_model_raw,X_test,y_test)
    dict = {'Model' : 'rf_model_raw','Exact Accuracy': exact,'One off Accuracy':one_off}
    model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)




    #st.write('Training LSTM on stripped hold info')
    #lstm_model_stripped = Sequential()
    #lstm_model_stripped.add(Embedding(df_climbing_base.shape[0], 128, input_length=features.shape[1]))
    #lstm_model_stripped.add(SpatialDropout1D(0.2))
    #lstm_model_stripped.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    #lstm_model_stripped.add(Dense(17, activation='softmax'))
    #lstm_model_stripped.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #epochs = 50
    #batch_size = 1024

    #history_lstm_stripped = lstm_model_stripped.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    #pickle.dump(lstm_model_stripped, open(f"{model_path}lstm_model_stripped.pickle", "wb"))

    #exact, one_off = kilter_utils.make_predictions(lstm_model_stripped,X_test,y_test)
    #dict = {'Model' : 'lstm_model_stripped','Exact Accuracy': exact,'One off Accuracy':one_off}
    #model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    #st.write('Training LSTM on stripped raw info')
    #lstm_model_raw = Sequential()
    #lstm_model_raw.add(Embedding(df_climbing_raw.shape[0], 128, input_length=features.shape[1]))
    #lstm_model_raw.add(SpatialDropout1D(0.2))
    #lstm_model_raw.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    #lstm_model_raw.add(Dense(17, activation='softmax'))
    #lstm_model_raw.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #epochs = 50
    #batch_size = 1024

    #history_lstm_raw = lstm_model_raw.fit(X_train_raw, y_train_raw, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    #pickle.dump(lstm_model_raw, open(f"{model_path}lstm_model_raw.pickle", "wb"))

    #exact, one_off = kilter_utils.make_predictions(lstm_model_raw,X_test,y_test)
    #dict = {'Model' : 'lstm_model_raw','Exact Accuracy': exact,'One off Accuracy':one_off}
    #model_accuracy_df = model_accuracy_df.append(dict, ignore_index=True)

    st.write('Training Generative Adverserial Network (GAN) on raw info')
    kilter_GAN = GANGenerator(gen_x_times=1.1,
                                cat_cols=None,
                                bot_filter_quantile=0.001,
                                top_filter_quantile=0.999,
                                is_post_process=True,

                                adversarial_model_params={
                                        "metrics": "rmse",
                                        "max_depth": 2,
                                        "max_bin": 100,
                                        "learning_rate": 0.02,
                                        "random_state": 42,
                                        "n_estimators": 500,
                                        },

                                pregeneration_frac=2,
                                only_generated_data=False,

                                gan_params = {
                                    "batch_size": 500,
                                    "patience": 25,
                                    "epochs" : 500,
                                }
                               )

    pickle.dump(kilter_GAN, open(f"{model_path}kilter_GAN.pickle", "wb"))


    st.dataframe(model_accuracy_df)

    st.write('Done')
