import kilter_utils
import tabgan
from tabgan.sampler import GANGenerator
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout


import pandasql as ps

import cv2

import kilter_utils

st.set_page_config(layout="wide")

if 'preloading' not in st.session_state:
    st.session_state['preloading'] = False

if st.session_state['preloading'] == False:
    with st.spinner('Loading in necessary files, this will only run on the first instance of loading this page'):
    #st.write('Loading in necessary files, this will only run on the first instance of loading this page')


        data_path = "C:/Users/Declan/Documents/DataScienceProjects/KilterBoard_project/streamlit_app/data/"
        model_path = "C:/Users/Declan/Documents/DataScienceProjects/KilterBoard_project/streamlit_app/models/"

        # load in models
        xgb_model_stripped = pickle.load(open(f"{model_path}xgb_model_stripped.pickle", "rb"))
        if 'xgb_model_stripped' not in st.session_state:
            st.session_state.xgb_model_stripped = xgb_model_stripped

        xgb_model_raw = pickle.load(open(f"{model_path}xgb_model_raw.pickle", "rb"))
        if 'xgb_model_raw' not in st.session_state:
            st.session_state.xgb_model_saw = xgb_model_raw

        rf_model_stripped = pickle.load(open(f"{model_path}rf_model_stripped.pickle", "rb"))
        if 'rf_model_stripped' not in st.session_state:
            st.session_state.rf_model_stripped = rf_model_stripped

        rf_model_raw = pickle.load(open(f"{model_path}rf_model_raw.pickle", "rb"))
        if 'rf_model_raw' not in st.session_state:
            st.session_state.rf_model_raw = rf_model_raw

        lstm_model_stripped = pickle.load(open(f"{model_path}lstm_model_stripped.pickle", "rb"))
        if 'lstm_model_stripped' not in st.session_state:
            st.session_state.lstm_model_stripped = lstm_model_stripped

        lstm_model_raw = pickle.load(open(f"{model_path}lstm_model_raw.pickle", "rb"))
        if 'lstm_model_raw' not in st.session_state:
            st.session_state.lstm_model_raw = lstm_model_raw


        # board Image
        board_path = f"{data_path}full_board_commercial.png"

        # get dataframe
        df_climbing = kilter_utils.get_df_for_GAN(data_path)

        #
        df_climbing = kilter_utils.split_frames(df_climbing)
        df_climbing  = kilter_utils.raw_holds_to_basic(df_climbing)

        # Perform index based encoding and retrieve token to hold dictionary
        X_df_raw, word_to_idx_raw = kilter_utils.perform_index_based_tokenization(data_path, df_climbing, base=False)

        # concat df with sequence to tokenized holds using index based encoding
        df_climbing_raw = pd.concat([df_climbing, X_df_raw], axis=1)

        # Split into training columns and target
        features_raw, target_raw = kilter_utils.split_df_into_features_and_target(df_climbing_raw)

        #st.write('Split train test')
        # further split into test/train
        df_x_train, df_x_test, df_y_train, df_y_test = kilter_utils.train_test_split(
            features_raw,
            target_raw,
            test_size=0.20,
            random_state=42,
        )

        # Create dataframe versions for tabular GAN
        df_x_test, df_y_test = df_x_test.reset_index(drop=True),df_y_test.reset_index(drop=True)

        # Convert to df
        df_y_train = pd.DataFrame(df_y_train)
        df_y_test = pd.DataFrame(df_y_test)

        # Pandas to Numpy
        x_train = df_x_train.values
        x_test = df_x_test.values
        y_train = df_y_train.values
        y_test = df_y_test.values


        # load the neural network
        kilter_GAN = pickle.load(open(f"{model_path}kilter_GAN.pickle", "rb"))



        # GAN
        gen_x,gen_y = kilter_GAN.generate_data_pipe(df_x_train,
                                                        df_y_train,
                                                        df_x_test,
                                                        deep_copy=True,
                                                        only_adversarial=True, # was false
                                                        use_adversarial=True)


        # #
        inverse_dict = {v: k for k, v in word_to_idx_raw.items()}

        # create sessions state for df
        if 'new_climbs_df' not in st.session_state:
            st.session_state['new_climbs_df'] = 'empty'

        new_climbs_df = kilter_utils.convert_index_to_hold_df(gen_x,gen_y,inverse_dict)
        st.session_state['new_climbs_df'] = new_climbs_df


        # df of angle and grades
        if 'distribution_df' not in st.session_state:
            st.session_state['distribution_df'] = 'empty'


        distribution_df = kilter_utils.generated_holds_distribution_df(new_climbs_df)
        st.session_state['distribution_df'] = distribution_df

        st.session_state['preloading'] = True

        #st.write('state of preload after everything')
        #st.write(st.session_state['preloading'])
        st.experimental_rerun()

else:
    data_path = "C:/Users/Declan/Documents/DataScienceProjects/KilterBoard_project/streamlit_app/data/"
    board_path = f"{data_path}full_board_commercial.png"
    board_image = cv2.imread(board_path)

    new_climbs_df = st.session_state['new_climbs_df']

    distribution_df = st.session_state['distribution_df']

    angle_options = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60','65','70']
    #grade_options = range(0,17)


    col1,col2 = st.columns([2,3])


    if 'angle' not in st.session_state:
        st.session_state['angle'] = 0

    if 'grade' not in st.session_state:
        st.session_state['grade'] = 0

    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        angle = st.select_slider('Choose an angle', options =  angle_options)
        filtered_angles_df = kilter_utils.filter_by_angle(new_climbs_df,angle)

        available_grades = filtered_angles_df['v_grade'].unique()
        available_grades.sort()
        grade = st.select_slider('Choose a v grade',  options =  available_grades)

        #grade_angle_of_interest = kilter_utils.filter_grade_and_angle(new_climbs_df, angle, grade)
        grade_angle_of_interest = kilter_utils.filter_by_grade(filtered_angles_df,grade)
        grade_angle_of_interest= grade_angle_of_interest.reset_index(drop=True) #grade_angle_of_interest =

        st.write("")



        # Test
        if 'generted_climb_idx' not in st.session_state:
            st.session_state['generted_climb_idx'] = 0

        try:

            model_grade_df = pd.DataFrame(columns=['Model','Grade'])

            if grade_angle_of_interest.shape[0] == 1:
                generted_climb_idx = 0
                st.write('Only one climb at this grade and angle')
            else:
                generted_climb_idx = st.select_slider('Cycle through available climbs',  options =  range(0,grade_angle_of_interest.shape[0]))

            to_predict_raw = kilter_utils.preprocess_for_single_grade_prediction_raw(grade_angle_of_interest,generted_climb_idx, data_path)
            to_predict_stripped = kilter_utils.preprocess_for_single_grade_prediction_stripped(grade_angle_of_interest,generted_climb_idx, data_path)

            # run through LSTM_raw
            model_path = "C:/Users/Declan/Documents/DataScienceProjects/KilterBoard_project/streamlit_app/models/"

            xgb_model_stripped = st.session_state.xgb_model_stripped
            xgb_model_raw = st.session_state.xgb_model_saw
            rf_model_stripped = st.session_state.rf_model_stripped
            rf_model_raw = st.session_state.rf_model_raw
            lstm_model_stripped = st.session_state.lstm_model_stripped
            lstm_model_raw = st.session_state.lstm_model_raw


            y_pred = xgb_model_stripped.predict(np.asarray(to_predict_stripped).astype(np.float32)) #to_predict_stripped
            model_grade_df= model_grade_df.append({'Model':'xgb_model_stripped', 'Grade': y_pred}, ignore_index=True)
            #st.write('xgb_model_stripped', y_pred)

            y_pred = xgb_model_raw.predict(np.asarray(to_predict_raw).astype(np.float32))
            model_grade_df = model_grade_df.append({'Model':'xgb_model_raw', 'Grade': y_pred}, ignore_index=True)
            #st.write('xgb_model_raw', y_pred)



            y_pred = rf_model_stripped.predict(np.asarray(to_predict_stripped).astype(np.float32)) #to_predict_stripped
            model_grade_df = model_grade_df.append({'Model':'rf_model_stripped', 'Grade': y_pred}, ignore_index=True)

            y_pred = rf_model_raw.predict(np.asarray(to_predict_stripped).astype(np.float32)) #to_predict_stripped
            model_grade_df = model_grade_df.append({'Model':'rf_model_raw', 'Grade': y_pred}, ignore_index=True)




            y_pred = lstm_model_stripped.predict(np.asarray(to_predict_stripped).astype(np.float32))
            y_pred = np.argmax(y_pred, axis=1)
            model_grade_df = model_grade_df.append({'Model':'lstm_model_stripped', 'Grade': y_pred}, ignore_index=True)
            #st.write('LSTM stripped', y_pred)

            y_pred = lstm_model_raw.predict(np.asarray(to_predict_raw).astype(np.float32))
            y_pred = np.argmax(y_pred, axis=1)
            model_grade_df = model_grade_df.append({'Model':'lstm_model_raw', 'Grade': y_pred}, ignore_index=True)
            #st.write('LSTM raw', y_pred)


            model_grade_df = model_grade_df.reset_index(drop=True)
            st.dataframe(model_grade_df)


        except:

            st.write('Error occured, change values to make it go away')




    with col2:

        try:
            fig = kilter_utils.plot_climb(data_path,grade_angle_of_interest, generted_climb_idx) #random_num
            st.write(f"Showing climb at a suggested grade of v{grade} at an angle of {angle} degrees")
            fig

            st.session_state['angle'] = angle
            st.session_state['grade'] = grade
        except:
            st.session_state['angle'] = angle
            st.session_state['grade'] = grade
            pass


        try:
            st.session_state['generted_climb_idx'] = generted_climb_idx
            st.session_state['angle'] = angle
            st.session_state['grade'] = grade
        except:
            pass

st.write('Note on model meaning')
st.write('The sequences are generated with relevant hold type e.g. hand, foot etc. Models with stripped have this value removed so only the hold id is uses. For models with raw have the hold type encoded in the sequence')
