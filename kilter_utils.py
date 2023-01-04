import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json
import matplotlib.image as mpimg
from PIL import Image
import sqlite3
import pandas as pd
import pandasql as ps
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import json
from collections import Counter
import cv2
import matplotlib.cm     as cm
import seaborn as sns
import matplotlib.pylab as plt
import sqlite3

import pandasql as ps
import plotly.express as px
import plotly.graph_objects as go
import json
import seaborn as sns


import numpy as np

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

#!pip install xgboost
import xgboost as xgb
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# LSTM
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout


def read_in_sqllite_file(path_to_apk):
    return sqlite3.connect(f"{path_to_apk}db.sqlite3")

def get_list_holds_type_removed(series):

    # get all holds
    df_holds = list(series)
    total_list_of_holds = "".join([frame for frame in df_holds])

    # split into each hold
    total_list_of_holds = total_list_of_holds.split('p')

    # strip r.. off end of hold
    total_list_of_holds_type_removed = [hold[:-3] for hold in total_list_of_holds[1:]]

    # Keep intact
    total_list_of_holds_type_included = [hold for hold in total_list_of_holds[1:]]


    return total_list_of_holds_type_removed, total_list_of_holds_type_included

def count_holds(list_holds):

    df_counter = pd.DataFrame.from_dict(Counter(list_holds), orient='index').reset_index()
    df_counter = df_counter.set_axis(['hold_id', 'count'], axis=1, inplace=False)

    return df_counter

def read_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        my_data = json.load(f)

    return my_data

def concat_hold_dataframes(json_file):

    df_mainholds = pd.DataFrame.from_dict(json_file[0][0]['mainholds'])
    df_aux = pd.DataFrame.from_dict(json_file[0][1]['auxillary'])
    df_kickboard = pd.DataFrame.from_dict(json_file[0][2]['kickboard'])

    df_holds = pd.concat([df_mainholds,df_aux,df_kickboard])

    return df_holds # list(set(df_holds))

def list_of_holds_in_df(df):
    return list(set(df))


def mainholds_image_no_kickboard(path):

    mainholds_img = Image.open(f"{path}mainboard_transparent.png")

    width, height = mainholds_img.size

    # Setting the points for cropped image
    left = 0
    top = 0
    right = width
    bottom = height-100

    mainholds_img_no_kickboard = mainholds_img.crop((left, top, right, bottom))

    im1 = mainholds_img_no_kickboard.save(f"{path}main_holds_no_kb_transparent.png", transparent=True)

def fill_missing_counts(df):

    df['count'] = df['count'].fillna(0)
    df = df[df['grid'].notna()]

    return df

def generate_hold_usage_data(df, num_unique_climbs):

    max_count = max(df['count'])
    df['normalized_1'] = round(df['count']/max_count,2)

    # normalize to 255
    df['normalized_255'] = round(df['count']*255,0)

    # %
    df['percent_hold'] = round(df['count']*100/num_unique_climbs,5)

    return df

def mainholds_coloured_dots_heatmap(path, df, output_name):

    image_board = cv2.imread(f"{path}combined.png")


    for idx in range(df.shape[0]):

        coor = df['coords'][idx]
        colour = cm.jet(df['normalized_1'][idx])

        colour1, colour2, colour3,_  = colour

        image = cv2.circle(image_board, coor, 5, (colour1*255, colour2*255, colour3*255), 6)

    cv2.imwrite(f"{path}/{output_name}.png", image)

def mainholds_heatmap(path, df, show = False):

    fig, ax = plt.subplots(figsize=(11.05,11.70))
    df = df.loc[df['set'] == 'main']
    plot_1 = df['percent_hold'].values.reshape(18,17)

    sns.heatmap(plot_1, linewidth=0.3, cbar=False, cmap=sns.cubehelix_palette(as_cmap=True))
    ax.tick_params(left=False, bottom=False)

    ax.set(xticklabels=[])  # remove the tick labels
    ax.tick_params(bottom=False)  # remove the

    ax.set(yticklabels=[])  # remove the tick labels
    ax.tick_params(left=False)  # remove the

    plt.savefig(f"{path}main_holds_heatmap.png", transparent=True)

    if show == True:
        plt.show()

def mainholds_heatmap_overlayed(path, df):

    mainholds_img_no_kickboard = Image.open(f"{path}main_holds_no_kb_transparent.png")

    fig, ax = plt.subplots(figsize=(11.05,11.70))
    df = df.loc[df['set'] == 'main']
    plot_1 = df['percent_hold'].values.reshape(18,17)

    width, height = mainholds_img_no_kickboard.size

    left = 30
    right = width-left
    top = 0
    bot =  height - top+15#-10#23

    mainholds_img_no_kickboard = mainholds_img_no_kickboard.crop((left, top, right, bot))
    mainholds_img_no_kickboard = mainholds_img_no_kickboard.resize((1105,1070))


    hmax = sns.heatmap(plot_1,
                alpha = 0.5,
                zorder = 2,
                linewidth=0.3,
                cbar=True
                )

    hmax.set(xticklabels=[])  # remove the tick labels
    hmax.set(yticklabels=[])  # remove the tick labels
    hmax.tick_params(left=False, bottom=False)  # remove the ticks

#fig =
    hmax.imshow(mainholds_img_no_kickboard,
                      aspect = hmax.get_aspect(),
                      extent = hmax.get_xlim() + hmax.get_ylim(),
                      zorder = 2)

    return fig



def prepare_count_df(list_of_holds,phrase):

    df_counter = [hold[:-3] for hold in list_of_holds[1:] if phrase in hold]

    df_counter = pd.DataFrame.from_dict(Counter(df_counter), orient='index').reset_index()

    df_counter = df_counter.set_axis(['hold_id', 'count'], axis=1, inplace=False)


    return df_counter

def merge_count_df_with_df(count_df, merge_df):

    #merge_df = merge_df.drop(['count','normalized_1', 'normalized_255','percent_hold'], axis=1)

    df = pd.merge(
                 merge_df,
                 count_df,
                 how="outer",
                 on='hold_id'
                 )

    return df

def aux_holds_heatmap_overlayed(path, df):

    # filter by hold type
    df = df.loc[df['set'] == 'auxillary']

    # create series
    percent_hold_series = df['percent_hold']

    # create zeros array twice the ssize of the original array
    out = np.zeros((1,2*135),dtype=percent_hold_series.dtype)

    #for value in out


    # make every second value the intended value
    out[:,::2] = percent_hold_series # -  wrong

    # reshape to desired shape
    plot_1 = out.reshape(15,18)

    # fix always starting with 1 value , every second row starts at 0 need to shift one to the right
    for idx, array in enumerate(plot_1):

        if idx == 0 or idx % 2 == 0:
            pass
        else:
            plot_1[idx] = np.roll(plot_1[idx],1)

    auxillary_holds_no_kb_transparent = Image.open(f"{path}auxillary_holds_no_kb_transparent.png")

    fig, ax = plt.subplots(figsize=(11.05,11.70))


    width, height = auxillary_holds_no_kb_transparent.size

    left = 0
    right = width-left
    top = 10
    bot =  height-15#-100# - top+15#-10#23

    auxillary_holds_no_kb_transparent = auxillary_holds_no_kb_transparent.crop((left, top, right, bot))
    auxillary_holds_no_kb_transparent = auxillary_holds_no_kb_transparent.resize((1000,1000))


    hmax = sns.heatmap(plot_1,
                alpha = 0.5,
                zorder = 2,
                linewidth=0.3,
                cbar=True
                )

    hmax.set(xticklabels=[])  # remove the tick labels
    hmax.set(yticklabels=[])  # remove the tick labels
    hmax.tick_params(left=False, bottom=False)  # remove the ticks


    fig1 = hmax.imshow(auxillary_holds_no_kb_transparent,
                      aspect = hmax.get_aspect(),
                      extent = hmax.get_xlim() + hmax.get_ylim(),
                      zorder = 2)

    return fig



def get_list_holds_type_removed(series):

    # get all holds
    df_holds = list(series)
    total_list_of_holds = "".join([frame for frame in df_holds])

    # split into each hold
    total_list_of_holds = total_list_of_holds.split('p')

    # strip r.. off end of hold
    total_list_of_holds_type_removed = [hold[:-3] for hold in total_list_of_holds[1:]]

    # Keep intact
    total_list_of_holds_type_included = [hold for hold in total_list_of_holds[1:]]


    return total_list_of_holds_type_removed, total_list_of_holds_type_included


def prepare_hold_heatmap_angle_grade_df(path, df, angle, grade):

    if angle =='None' and grade == 'None':
        angle_grade = df
        #st.write('Showing heatmap for all climbs ')

    else:
        filtered_values = np.where((df['v_grade'] == str(grade)) & (df['board_angle'] == angle))
        angle_grade = (df.loc[filtered_values])
        #st.write(f"Showing heatmap for climbs of grade {grade} at {angle} degrees. {angle_grade.shape[0]} climbs found (climbs must have at least 5 ascents to count)")
        st.write(f"{angle_grade.shape[0]} climbs found (climbs must have at least 5 ascents to count)")


    #st.write(f"{angle_grade.shape[0]} climbs found (climbs must have at least 5 ascents to count)")

    list_holds_strip, list_holds_raw = get_list_holds_type_removed(angle_grade['frames'])

    df_counter_strip = count_holds(list_holds_strip)
    df_counter_raw = count_holds(list_holds_raw)

    holds_json = read_json(f"{path}/all_holds.json")
    holds_df = concat_hold_dataframes(holds_json)
    list_of_holds = list_of_holds_in_df(holds_df['hold_id'])

    # Convert to relevant dtype
    df_counter_strip['hold_id'] = df_counter_strip['hold_id'].astype(str)
    df_counter_raw['hold_id'] = df_counter_raw['hold_id'].astype(str)


    # Convert to relevant dtype
    holds_df['hold_id'] = holds_df['hold_id'].astype(str)

    stripped_df = pd.merge(
                            holds_df,
                            df_counter_strip,
                            how="outer",
                            on='hold_id'
                        )

    stripped_df = fill_missing_counts(stripped_df)


    # find number of unique climbs
    num_unique_climbs = len(angle_grade['uuid'].unique())

    # normalise to 1, 255 and find %
    stripped_df = generate_hold_usage_data(stripped_df, num_unique_climbs)

    return stripped_df


# grade climbs
def split_frames(df):

    df['holds_raw'] = df['frames'].str.split('p')
    df['holds_raw']  = [hold for hold in df['holds_raw']]

    for idx,_ in enumerate(df['holds_raw']):
        df['holds_raw'][idx] = (" ".join([hold for hold in df['holds_raw'][idx][1:]]))
        df['holds_raw'][idx] = df['holds_raw'][idx].split(" ")

    return df

def raw_holds_to_basic(df):

    df['holds'] = 0
    df['hold_type'] = 0

    for idx,row in enumerate(df['holds_raw']):
        df['holds'][idx] = [hold[:4] for hold in df['holds_raw'][idx]]
        df['hold_type'][idx] = [hold[-2:] for hold in df['holds_raw'][idx]]

    return df

def perform_index_based_tokenization(path, df, base=True):

    # get holds dataframe
    holds_json = read_json(f"{path}/all_holds.json")
    holds_df = concat_hold_dataframes(holds_json)


    # converting text to integers using index based tokenisation
    if base == False:
        token_docs = df['holds_raw'].tolist()
        all_tokens = set([str(word) for sentence in token_docs for word in holds_df['hold_id']])

        all_tokens_all_combos = []

        for hold in all_tokens:

            for hold_type in ['r12', 'r13', 'r14', 'r15']:
                all_tokens_all_combos.append(hold+hold_type)

        all_tokens_all_combos = set((all_tokens_all_combos)) # was not int
        word_to_idx = {token:idx+1 for idx, token in enumerate(all_tokens_all_combos)}
    else:

        token_docs = df['holds'].tolist()
        all_tokens = set([str(word) for sentence in token_docs for word in holds_df['hold_id']])
        #st.write('stripped')
        #st.write(all_tokens)
        #st.write(type(all_tokens))
        word_to_idx = {token:idx+1 for idx, token in enumerate(all_tokens)}

        #st.write('word_to_idx')
        #st.write(word_to_idx)
        #st.write(type(word_to_idx))

    # converting the docs to their token ids+

    index_based_tokenisation = np.array([[word_to_idx[token] for token in token_doc] for token_doc in token_docs], dtype=object)

    # padding the sequences to max length
    index_based_tokenisation_padded = pad_sequences(index_based_tokenisation, padding="post")

    # converting to pandas df
    index_based_tokenisation_df = pd.DataFrame(index_based_tokenisation_padded)

    return index_based_tokenisation_df, word_to_idx

def split_df_into_features_and_target(df):
    X = df.drop(['frames', 'v_grade','holds_raw','holds','hold_type'], axis=1) # ,'sequence_indexed' df_climbing[['board_angle','quality_average','hold_type']]
    Y = df['v_grade'].astype(int)

    return X,Y


def pad_for_grade_predict(df,df_with_sequence):

    if df.shape[1] <35:

        for value in range(df.shape[1], 35):
            df[str(value)] = 0

        df = pd.DataFrame(df).reset_index(drop =True)

        df = pd.concat([df_with_sequence.iloc[:,:2],df],axis = 1)
    return df

def preprocess_for_single_grade_prediction_raw(df, idx,data_path):

    df_pred = df.iloc[idx]
    df_pred = pd.DataFrame(df_pred).transpose()
    df_pred = df_pred.rename(columns={"sequence": "holds_raw"})
    df_pred = df_pred.iloc[: , :-1].reset_index(drop=True)
    output,_ = perform_index_based_tokenization(data_path, df_pred, base=False)
    processed_df = pad_for_grade_predict(output,df_pred)

    return processed_df

def preprocess_for_single_grade_prediction_stripped(df, idx, data_path):

    df_pred = df.iloc[idx]
    df_pred = pd.DataFrame(df_pred).transpose()
    df_pred = df_pred.rename(columns={"sequence": "holds"})
    df_pred = df_pred.iloc[: , :-1].reset_index(drop=True)

    df_pred =  df_pred.rename(columns={"holds": "holds_raw"})
    df_pred = raw_holds_to_basic(df_pred)

    #st.write("after", df_pred)

    output,_ = perform_index_based_tokenization(data_path, df_pred, base=True)
    processed_df = pad_for_grade_predict(output,df_pred)

    return processed_df

def split_data(X,Y):
    seed = 42
    test_size = 0.15
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

def make_predictions(model, X_test,y_test):

    # Exact grade

    # make predictions for test data
    #st.write(X_test)
    y_pred = model.predict(X_test)

    # try to access shape of lstm output which is an array
    #st.write(y_pred.shape)

    try:
        if y_pred.shape[1] == 17:
            y_pred = np.argmax(y_pred, axis=1)
    except:
        pass

    predictions = [int(value) for value in y_pred] #was round

    # evaluate predictions
    exact_accuracy = accuracy_score(y_test, predictions)*100

    st.write()
    # One off grade +/- 1
    # Gan outputs array which breaks on reset index
    try:
        y_test_series = y_test.reset_index(drop=True)
    except:
        y_test_series = y_test


    correct = 0
    total = 0

    for idx,value in enumerate(predictions):

        if predictions[idx] == y_test_series[idx] or predictions[idx] == y_test_series[idx]-1 or predictions[idx] == y_test_series[idx]+1:
            correct +=1

        total +=1

    one_out_accuracy = correct/total * 100.0
    #st.write("One out grade accuracy: %.2f%%" % (one_out_accuracy))

    return exact_accuracy, one_out_accuracy

# Create climbs
def get_df_for_GAN(data_path):

    cnx = read_in_sqllite_file(data_path)
    df_climbing = pd.read_sql_query("SELECT board_angle, frames, quality_average, v_grade FROM (  SELECT board_angle, grade, boulder_name, TRIM(substr(boulder_name, instr(boulder_name,'/')+2)) AS v_grade, frames, name, ascensionist_count ,quality_average  FROM ( SELECT climb_stats.angle AS board_angle, round(difficulty_average,0) AS grade, frames, name, ascensionist_count, quality_average       FROM climbs  INNER JOIN climb_stats ON climbs.uuid = climb_stats.climb_uuid  WHERE is_listed = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140 and frames NOT LIKE '%r2%'and frames NOT LIKE '%r3%') t1 INNER JOIN difficulty_grades ON difficulty_grades.difficulty = t1.grade ORDER BY board_angle ASC, grade ASC) WHERE ascensionist_count >=5  ", cnx)

    return df_climbing

def convert_index_to_hold_df(df_x, df_y,inverse_dict):
    # copy orig df - for some reasons it breaks otherwise
    df_copy = df_x.copy()

    # add empty column to add hold sequence to
    df_copy['sequence'] = ""

    df_copy=df_copy.merge(df_y.to_frame(), left_index=True, right_index=True)

    # iterate over rows
    for index, row in df_copy.iterrows():

        # empty sequence for each row
        sequence_list = []

        # skip irrevant columns - just want the index encoded columns
        column_of_interest = row[2:-2]

        # Loop over holds in row
        for value in column_of_interest:

            # skip 0
            if value == 0:
                pass

            else:
                # find what the encoded value equivalent hols is
                hold_info = str(inverse_dict[value])
                #append to list
                sequence_list.append(hold_info)

        # set index of empty column to decoded sequence
        df_copy['sequence'][index] = sequence_list

    return df_copy.iloc[:,[0,1,-2,-1]]

def generated_holds_distribution_df(df):
    df = df[['board_angle','quality_average', 'v_grade']]
    df_distribution = ps.sqldf("SELECT COUNT(*) AS num_climbs, v_grade, board_angle FROM df GROUP BY v_grade, board_angle")

    return df_distribution

def filter_grade_and_angle(df, angle, grade):

    angle = int(angle)
    #st.write('type of angle', type(angle))
    #st.write('type of grade', type(grade))
    #df = ps.sqldf(f"SELECT * FROM df WHERE v_grade = {grade} and board_angle = {angle}")
    df_angle_filtered = df.loc[df['board_angle'] == angle]
    df_grade_and_angle_filtered = df_angle_filtered.loc[df_angle_filtered['v_grade'] == grade]
    df_grade_and_angle_filtered.sort_values(by = ['quality_average'], ascending=False)

    return df_grade_and_angle_filtered

def filter_by_angle(df, angle):
    angle = int(angle)
    df_angle_filtered = df.loc[df['board_angle'] == angle]
    #df_grade_and_angle_filtered = df_angle_filtered.loc[df_angle_filtered['v_grade'] == grade]
    #df_grade_and_angle_filtered.sort_values(by = ['quality_average'], ascending=False)

    return df_angle_filtered

def filter_by_grade(df, grade):
    df_grade_and_angle_filtered = df.loc[df['v_grade'] == grade]
    df_grade_and_angle_filtered.sort_values(by = ['quality_average'], ascending=False)

    return df_grade_and_angle_filtered


def plot_climb(data_path, df, num):

    board_path = f"{data_path}full_board_commercial.png"
    board_image = cv2.imread(board_path)

    for hold in df['sequence'][num]:

        holds_json = read_json(f"{data_path}/all_holds.json")
        test_df = concat_hold_dataframes(holds_json)

        hold_id,hold_type = hold.split("r")

        # find where hold occurs in test_df
        corresponding_row = test_df.loc[test_df['hold_id'] == int(hold_id)]
        coords = corresponding_row['coords'].values[0]

        # draw on image
        center_coordinates = (coords[0], coords[1])
        radius = 30
        thickness = 2

        if hold_type == str(12):
            color = (0,255,0) #start

        if hold_type == str(13): # hands
            color = (255,255,0)

        if hold_type == str(14): # end
            color = (255,0,255)

        if hold_type == str(15): # feet
            color = (0,255,255)


        image = cv2.circle(board_image, center_coordinates, radius, color, thickness)

    fig=plt.figure()
    fig.set_size_inches(10, 10)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=False)
    plt.axis('off')

    #st.write(f"## Showing climb at grade {grade} at angle {angle}")
    #fig
    return fig


def predict_grade(model, climb):

    holds_json = read_json(f"{path}/all_holds.json")
    holds_df = concat_hold_dataframes(holds_json)

    y_pred = model.predict(climb['sequence'])
