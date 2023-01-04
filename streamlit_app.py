import streamlit as st
import os
from PIL import Image
import kilter_utils

st.write(os.getcwd())
path = f"{os.getcwd()}/"
#path = "C:/Users/Declan/Documents/DataScienceProjects/KilterBoard_project/streamlit_app/"

#st.set_page_config(
#    page_title="Main Page"
#)

'''## Kilter Board Analysis'''
'''### By Declan Stockdale-Garbutt'''

'''
#
#
## What is a Kilter Board ?

A Kilter board is a standardized climbing board that is popular in commercial gyms and can be used to create and share climbing routes with other climbers around the world. It is adjustable from 5 to 70 degrees in 5 degree increments, providing a wide range of options for training at different angles and difficulties. Each hold on the board can be backlight by a LED for easier route navigation. The Kilter board is available in various sizes, with the most common being the 12x12 size that is widely used in gyms that have Kilter boards. Notable users of the Kilter board include world-renowned climbers such as Alexander Megos and Jimmy Webb, among others.
'''

'''
## Purpose of this project

An analysis of various climbs at various angle and grades is performed, along with details about setters.
A climbing generator using a Generative Adverserial Network has also been utilized to create new climbs which can be used as is or as a template for addititoal climbs

'''

'''
##
'''
# display kilter board
image = Image.open(f"{path}data/full_board_commercial.png")
st.image(image,'12x12 Kilter Board' )#.resize(image.size[0]*0.8,image.size[1]*0.8)
