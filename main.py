import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pages import arima_page1, arima_page2
from utils import preprocess
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

path = 'data/data throughput only.csv'
data, bts_names = preprocess.load_data(path)
cols = ['dl_avg', 'ul_avg', 'dl_peak', 'ul_peak']

##################################Layout##################################
select_page= st.sidebar.selectbox(
    "What kind of model you want to use?",
    ("Baseline", "Modified")
)
##################################row 1##################################
lc_1, mc_1, rc_1 = st.columns(3)
with lc_1:
    select_data = st.selectbox(
        'Which data you want to predict?',
         cols)
with mc_1:
    select_model = st.selectbox(
        'Which model you want to use?',
        ['ARIMA', 'SARIMAX']
    )
with rc_1:
    select_bts = st.selectbox(
        'Which bts you want to see?',
        bts_names
    )
##################################row 2##################################
start_date, end_date = st.select_slider(
    'Select a range of day to predict',
    options=[i for i in range(1, 31)],
    value=(25, 30)
)
selected_data = preprocess.get_data(data, select_bts, select_data)
##################################row 3##################################
if select_page == 'Modified':
    arima_page2.modify_model(selected_data, select_model, start_date, end_date, select_bts, select_data)
else:
    arima_page1.baseline_model(selected_data, select_model, start_date, end_date, select_bts, select_data)

##################################row 4##################################
lc_4, rc_4 = st.columns(2)
with lc_4:
    with st.container():
        st.header('MAE')
        mae = mean_absolute_error(selected_data.iloc[start_date*24:end_date*24]['y'], selected_data.iloc[start_date*24:end_date*24]['forecast'])
        st.subheader(f'{mae}')


with rc_4:
    with st.container():
        st.header('MSE')
        mse = mean_squared_error(selected_data.iloc[start_date*24:end_date*24]['y'], selected_data.iloc[start_date*24:end_date*24]['forecast'])
        st.subheader(f'{mse}')