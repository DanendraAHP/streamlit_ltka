from utils import model, preprocess, visualize
import streamlit as st

def baseline_model(data, select_model, start_date, end_date, select_bts, select_data):
    ##################################Select Data BTS with specific data type and the model prediction##################################
    if select_model == 'SARIMAX':
        selected_data = model.sarimax(data, start_date, end_date, 1, 1, 1, 24)
    else:
        selected_data = model.arima(data, start_date, end_date, 1, 1, 1)
    ##################################row 3##################################
    with st.container():
        st.title(f'Line chart {select_data} for {select_bts}')
        fig = visualize.visualize_model_prediction(selected_data)
        st.pyplot(fig)
    
