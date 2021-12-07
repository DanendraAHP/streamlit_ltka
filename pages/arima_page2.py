from utils import model, preprocess, visualize
import streamlit as st

def modify_model(data, select_model, start_date, end_date, select_bts, select_data):
    ##################################Select d,p,q and seasonality##################################
    lc_1, lc_2, rc_1, rc_2 = st.columns(4)
    with lc_1:
        select_d = st.slider(
            'd value?', 0, 10, 0
        )
    with lc_2:
        select_p = st.slider(
            'p value?', 0, 10, 0
        )
    with rc_1:
        select_q = st.slider(
            'q value?', 0, 10, 0
        )
    with rc_2:
        select_season = st.slider(
            'seasonality value?', 0, 48, 24
        )
    ##Visualize
    fig1 = visualize.plot_qd(data['y'], select_d)
    st.header('Result of d value')
    st.pyplot(fig1)
    
    fig2 = visualize.plot_p(data['y'], select_p)
    st.header('Result of p value')
    st.pyplot(fig2)

    fig3 = visualize.plot_qd(data['y'], select_q)
    st.header('Result of q value')
    st.pyplot(fig3)
    ##################################Select model##################################
    if select_model == 'SARIMAX':
        selected_data = model.sarimax(data, start_date, end_date, select_d, select_p, select_q, select_season)
    else:
        selected_data = model.arima(data, start_date, end_date, select_d, select_p, select_q)
    ##################################row 3##################################
    with st.container():
        st.title(f'Line chart {select_data} for {select_bts}')
        fig4 = visualize.visualize_model_prediction(selected_data)
        st.pyplot(fig4)