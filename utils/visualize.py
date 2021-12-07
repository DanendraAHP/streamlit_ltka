import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def visualize_model_prediction(bts_data):
    fig ,ax = plt.subplots() 
    bts_data[['y','forecast']].plot(figsize=(12,8), ax=ax)
    return fig

def plot_qd(data, n):
    data = preprocess.diff_data(data, n)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(data)
    ax[0].set_title(f'{n} Order Differencing')
    plot_acf(data, ax=ax[1])
    return fig

def plot_p(data, n):
    data = preprocess.diff_data(data, n)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(data)
    axes[0].set_title(f'{n} Order Differencing')
    axes[1].set(ylim=(0,5))
    plot_pacf(data, ax=axes[1])
    return fig