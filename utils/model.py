from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pandas as pd
import numpy as np

def arima(bts_data, start, end, d, p ,q):
    model=ARIMA(bts_data['y'],order=(d, p, q))
    results=model.fit()
    bts_data['forecast']=results.predict(start=start*24,end=end*24,dynamic=True)
    return bts_data

def sarimax(bts_data, start, end, d, p, q, season):
    model=sm.tsa.statespace.SARIMAX(bts_data['y'],order=(d, p, q),seasonal_order=(d, p, q, season))
    results=model.fit()
    bts_data['forecast']=results.predict(start=start*24,end=end*24,dynamic=True)
    return bts_data