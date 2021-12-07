import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df = df.rename(columns={
        'User DL Average Throughput_LTE(kB/s)' : 'dl_avg',
        'User UL Average Throughput_LTE(kB/s)' : 'ul_avg',
        'Cell DL Peak Throughput_LTE(MB/s)' : 'dl_peak',
        'Cell UL Peak Throughput_LTE(MB/s)' : 'ul_peak'
    })
    df = df[df.columns[:-1]]
    df = df.dropna()
    df['eNodeB Name'] = df['eNodeB Name'].apply(lambda x: str(x).lower())
    df['Time']=pd.to_datetime(df['Time'])
    
    filtered = df[df['eNodeB Name'].map(df['eNodeB Name'].value_counts()) == 720]
    filtered = filtered.rename(columns={'Time':'ds'})
    bts_name = filtered['eNodeB Name'].unique()
    
    return filtered, bts_name

# get data
def get_data(bts_data, bts_name, column):
    bts_data = bts_data[bts_data['eNodeB Name']==bts_name]
    bts_data_col = bts_data.rename(columns={f'{column}':'y'})
    bts_data_col = bts_data_col[['ds', 'y']]
    bts_data_col = bts_data_col.set_index('ds')
    bts_data_col.index = pd.DatetimeIndex(bts_data_col.index).to_period('H')
    return bts_data_col

def diff_data(data, n):
    for i in range(n):
        data = data.diff()
    return data.values