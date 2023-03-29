import pandas as pd
def data_preprocessing(data, univ=[]):
    data.columns = data.iloc[6]
    data = data.drop(range(0, 13), axis=0)
    data = data.rename(columns={'Code':'date'}).rename_axis("종목코드", axis="columns").set_index('date')
    data.index = pd.to_datetime(data.index)
    if len(univ)!=0:
        data = data[univ]
    return data
