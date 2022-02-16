import pandas_datareader as pdr
import requests
import pandas as pd
from bs4 import BeautifulSoup

# 네이버 차트에서 수정주가(종가, 시가)
def get_data_naver(company_code):
    # count=3000에서 3000은 과거 3,000 영업일간의 데이터를 의미. 사용자가 조절 가능
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count=300000&requestType=0".format(company_code)
    get_result = requests.get(url)
    bs_obj = BeautifulSoup(get_result.content, "html.parser")

    # information
    inf = bs_obj.select('item')
    columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_inf = pd.DataFrame([], columns=columns, index=range(len(inf)))

    for i in range(len(inf)):
        df_inf.iloc[i] = str(inf[i]['data']).split('|')
    df_inf.index = pd.to_datetime(df_inf['date'])
    return df_inf.drop('date', axis=1).astype(float)
def get_naver_open_close(company_codes):
    output = pd.DataFrame()
    if type(company_codes)==str:
        company_code=company_codes

        df = get_data_naver(company_code)[['Open', 'Close']].stack().reset_index().rename(columns={0:company_code})
        df['level_1'] = df['level_1'].apply(lambda x:"-16" if x=='Close' else "-09")
        df.index = pd.to_datetime(df['date'].astype(str) + df['level_1'])
        output = df[[company_code]]
    else:
        for company_code in company_codes:
            df = get_data_naver(company_code)[['Open', 'Close']].stack().reset_index().rename(columns={0: company_code})
            df['level_1'] = df['level_1'].apply(lambda x: "-16" if x == 'Close' else "-09")
            df.index = pd.to_datetime(df['date'].astype(str) + df['level_1'])
            output = pd.concat([output, df[[company_code]]], axis=1)
    return output
def get_naver_close(company_codes):
    output = pd.DataFrame()
    if type(company_codes)==str:
        company_code=company_codes
        df = get_data_naver(company_code)[['Close']].rename(columns={'Close':company_code})
        output = df[[company_code]]
    else:
        for company_code in company_codes:
            company_code = company_codes
            df = get_data_naver(company_code)[['Close']].rename(columns={'Close': company_code})
            output = pd.concat([output, df[[company_code]]], axis=1)
    return output


# 야후 수정주가 가져오기
def get_all_yahoo_data(name):
    return pdr.get_data_yahoo(name, start='1980-01-01').rename_axis('date', axis=0).sort_index()
def get_data_yahoo_close(symbols):
    if type(symbols) ==str:
        df = get_all_yahoo_data(symbols)[['Adj Close']].rename(columns={'Adj Close':symbols})
    else:
        df = get_all_yahoo_data(symbols)['Adj Close']
    return df
def get_data_yahoo_open_close(name):
    df = get_all_yahoo_data(name)

    close = df['Adj Close']
    open = df['Adj Close'] * df['Open'] / df['Close']
    open.index = pd.to_datetime(open.index.astype(str) + '-09')
    close.index = pd.to_datetime(close.index.astype(str) + '-16')
    ans = pd.concat([open,close]).sort_index()
    if type(name) == str:
        ans.name = name
        ans = pd.DataFrame(ans)
    else:
        ans.index.name = 'date'
        ans.columns.name = None
    return ans