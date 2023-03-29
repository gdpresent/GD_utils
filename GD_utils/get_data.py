import pandas_datareader as pdr
from datetime import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf
yf.pdr_override()
# 네이버 차트에서 수정주가(종가, 시가)
def get_data_naver(company_code):
    # count=3000에서 3000은 과거 3,000 영업일간의 데이터를 의미. 사용자가 조절 가능
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count=3000000&requestType=0".format(company_code)
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
def get_all_yahoo_data_old(name):
    return pdr.get_data_yahoo(name, start='1971-01-01').rename_axis('date', axis=0).sort_index()
def get_all_yahoo_data(name, stt='1927-12-30'):
    def unix_date(date):
        epoch = datetime(1970, 1, 1)  # 유닉스 기준일
        t = datetime.strptime(date, '%Y-%m-%d')
        diff = t - epoch
        return (diff.days * 24 * 3600 + diff.seconds)
    start_day = unix_date(stt)
    end_day = unix_date(datetime.today().strftime('%Y-%m-%d'))  # 오늘 날짜 까지

    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + name + '?period1=' + str(start_day) + '&period2=' + str(end_day) + '&interval=1d&events=history'
    df = pd.read_csv(url, parse_dates=True, index_col='Date').rename_axis('date', axis=0).sort_index()
    # pdr.data.get_data_yahoo(name, start='1920-01-01')
    return df
def get_data_yahoo_close(symbols, stt='1927-12-30'):
    if type(symbols) ==str:
        df = get_all_yahoo_data(symbols, stt)[['Adj Close']].rename(columns={'Adj Close':symbols})
    else:
        df = get_all_yahoo_data(symbols, stt)['Adj Close']
    return df
def get_data_stooq_close(symbols, stt='1970-01-01'):
    if type(symbols) ==str:
        df = pdr.get_data_stooq(symbols, start=stt).rename_axis('date', axis=0).sort_index()[['Close']].rename(columns={'Close':symbols})
    else:
        df=pd.DataFrame()
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

# 미국 매크로
def get_US_macro(today_not_str=pd.Timestamp.now()):
    # investing.com 직접 받을 데이터 .... ㅈㅅ
    # todo 2. CHINA PMI : https://www.investing.com/economic-calendar/chinese-manufacturing-pmi-594
    # PMICH = pd.read_csv('./data/macro/PMI_CH.csv', parse_dates=['date'], index_col='date').add_prefix('PMICH_')
    # todo 3. US ISM: https://www.investing.com/economic-calendar/ism-manufacturing-pmi-173
    # ISMUS = pd.read_csv('./data/macro/ISM_US.csv', parse_dates=['date'], index_col='date').add_prefix('ISMUS_')

    # if ISMUS.index.max() < pd.to_datetime(self.today.strftime('%Y-%m-%d')):
    #     print('직접 다운받을 데이터 추가 후 사용하십시오.')
    #     print(r'CHI PMI : https://www.investing.com/economic-calendar/chinese-manufacturing-pmi-594')
    #     print(r'USA PMI : https://www.investing.com/economic-calendar/ism-manufacturing-pmi-173')

    # investing.com 크롤링으로 받을 데이터
    # ch_name = {'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}
    # GOLD = investpy.get_commodity_historical_data('Gold', from_date='01/01/2000', to_date=today_not_str.strftime('%d/%m/%Y')).rename(columns=ch_name).add_prefix('Gold_')
    # SLVER = investpy.get_commodity_historical_data('Silver', from_date='01/01/2000', to_date=today_not_str.strftime('%d/%m/%Y')).rename(columns=ch_name).add_prefix('Silver_')
    # COPPER = investpy.get_commodity_historical_data('COPPER', from_date='01/01/2000', to_date=today_not_str.strftime('%d/%m/%Y')).rename(columns=ch_name).add_prefix('Copper_')
    # WTI = investpy.get_commodity_historical_data('Crude Oil WTI', from_date='01/01/2000', to_date=today_not_str.strftime('%d/%m/%Y')).rename(columns=ch_name).add_prefix('WTI_')
    # KRWYEN = investpy.get_currency_cross_historical_data('KRW/JPY', from_date='01/01/2000', to_date=today_not_str.strftime('%d/%m/%Y')).rename(columns=ch_name).add_prefix('KRWYEN_')
    # KRWUSD = investpy.get_currency_cross_historical_data('KRW/USD', from_date='01/01/2000', to_date=today_not_str.strftime('%d/%m/%Y')).rename(columns=ch_name).add_prefix('KRWUSD_')
    # GOLD.index.name = 'date'
    # SLVER.index.name = 'date'
    # COPPER.index.name = 'date'
    # WTI.index.name = 'date'
    # KRWYEN.index.name = 'date'
    # KRWUSD.index.name = 'date'

    # yahoo에서 데이터 : S&P500 / NASDAQ / KOSPI
    # SNP500 = get_data_yahoo_close('^GSPC').rename(columns={'^GSPC':'SNP500'}).sort_index()
    # KOSPI = get_data_yahoo_close('^KS11').rename(columns={'^GSPC': 'SNP500'}).sort_index()
    DI = get_data_yahoo_close('DX-Y.NYB').rename(columns={'DX-Y.NYB': 'DollarIndex'}).rename_axis(index='date')[['DollarIndex']]
    DI.index = pd.to_datetime(DI.index, format='%Y-%m-%d')

    NASDAQ = get_data_yahoo_close('^IXIC').rename(columns={'^IXIC': 'NASDAQ'})[['NASDAQ']]
    NASDAQ.index = pd.to_datetime(NASDAQ.index, format='%Y-%m-%d')
    NASDAQ.index.name = 'date'
    NASDAQ.columns = ['NASDAQ']
    SNP500 = get_data_yahoo_close('^GSPC').rename(columns={'^GSPC': 'SNP500'})[['SNP500']]
    # SNP500 = nx.get_data.get_data_stooq('^SPX', start_date=KOSPI.index.min(), end_date=KOSPI.index.max()).rename(columns={'^GSPC':'SNP500'})[['Close']]
    SNP500.index = pd.to_datetime(SNP500.index, format='%Y-%m-%d')
    SNP500.index.name = 'date'
    SNP500.columns = ['SNP500']
    # self.SNP500 = SNP500.copy()
    End_Month_Dates = SNP500.groupby(pd.Grouper(freq='BM')).tail(1)


    # todo: FRED에서 받을 데이터
    from cif import cif
    data, subjects, measures = cif.createDataFrameFromOECD(countries=['USA'], dsname='MEI', frequency='M')
    data.index = pd.to_datetime(data.index)
    data = data.loc[End_Month_Dates.index.min():]
    data.index.name = 'date'

    # M2: https://fred.stlouisfed.org/series/M2SL # (Billions of Dollars, Seasonally Adjusted, Monthly)
    #  These data are released on the fourth Tuesday of every month, generally at 1:00 p.m.
    #  Publication may be shifted to the next business day when the regular publication date falls on a federal holiday.
    #  40영업일 래깅 -> 다음달 말일에 최초 관측 가능하다고 하자


    # unemployment: https://fred.stlouisfed.org/series/UNRATE # (Billions of Dollars, Seasonally Adjusted, Monthly)
    # release: 당월 익월 첫번째 금요일인 영업일
    # 익월 마지막날에 최초관측할 수 있었다고 하자.
    unemployment = pdr.get_data_fred('UNRATE', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'UNRATE': 'unemployment'}).rename_axis('date', axis=0)
    unemployment.loc[unemployment.index[-1] + pd.DateOffset(months=1)] = unemployment.iloc[-1]
    unemployment = unemployment.shift(1)
    unemployment = pd.concat([unemployment, End_Month_Dates], axis=1).fillna(method='ffill').groupby(pd.Grouper(freq='BM')).tail(1).drop('SNP500', axis=1)

    # BEI: https://fred.stlouisfed.org/series/T10YIE # (Percent, Not Seasonally Adjusted, Daily)
    # (10-Year Breakeven Inflation Rate)
    # 래깅 필요 없음
    BEI = pdr.get_data_fred('T10YIE', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'T10YIE': 'BEI'}).rename_axis('date', axis=0)


    # CPI(Consumer Price Index: All Items for the United States)
    # : https://fred.stlouisfed.org/series/USACPIALLMINMEI # (Index 2015=100, Not Seasonally Adjusted, Monthly)
    # release: 익월 중순 15일 전후 -> 익월말일에 최초관측가능하다고 하자
    CPI = data[("USA",'CPALTT01','IXOBSA')].rename("CPI")
    CPI.loc[CPI.index[-1] + pd.DateOffset(months=1)] = CPI.iloc[-1]
    CPI = CPI.shift(1).dropna()

    CPI_YoY = CPI.pct_change(12).rename('CPI_YoY')
    CPI_MoM = CPI.pct_change(1).rename('CPI_MoM')
    # CPI_YoY_z48 = CPI.pct_change(12).sub(CPI.pct_change(12).rolling(min_periods=48, window=48).mean()).div(CPI.pct_change(12).rolling(min_periods=48, window=48).std()).rename('CPI_YoY_z48')
    # CPI_YoY_z60 = CPI.pct_change(12).sub(CPI.pct_change(12).rolling(min_periods=60, window=60).mean()).div(CPI.pct_change(12).rolling(min_periods=60, window=60).std()).rename('CPI_YoY_z60')
    # CPI_YoY_z72 = CPI.pct_change(12).sub(CPI.pct_change(12).rolling(min_periods=72, window=72).mean()).div(CPI.pct_change(12).rolling(min_periods=72, window=72).std()).rename('CPI_YoY_z72')
    CPI = pd.concat([CPI, CPI_YoY, CPI_MoM], axis=1) # CPI_YoY_z48, CPI_YoY_z60, CPI_YoY_z72
    CPI = pd.concat([CPI, End_Month_Dates], axis=1).fillna(method='ffill').groupby(pd.Grouper(freq='BM')).tail(1).drop('SNP500', axis=1)

    # PCE: https://fred.stlouisfed.org/series/PCE # (Billions of Dollars, Seasonally Adjusted Annual Rate, Monthly)
    # (Personal Consumption Expenditures)
    # release: 당월 익월 마지막 금요일 영업일
    # 40영업일래깅
    # PCE = pdr.get_data_fred('PCE', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d'))  # .rename(columns={'PCE':'PCE'})
    # PCE.index.name = 'date'

    # VIX: https://fred.stlouisfed.org/series/VIXCLS # (Index, Not Seasonally Adjusted, Daily, Close)
    # 래깅 0
    VIX = pdr.get_data_fred('VIXCLS', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'VIXCLS': 'VIX'})
    VIX.index.name = 'date'
    # TED spread: https://fred.stlouisfed.org/series/TEDRATE # (Percent, Not Seasonally Adjusted, Daily)

    # (T-Treasury, ED-Euro Dollar, 미국을 제외한 지역에서의 은행이나 미국은행들의 해외지사에 예금된 미국 달러
    # TED 스프레드: 미국 외 지역 혹은 미국 은행들의 해외지점에 예금된 미 달러화의 대외거래 금리인 리보금리와 미국 단기국채 금리의 차이)
    # he series is lagged by one week because the LIBOR series is lagged by one week due to an agreement with the source.
    # Starting with the update on June 21, 2019, the Treasury bond data used in calculating interest rate spreads is obtained directly from the U.S. Treasury Department.
    # 7영업일 래깅
    # TEDspread = pdr.get_data_fred('TEDRATE', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'TEDRATE': 'TEDspread'})
    # TEDspread.index.name = 'date'

    # Leading Indicators OECD: Reference series: Gross Domestic Product (GDP): Normalised for the United States:
    # 3개월 뒤(익익익월) 말일자에 최초 관측할 수 있었다고 하자.
    # CLI = pdr.get_data_fred('USALORSGPNOSTSAM', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'USALORSGPNOSTSAM': 'CLI'})
    US_GDP = data[('USA', 'LORSGPNO', 'STSA')].rename("GDP")
    US_GDP.loc[US_GDP.index[-1] + pd.DateOffset(months=1)] = US_GDP.iloc[-1]
    US_GDP = US_GDP.shift(3)
    GDP_YoY = US_GDP.pct_change(12).rename('GDP_YoY')
    GDP_MoM = US_GDP.pct_change(1).rename('GDP_MoM')
    # CPI_YoY_z48 = CPI.pct_change(12).sub(CPI.pct_change(12).rolling(min_periods=48, window=48).mean()).div(CPI.pct_change(12).rolling(min_periods=48, window=48).std()).rename('CPI_YoY_z48')
    # CPI_YoY_z60 = CPI.pct_change(12).sub(CPI.pct_change(12).rolling(min_periods=60, window=60).mean()).div(CPI.pct_change(12).rolling(min_periods=60, window=60).std()).rename('CPI_YoY_z60')
    # CPI_YoY_z72 = CPI.pct_change(12).sub(CPI.pct_change(12).rolling(min_periods=72, window=72).mean()).div(CPI.pct_change(12).rolling(min_periods=72, window=72).std()).rename('CPI_YoY_z72')
    GDP = pd.concat([US_GDP, GDP_YoY, GDP_MoM], axis=1) #, CPI_YoY_z48, CPI_YoY_z60, CPI_YoY_z72
    GDP = pd.concat([GDP, End_Month_Dates], axis=1).fillna(method='ffill').groupby(pd.Grouper(freq='BM')).tail(1).drop('SNP500', axis=1)

    # CLI: Leading Indicators OECD: Leading indicators: CLI: Normalised for the United States
    # https://fred.stlouisfed.org/series/USALOLITONOSTSAM
    # 익월 말일자에 최초 관측할 수 있었다고 하자.
    CLI_Norm = data[('USA', 'LOLITONO', 'STSA')].rename("CLI_Norm")
    CLI_Norm.loc[CLI_Norm.index[-1] + pd.DateOffset(months=1)] = CLI_Norm.iloc[-1]
    CLI_Norm = CLI_Norm.shift(1)
    CLI_Norm = pd.concat([CLI_Norm, End_Month_Dates], axis=1).fillna(method='ffill').groupby(pd.Grouper(freq='BM')).tail(1).drop('SNP500', axis=1)

    # CLI_Amp = data[('USA', 'LOLITOAA', 'STSA')].rename("CLI_Amp")
    # CLI_Amp.loc[CLI_Amp.index[-1] + pd.DateOffset(months=1)] = CLI_Amp.iloc[-1]
    # CLI_Amp = CLI_Amp.shift(1)
    # CLI_Amp = pd.concat([CLI_Amp, End_Month_Dates], axis=1).fillna(method='ffill').groupby(pd.Grouper(freq='BM')).tail(1).drop('SNP500', axis=1)


    USBond_1Y = pdr.get_data_fred('DGS1', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'DGS1': 'fredUSBond1Y'}).rename_axis(index='date')
    USBond_2Y = pdr.get_data_fred('DGS2', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'DGS2': 'fredUSBond2Y'}).rename_axis(index='date')
    USBond_3Y = pdr.get_data_fred('DGS3', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'DGS3': 'fredUSBond3Y'}).rename_axis(index='date')
    USBond_5Y = pdr.get_data_fred('DGS5', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'DGS5': 'fredUSBond5Y'}).rename_axis(index='date')
    USBond_10Y = pdr.get_data_fred('DGS10', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'DGS10': 'fredUSBond10Y'}).rename_axis(index='date')
    USBond_30Y = pdr.get_data_fred('DGS30', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'DGS30': 'fredUSBond30Y'}).rename_axis(index='date')

    # BBB 이상 회사채(잔존만기 1년 이상) - > OAS로 Embedded Option Adjusted
    Credit_Spread = pdr.get_data_fred('BAMLC0A0CM', start='1978-01-01', end=today_not_str.strftime('%Y-%m-%d')).rename(columns={'BAMLC0A0CM': 'CreditSpread'}).rename_axis(index='date')



    DATA = pd.merge(SNP500.add_suffix("(D)"), NASDAQ.add_suffix("(D)"), how='outer', on='date')
    # DATA = DATA.merge(GOLD, how='outer', on='date')
    # DATA = DATA.merge(SLVER, how='outer', on='date')
    # DATA = DATA.merge(COPPER, how='outer', on='date')
    # DATA = DATA.merge(WTI, how='outer', on='date')
    # DATA = DATA.merge(PMICH, how='outer', on='date')
    # DATA = DATA.merge(ISMUS, how='outer', on='date')
    DATA = DATA.merge(DI.add_suffix("(D)"), how='outer', on='date')
    # DATA = DATA.merge(KRWYEN, how='outer', on='date')
    # DATA = DATA.merge(KRWUSD, how='outer', on='date')
    # DATA = DATA.merge(PCE, how='outer', on='date')
    DATA = DATA.merge(VIX.add_suffix("(D)"), how='outer', on='date')
    # DATA = DATA.merge(TEDspread, how='outer', on='date')
    DATA = DATA.merge(USBond_1Y.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(USBond_2Y.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(USBond_3Y.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(USBond_5Y.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(USBond_10Y.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(USBond_30Y.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(BEI.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.merge(Credit_Spread.add_suffix("(D)"), how='outer', on='date')
    DATA = DATA.sort_index().fillna(method='ffill')
    DATA = DATA.merge(CLI_Norm.add_suffix("(M)"), how='outer', on='date')
    # DATA = DATA.merge(CLI_Amp.add_suffix("(M)"), how='outer', on='date')
    DATA = DATA.merge(unemployment.add_suffix("(M)"), how='outer', on='date')
    DATA = DATA.merge(CPI.add_suffix("(M)"), how='outer', on='date')
    DATA = DATA.merge(GDP.add_suffix("(M)"), how='outer', on='date')

    DATA = DATA.sort_index()
    DATA = DATA.loc[SNP500.index]
    # DATA = DATA[DATA.index >= '2000-01-01']
    print(f'Macro Data columns = \n{DATA.columns}')
    return DATA

# 한국은행 Open API
def get_BOK_macro_dict(key):
    dic = {
        "금리": {'KORIBOR3M': '010150000', 'KORIBOR6M': '010151000', 'KORIBOR12M': '010152000', 'CD91D': '010502000',
               'CP91D': '010503000', '국민주택채권1종5Y': '010503500', '국고채1Y': '010190000', '국고채2Y': '010195000',
               '국고채3Y': '010200000', '국고채5Y': '010200001', '국고채10Y': '010210000', '국고채20Y': '010220000',
               '국고채30Y': '010230000', '통안증권91D': '010400001', '통안증권1Y': '010400000', '통안증권2Y': '010400002',
               '산금채1Y': '010260000', '회사채3YAAm': '010300000', '회사채3YBBBm': '010320000', '회사채AAm민평수익률': '010310000',
               'MMF7D': '010501000', 'CMA수시형': '010504000', 'ID': "817Y002", "PERIOD": "D"},
        "경기실사지수(실적)": {
            '전산업': '99988',
            '제조업': 'C0000',
            '대기업': 'X5000',
            '중소기업': 'X6000',
            '중화학공업': 'X3000',
            '경공업': 'X4000',
            '수출기업': 'X8000',
            '내수기업': 'X9000',
            '비제조업': 'Y9900',
            '서비스업': 'Y9950',
            'ID': "512Y013", "PERIOD": "M"},
        "경기실사지수(전망)": {
            '전산업': '99988',
            '제조업': 'C0000',
            '대기업': 'X5000',
            '중소기업': 'X6000',
            '중화학공업': 'X3000',
            '경공업': 'X4000',
            '수출기업': 'X8000',
            '내수기업': 'X9000',
            '비제조업': 'Y9900',
            '서비스업': 'Y9950',
            'ID': "512Y015", "PERIOD": "M"},
        "GDP성장률": {'한국': 'KOR', '호주': 'AUS', '오스트리아': 'AUT', '벨기에': 'BEL', '캐나다': 'CAN', '칠레': 'CHL', '중국': 'CHN',
                   '체코': 'CZE', '덴마크': 'DNK', '에스토니아': 'EST', '핀란드': 'FIN', '프랑스': 'FRA', '독일': 'DEU', '그리스': 'GRC',
                   '헝가리': 'HUN', '아이슬란드': 'ISL', '인도네시아': 'IDN', '아일랜드': 'IRL', '이스라엘': 'ISR', '이탈리아': 'ITA',
                   '일본': 'JPN', '라트비아': 'LVA', '룩셈부르크': 'LUX', '멕시코': 'MEX', '네덜란드': 'NLD', '뉴질랜드': 'NZL',
                   '노르웨이': 'NOR', '폴란드': 'POL', '포르투갈': 'PRT', '러시아': 'RUS', '슬로바키아': 'SVK', '슬로베니아': 'SVN',
                   '스페인': 'ESP', '스웨덴': 'SWE', '스위스': 'CHE', '터키': 'TUR', '영국': 'GBR', "ID": '902Y015',
                   'PERIOD': 'Q'},
        "소비자물가지수": {'한국': 'KR',
                    '호주': 'AU', '오스트리아': 'AT', '벨기에': 'BE', '브라질': 'BR', '캐나다': 'CA', '칠레': 'CL', '중국': 'CN',
                    '체코': 'CZ', '덴마크': 'DK', '에스토니아': 'EE', '핀란드': 'FI', '프랑스': 'FR', '독일': 'DE', '그리스': 'GR',
                    '헝가리': 'HU', '아이슬란드': 'IS', '인도': 'IN', '인도네시아': 'ID', '아일랜드': 'IE', '이스라엘': 'IL', '이탈리아': 'IT',
                    '일본': 'JP', '라트비아': 'LV', '룩셈부르크': 'LU', '멕시코': 'MX', '네덜란드': 'NL', '뉴질랜드': 'NZ', '노르웨이': 'NO',
                    '폴란드': 'PL', '포르투갈': 'PT', '러시아': 'RU', '슬로바키아': 'SK', '슬로베니아': 'SI', '남아프리카공화국': 'ZA',
                    '스페인': 'ES', '스웨덴': 'SE', '스위스': 'CH', '터키': 'TR', '영국': 'GB', "ID": "902Y008", "PERIOD": "M"},
        "한국소비자물가지수": {
                        '총지수': '0',
                        '식료품및비주류음료': 'A',
                        '주류및담배': 'B',
                        '의류및신발': 'C',
                        '주택,수도,전기및연료': 'D',
                        '가정용품및가사서비스': 'E',
                        '보건': 'F',
                        '교통': 'G',
                        '통신': 'H',
                        '오락및문화': 'I',
                        '교육': 'J',
                        '음식및숙박': 'K',
                        '기타상품및서비스': 'L',
                        "ID": "901Y009", "PERIOD": "M"},
        "실업률": {'한국': 'KOR', '호주': 'AUS', '오스트리아': 'AUT', '벨기에': 'BEL', '캐나다': 'CAN', '칠레': 'CHL', '체코': 'CZE',
                '덴마크': 'DNK', '에스토니아': 'EST', '핀란드': 'FIN', '프랑스': 'FRA', '독일': 'DEU', '그리스': 'GRC', '헝가리': 'HUN',
                '아이슬란드': 'ISL', '아일랜드': 'IRL', '이스라엘': 'ISR', '이탈리아': 'ITA', '일본': 'JPN', '룩셈부르크': 'LUX',
                '멕시코': 'MEX', '네덜란드': 'NLD', '뉴질랜드': 'NZL', '노르웨이': 'NOR', '폴란드': 'POL', '포르투갈': 'PRT',
                '슬로바키아': 'SVK', '슬로베니아': 'SVN', '스페인': 'ESP', '스웨덴': 'SWE', '스위스': 'CHE', '터키': 'TUR', '영국': 'GBR',
                "ID": "908Y021", "PERIOD": "M"},
        "환율": {'원달러': '0000001', '원위안': '0000053', '원엔': '0000002', "ID": '731Y001', 'PERIOD': 'D'},
        "국제환율": {'일본엔달러': '0000002', '달러유로': '0000003', '독일마르크달러': '0000004', '프랑스프랑달러': '0000005',
                 '이태리리라달러': '0000006', '벨기에프랑달러': '0000007', '오스트리아실링달러': '0000008', '네덜란드길더달러': '0000009',
                 '스페인페세타달러': '0000010', '핀란드마르카달러': '0000011', '달러영국파운드': '0000012', '캐나다달러달러': '0000013',
                 '스위스프랑달러': '0000014', '달러호주달러': '0000017', '달러뉴질랜드달러': '0000026',
                 '중국위안달러': '0000027', '홍콩위안달러': '0000030', '홍콩달러달러': '0000015', '대만달러달러': '0000031',
                 '몽골투그릭달러': '0000032', '카자흐스탄텡게달러': '0000033',
                 '태국바트달러': '0000028', '싱가폴달러달러': '0000024', '인도네시아루피아달러': '0000029', '말레이지아링기트달러': '0000025',
                 '필리핀페소달러': '0000034', '베트남동달러': '0000035', '브루나이달러달러': '0000036',
                 '인도루피달러': '0000037', '파키스탄루피달러': '0000038', '방글라데시타카달러': '0000039', '멕시코 페소달러': '0000040',
                 '브라질헤알달러': '0000041', '아르헨티나페소달러': '0000042', '스웨덴크로나달러': '0000016', '덴마크크로네달러': '0000018',
                 '노르웨이크로네달러': '0000019', '러시아루블달러': '0000043', '헝가리포린트달러': '0000044', '폴란트즈워티달러': '0000045',
                 '체코코루나달러': '0000046', '사우디아라비아리알달러': '0000020', '카타르리얄달러': '0000047',
                 '이스라엘셰켈달러': '0000048', '요르단디나르달러': '0000049', '쿠웨이트디나르달러': '0000021', '바레인디나르달러': '0000022',
                 '아랍연방토후국 더히람달러': '0000023', '터키리라달러': '0000050', '남아프리카공화국랜드달러': '0000051', "ID": '731Y002',
                 'PERIOD': 'D'}
    }
    return dic[key]
def get_BOK_macro(cls, code, AuthKey):
    # cls, code, AuthKey='금리', 'CD91D', your_key
    today = pd.Timestamp.today()
    def quarter_to_date(inp):
        # inp = date[i].text
        if inp[-1] == '1':
            output = '0331'
        elif inp[-1] == '2':
            output = '0630'
        elif inp[-1] == '3':
            output = '0930'
        else:
            output = '1231'
        return inp[:-1] + output

        # date[i].text[:-1] + date[i].text[-1]


    code_dict = get_BOK_macro_dict(cls)
    ID = code_dict['ID']
    PERIOD = code_dict['PERIOD']

    if PERIOD == 'M':
        Date_D = today.strftime('%Y%m')
        STT_D = '199901'
        format_D = '%Y%m'
    elif PERIOD == 'D':
        Date_D = today.strftime('%Y%m%d')
        STT_D = '19990101'
        format_D = '%Y%m%d'
    elif PERIOD == 'Q':
        Date_D = today.strftime('%Y%m')
        STT_D = '1999'
        format_D = '%Y%m%d'
    else:
        Date_D = today.strftime('%Y')
        STT_D = '1999'
        format_D = 'A'


    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{AuthKey}/xml/kr/1/10000/{ID}/{PERIOD}/{STT_D}/{Date_D}/{code_dict[code]}"
    get_result = requests.get(url)
    if get_result.status_code == 200:
        try:
            bs_obj = BeautifulSoup(get_result.text, "html.parser")
            value = bs_obj.find_all('data_value')
            date = bs_obj.select('time')
            df_output = pd.DataFrame([], columns=["date", code], index=range(len(value)))
            for i in range(len(value)):
                if PERIOD == 'Q':

                    df_output.iloc[i, 0] = pd.to_datetime(quarter_to_date(date[i].text), format=format_D)
                else:
                    df_output.iloc[i, 0] = pd.to_datetime(date[i].text, format=format_D)
                df_output.iloc[i, 1] = float(value[i].text)

            return df_output.set_index('date')

            ## 호출결과에 오류가 있었는지 확인합니다.
        except Exception as e:
            print(str(e))
    else:
        print('get_result.status_code is not equal to 200')

        ##예외가 발생했을때 처리합니다.
def get_KRmacro_data(your_key):
    """
    # AuthKey 오류시 한국은행 경제통제시스템에서 갱신(My page -> 인증키 발급 내역)
    # https://ecos.bok.or.kr/jsp/openapi/OpenApiController.jsp?t=myAuthKey
    """

    # todo: 한국은행 데이터

    KRCD_3M = get_BOK_macro('금리', 'CD91D', your_key).rename(columns={"CD91D": 'KRCD3M'}) # daily
    KRCP_3M = get_BOK_macro('금리', 'CP91D', your_key).rename(columns={"CP91D": 'KRCP3M'}) # daily
    KRMonStab_1Y = get_BOK_macro('금리', '통안증권1Y', your_key).rename(columns={"통안증권1Y": 'KRMonStab1Y'}) #daily
    KRBond_1Y = get_BOK_macro('금리', '국고채1Y', your_key).rename(columns={"국고채1Y": 'KRBond1Y'}) #daily
    KRBond_2Y = get_BOK_macro('금리', '국고채2Y', your_key).rename(columns={"국고채1Y": 'KRBond2Y'}) #daily
    KRBond_3Y = get_BOK_macro('금리', '국고채3Y', your_key).rename(columns={"국고채3Y": 'KRBond3Y'}) #daily
    KRBond_5Y = get_BOK_macro('금리', '국고채5Y', your_key).rename(columns={"국고채5Y": 'KRBond5Y'}) #daily
    KRBond_10Y = get_BOK_macro('금리', '국고채10Y', your_key).rename(columns={"국고채10Y": 'KRBond10Y'}) #daily
    KRBond_20Y = get_BOK_macro('금리', '국고채20Y', your_key).rename(columns={"국고채20Y": 'KRBond20Y'}) #daily
    KRBond_30Y = get_BOK_macro('금리', '국고채30Y', your_key).rename(columns={"국고채30Y": 'KRBond20Y'}) #daily
    KRExRate = get_BOK_macro('환율', '원달러', your_key).rename(columns={"한국": 'KR원달러'}) #daily
    KRCPI = get_BOK_macro('소비자물가지수', '한국', your_key).rename(columns={"한국": 'KRCPI'}) #monthly


    # KRM2 = get_BOK_macro('통화', 'M2(평잔)',your_key).rename(columns={"M2(평잔)": 'KRM2'}) #monthly
    # KRM2SL = get_BOK_macro('통화', '계절조정M2(평잔)',your_key).rename(columns={"계절조정M2(평잔)": 'KRM2SL'}) #monthly
    # KRLoan_tot = get_BOK_macro('대출', '총대출(당좌대출제외)',your_key).rename(columns={"총대출(당좌대출제외)": 'KR총대출'}) #monthly
    # KRLoan_HH = get_BOK_macro('대출', '가계대출',your_key).rename(columns={"가계대출": 'KR가계대출'}) #monthly
    # KRBSI_manuf = get_BOK_macro('경기실사지수(전망)', '제조업',your_key).rename(columns={"제조업": 'KR경기실사지수_제조업'}) #monthly
    # KRBSI_expt = get_BOK_macro('경기실사지수(전망)', '수출기업',your_key).rename(columns={"수출기업": 'KR경기실사지수_수출기업'}) #monthly
    # KRGDP_growth = get_BOK_macro('GDP성장률', '한국',your_key).rename(columns={"한국": 'KRGDPgrowth'}) #quarterly
    # KRUnrate = get_BOK_macro('실업률', '한국',your_key).rename(columns={"한국": 'KRUnemployment'})#monthly

    output = pd.concat(
                        [
                            KRCD_3M,
                            KRCP_3M,
                            KRMonStab_1Y,
                            KRBond_1Y,
                            KRBond_2Y,
                            KRBond_3Y,
                            KRBond_5Y,
                            KRBond_10Y,
                            KRBond_20Y,
                            KRBond_30Y,
                            KRExRate,
                            KRCPI,
                        ], axis=1
                      )

    return output
def get_CPI(your_key):
    return get_BOK_macro('소비자물가지수', '한국', your_key).rename(columns={"한국": 'KRCPI'}) #monthly
def get_KRCPI_total(your_key):
    return get_BOK_macro('한국소비자물가지수', '총지수', your_key).rename(columns={"한국": 'KRCPI'}) #monthly

# 공공데이터 포털 API
def get_Export_PublicDataPortal(your_key, today=pd.Timestamp.now().strftime("%Y-%m")):
    _col_replace_ = {'balpayments': '무역수지(달러)', 'expcnt': '수출건수', 'expdlr': '수출금액(달러)', 'impcnt': '수입건수', 'impdlr': '수입금액(달러)', 'year': 'date'}
    stt_date = pd.to_datetime('1989-01')
    today_ym = pd.to_datetime(today)
    def partition(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i: i + size]
    dates_range = [x.strftime("%Y%m") for x in pd.date_range(start=stt_date, end=today_ym, freq='MS')]

    output = pd.DataFrame()
    for year_list in list(partition(dates_range, 12)):
        ImExPort_Data_API = f'https://apis.data.go.kr/1220000/Newtrade/getNewtradeList?serviceKey={your_key}&strtYymm={year_list[0]}&endYymm={year_list[-1]}'
        response = requests.get(ImExPort_Data_API, verify=False)
        soup = BeautifulSoup(response.content,'html.parser')
        data = soup.find_all('item')

        data_temp=pd.DataFrame()
        for item in data:
            dict_update={}
            for k,v in _col_replace_.items():
                value = item.find(k).get_text()

                if value == '총계':
                    continue

                if _col_replace_[k]!='date':
                    dict_update[v] = float(value)
                elif _col_replace_[k]=='date':
                    dict_update[v] = pd.Timestamp(value)
            data_temp=pd.concat([data_temp,pd.DataFrame([dict_update])], axis=0)
        output = pd.concat([output, data_temp.dropna(axis=0)], axis=0)
    # output.set_index('date').to_excel(f'./수출입총괄_{today}.xlsx')
    return output

# 거래일 정보
def get_trading_calendar_from_now(n_days_from_now=5, this_week=False):
    import pandas_market_calendars as mcal
    KR_cal_cls = mcal.get_calendar('XKRX')
    NYSE_cal_cls = mcal.get_calendar('NYSE')
    NAS_cal_cls = mcal.get_calendar('NASDAQ')

    now=pd.Timestamp.now()
    dt_list = pd.date_range(now.strftime("%Y-%m-%d"), periods=n_days_from_now, freq='D')
    dt_df = pd.DataFrame(index=dt_list)
    dt_df['week_num'] = [x.week for x in dt_list]

    if this_week:
        dt_df = dt_df[dt_df['week_num']==dt_df['week_num'].min()]

    KR_cal=KR_cal_cls.schedule(start_date=dt_df.index[0], end_date=dt_df.index[-1]).rename(columns=lambda x:x.replace('market', 'KRX'))
    NYSE_cal=NYSE_cal_cls.schedule(start_date=dt_df.index[0], end_date=dt_df.index[-1]).rename(columns=lambda x:x.replace('market', 'NYSE'))
    NAS_cal=NAS_cal_cls.schedule(start_date=dt_df.index[0], end_date=dt_df.index[-1]).rename(columns=lambda x:x.replace('market', 'NASDAQ'))

    # GMT+9:00:00
    KR_cal_GMT9=KR_cal+pd.DateOffset(hours=9)
    NYSE_cal_GMT9=NYSE_cal+pd.DateOffset(hours=9)
    NAS_cal_GMT9=NAS_cal+pd.DateOffset(hours=9)

    KR_cal_GMT9 = KR_cal_GMT9.filter(regex='open$|close$', axis=1)
    NYSE_cal_GMT9 = NYSE_cal_GMT9.filter(regex='open$|close$', axis=1)
    NAS_cal_GMT9 = NAS_cal_GMT9.filter(regex='open$|close$', axis=1)

    return pd.concat([KR_cal_GMT9, NYSE_cal_GMT9, NAS_cal_GMT9], axis=1)
def get_trading_calendar_historical(stt_date="1999-12-31", end_date=pd.Timestamp.today().strftime("%Y-%m-%d")):
    import pandas_market_calendars as mcal
    KR_cal_cls = mcal.get_calendar('XKRX')
    NYSE_cal_cls = mcal.get_calendar('NYSE')
    NAS_cal_cls = mcal.get_calendar('NASDAQ')


    KR_cal=KR_cal_cls.schedule(start_date=stt_date, end_date=end_date).rename(columns=lambda x:x.replace('market', 'KRX'))
    NYSE_cal=NYSE_cal_cls.schedule(start_date=stt_date, end_date=end_date).rename(columns=lambda x:x.replace('market', 'NYSE'))
    NAS_cal=NAS_cal_cls.schedule(start_date=stt_date, end_date=end_date).rename(columns=lambda x:x.replace('market', 'NASDAQ'))

    # GMT+9:00:00
    KR_cal_GMT9=KR_cal+pd.DateOffset(hours=9)
    NYSE_cal_GMT9=NYSE_cal+pd.DateOffset(hours=9)
    NAS_cal_GMT9=NAS_cal+pd.DateOffset(hours=9)

    KR_cal_GMT9 = KR_cal_GMT9.filter(regex='open$|close$', axis=1)
    NYSE_cal_GMT9 = NYSE_cal_GMT9.filter(regex='open$|close$', axis=1)
    NAS_cal_GMT9 = NAS_cal_GMT9.filter(regex='open$|close$', axis=1)

    return pd.concat([KR_cal_GMT9, NYSE_cal_GMT9, NAS_cal_GMT9], axis=1)
def get_economic_schedule(date_target=None):
    from requests_html import HTMLSession
    date = pd.Timestamp.now().strftime("%Y%m%d")
    if date_target:
        date = pd.to_datetime(date_target).strftime("%Y%m%d")
    date_dash = pd.to_datetime(date).strftime("%Y-%m-%d")

    # url = f"https://invest.kiwoom.com/inv/calendar?date={20221208}"
    url = f"https://invest.kiwoom.com/inv/calendar?date={int(date)}"

    session = HTMLSession()
    r = session.get(url)

    r.html.render()
    bs_obj = BeautifulSoup(r.html.html, "html.parser")

    # col = bs_obj.select('h2#calendar-date')[0].text
    # col = bs_obj.select('div', {'class':'calendar_list_box > date', 'id':date_dash})
    # col = bs_obj.find_all('div', 'calendar_list_box', 'date', {'id': f"{date_dash}"})
    col = bs_obj.find_all('div', {"class":'calendar_list_box', 'id': f"{date_dash}"})[0].find_all('h3','date')[0].text

    # objs = bs_obj.select('div.calendar_info_list > li')
    objs = bs_obj.find_all('div', {"class":'calendar_list_box', 'id': f"{date_dash}"})[0].find_all('li')
    output = pd.DataFrame(columns=[col])
    for i in range(len(objs)):
        output.loc[i, col] = objs[i].text
    output['date'] = pd.to_datetime(date)
    output = output.set_index('date')
    return output