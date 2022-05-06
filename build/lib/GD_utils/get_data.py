import pandas_datareader as pdr
import requests
import pandas as pd
from bs4 import BeautifulSoup
import investpy

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


# 미국 금리
def get_US_Bond_all():
    ch_name = {'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}
    # investpy.get_bonds_list('united states')
    today =pd.Timestamp.today().strftime('%d/%m/%Y')
    USBond_1Y = investpy.get_bond_historical_data(bond='U.S. 1Y', from_date='01/01/2000', to_date=today).rename(columns=ch_name).add_prefix('USBond1Y_').rename_axis('date', axis=0)
    USBond_2Y = investpy.get_bond_historical_data(bond='U.S. 2Y', from_date='01/01/2000', to_date=today).rename(columns=ch_name).add_prefix('USBond2Y_').rename_axis('date', axis=0)
    USBond_3Y = investpy.get_bond_historical_data(bond='U.S. 3Y', from_date='01/01/2000', to_date=today).rename(columns=ch_name).add_prefix('USBond3Y_').rename_axis('date', axis=0)
    USBond_5Y = investpy.get_bond_historical_data(bond='U.S. 5Y', from_date='01/01/2000', to_date=today).rename(columns=ch_name).add_prefix('USBond5Y_').rename_axis('date', axis=0)
    USBond_10Y = investpy.get_bond_historical_data(bond='U.S. 10Y', from_date='01/01/2000', to_date=today).rename(columns=ch_name).add_prefix('USBond10Y_').rename_axis('date', axis=0)
    USBond_30Y = investpy.get_bond_historical_data(bond='U.S. 30Y', from_date='01/01/2000', to_date=today).rename(columns=ch_name).add_prefix('USBond30Y_').rename_axis('date', axis=0)

    output = pd.concat([
        USBond_1Y,
        USBond_2Y,
        USBond_3Y,
        USBond_5Y,
        USBond_10Y,
        USBond_30Y,
    ], axis=1)
    return output
def get_US_Bond_close():
    # investpy.get_bonds_list('united states')
    today =pd.Timestamp.today().strftime('%d/%m/%Y')

    USBond_1Y = investpy.get_bond_historical_data(bond='U.S. 1Y', from_date='01/01/2000', to_date=today)[['Close']].rename(columns={"Close":"USBond1Y"}).rename_axis('date', axis=0)
    USBond_2Y = investpy.get_bond_historical_data(bond='U.S. 2Y', from_date='01/01/2000', to_date=today)[['Close']].rename(columns={"Close":"USBond2Y"}).rename_axis('date', axis=0)
    USBond_3Y = investpy.get_bond_historical_data(bond='U.S. 3Y', from_date='01/01/2000', to_date=today)[['Close']].rename(columns={"Close":"USBond3Y"}).rename_axis('date', axis=0)
    USBond_5Y = investpy.get_bond_historical_data(bond='U.S. 5Y', from_date='01/01/2000', to_date=today)[['Close']].rename(columns={"Close":"USBond5Y"}).rename_axis('date', axis=0)
    USBond_10Y = investpy.get_bond_historical_data(bond='U.S. 10Y', from_date='01/01/2000', to_date=today)[['Close']].rename(columns={"Close":"USBond10Y"}).rename_axis('date', axis=0)
    USBond_30Y = investpy.get_bond_historical_data(bond='U.S. 30Y', from_date='01/01/2000', to_date=today)[['Close']].rename(columns={"Close":"USBond30Y"}).rename_axis('date', axis=0)

    output = pd.concat([
                        USBond_1Y,
                        USBond_2Y,
                        USBond_3Y,
                        USBond_5Y,
                        USBond_10Y,
                        USBond_30Y,
                        ], axis=1)
    return output

# 미국 매크로
def get_US_macro():
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    # M2: https://fred.stlouisfed.org/series/M2SL # (Billions of Dollars, Seasonally Adjusted, Monthly)
    #  These data are released on the fourth Tuesday of every month, generally at 1:00 p.m.
    #  Publication may be shifted to the next business day when the regular publication date falls on a federal holiday.
    #  40영업일 래깅
    M2 = pdr.get_data_fred('M2SL', start='1990-01-01', end=today).rename(columns={'M2SL': 'M2'}).rename_axis('date', axis=0)

    # unemployment: https://fred.stlouisfed.org/series/UNRATE # (Billions of Dollars, Seasonally Adjusted, Monthly)
    # release: 당월 익월 첫번째 금요일인 영업일
    unemployment = pdr.get_data_fred('UNRATE', start='1990-01-01', end=today).rename(columns={'UNRATE': 'unemployment'}).rename_axis('date', axis=0)

    # BEI: https://fred.stlouisfed.org/series/T10YIE # (Percent, Not Seasonally Adjusted, Daily)
    # (10-Year Breakeven Inflation Rate)
    # 래깅 필요 없음
    BEI = pdr.get_data_fred('T10YIE', start='1990-01-01', end=today).rename(columns={'T10YIE': 'BEI'}).rename_axis('date', axis=0)

    # CPI: https://fred.stlouisfed.org/series/CPALTT01USM657N # (Growth Rate Previous Period, Not Seasonally Adjusted, Monthly)
    # (Consumer Price Index)
    # release: 당월 익익월 8번째 영업일
    # 50영업일래깅
    # CPI = pdr.get_data_fred('CPALTT01USM657N',start='1990-01-01').rename(columns={'CPALTT01USM657N':'CPI'})
    CPI = pdr.get_data_fred('CPALTT01USM657N', start='1990-01-01', end=today).rename(columns={'CPALTT01USM657N': 'CPI'}).rename_axis('date', axis=0)

    # PCE: https://fred.stlouisfed.org/series/PCE # (Billions of Dollars, Seasonally Adjusted Annual Rate, Monthly)
    # (Personal Consumption Expenditures)
    # release: 당월 익월 마지막 금요일 영업일
    # 40영업일래깅
    PCE = pdr.get_data_fred('PCE', start='1990-01-01', end=today).rename_axis('date', axis=0)

    # VIX: https://fred.stlouisfed.org/series/VIXCLS # (Index, Not Seasonally Adjusted, Daily, Close)
    # 래깅 0
    VIX = pdr.get_data_fred('VIXCLS', start='1990-01-01', end=today).rename(columns={'VIXCLS': 'VIX'}).rename_axis('date', axis=0)

    # TED spread: https://fred.stlouisfed.org/series/TEDRATE # (Percent, Not Seasonally Adjusted, Daily)
    # (T-Treasury, ED-Euro Dollar, 미국을 제외한 지역에서의 은행이나 미국은행들의 해외지사에 예금된 미국 달러
    # TED 스프레드: 미국 외 지역 혹은 미국 은행들의 해외지점에 예금된 미 달러화의 대외거래 금리인 리보금리와 미국 단기국채 금리의 차이)
    # he series is lagged by one week because the LIBOR series is lagged by one week due to an agreement with the source.
    # Starting with the update on June 21, 2019, the Treasury bond data used in calculating interest rate spreads is obtained directly from the U.S. Treasury Department.
    # 7영업일 래깅
    TEDspread = pdr.get_data_fred('TEDRATE', start='1990-01-01', end=today).rename(columns={'TEDRATE': 'TEDspread'}).rename_axis('date', axis=0)


    # Leading Indicators OECD: Reference series: Gross Domestic Product (GDP): Normalised for the United States:
    # https://fred.stlouisfed.org/series/USALORSGPNOSTSAM # (Index, Seasonally Adjusted, Monthly)
    # release: 당월 익익익익월 8번째 영업일
    # 90영업일 래깅
    CLI = pdr.get_data_fred('USALORSGPNOSTSAM', start='1990-01-01', end=today).rename(columns={'USALORSGPNOSTSAM': 'CLI'}).rename_axis('date', axis=0)

    output = pd.concat([
                        M2,
                        unemployment,
                        BEI,
                        CPI,
                        PCE,
                        VIX,
                        TEDspread,
                        CLI,
                        ], axis=1)

    return output

# 한국 금리 및 매크로
def get_KRmacro_data(your_key):
    """
    # AuthKey 오류시 한국은행 경제통제시스템에서 갱신(My page -> 인증키 발급 내역)
    # https://ecos.bok.or.kr/jsp/openapi/OpenApiController.jsp?t=myAuthKey
    """
    def get_BOK_macro_dict(key):
        dic = {
            "금리": {'KORIBOR3M': '010150000', 'KORIBOR6M': '010151000', 'KORIBOR12M': '010152000', 'CD91D': '010502000',
                   'CP91D': '010503000', '국민주택채권1종5Y': '010503500', '국고채1Y': '010190000', '국고채2Y': '010195000',
                   '국고채3Y': '010200000', '국고채5Y': '010200001', '국고채10Y': '010210000', '국고채20Y': '010220000',
                   '국고채30Y': '010230000', '통안증권91D': '010400001', '통안증권1Y': '010400000', '통안증권2Y': '010400002',
                   '산금채1Y': '010260000', '회사채3YAAm': '010300000', '회사채3YBBBm': '010320000', '회사채AAm민평수익률': '010310000',
                   'MMF7D': '010501000', 'CMA수시형': '010504000', 'ID': "060Y001", "PERIOD": "DD"},
            "통화": {"화폐발행잔액(말잔)": "AAAA11", "화폐발행잔액(평잔)": "AAAA12", "본원통화(말잔)": "AAAA13", "본원통화(평잔)": "AAAA14",
                   "M1(말잔)": "AAAA15", "M1(평잔)": "AAAA16", "M1MMF(말잔)": "AAAA1N", "M1MMF(평잔)": "AAAA1O",
                   "M2(말잔)": "AAAA17",
                   "M2(평잔)": "AAAA18", "Lf(말잔)": "AAAA1D", "Lf(평잔)": "AAAA1E", "L(말잔)": "AAAA1F",
                   "계절조정M1(말잔)": "AAAA1H",
                   "계절조정M1(평잔)": "AAAA1I", "계절조정M2(말잔)": "AAAA1J", "계절조정M2(평잔)": "AAAA1K", "계절조정Lf(말잔)": "AAAA1L",
                   "계절조정Lf(평잔)": "AAAA1M", "계절조정L(말잔)": "AAAA1P", "예금은행총예금(말잔)": "AAAA21", "예금은행총예금(평잔)": "AAAA22",
                   "예금은행저축성예금(말잔)": "AAAA23", "예금은행저축성예금(평잔)": "AAAA24", "예금은행대출금(말잔)": "AAAA25",
                   "예금은행대출금(평잔)": "AAAA26",
                   "ID": "010Y002", "PERIOD": "MM"},
            "대출": {
                '총대출(당좌대출제외)': "BECBLB02", '기업대출': "BECBLB0201", '대기업대출': "BECBLB020101", '중소기업대출': "BECBLB020102",
                '운전자금대출': "BECBLB020103", '시설자금대출': "BECBLB020104", '가계대출': "BECBLB0202", '소액대출(500이하)': "BECBLB020201",
                '주택담보대출': "BECBLB020202", '예적금담보대출': "BECBLB020203", '보증대출': "BECBLB020204", '일반신용대출': "BECBLB020206",
                '집단대출': "BECBLB020207", '공공기타부문대출': "BECBLB03", '당좌대출': "BECBLB04", '총대출': "BECBLB01",
                "ID": "005Y006", "PERIOD": "MM"},
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
                'ID': "041Y013", "PERIOD": "MM"},
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
                'ID': "041Y014", "PERIOD": "MM"},
            "GDP성장률": {'한국': 'KOR', '호주': 'AUS', '오스트리아': 'AUT', '벨기에': 'BEL', '캐나다': 'CAN', '칠레': 'CHL', '중국': 'CHN',
                       '체코': 'CZE', '덴마크': 'DNK', '에스토니아': 'EST', '핀란드': 'FIN', '프랑스': 'FRA', '독일': 'DEU', '그리스': 'GRC',
                       '헝가리': 'HUN', '아이슬란드': 'ISL', '인도네시아': 'IDN', '아일랜드': 'IRL', '이스라엘': 'ISR', '이탈리아': 'ITA',
                       '일본': 'JPN', '라트비아': 'LVA', '룩셈부르크': 'LUX', '멕시코': 'MEX', '네덜란드': 'NLD', '뉴질랜드': 'NZL',
                       '노르웨이': 'NOR', '폴란드': 'POL', '포르투갈': 'PRT', '러시아': 'RUS', '슬로바키아': 'SVK', '슬로베니아': 'SVN',
                       '스페인': 'ESP', '스웨덴': 'SWE', '스위스': 'CHE', '터키': 'TUR', '영국': 'GBR', "ID": 'I10Y041',
                       'PERIOD': 'QQ'},
            "소비자물가지수": {'한국': 'KR',
                        '호주': 'AU', '오스트리아': 'AT', '벨기에': 'BE', '브라질': 'BR', '캐나다': 'CA', '칠레': 'CL', '중국': 'CN',
                        '체코': 'CZ', '덴마크': 'DK', '에스토니아': 'EE', '핀란드': 'FI', '프랑스': 'FR', '독일': 'DE', '그리스': 'GR',
                        '헝가리': 'HU', '아이슬란드': 'IS', '인도': 'IN', '인도네시아': 'ID', '아일랜드': 'IE', '이스라엘': 'IL', '이탈리아': 'IT',
                        '일본': 'JP', '라트비아': 'LV', '룩셈부르크': 'LU', '멕시코': 'MX', '네덜란드': 'NL', '뉴질랜드': 'NZ', '노르웨이': 'NO',
                        '폴란드': 'PL', '포르투갈': 'PT', '러시아': 'RU', '슬로바키아': 'SK', '슬로베니아': 'SI', '남아프리카공화국': 'ZA',
                        '스페인': 'ES', '스웨덴': 'SE', '스위스': 'CH', '터키': 'TR', '영국': 'GB', "ID": "I10Y022", "PERIOD": "MM"},
            "실업률": {'한국': 'KOR', '호주': 'AUS', '오스트리아': 'AUT', '벨기에': 'BEL', '캐나다': 'CAN', '칠레': 'CHL', '체코': 'CZE',
                    '덴마크': 'DNK', '에스토니아': 'EST', '핀란드': 'FIN', '프랑스': 'FRA', '독일': 'DEU', '그리스': 'GRC', '헝가리': 'HUN',
                    '아이슬란드': 'ISL', '아일랜드': 'IRL', '이스라엘': 'ISR', '이탈리아': 'ITA', '일본': 'JPN', '룩셈부르크': 'LUX',
                    '멕시코': 'MEX', '네덜란드': 'NLD', '뉴질랜드': 'NZL', '노르웨이': 'NOR', '폴란드': 'POL', '포르투갈': 'PRT',
                    '슬로바키아': 'SVK', '슬로베니아': 'SVN', '스페인': 'ESP', '스웨덴': 'SWE', '스위스': 'CHE', '터키': 'TUR', '영국': 'GBR',
                    "ID": "I10Y052", "PERIOD": "MM"},
            "환율": {'원달러': '0000003', '원위안': '0000010', '원엔': '0000006', "ID": '036Y003', 'PERIOD': 'DD'},
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
                     '아랍연방토후국 더히람달러': '0000023', '터키리라달러': '0000050', '남아프리카공화국랜드달러': '0000051', "ID": '036Y002',
                     'PERIOD': 'DD'}
        }
        return dic[key]
    def get_BOK_macro(cls, code, AuthKey):
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

        if PERIOD == 'MM':
            Date_D = today.strftime('%Y%m')
            STT_D = '199901'
            format_D = '%Y%m'
        elif PERIOD == 'DD':
            Date_D = today.strftime('%Y%m%d')
            STT_D = '19990101'
            format_D = '%Y%m%d'
        elif PERIOD == 'QQ':
            Date_D = today.strftime('%Y%m')
            STT_D = '1999'
            format_D = '%Y%m%d'
        else:
            Date_D = today.strftime('%Y')
            STT_D = '1999'
            format_D = '%Y'

        url = f"http://ecos.bok.or.kr/api/StatisticSearch/{AuthKey}/xml/kr/1/10000/{ID}/{PERIOD}/{STT_D}/{Date_D}/{code_dict[code]}"
        get_result = requests.get(url)
        if get_result.status_code == 200:
            try:
                bs_obj = BeautifulSoup(get_result.text, "html.parser")
                value = bs_obj.find_all('data_value')
                date = bs_obj.select('time')
                df_output = pd.DataFrame([], columns=["date", code], index=range(len(value)))
                for i in range(len(value)):
                    if PERIOD == 'QQ':

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