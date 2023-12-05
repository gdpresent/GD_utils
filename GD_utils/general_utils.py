import os
import pandas as pd
import gzip, pickle
import time
import requests
from bs4 import BeautifulSoup

def data_preprocessing(data, univ=[]):
    data.columns = data.iloc[6]
    data = data.drop(range(0, 13), axis=0)
    data = data.rename(columns={'Code':'date'}).rename_axis("종목코드", axis="columns").set_index('date')
    data.index = pd.to_datetime(data.index)
    if len(univ)!=0:
        data = data[univ]
    return data
def read_gzip_pickle(location):
    start = time.time() # 시작 시간 저장
    with gzip.open(location, 'rb') as l:
        read = pickle.load(l)
    print(f'Loading Complete({round((time.time() - start) / 60, 2)}min): {location}')
    return read
def save_as_gzip_pickle(location, file):
    start = time.time()  # 시작 시간 저장
    with gzip.open(location, 'wb') as f:
        pickle.dump(file, f)
        print(f'Saving Complete({round((time.time() - start) / 60, 2)}min): {location}')
def save_as_pd_parquet(location, pandas_df_form):
    if location[-4:]!='.hd5':
        location = location+'.hd5'
    start = time.time()  # 시작 시간 저장
    pandas_df_form.to_parquet(f'{location}')
    print(f'Saving Complete({round((time.time() - start) / 60, 2)}min): {location}')
def read_pd_parquet(location):
    start = time.time() # 시작 시간 저장
    read = pd.read_parquet(location)
    print(f'Loading Complete({round((time.time() - start) / 60, 2)}min): {location}')
    return read

class get_processed_data:
    def __init__(self, today):
        self.today = today
        self.original_path = os.getcwd()
        self.chdir_to_DB_path()
    def chdir_to_DB_path(self):
        # original_path = os.getcwd()
        cnt_ = 0
        print(f'현재경로: {os.getcwd()}')
        while 'database' not in os.listdir(os.getcwd()):
            os.chdir('..')
            print(f'기준경로 변경 ==> {os.getcwd()}')
            if cnt_ > 100:
                print('경로설정확인요망')
                raise FileExistsError
        os.chdir('./database')
        print(f'경로변경완료: {os.getcwd()}')
    def chdir_to_original_path(self):
        print(f'경로 원위치 ==> {self.original_path}')
        os.chdir(self.original_path)

    def gen_Economic_Phase(self, EXP_XoX, CPI_XoX, Rolling_Month):
        import GD_utils as gdu
        # EXP_XoX=12
        # CPI_XoX=12
        # Rolling_Month=12*6
        self.chdir_to_DB_path()
        def get_BOK_macro_dict(key):
            dic = {
                "금리": {'KORIBOR3M': '010150000', 'KORIBOR6M': '010151000', 'KORIBOR12M': '010152000',
                       'CD91D': '010502000',
                       'CP91D': '010503000', '국민주택채권1종5Y': '010503500', '국고채1Y': '010190000', '국고채2Y': '010195000',
                       '국고채3Y': '010200000', '국고채5Y': '010200001', '국고채10Y': '010210000', '국고채20Y': '010220000',
                       '국고채30Y': '010230000', '통안증권91D': '010400001', '통안증권1Y': '010400000', '통안증권2Y': '010400002',
                       '산금채1Y': '010260000', '회사채3YAAm': '010300000', '회사채3YBBBm': '010320000',
                       '회사채AAm민평수익률': '010310000',
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
                "GDP성장률": {'한국': 'KOR', '호주': 'AUS', '오스트리아': 'AUT', '벨기에': 'BEL', '캐나다': 'CAN', '칠레': 'CHL',
                           '중국': 'CHN',
                           '체코': 'CZE', '덴마크': 'DNK', '에스토니아': 'EST', '핀란드': 'FIN', '프랑스': 'FRA', '독일': 'DEU',
                           '그리스': 'GRC',
                           '헝가리': 'HUN', '아이슬란드': 'ISL', '인도네시아': 'IDN', '아일랜드': 'IRL', '이스라엘': 'ISR',
                           '이탈리아': 'ITA',
                           '일본': 'JPN', '라트비아': 'LVA', '룩셈부르크': 'LUX', '멕시코': 'MEX', '네덜란드': 'NLD', '뉴질랜드': 'NZL',
                           '노르웨이': 'NOR', '폴란드': 'POL', '포르투갈': 'PRT', '러시아': 'RUS', '슬로바키아': 'SVK', '슬로베니아': 'SVN',
                           '스페인': 'ESP', '스웨덴': 'SWE', '스위스': 'CHE', '터키': 'TUR', '영국': 'GBR', "ID": '902Y015',
                           'PERIOD': 'Q'},
                "소비자물가지수": {'한국': 'KR',
                            '호주': 'AU', '오스트리아': 'AT', '벨기에': 'BE', '브라질': 'BR', '캐나다': 'CA', '칠레': 'CL',
                            '중국': 'CN',
                            '체코': 'CZ', '덴마크': 'DK', '에스토니아': 'EE', '핀란드': 'FI', '프랑스': 'FR', '독일': 'DE',
                            '그리스': 'GR',
                            '헝가리': 'HU', '아이슬란드': 'IS', '인도': 'IN', '인도네시아': 'ID', '아일랜드': 'IE', '이스라엘': 'IL',
                            '이탈리아': 'IT',
                            '일본': 'JP', '라트비아': 'LV', '룩셈부르크': 'LU', '멕시코': 'MX', '네덜란드': 'NL', '뉴질랜드': 'NZ',
                            '노르웨이': 'NO',
                            '폴란드': 'PL', '포르투갈': 'PT', '러시아': 'RU', '슬로바키아': 'SK', '슬로베니아': 'SI', '남아프리카공화국': 'ZA',
                            '스페인': 'ES', '스웨덴': 'SE', '스위스': 'CH', '터키': 'TR', '영국': 'GB', "ID": "902Y008",
                            "PERIOD": "M"},
                "KR소비자물가지수": {'국내': '0', 'ID': "901Y009", "PERIOD": "M"},
                "실업률": {'한국': 'KOR', '호주': 'AUS', '오스트리아': 'AUT', '벨기에': 'BEL', '캐나다': 'CAN', '칠레': 'CHL',
                        '체코': 'CZE',
                        '덴마크': 'DNK', '에스토니아': 'EST', '핀란드': 'FIN', '프랑스': 'FRA', '독일': 'DEU', '그리스': 'GRC',
                        '헝가리': 'HUN',
                        '아이슬란드': 'ISL', '아일랜드': 'IRL', '이스라엘': 'ISR', '이탈리아': 'ITA', '일본': 'JPN', '룩셈부르크': 'LUX',
                        '멕시코': 'MEX', '네덜란드': 'NLD', '뉴질랜드': 'NZL', '노르웨이': 'NOR', '폴란드': 'POL', '포르투갈': 'PRT',
                        '슬로바키아': 'SVK', '슬로베니아': 'SVN', '스페인': 'ESP', '스웨덴': 'SWE', '스위스': 'CHE', '터키': 'TUR',
                        '영국': 'GBR',
                        "ID": "908Y021", "PERIOD": "M"},
                "환율": {'원달러': '0000001', '원위안': '0000053', '원엔': '0000002', "ID": '731Y001', 'PERIOD': 'D'},
                "국제환율": {'일본엔달러': '0000002', '달러유로': '0000003', '독일마르크달러': '0000004', '프랑스프랑달러': '0000005',
                         '이태리리라달러': '0000006', '벨기에프랑달러': '0000007', '오스트리아실링달러': '0000008', '네덜란드길더달러': '0000009',
                         '스페인페세타달러': '0000010', '핀란드마르카달러': '0000011', '달러영국파운드': '0000012', '캐나다달러달러': '0000013',
                         '스위스프랑달러': '0000014', '달러호주달러': '0000017', '달러뉴질랜드달러': '0000026',
                         '중국위안달러': '0000027', '홍콩위안달러': '0000030', '홍콩달러달러': '0000015', '대만달러달러': '0000031',
                         '몽골투그릭달러': '0000032', '카자흐스탄텡게달러': '0000033',
                         '태국바트달러': '0000028', '싱가폴달러달러': '0000024', '인도네시아루피아달러': '0000029',
                         '말레이지아링기트달러': '0000025',
                         '필리핀페소달러': '0000034', '베트남동달러': '0000035', '브루나이달러달러': '0000036',
                         '인도루피달러': '0000037', '파키스탄루피달러': '0000038', '방글라데시타카달러': '0000039', '멕시코 페소달러': '0000040',
                         '브라질헤알달러': '0000041', '아르헨티나페소달러': '0000042', '스웨덴크로나달러': '0000016', '덴마크크로네달러': '0000018',
                         '노르웨이크로네달러': '0000019', '러시아루블달러': '0000043', '헝가리포린트달러': '0000044', '폴란트즈워티달러': '0000045',
                         '체코코루나달러': '0000046', '사우디아라비아리알달러': '0000020', '카타르리얄달러': '0000047',
                         '이스라엘셰켈달러': '0000048', '요르단디나르달러': '0000049', '쿠웨이트디나르달러': '0000021',
                         '바레인디나르달러': '0000022',
                         '아랍연방토후국 더히람달러': '0000023', '터키리라달러': '0000050', '남아프리카공화국랜드달러': '0000051',
                         "ID": '731Y002',
                         'PERIOD': 'D'}
            }
            return dic[key]
        def get_BOK_macro(cls, code, AuthKey):
            # cls, code, AuthKey='금리', 'CD91D', "Z034ROZNL01ZJFFCB3K0"
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

            url = f"http://ecos.bok.or.kr/api/StatisticSearch/{AuthKey}/xml/kr/1/100000/{ID}/{PERIOD}/{STT_D}/{Date_D}/{code_dict[code]}"
            get_result = requests.get(url)
            if get_result.status_code == 200:
                try:
                    bs_obj = BeautifulSoup(get_result.text, "html.parser")
                    value = bs_obj.find_all('data_value')
                    date = bs_obj.select('time')
                    df_output = pd.DataFrame([], columns=["date", code], index=range(len(value)))
                    # for i in tqdm(range(len(value)), '----- loading data -----'):
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
        def get_CPI(your_key):
            return get_BOK_macro('KR소비자물가지수', '국내', your_key).rename(columns={"국내": 'KRCPI'})  # monthly

        CPI = get_CPI("Z034ROZNL01ZJFFCB3K0").sort_index()
        CPI.loc[CPI.index[-1] + pd.DateOffset(months=1)] = CPI.iloc[-1, 0]
        CPI = CPI.shift(1)
        CPI = CPI['KRCPI']

        # EXPORT = pd.read_excel(f'./../../database/Excel_data/ExpImp_{today}.xls')  # 익월 1일
        # EXPORT['기간'] = pd.to_datetime(EXPORT['기간'].astype(str).apply(lambda x: x[:7] + "0" if len(x) <= 6 else x[:7]), format="%Y.%m")
        # EXPORT['수출금액'] = EXPORT['수출금액'].astype(str).str.replace(',', '').astype(float)
        # # EXPORT['수출건수'] = EXPORT['수출건수'].astype(str).str.replace(',', '').astype(float)
        # EXPORT = EXPORT.rename(columns={"기간": "date"}).set_index('date').sort_index()
        # EXPORT.loc[EXPORT.index[-1] + pd.DateOffset(months=1)] = EXPORT.iloc[-1, 0]
        # EXPORT = EXPORT.shift(1)

        if not os.path.exists(f'./Excel_data/ExpImp_{self.today}.pickle'):
            EXPORT_tmp = gdu.get_data.get_Export_PublicDataPortal('Z7AubMAAhdoq2sLF3JiHlGXoJfjBedvBF%2BvmPH20t3wlWI6lVbcot1gPZPkI6nuP6vkJywAkQV5tmcfkNS3JYw%3D%3D')
            EXPORT_tmp = EXPORT_tmp.set_index('date')
            save_as_gzip_pickle(f'./Excel_data/ExpImp_{self.today}.pickle', EXPORT_tmp)
        else:
            EXPORT_tmp = read_gzip_pickle(f'./Excel_data/ExpImp_{self.today}.pickle')
        # EXPORT = EXPORT_tmp.loc["2000-01-01":].shift(1).rename(columns=lambda x: x.replace('(달러)', ''))
        # EXPORT = EXPORT_tmp.loc["2000-01-01":].rename(columns=lambda x:x.replace('(달러)', ''))
        # EXPORT = EXPORT_tmp.shift(1).rename(columns=lambda x:x.replace('(달러)', ''))
        EXPORT = EXPORT_tmp.rename(columns=lambda x:x.replace('(달러)', ''))
        EXPORT = EXPORT['수출금액']

        EXPORT_MoM = EXPORT.pct_change(EXP_XoX)  # 원래는 1을 넣어야하는데;;
        CPI_YoY = CPI.pct_change(CPI_XoX)  # 물가상승률 YoY
        Export_zscore = EXPORT_MoM.sub(EXPORT_MoM.rolling(min_periods=Rolling_Month, window=Rolling_Month).mean()).div(EXPORT_MoM.rolling(min_periods=Rolling_Month, window=Rolling_Month).std())
        CPI_zscore = CPI_YoY.sub(CPI_YoY.rolling(min_periods=Rolling_Month, window=Rolling_Month).mean()).div(CPI_YoY.rolling(min_periods=Rolling_Month, window=Rolling_Month).std())

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(14,10))
        # ax1 = fig.add_subplot(111)
        # ax1_l = ax1.plot(EXPORT_MoM.index, EXPORT_MoM, color='blue', linestyle='-', label='EXPORT_YoY')
        # ax2 = ax1.twinx()
        # ax2_l=ax2.plot(Export_zscore.index, Export_zscore, color='r', linestyle='-', label='EXPORT_YoY_zscore')
        # lns = ax1_l+ax2_l
        # labs = [l.get_label() for l in lns]
        # ax1.legend(lns, labs, loc=0)
        # plt.grid()
        # # plt.show()
        # plt.savefig('./ExportYoY_Zscore.png')

        DATA = pd.concat([Export_zscore.rename('Export_zscore'), CPI_zscore.rename('CPI_zscore')], axis=1)
        DATA.loc[(DATA['Export_zscore'] >= 0) & (DATA['CPI_zscore'] < 0), 'EconomicPhase'] = 'Recovery'  # 0
        DATA.loc[(DATA['Export_zscore'] >= 0) & (DATA['CPI_zscore'] >= 0), 'EconomicPhase'] = 'Expansion'  # 1
        DATA.loc[(DATA['Export_zscore'] < 0) & (DATA['CPI_zscore'] >= 0), 'EconomicPhase'] = 'Slowdown'  # 2
        DATA.loc[(DATA['Export_zscore'] < 0) & (DATA['CPI_zscore'] < 0), 'EconomicPhase'] = 'Contraction'  # 3
        save_as_gzip_pickle(f'./Excel_data/Economic_Phase_E{EXP_XoX}C{CPI_XoX}R{Rolling_Month}_{self.today}.pickle', DATA)

        self.chdir_to_original_path()
        return DATA
    def gen_Sector_info(self):
        # EXP_XoX=12
        # CPI_XoX=12
        # Rolling_Month=12*6

        self.chdir_to_DB_path()
        if not os.path.exists(f'./Excel_data/S_Sector_info_{self.today}.pickle'):
            L_info_tmp = pd.read_excel(f'./Excel_data/Sector_info_{self.today}.xlsx', sheet_name='대')
            M_info_tmp = pd.read_excel(f'./Excel_data/Sector_info_{self.today}.xlsx', sheet_name='중')
            S_info_tmp = pd.read_excel(f'./Excel_data/Sector_info_{self.today}.xlsx', sheet_name='소')
            L_info = data_preprocessing(L_info_tmp).stack()
            M_info = data_preprocessing(M_info_tmp).stack()
            S_info = data_preprocessing(S_info_tmp).stack()
            save_as_gzip_pickle(f'./Excel_data/L_Sector_info_{self.today}.pickle', L_info)
            save_as_gzip_pickle(f'./Excel_data/M_Sector_info_{self.today}.pickle', M_info)
            save_as_gzip_pickle(f'./Excel_data/S_Sector_info_{self.today}.pickle', S_info)
        else:
            L_info = read_gzip_pickle(f'./Excel_data/L_Sector_info_{self.today}.pickle')
            M_info = read_gzip_pickle(f'./Excel_data/M_Sector_info_{self.today}.pickle')
            S_info = read_gzip_pickle(f'./Excel_data/S_Sector_info_{self.today}.pickle')
        self.chdir_to_original_path()
        return L_info.rename('sector'), M_info.rename('sector'), S_info.rename('sector')
    def gen_SectorCode_info(self):
        # EXP_XoX=12
        # CPI_XoX=12
        # Rolling_Month=12*6

        self.chdir_to_DB_path()
        if not os.path.exists(f'./Excel_data/S_SectorCode_info_{self.today}.pickle'):
            L_info_tmp = pd.read_excel(f'./Excel_data/SectorCode_info_{self.today}.xlsx', sheet_name='대')
            M_info_tmp = pd.read_excel(f'./Excel_data/SectorCode_info_{self.today}.xlsx', sheet_name='중')
            S_info_tmp = pd.read_excel(f'./Excel_data/SectorCode_info_{self.today}.xlsx', sheet_name='소')
            L_info = data_preprocessing(L_info_tmp).stack()
            M_info = data_preprocessing(M_info_tmp).stack()
            S_info = data_preprocessing(S_info_tmp).stack()
            save_as_gzip_pickle(f'./Excel_data/L_SectorCode_info_{self.today}.pickle', L_info)
            save_as_gzip_pickle(f'./Excel_data/M_SectorCode_info_{self.today}.pickle', M_info)
            save_as_gzip_pickle(f'./Excel_data/S_SectorCode_info_{self.today}.pickle', S_info)
        else:
            L_info = read_gzip_pickle(f'./Excel_data/L_SectorCode_info_{self.today}.pickle')
            M_info = read_gzip_pickle(f'./Excel_data/M_SectorCode_info_{self.today}.pickle')
            S_info = read_gzip_pickle(f'./Excel_data/S_SectorCode_info_{self.today}.pickle')
        self.chdir_to_original_path()
        return L_info.rename('sector_code'), M_info.rename('sector_code'), S_info.rename('sector_code')
