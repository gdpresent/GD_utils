import GD_utils as gdu
import pandas as pd
import numpy as np
BM = gdu.get_data.get_naver_close("KOSPI")


test_df = pd.read_pickle("./test_df_top_300_sector.pickle")
test_df_day = pd.read_pickle("./test_df_sector_day.pickle")
test_df_day_price = test_df_day.pivot(index='date', columns='종목코드', values='수정주가')

sectors6 = test_df.groupby(['date', 'sector'])['종목코드'].count().groupby('sector').mean().apply(lambda x:np.nan if x<10 else 1).dropna().index
from GD_utils.factor_calculator import FactorAnalysis
self = FactorAnalysis(test_df, test_df_day_price,BM)
asdadsd

self.factor_report('매출총이익_매출액', False, outputname='./UnnamedReport')
