import GD_utils as gdu
import pandas as pd
BM = gdu.get_data.get_naver_close("KOSPI")


test_df = pd.read_pickle("./test_df")
test_df_day = pd.read_pickle("./test_df_day")
test_df_day = pd.pivot(test_df_day, index='date', columns='종목코드', values='수정주가')

col_name1,col_name2,col_name3, drtion1,drtion2,drtion3='매출원가_매출액','매출총이익_매출액','법인세비용_매출액',False,False,False

from GD_utils.factor_calculator import FactorAnalysis
self = FactorAnalysis(test_df, test_df_day, BM)

self.three_factor_decompose_report('매출원가_매출액',False,'매출총이익_매출액',False,'법인세비용_매출액',False, outputname='./UnnamedReport')

self.factor_report('매출총이익_매출액', False, outputname='./UnnamedReport')
