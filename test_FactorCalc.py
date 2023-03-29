import GD_utils as gdu
import pandas as pd
BM = gdu.get_data.get_naver_close("KOSPI")


test_df = pd.read_pickle("./test_df_top_all_sector.pickle")
test_df_day = pd.read_pickle("./test_df_sector_day.pickle")
test_df_day = pd.pivot(test_df_day, index='date', columns='종목코드', values='수정주가')


AA=pd.Series(test_df.filter(like='type_1').columns)

from GD_utils.factor_calculator import FactorAnalysis
self = FactorAnalysis(test_df, test_df_day, BM)
self.three_factor_decompose_report(
                                   col_name1='영업이익(TTM)_type_1_YoY_QoQ',
                                   drtion1=False,
                                   col_name2='ROE(지배, TTM)_type_1',
                                   drtion2=False,
                                   col_name3='EV/EBITDA(TTM)_type_1',
                                   drtion3=False,
                                   outputname='./UnnamedReport',
                                   display=True
                                   )
print("OK")
# self.factor_report('매출총이익_매출액', False, outputname='./UnnamedReport')
