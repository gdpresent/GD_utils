import GD_utils as gdu
import pandas as pd
from GD_utils.factor_calculator import FactorAnalysis


# 예제 데이터 호출
test_df = pd.read_pickle("./test_df_top_all_sector.pickle")
test_df_day = pd.read_pickle("./test_df_sector_day.pickle")#.pivot(index='date', columns='종목코드', values='수정주가')

# 사용되는 BM
BM = gdu.get_data.get_naver_close("KOSPI")

# 클래스 호출
Factor_Rank_Sum_CLS = FactorAnalysis(test_df, test_df_day, BM)

# 원팩터 리포트
Factor_Rank_Sum_CLS.factor_report('ROE(지배, TTM)_type_1', False, outputname='./UnnamedReport')
# 3-팩터 rank sum 전략 리포트
Factor_Rank_Sum_CLS.three_factor_decompose_report(col_name1='영업이익(TTM)_type_1_YoY_QoQ',
                                                  drtion1=False,
                                                  col_name2='ROE(지배, TTM)_type_1',
                                                  drtion2=False,
                                                  col_name3='EV/EBITDA(TTM)_type_1',
                                                  drtion3=False,
                                                  outputname='./UnnamedReport',
                                                  display=True)




