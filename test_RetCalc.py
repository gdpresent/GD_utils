import gzip, pickle
import GD_utils as gdu
import pandas as pd
from GD_utils.portfolio_calculator import PortfolioAnalysis
BM = gdu.get_data.get_naver_close("KOSPI")

w_df = pd.read_excel('./test_w_df_20220404.xlsx', index_col='date', parse_dates=['date'])
with gzip.open(f'./test_df_day_20220404.pickle', 'rb') as l:
    test_df_day = pickle.load(l)
gdu.data = test_df_day.pivot(index='date', columns='종목코드', values='수정주가')
gdu.data['CASH']=1

self = gdu.return_calculator(ratio_df=w_df, cost=0.00365)
adsdas
# gdu.basic_report(self.backtest_cumulative_return.loc["2022-03-24":"2022-03-31"])
# gdu.calc_return.calc_return(ratio_df=w_df, cost=0.00365)