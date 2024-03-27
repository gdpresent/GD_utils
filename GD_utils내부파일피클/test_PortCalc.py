import GD_utils as gdu
import pandas as pd
from GD_utils.portfolio_calculator import PortfolioAnalysis
BM = gdu.get_data.get_naver_close("KOSPI")


test_df = pd.read_excel("./test_price.xlsx", index_col='date', parse_dates=['date']).dropna()

self = PortfolioAnalysis(daily_return=test_df[test_df.columns[-2:]].dropna().pct_change(), last_BM=True)


gdu.single_report(test_df[test_df.columns[-2:]], last_BM=True)
gdu.report(test_df, last_BM=True)
gdu.basic_report(test_df)

# self.factor_report('매출총이익_매출액', False, outputname='./UnnamedReport')
