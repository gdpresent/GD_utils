import gzip, pickle
import pandas as pd
import numpy as np
import GD_utils as gdu
import time
import multiprocessing
if __name__ == "__main__":
    from GD_utils.portfolio_calculator import PortfolioAnalysis
    BM = gdu.get_data.get_naver_close("KOSPI")

    w_df = pd.read_excel('./test_w_df_20220404.xlsx', index_col='date', parse_dates=['date'])
    with gzip.open(f'./test_df_day_20220404.pickle', 'rb') as l:
        test_df_day = pickle.load(l)
    gdu.data = test_df_day.pivot(index='date', columns='종목코드', values='수정주가')
    gdu.data['CASH']=1
    self=gdu.return_calculator_v2(ratio_df=w_df, cost=0.0)
    start = time.time()  # 시작 시간 저장
    print(gdu.return_calculator_v2(ratio_df=w_df, cost=0.0, n_day_after=0).portfolio_cumulative_return)
    print(f"데이터 처리1: {round((time.time() - start), 2)}초 소요")