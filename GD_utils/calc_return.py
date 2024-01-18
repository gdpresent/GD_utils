from GD_utils.return_calculator import *
from GD_utils.return_calculator_old import calculator


# 2024-01-18 디버그 완료
def calc_return(ratio_df, cost, n_days_after=0):
    return return_calculator_v2(ratio_df, cost, n_days_after).backtest_cumulative_return
def calc_return_contribution(ratio_df, cost, n_days_after=0):
    return return_calculator_v2(ratio_df, cost, n_days_after).daily_ret_cntrbtn

def calc_return_old(ratio_df, cost, n_days_after=0):
    return return_calculator(ratio_df, cost, n_days_after).backtest_cumulative_return
def calc_return_contribution_old(ratio_df, cost, n_days_after=0):
    return return_calculator(ratio_df, cost, n_days_after).daily_ret_cntrbtn

def calc_return_oldold_vers(ratio_df, cost):
    return calculator(ratio_df, cost).backtest_cumulative_return