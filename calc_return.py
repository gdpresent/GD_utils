import GD_utils as gdu
from GD_utils.return_calculator import calculator
def calc_return(ratio_df, cost):
    return calculator(ratio_df, cost).backtest_cumulative_return