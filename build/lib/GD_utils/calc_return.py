from GD_utils.return_calculator import return_calculator
def calc_return(ratio_df, cost):
    return return_calculator(ratio_df, cost).backtest_cumulative_return