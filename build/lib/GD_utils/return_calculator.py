import pandas as pd
import numpy as np
import GD_utils as gdu
class return_calculator:
    def __init__(self, ratio_df, cost=0.00):
        """
        :param ratio_df:    (
                            index   - rebalancing dates
                            columns - Symbols(the same with imported price data)
                            value   - weight
                            )

        :param cost:        (
                            [%]
                            ex. if do you want to apply trading cost of 0.73%, cost=0.0074
                            )
        """
        ratio_df[ratio_df==0] = np.nan
        self.rb_dt = ratio_df.copy()
        self.rb_dts = ratio_df.index
        self.p_dt = gdu.data.copy().loc[self.rb_dts[0]:,ratio_df.columns]
        self.cost = cost

        gr_p = self.get_df_grouped(self.p_dt, self.rb_dts)

        # 회전율 관련
        self.daily_ratio = gr_p.apply(self.calc_bt_daily_ratio)
        self.rb_tr_ratio = self.calc_rb_turnover(self.daily_ratio)
        self.rb_tr_ratio_stockwise = self.calc_rb_turnover_stockwise(self.daily_ratio)

        # 수익률 관련
        # periodic cumulative return
        # self.cmpd_rtn_period = gr_p.apply(self.calc_compound_return)

        # back-test daily return
        self.backtest_daily_return = gr_p.apply(self.calc_bt_compound_return).droplevel(0)
        # back-test daily cumulative return
        self.backtest_cumulative_return = self.backtest_daily_return.add(1).cumprod()

        # todo: 수익률 기여도 진행중
        # self.backtest_daily_return_ret_dist = gr_p.apply(self.calc_bt_ret_distribution)
        # self.backtest_daily_return_ret_dist['Portfolio Return']

        # AA2 = self.backtest_daily_return_ret_dist.add(1).cumprod()
        # AA3=AA2.loc["2022-03-24":"2022-03-31"].dropna(how='all', axis=1)
        # AA4=AA3.pct_change(5)
        # AA5=self.backtest_cumulative_return.loc["2022-03-24":"2022-03-31"].pct_change(5)
        # self.backtest_daily_return_ret_dist.add(1).cumprod()['Portfolio Return']


    def get_df_grouped(self, df, dts):
        # df, dts = self.p_dt.copy(), self.rb_dts.copy()
        df.loc[dts,'gr_idx'] = dts
        df['gr_idx'] = df['gr_idx'].fillna(method='ffill')
        return df.groupby('gr_idx')
    def calc_compound_return(self, grouped_price):
        return grouped_price.drop('gr_idx', axis=1).pct_change().fillna(0).add(1).cumprod().sub(1)
    def calc_bt_compound_return(self, grouped_price):
        # grouped_price = gr_p.get_group([x for x in gr_p.groups.keys()][0])
        gr_rtn = grouped_price.set_index('gr_idx').pct_change().fillna(0).add(1).cumprod().sub(1)
        output = gr_rtn.mul(self.rb_dt.loc[gr_rtn.index[0]]).dropna(axis=1,how='all').add(1).pct_change().sum(1).fillna(0)
        output.index = grouped_price.index

        # apply trading cost
        rb_d = output.index[0]
        output.loc[rb_d] = output.loc[rb_d] - self.cost*self.rb_tr_ratio.loc[rb_d]
        return output
    def calc_bt_ret_distribution(self, grouped_price):
        # grouped_price = gr_p.get_group([x for x in gr_p.groups.keys()][0])
        gr_rtn = grouped_price.set_index('gr_idx').pct_change().fillna(0).add(1).cumprod().sub(1)
        output = gr_rtn.mul(self.rb_dt.loc[gr_rtn.index[0]]).dropna(axis=1,how='all')#.add(1).pct_change().sum(1).fillna(0)
        output.index = grouped_price.index


        # apply trading cost
        rb_d = output.index[0]
        output.loc[rb_d] = output.loc[rb_d] - self.cost*self.rb_tr_ratio_stockwise.loc[rb_d].dropna()

        # stock-wise decomposing
        output['Portfolio Return'] = output.add(1).pct_change().sum(1)
        first_line = output.iloc[0]

        output = output.add(1).pct_change()
        output.iloc[0]=first_line
        return output
    def calc_bt_daily_ratio(self, grouped_price):
        gr_rtn = grouped_price.set_index('gr_idx').pct_change().fillna(0).add(1).cumprod()#.sub(1)
        output = gr_rtn.mul(self.rb_dt.loc[gr_rtn.index[0]])#.add(1).pct_change().sum(1).fillna(0)
        output.index = grouped_price.index
        return output
    def calc_rb_turnover(self, daily_account_ratio):
        rb_ratio_diff = daily_account_ratio.diff(1).loc[self.rb_dts]
        rb_ratio_diff.loc[self.rb_dts[0], self.rb_dt.columns] = self.rb_dt.iloc[0]
        return abs(rb_ratio_diff).sum(1)
    def calc_rb_turnover_stockwise(self, daily_account_ratio):
        rb_ratio_diff = daily_account_ratio.diff(1).loc[self.rb_dts]
        rb_ratio_diff.loc[self.rb_dts[0], self.rb_dt.columns] = self.rb_dt.iloc[0]
        return abs(rb_ratio_diff)