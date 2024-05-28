import os

import pandas as pd
import numpy as np
import GD_utils as gdu
import time


class return_calculator_v1:
    def __init__(self, ratio_df, cost=0.00, n_day_after=1):

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
        price_df = gdu.data.copy()
        ratio_df = ratio_df.apply(lambda x: x.replace(0, np.nan))
        self._cost = cost

        # 지정된 리밸런싱날짜
        self._rb_dt = ratio_df.copy()
        self._rb_dts = self._rb_dt.index

        # 지정된 리밸런싱 날짜 +n 영업일(실제리밸런싱날짜의 다음날 - 수익률기반으로 계산하기 위하여)
        self._rb_dt_nd_dly = ratio_df.copy()
        self._rb_dts_nd_dly = self.get_nday_delay(self._rb_dts, n_day_after+1)
        self._rb_dt_nd_dly.index = self._rb_dts_nd_dly

        # 실제 리밸런싱 날짜(실제 리밸런싱 날짜의 Turnover Ratio를 계산하기 위하여)
        self._rb_dts_1day_ago = self.get_nday_ago(self._rb_dts_nd_dly, -1)

        # 가격데이터
        # self._p_dt = price_df.loc[:,ratio_df.columns]

        # 일별수익률
        self._rnt_dt = price_df.pct_change().mask(price_df.isnull(), np.nan).loc[:,ratio_df.columns]
        self._rnt_dt.iloc[0]=0

        # gr_p = self.get_df_grouped(self._p_dt, self._rb_dts_nd_dly) # 가격기반 수익률계산에서, 일별수익률 기반 계산으로 변경함
        gr_rtn = self.get_df_grouped(self._rnt_dt, self._rb_dts_nd_dly)

        # 회전율 관련
        self._daily_ratio = gr_rtn.apply(self.calc_bt_daily_ratio) # 실제 일별 내 계좌 잔고의 종목별 비중[%]
        self._rb_tr_ratio = self.calc_rb_turnover(self._daily_ratio) # 실제 리밸런싱 날짜의 회전율
        self._rb_tr_ratio_stockwise = self.calc_rb_turnover_stockwise(self._daily_ratio) # 실제 리밸런싱 날짜의 종목별 회전율

        # back-test daily return
        self.backtest_daily_return = gr_rtn.apply(self.calc_bt_compound_return).droplevel(0)
        # back-test daily cumulative return
        self.backtest_cumulative_return = self.backtest_daily_return.add(1).cumprod()

        # 수익률 기여도
        self.daily_ret_cntrbtn = gr_rtn.apply(self.calc_ret_cntrbtn)
        # 확인: self.daily_ret_cntrbtn.sum(1).add(1).cumprod()

    def get_nday_delay(self, rb_dts, n=0):
        _rb_dts_1d_dly = []
        for idx in rb_dts:
            try:
                _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[n])
            except:
                print(f'가격데이터 불충분 > {n-1}일 후 리밸런싱이 기입 -> BUT \n{gdu.data.loc[idx:]}')
                print(f'따라서 마지막날짜: {gdu.data.loc[idx:].index[-1]}로 대체하였습니다.')
                _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_nday_ago(self, rb_dts, n=-1):
        # rb_dts, n = self._rb_dts, n_day_after + 1
        # rb_dts, n = self._rb_dts_nd_dly, -1
        _rb_dts_1d_dly = []
        for idx in rb_dts:
            _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[n-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_df_grouped(self, df, dts):
        # df, dts = self._rnt_dt.copy(), self._rb_dts_nd_dly.copy()
        df.loc[dts,'gr_idx'] = dts
        df['gr_idx'] = df['gr_idx'].fillna(method='ffill')
        return df.groupby('gr_idx')
    def calc_compound_return(self, grouped_price):
        return grouped_price.drop('gr_idx', axis=1).pct_change().fillna(0).add(1).cumprod().sub(1)
    def calc_bt_compound_return(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][1])
        gr_comp_rtn = grouped_return.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1) # input된 것은 일별수익률임 따라서 복리수익률을 만들어 준 이후,
        daily_comp_rtn = gr_comp_rtn.mul(self._rb_dt_nd_dly.loc[gr_comp_rtn.index[0]]).dropna(axis=1, how='all')  # "복리수익률" * "리밸런싱때 조정한 비중" = 포트폴리오 종목별 일별 (누적)복리수익률
        daily_comp_rtn.index = grouped_return.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0] # pct_change를 할 때 첫 줄이 날아가기 때문에 남겨놓아야 하는 첫 번째날 수익률(리밸런싱 하루 이후 포트폴리오 종목별 수익률 backup)
        daily_comp_rtn = daily_comp_rtn.add(1).pct_change() # 포트폴리오 종목별 일별 수익률
        daily_comp_rtn.iloc[0] = first_line


        tr_applied_here = self._rb_tr_ratio_stockwise.copy() # 회전율은 하루 전으로 날짜가 잡혀있고, 여기 계산된 수익률은 하루 뒤로 밀려있음(수익률이기 때문에) / 날짜를 맞춰주기 위한 조작
        tr_applied_here.index = self._rb_dts_nd_dly

        # apply trading cost0
        rb_d = daily_comp_rtn.index[0]
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost * tr_applied_here.loc[rb_d]
        output = daily_comp_rtn.sum(1)
        return output
    def calc_ret_cntrbtn(self, grouped_price):
        # grouped_price = gr_rtn.get_group([x for x in gr_p.groups.keys()][0])
        gr_comp_rtn = grouped_price.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1)
        daily_comp_rtn = gr_comp_rtn.mul(self._rb_dt_nd_dly.loc[gr_comp_rtn.index[0]]).dropna(axis=1,how='all')#.add(1).pct_change().sum(1).fillna(0)
        daily_comp_rtn.index = grouped_price.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0]
        daily_comp_rtn = daily_comp_rtn.add(1).pct_change()  # .sum(1)
        daily_comp_rtn.iloc[0] = first_line

        tr_applied_here = self._rb_tr_ratio_stockwise.copy()
        tr_applied_here.index = self._rb_dts_nd_dly

        # apply trading cost
        rb_d = daily_comp_rtn.index[0]
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost*tr_applied_here.loc[rb_d]

        return daily_comp_rtn
    def calc_bt_daily_ratio(self, grouped_return):
        # grouped_price = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][2])
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_rtn_ = grouped_return.set_index('gr_idx').dropna(how='all', axis=1).add(1).cumprod()
        output = gr_rtn_.mul(self._rb_dt_nd_dly.loc[gr_rtn_.index[0]])#.add(1).pct_change().sum(1).fillna(0)
        output = output.div(output.sum(1), axis=0)
        output.index = grouped_return.index
        return output
    def calc_rb_turnover(self, daily_account_ratio):
        # daily_account_ratio = self._daily_ratio.copy()
        past_account_ratio = daily_account_ratio.loc[self._rb_dts_1day_ago[1:]]
        now_ratio_target = self._rb_dt_nd_dly.loc[self._rb_dts_nd_dly]
        now_ratio_target.index = self._rb_dts_1day_ago
        rb_ratio_diff = now_ratio_target.sub(past_account_ratio, fill_value=0)
        return abs(rb_ratio_diff).sum(1)
    def calc_rb_turnover_stockwise(self, daily_account_ratio):
        past_account_ratio = daily_account_ratio.loc[self._rb_dts_1day_ago[1:]]
        now_ratio_target = self._rb_dt_nd_dly.loc[self._rb_dts_nd_dly]
        now_ratio_target.index = self._rb_dts_1day_ago
        rb_ratio_diff = now_ratio_target.sub(past_account_ratio, fill_value=0)
        return abs(rb_ratio_diff)
class return_calculator_Faster:
    def __init__(self, ratio_df, cost=0.00, n_day_after=0):
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
        price_df = gdu.data.copy()
        ratio_df = ratio_df.apply(lambda x: x.replace(0, np.nan))
        self._cost = cost

        # 지정된 리밸런싱날짜
        self._rb_dt = ratio_df.copy()
        self._rb_dts = self._rb_dt.index

        # 지정된 리밸런싱 날짜 +n 영업일(실제리밸런싱날짜의 다음날 - 수익률기반으로 계산하기 위하여)
        self._rb_dt_nd_dly = ratio_df.copy()
        self._rb_dts_nd_dly = self.get_nday_delay(self._rb_dts, n_day_after+1)
        self._rb_dt_nd_dly.index = self._rb_dts_nd_dly

        # 실제 리밸런싱 날짜(실제 리밸런싱 날짜의 Turnover Ratio를 계산하기 위하여)
        self._rb_dts_1day_ago = self.get_nday_ago(self._rb_dts_nd_dly, -1)

        # 가격데이터
        # self._p_dt = price_df.loc[:,ratio_df.columns]

        # 일별수익률
        self._rnt_dt = price_df.pct_change().mask(price_df.isnull(), np.nan).loc[:,ratio_df.columns]
        self._rnt_dt.iloc[0]=0


        # gr_p = self.get_df_grouped(self._p_dt, self._rb_dts_nd_dly) # 가격기반 수익률계산에서, 일별수익률 기반 계산으로 변경함
        gr_rtn = self.get_df_grouped(self._rnt_dt, self._rb_dts_nd_dly)

        # 회전율 관련
        self._daily_ratio = gr_rtn.apply(self.calc_bt_daily_ratio) # 실제 일별 내 계좌 잔고의 종목별 비중[%]
        self._rb_tr_ratio = self.calc_rb_turnover(self._daily_ratio) # 실제 리밸런싱 날짜의 회전율
        self._rb_tr_ratio_stockwise = self.calc_rb_turnover_stockwise(self._daily_ratio) # 실제 리밸런싱 날짜의 종목별 회전율

        # back-test daily return
        self.backtest_daily_return = gr_rtn.apply(self.calc_bt_compound_return).droplevel(0)
        # back-test daily cumulative return
        self.backtest_cumulative_return = self.backtest_daily_return.add(1).cumprod()

        # 수익률 기여도
        self.daily_ret_cntrbtn = gr_rtn.apply(self.calc_ret_cntrbtn)
        # 확인: self.daily_ret_cntrbtn.sum(1).add(1).cumprod()

    def get_nday_delay(self, rb_dts, n=0):
        _rb_dts_1d_dly = []
        for idx in rb_dts:
            try:
                _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[n])
            except:
                print(f'가격데이터 불충분 > {n-1}일 후 리밸런싱이 기입 -> BUT \n{gdu.data.loc[idx:]}')
                print(f'따라서 마지막날짜: {gdu.data.loc[idx:].index[-1]}로 대체하였습니다.')
                _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_nday_ago(self, rb_dts, n=-1):
        # rb_dts, n = self._rb_dts, n_day_after + 1
        # rb_dts, n = self._rb_dts_nd_dly, -1
        _rb_dts_1d_dly = []
        for idx in rb_dts:
            _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[n-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_df_grouped(self, df, dts):
        # df, dts = self._rnt_dt.copy(), self._rb_dts_nd_dly.copy()
        df.loc[dts,'gr_idx'] = dts
        df['gr_idx'] = df['gr_idx'].fillna(method='ffill')
        return df.groupby('gr_idx')
    def calc_compound_return(self, grouped_price):
        return grouped_price.drop('gr_idx', axis=1).pct_change().fillna(0).add(1).cumprod().sub(1)
    def calc_bt_compound_return(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][1])
        gr_comp_rtn = grouped_return.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1) # input된 것은 일별수익률임 따라서 복리수익률을 만들어 준 이후,
        daily_comp_rtn = gr_comp_rtn.mul(self._rb_dt_nd_dly.loc[gr_comp_rtn.index[0]]).dropna(axis=1, how='all')  # "복리수익률" * "리밸런싱때 조정한 비중" = 포트폴리오 종목별 일별 (누적)복리수익률
        daily_comp_rtn.index = grouped_return.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0] # pct_change를 할 때 첫 줄이 날아가기 때문에 남겨놓아야 하는 첫 번째날 수익률(리밸런싱 하루 이후 포트폴리오 종목별 수익률 backup)
        daily_comp_rtn = daily_comp_rtn.add(1).pct_change() # 포트폴리오 종목별 일별 수익률
        daily_comp_rtn.iloc[0] = first_line


        tr_applied_here = self._rb_tr_ratio_stockwise.copy() # 회전율은 하루 전으로 날짜가 잡혀있고, 여기 계산된 수익률은 하루 뒤로 밀려있음(수익률이기 때문에) / 날짜를 맞춰주기 위한 조작
        tr_applied_here.index = self._rb_dts_nd_dly

        # apply trading cost0
        rb_d = daily_comp_rtn.index[0]
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost * tr_applied_here.loc[rb_d]
        output = daily_comp_rtn.sum(1)
        return output
    def calc_ret_cntrbtn(self, grouped_price):
        # grouped_price = gr_rtn.get_group([x for x in gr_p.groups.keys()][0])
        gr_comp_rtn = grouped_price.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1)
        daily_comp_rtn = gr_comp_rtn.mul(self._rb_dt_nd_dly.loc[gr_comp_rtn.index[0]]).dropna(axis=1,how='all')#.add(1).pct_change().sum(1).fillna(0)
        daily_comp_rtn.index = grouped_price.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0]
        daily_comp_rtn = daily_comp_rtn.add(1).pct_change()  # .sum(1)
        daily_comp_rtn.iloc[0] = first_line

        tr_applied_here = self._rb_tr_ratio_stockwise.copy()
        tr_applied_here.index = self._rb_dts_nd_dly

        # apply trading cost
        rb_d = daily_comp_rtn.index[0]
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost*tr_applied_here.loc[rb_d]

        return daily_comp_rtn
    def calc_bt_daily_ratio(self, grouped_return):
        # grouped_price = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][2])
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_rtn_ = grouped_return.set_index('gr_idx').dropna(how='all', axis=1).add(1).cumprod()
        output = gr_rtn_.mul(self._rb_dt_nd_dly.loc[gr_rtn_.index[0]])#.add(1).pct_change().sum(1).fillna(0)
        output = output.div(output.sum(1), axis=0)
        output.index = grouped_return.index
        return output
    def calc_rb_turnover(self, daily_account_ratio):
        # daily_account_ratio = self._daily_ratio.copy()
        past_account_ratio = daily_account_ratio.loc[self._rb_dts_1day_ago[1:]]
        now_ratio_target = self._rb_dt_nd_dly.loc[self._rb_dts_nd_dly]
        now_ratio_target.index = self._rb_dts_1day_ago
        rb_ratio_diff = now_ratio_target.sub(past_account_ratio, fill_value=0)
        return abs(rb_ratio_diff).sum(1)
    def calc_rb_turnover_stockwise(self, daily_account_ratio):
        past_account_ratio = daily_account_ratio.loc[self._rb_dts_1day_ago[1:]]
        now_ratio_target = self._rb_dt_nd_dly.loc[self._rb_dts_nd_dly]
        now_ratio_target.index = self._rb_dts_1day_ago
        rb_ratio_diff = now_ratio_target.sub(past_account_ratio, fill_value=0)
        return abs(rb_ratio_diff)


# 2024-01-22 디버그 완료
class return_calculator:
    def __init__(self, ratio_df, cost, n_day_after=0):
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
        try:gdu.data.loc[ratio_df.index]
        except: raise ValueError('비영업일 리밸런싱 오류')

        price_df = gdu.data.copy()
        self.ratio_df = ratio_df.apply(lambda x: x.replace(0, np.nan))
        self._cost = cost

        # buy & sell dates
        self.b_dts = self.get_nday_delay(self.ratio_df.index, n_day_after+1) # 수익률로 계산하기 때문에 더하기 1 필요
        self.s_dts = self.get_nday_delay(self.ratio_df.index, n_day_after)[1:]
        self.ratio_df_buydate = self.ratio_df.copy()
        self.ratio_df_buydate.index = self.b_dts
        self.ratio_df_selldate = self.ratio_df.iloc[1:]

        # 일별수익률
        _rnt_dt = price_df.pct_change().loc[:,self.ratio_df.columns]

        # 리밸런싱 날짜별로 잘 그루핑을 해놓자
        gr_rtn = self.get_df_grouped(_rnt_dt)
        # self.grouped_return = gr_rtn.apply(lambda x:x).drop('gr_idx', axis=1)

        # 회전율 관련
        self.daily_account_ratio = gr_rtn.apply(self.calc_bt_daily_ratio)#.droplevel(0) # 실제 일별 내 계좌 잔고의 종목별 비중[%]
        self.daily_account_ratio.loc[self.ratio_df.index[0]] = self.ratio_df.iloc[0] # 첫날 비중은 그대로
        self.daily_account_ratio= self.daily_account_ratio.sort_index()
        self.stockwise_turnover_ratio = self.calc_rb_turnover(self.daily_account_ratio) # 실제 리밸런싱 날짜의 종목별 회전율
        self.portfolio_turnover_ratio = abs(self.stockwise_turnover_ratio).sum(1).div(2) # 실제 리밸런싱 날짜의 회전율

        self.portfolio_daily_return = gr_rtn.apply(self.calc_ret_cntrbtn).droplevel(0).sort_index() # 
        self.portfolio_cumulative_return = self.portfolio_daily_return.add(1).cumprod()


    def get_nday_delay(self, rb_dts, n=0):
        _rb_dts_1d_dly = []
        if n>=0:
            for idx in rb_dts:
                try:
                    _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[n])
                except:
                    print('리밸런싱 불가능한 날 존재')
                    pass
                    # print(f'가격데이터 불충분 > {n-1}일 후 리밸런싱이 기입 -> BUT \n{gdu.data.loc[idx:]}')
                    # print(f'따라서 마지막날짜: {gdu.data.loc[idx:].index[-1]}로 대체하였습니다.')
                    # _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[-1])
        else:
            for idx in rb_dts:
                _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[n-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_nday_ago(self, rb_dts, n=-1):
        # rb_dts, n = self._rb_dts, n_day_after + 1
        # rb_dts, n = self._rb_dts_nd_dly, -1
        _rb_dts_1d_dly = []
        for idx in rb_dts:
            _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[n-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_df_grouped(self, df):
        # df = _rnt_dt.copy()
        df.loc[self.b_dts,'gr_idx'] = range(len(self.b_dts))
        df.loc[self.s_dts,'gr_idx'] = range(len(self.s_dts))
        for i in range(int(df['gr_idx'].max())):
            idx = df[df['gr_idx']==i].index
            idx_s, idx_e = idx[0],idx[-1]
            df.loc[idx_s:idx_e, 'gr_idx'] = self.b_dts[i]
        max_i = max(len(self.b_dts), len(self.s_dts))-1
        max_idx = df[df['gr_idx'] == max_i].index[0]
        df.loc[max_idx:, 'gr_idx'] = self.b_dts[max_i]
        df=df.dropna(subset='gr_idx', axis=0)
        # df['gr_idx'] = df['gr_idx'].fillna(method='ffill')
        return df.groupby('gr_idx')
    def calc_ret_cntrbtn(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_comp_rtn = grouped_return.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1) # input된 것은 일별수익률임 따라서 복리수익률을 만들어 준 이후,
        daily_comp_rtn = gr_comp_rtn.mul(self.ratio_df_buydate.loc[gr_comp_rtn.index[0]]).dropna(axis=1, how='all')  # "복리수익률" * "리밸런싱때 조정한 비중" = 포트폴리오 종목별 일별 (누적)복리수익률
        daily_comp_rtn.index = grouped_return.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0] # pct_change를 할 때 첫 줄이 날아가기 때문에 남겨놓아야 하는 첫 번째날 수익률(리밸런싱 하루 이후 포트폴리오 종목별 수익률 backup)
        daily_rtn = daily_comp_rtn.add(1).pct_change() # 포트폴리오 종목별 일별 수익률
        daily_rtn.iloc[0] = first_line

        rb_d = daily_rtn.index[0]
        tr_applied_here = self.stockwise_turnover_ratio.loc[rb_d].dropna() # 회전율은 하루 전으로 날짜가 잡혀있고, 여기 계산된 수익률은 하루 뒤로 밀려있음(수익률이기 때문에) / 날짜를 맞춰주기 위한 조작

        # apply trading cost0
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost * tr_applied_here

        daily_port_rtn = daily_comp_rtn.sum(1).add(1).pct_change()
        daily_port_rtn.loc[rb_d] = daily_comp_rtn.sum(1).loc[rb_d]
        return daily_port_rtn
    def calc_bt_daily_ratio(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_rtn_ = grouped_return.set_index('gr_idx').dropna(how='all', axis=1).add(1).cumprod()
        output = gr_rtn_.mul(self.ratio_df_buydate.loc[gr_rtn_.index[0]])#.add(1).pct_change().sum(1).fillna(0)
        output = output.div(output.sum(1), axis=0)
        output.index = grouped_return.index
        return output
    def calc_rb_turnover(self, daily_account_ratio):
        # daily_account_ratio = _daily_ratio.copy()
        s_dt_account_ratio = daily_account_ratio.loc[self.s_dts]
        b_dt_ratio_target = self.ratio_df_buydate.loc[self.b_dts]
        s_dt_account_ratio.loc[b_dt_ratio_target.index[0]]=0
        s_dt_account_ratio = s_dt_account_ratio.sort_index()
        s_dt_account_ratio.index = b_dt_ratio_target.index

        rb_ratio_diff = b_dt_ratio_target.sub(s_dt_account_ratio, fill_value=0)
        return rb_ratio_diff
class retcnt_calculator:
    def __init__(self, ratio_df, cost=0.00, n_day_after=0):
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
        try:gdu.data.loc[ratio_df.index]
        except: raise ValueError('비영업일 리밸런싱 오류')

        price_df = gdu.data.copy()
        self.ratio_df = ratio_df.apply(lambda x: x.replace(0, np.nan))
        self._cost = cost

        # buy & sell dates
        self.b_dts = self.get_nday_delay(self.ratio_df.index, n_day_after+1) # 수익률로 계산하기 때문에 더하기 1 필요
        self.s_dts = self.get_nday_delay(self.ratio_df.index, n_day_after)[1:]
        self.ratio_df_buydate = self.ratio_df.copy()
        self.ratio_df_buydate.index = self.b_dts
        self.ratio_df_selldate = self.ratio_df.iloc[1:]

        # 일별수익률
        _rnt_dt = price_df.pct_change().loc[:,ratio_df.columns].fillna(0)

        # 리밸런싱 날짜별로 잘 그루핑을 해놓자
        gr_rtn = self.get_df_grouped(_rnt_dt)
        self.grouped_return = gr_rtn.apply(lambda x:x).drop('gr_idx', axis=1)

        # 회전율 관련
        self.daily_account_ratio = gr_rtn.apply(self.calc_bt_daily_ratio)#.droplevel(0) # 실제 일별 내 계좌 잔고의 종목별 비중[%]
        self.daily_account_ratio.loc[self.ratio_df.index[0]] = self.ratio_df.iloc[0] # 첫날 비중은 그대로
        self.daily_account_ratio= self.daily_account_ratio.sort_index()
        self.rebal_account_ratio, self.before_account_ratio,self.stockwise_turnover_ratio = self.calc_rb_turnover(self.daily_account_ratio) # 실제 리밸런싱 날짜의 종목별 회전율
        self.portfolio_turnover_ratio = abs(self.stockwise_turnover_ratio).sum(1).div(2) # 실제 리밸런싱 날짜의 회전율

        # 수익률 기여도 & 포트폴리오 수익률
        daily_cntrbtn_port_concated = [self.calc_ret_cntrbtn_and_port_ret(group) for _, group in gr_rtn]
        # 종목별 일자별 수익률기여도
        self.stockwise_daily_return_contribution = pd.concat([daily_cntrbtn_tmp[0] for daily_cntrbtn_tmp in daily_cntrbtn_port_concated], axis=0).sort_index()
        # 리밸런싱별 종목별 수익률기여도
        self.stockwise_period_return_contribution = self.stockwise_daily_return_contribution.loc[self.s_dts]

        self.portfolio_daily_return = pd.concat([daily_ret_port[1] for daily_ret_port in daily_cntrbtn_port_concated],axis=0).sort_index()
        if ratio_df.index[0] not in self.portfolio_daily_return.index:
            self.portfolio_daily_return.loc[ratio_df.index[0]]=0
            self.stockwise_period_return_contribution.loc[ratio_df.index[0]]=0
            self.portfolio_daily_return = self.portfolio_daily_return.sort_index()
            self.stockwise_period_return_contribution = self.stockwise_period_return_contribution.sort_index()
        self.portfolio_cumulative_return = self.portfolio_daily_return.add(1).cumprod()


    def get_nday_delay(self, rb_dts, n=0):
        # rb_dts, n = P_w_pvt_input.index.copy(), n_day_after+1
        _rb_dts_1d_dly = []
        if n>=0:
            for idx in rb_dts:
                try:
                    _rb_dts_1d_dly.append(gdu.data.loc[idx:].index[n])
                except:
                    # raise ValueError('(가격데이터 불충분) 리밸런싱 불가능한 날 존재')
                    print('(가격데이터 불충분) 리밸런싱 불가능한 날 존재')
                    _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[-1])
        else:
            for idx in rb_dts:
                _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[n-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_nday_ago(self, rb_dts, n=-1):
        # rb_dts, n = self._rb_dts, n_day_after + 1
        # rb_dts, n = self._rb_dts_nd_dly, -1
        _rb_dts_1d_dly = []
        for idx in rb_dts:
            _rb_dts_1d_dly.append(gdu.data.loc[:idx].index[n-1])
        return pd.DatetimeIndex(_rb_dts_1d_dly)
    def get_df_grouped(self, df):
        # df = _rnt_dt.copy()
        df.loc[self.b_dts,'gr_idx'] = range(len(self.b_dts))
        df.loc[self.s_dts,'gr_idx'] = range(len(self.s_dts))
        max_i = int(df['gr_idx'].max())
        for i in range(max_i):
            idx = df[df['gr_idx']==i].index
            idx_s, idx_e = idx[0],idx[-1]
            df.loc[idx_s:idx_e, 'gr_idx'] = self.b_dts[i]

        max_idx = df[df['gr_idx'] == max_i].index[0]

        df.loc[max_idx:, 'gr_idx'] = self.b_dts[max_i]
        df=df.dropna(subset='gr_idx', axis=0)
        # df['gr_idx'] = df['gr_idx'].fillna(method='ffill')
        return df.groupby('gr_idx')
    def calc_ret_cntrbtn_and_port_ret(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_comp_rtn = grouped_return.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1) # input된 것은 일별수익률임 따라서 복리수익률을 만들어 준 이후,
        daily_comp_rtn = gr_comp_rtn.mul(self.ratio_df_buydate.loc[gr_comp_rtn.index[0]]).dropna(axis=1, how='all')  # "복리수익률" * "리밸런싱때 조정한 비중" = 포트폴리오 종목별 일별 (누적)복리수익률
        daily_comp_rtn.index = grouped_return.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0] # pct_change를 할 때 첫 줄이 날아가기 때문에 남겨놓아야 하는 첫 번째날 수익률(리밸런싱 하루 이후 포트폴리오 종목별 수익률 backup)
        daily_rtn = daily_comp_rtn.add(1).pct_change() # 포트폴리오 종목별 일별 수익률
        daily_rtn.iloc[0] = first_line

        rb_d = daily_rtn.index[0]
        tr_applied_here = self.stockwise_turnover_ratio.loc[rb_d].dropna() # 회전율은 하루 전으로 날짜가 잡혀있고, 여기 계산된 수익률은 하루 뒤로 밀려있음(수익률이기 때문에) / 날짜를 맞춰주기 위한 조작

        # apply trading cost0
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost * tr_applied_here

        daily_port_rtn = daily_comp_rtn.sum(1).add(1).pct_change()
        daily_port_rtn.loc[rb_d] = daily_comp_rtn.sum(1).loc[rb_d]
        # daily_port_rtn.add(1).cumprod()

        return [daily_comp_rtn, daily_port_rtn]
    def calc_bt_daily_ratio(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_rtn_ = grouped_return.set_index('gr_idx').dropna(how='all', axis=1).add(1).cumprod()
        output = gr_rtn_.mul(self.ratio_df_buydate.loc[gr_rtn_.index[0]])#.add(1).pct_change().sum(1).fillna(0)
        output = output.div(output.sum(1), axis=0)
        output.index = grouped_return.index
        return output
    def calc_rb_turnover(self, daily_account_ratio):
        # daily_account_ratio = _daily_ratio.copy()
        # daily_account_ratio = self.daily_account_ratio.copy()
        s_dt_account_ratio = daily_account_ratio.loc[self.s_dts]
        b_dt_ratio_target = self.ratio_df_buydate.loc[self.b_dts]
        s_dt_account_ratio.loc[b_dt_ratio_target.index[0]]=0
        s_dt_account_ratio = s_dt_account_ratio.sort_index()
        s_dt_account_ratio.index = b_dt_ratio_target.index

        before_account_ratio=s_dt_account_ratio.copy()
        rebal_account_ratio=b_dt_ratio_target.copy()
        rb_ratio_diff = b_dt_ratio_target.sub(s_dt_account_ratio, fill_value=0)
        return rebal_account_ratio, before_account_ratio, rb_ratio_diff
class BrinsonFachler_calculator(retcnt_calculator):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input, cost=0.00, n_day_after=0):
        """
        Asset_info: pandas Series
        index = 종목코드
        value = class
        """
        all_list = list(set(P_w_pvt_input.columns)|set(B_w_pvt_input.columns))



        # 시계열로 변하는 class를 고려하지 않음
        # 혹시라도 섹터가 구분되지 않는 종목이 섞여들어왔을 때
        self.Asset_info = Asset_info_input.copy()
        ambigous_list = list(set(all_list)-set(self.Asset_info.index))
        if len(ambigous_list)>0:
            print(f'class가 구분되지 않는 종목 {len(ambigous_list)}개')
            for cd in ambigous_list:
                self.Asset_info[cd] = 'unknown'

        price_df = gdu.data[all_list].copy()

        self.b_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after+1) # 수익률로 계산하기 때문에 더하기 1 필요
        self.s_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after)[1:]
        self.ratio_df_buydate = P_w_pvt_input.copy()
        self.ratio_df_buydate.index = self.b_dts
        self.ratio_df_selldate = P_w_pvt_input.iloc[1:]

        _rnt_dt = price_df.pct_change()
        gr_rtn = self.get_df_grouped(_rnt_dt)
        self.period_ExPost = gr_rtn.apply(lambda x: x.add(1).prod()-1)
        self.period_ExPost.index = P_w_pvt_input.index

        # 설정 확인
        P_w_pvt = P_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        B_w_pvt = B_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        # Portfolio는 BM 가만히 있을 때, 수시리밸런싱을 했을 수도 있으니까
        irre_rebal_date = P_w_pvt.index[~P_w_pvt.index.isin(B_w_pvt.index)]
        if len(irre_rebal_date) != 0:
            print(f'수시리밸런싱이 {len(irre_rebal_date)}회 있습니다: {list(irre_rebal_date)}')
            irre_rebal_B_w_tmp = retcnt_calculator(ratio_df=B_w_pvt).daily_account_ratio
            B_w_pvt = pd.concat([B_w_pvt, irre_rebal_B_w_tmp.loc[irre_rebal_date]], axis=0).sort_index()

        rP = P_w_pvt.mul(self.period_ExPost).sum(1)
        rB = B_w_pvt.mul(self.period_ExPost).sum(1)
        ######################## class별 변환
        P_cw_pvt, P_cr_pvt = self.convert_pvt_wrt_class(P_w_pvt)
        self.P_classweight_pvt = P_cw_pvt.copy()
        B_cw_pvt, B_cr_pvt = self.convert_pvt_wrt_class(B_w_pvt)

        # Allocation Effect
        wPj_minus_wBj = P_cw_pvt.sub(B_cw_pvt, fill_value=0)#.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        rBj_minus_rB = B_cr_pvt.sub(rB, axis=0)
        allocation_effect_tmp = wPj_minus_wBj.mul(rBj_minus_rB, fill_value=0).sum(1)

        # Selection Effect
        rPj_minus_rBj = P_cr_pvt.sub(B_cr_pvt, fill_value=0)
        selection_effect_tmp = rPj_minus_rBj.mul(B_cw_pvt,fill_value=0).sum(1)

        # Inter-action Effect
        interaction_effect_tmp = wPj_minus_wBj.mul(rPj_minus_rBj,fill_value=0).sum(1)

        self.allocation_effect = allocation_effect_tmp.shift(1)
        self.selection_effect = selection_effect_tmp.shift(1)
        self.interaction_effect = interaction_effect_tmp.shift(1)
        self.rB = rB.shift(1)
        self.rP = rP.shift(1)

        self.Bench_cls = retcnt_calculator(ratio_df=B_w_pvt, cost=cost, n_day_after=n_day_after)
        self.Port_cls = retcnt_calculator(ratio_df=P_w_pvt, cost=cost, n_day_after=n_day_after)

        # 확인
        # print('rP.sub(rB)',rP.sub(rB))
        # print('SUMMATION',allocation_effect_tmp.add(selection_effect_tmp).add(interaction_effect_tmp))
        # self.rP.sub(self.rB).add(1).cumprod()
        # self.allocation_effect.add(self.selection_effect).add(self.interaction_effect).add(1).cumprod()
        # return_calculator(B_w_pvt,cost=0, n_day_after=n_day_after).portfolio_cumulative_return.loc[:self.rB.index[-1]]
        # self.rB.add(1).cumprod()


        """
        # BHB(1986)와 DGTW(1997)를 기반
        TR = P_w_pvt.mul(period_ExPost).sum(1)
        SAA = B_w_pvt.mul(period_ExPost).sum(1)
        TAA = P_cw_pvt.sub(B_cw_pvt, fill_value=0).mul(B_cr_pvt).sum(1)
        AS = P_cw_pvt.mul(P_cr_pvt.sub(B_cr_pvt)).sum(1)
        TR
        SAA.add(TAA).add(AS)
        """
    def get_normed_pvt(self,_w_pvt, Ass_inf):
        # _w_pvt, Ass_inf = B_w_pvt.copy(), self.Asset_info.copy()
        # Asset_info_input DataFrame을 사용하여 P_w_pvt의 각 컬럼에 대한 클래스를 찾음
        class_for_columns = Ass_inf.loc[_w_pvt.columns].values.flatten()
        # 같은 클래스에 속하는 컬럼들의 합을 계산
        # cls=class_for_columns[0]
        class_sums = {cls: _w_pvt.loc[:, class_for_columns == cls].sum(axis=1) for cls in set(class_for_columns)}
        # P_w_pvt의 각 컬럼을 해당 클래스의 합으로 나눔
        P_w_pvt_normalized = _w_pvt.apply(lambda col: col / class_sums[class_for_columns[_w_pvt.columns.get_loc(col.name)]])
        return P_w_pvt_normalized
    def convert_pvt_wrt_class(self, w_pvt_):
        # w_pvt_ = P_w_pvt.copy()
        nw_pvt_ = self.get_normed_pvt(w_pvt_, self.Asset_info)
        cr_pvt_ = self.period_ExPost[w_pvt_.notna()].fillna(0).mul(nw_pvt_, fill_value=1).rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        cw_pvt_ = w_pvt_.rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        return cw_pvt_,cr_pvt_
class Modified_BrinsonFachler_calculator(retcnt_calculator):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input,Index_Daily_price_input, cost=0.00, n_day_after=0):
        """
        Asset_info: pandas Series
        index = 종목코드
        value = class
        """
        all_list = list(set(P_w_pvt_input.columns)|set(B_w_pvt_input.columns))

        # 시계열로 변하는 class를 고려하지 않음
        # 혹시라도 섹터가 구분되지 않는 종목이 섞여들어왔을 때
        self.Asset_info = Asset_info_input.copy()
        self.Index_Daily_price_input = Index_Daily_price_input.copy()
        ambigous_list = list(set(all_list)-set(self.Asset_info.index))
        if len(ambigous_list)>0:
            print(f'class가 구분되지 않는 종목 {len(ambigous_list)}개')
            for cd in ambigous_list:
                self.Asset_info[cd] = 'unknown'

        price_df = gdu.data[all_list].copy()

        self.b_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after+1) # 수익률로 계산하기 때문에 더하기 1 필요
        self.s_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after)[1:]
        self.ratio_df_buydate = P_w_pvt_input.copy()
        self.ratio_df_buydate.index = self.b_dts
        self.ratio_df_selldate = P_w_pvt_input.iloc[1:]

        _rnt_dt = price_df.pct_change()
        gr_rtn = self.get_df_grouped(_rnt_dt)
        self.period_ExPost = gr_rtn.apply(lambda x: x.add(1).prod()-1)
        self.period_ExPost.index = P_w_pvt_input.index

        # 설정 확인
        P_w_pvt = P_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        B_w_pvt = B_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        # Portfolio는 BM 가만히 있을 때, 수시리밸런싱을 했을 수도 있으니까
        irre_rebal_date = P_w_pvt.index[~P_w_pvt.index.isin(B_w_pvt.index)]
        if len(irre_rebal_date) != 0:
            print(f'수시리밸런싱이 {len(irre_rebal_date)}회 있습니다: {list(irre_rebal_date)}')
            irre_rebal_B_w_tmp = retcnt_calculator(ratio_df=B_w_pvt).daily_account_ratio
            B_w_pvt = pd.concat([B_w_pvt, irre_rebal_B_w_tmp.loc[irre_rebal_date]], axis=0).sort_index()
        self.Bench_cls = retcnt_calculator(ratio_df=B_w_pvt, cost=cost, n_day_after=n_day_after)
        self.Port_cls = retcnt_calculator(ratio_df=P_w_pvt, cost=cost, n_day_after=n_day_after)

        self.allocation_effect, self.selection_effect, self.interaction_effect, self.rP, self.rB, self.P_classweight_pvt = self.get_AA_effects(P_w_pvt, B_w_pvt)

        LATEST_Rebal = P_w_pvt.copy()
        IF_None_Rebal=self.Port_cls.before_account_ratio.copy()
        IF_None_Rebal.index=LATEST_Rebal.index
        self.Rebalancing_in_eff, self.Rebalancing_out_eff, self.rP_NonReb, self.rB_NonReb=self.get_Rebalancing_effects(LATEST_Rebal,IF_None_Rebal)

        # 확인
        # print('rP.sub(rB)',rP.sub(rB))
        # print('SUMMATION',allocation_effect_tmp.add(selection_effect_tmp).add(interaction_effect_tmp))
        # self.rP.sub(self.rB).add(1).cumprod()
        # self.allocation_effect.add(self.selection_effect).add(self.interaction_effect).add(1).cumprod()
        # return_calculator(B_w_pvt,cost=0, n_day_after=n_day_after).portfolio_cumulative_return.loc[:self.rB.index[-1]]
        # self.rB.add(1).cumprod()


        """
        # BHB(1986)와 DGTW(1997)를 기반
        TR = P_w_pvt.mul(period_ExPost).sum(1)
        SAA = B_w_pvt.mul(period_ExPost).sum(1)
        TAA = P_cw_pvt.sub(B_cw_pvt, fill_value=0).mul(B_cr_pvt).sum(1)
        AS = P_cw_pvt.mul(P_cr_pvt.sub(B_cr_pvt)).sum(1)
        TR
        SAA.add(TAA).add(AS)
        """
    def get_normed_pvt(self,_w_pvt, Ass_inf):
        # _w_pvt, Ass_inf = B_w_pvt.copy(), self.Asset_info.copy()
        # Asset_info_input DataFrame을 사용하여 P_w_pvt의 각 컬럼에 대한 클래스를 찾음
        class_for_columns = Ass_inf.loc[_w_pvt.columns].values.flatten()
        # 같은 클래스에 속하는 컬럼들의 합을 계산
        # cls=class_for_columns[0]
        class_sums = {cls: _w_pvt.loc[:, class_for_columns == cls].sum(axis=1) for cls in set(class_for_columns)}
        # P_w_pvt의 각 컬럼을 해당 클래스의 합으로 나눔
        P_w_pvt_normalized = _w_pvt.apply(lambda col: col / class_sums[class_for_columns[_w_pvt.columns.get_loc(col.name)]])
        return P_w_pvt_normalized
    def convert_pvt_wrt_class(self, w_pvt_):
        # w_pvt_ = P_w_pvt.copy()
        nw_pvt_ = self.get_normed_pvt(w_pvt_, self.Asset_info)
        cr_pvt_ = self.period_ExPost[w_pvt_.notna()].fillna(0).mul(nw_pvt_, fill_value=1).rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        cw_pvt_ = w_pvt_.rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        return cw_pvt_,cr_pvt_
    def get_AA_effects(self, P_w_pvt_, B_w_pvt_):
        # P_w_pvt_, B_w_pvt_ = LATEST_Rebal.copy(), IF_None_Rebal.copy()
        rP_ = P_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        rB_ = B_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        ######################## class별 변환
        P_cw_pvt_, P_cr_pvt_ = self.convert_pvt_wrt_class(P_w_pvt_)
        P_classweight_pvt = P_cw_pvt_.copy()
        B_cw_pvt_, B_cr_pvt_ = self.convert_pvt_wrt_class(B_w_pvt_)
        I_cr_pvt_ = self.Index_Daily_price_input.loc[P_cr_pvt_.index, P_cr_pvt_.columns].pct_change().shift(-1)

        # Allocation Effect
        wPj_minus_wBj_ = P_cw_pvt_.sub(B_cw_pvt_,
                                       fill_value=0)  # .rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        rBj_minus_rB_ = B_cr_pvt_.sub(rB_, axis=0)
        allocation_effect_tmp_ = wPj_minus_wBj_.mul(rBj_minus_rB_, fill_value=0).sum(1)

        # Selection Effect
        rPj_minus_rIj_ = P_cr_pvt_.sub(I_cr_pvt_, fill_value=0)
        selection_effect_tmp_ = rPj_minus_rIj_.mul(B_cw_pvt_, fill_value=0).sum(1)

        # Inter-action Effect
        rPj_minus_rBj_ = P_cr_pvt_.sub(B_cr_pvt_, fill_value=0)
        interaction_effect_tmp_ = P_cw_pvt_.mul(rPj_minus_rBj_, fill_value=0).sub(
            B_cw_pvt_.mul(rPj_minus_rIj_, fill_value=0), fill_value=0).sum(1)

        allocation_effect = allocation_effect_tmp_.shift(1)
        selection_effect = selection_effect_tmp_.shift(1)
        interaction_effect = interaction_effect_tmp_.shift(1)
        rB = rB_.shift(1)
        rP = rP_.shift(1)
        return allocation_effect, selection_effect, interaction_effect, rP, rB, P_classweight_pvt
    def get_Rebalancing_effects(self, P_w_pvt_, B_w_pvt_):
        # P_w_pvt_, B_w_pvt_ = LATEST_Rebal.copy(), IF_None_Rebal.copy()
        rP_ = P_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        rB_ = B_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)

        Port_change = P_w_pvt_.sub(B_w_pvt_, fill_value=0)
        Port_in = Port_change[Port_change>0]
        Port_out = Port_change[Port_change<0]

        Port_in_eff = Port_in.mul(self.period_ExPost).sum(1)
        Port_out_eff = Port_out.mul(self.period_ExPost).sum(1)
        # Port_in_eff.add(Port_out_eff)
        # rP_.sub(rB_)
        return Port_in_eff, Port_out_eff, rP_, rB_
class BrinsonHoodBeebower_calculator(retcnt_calculator):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input,Index_Daily_price_input, cost=0.00, n_day_after=0):
        """
        Asset_info: pandas Series
        index = 종목코드
        value = class
        """
        all_list = list(set(P_w_pvt_input.columns))


        self.Asset_info = Asset_info_input.copy()
        # 시계열로 변하는 class를 고려하지 않음
        # 혹시라도 섹터가 구분되지 않는 종목이 섞여들어왔을 때
        self.Index_Daily_price_input = Index_Daily_price_input.copy()
        ambigous_list = list(set(all_list)-set(self.Asset_info.index))
        if len(ambigous_list)>0:
            print(f'class가 구분되지 않는 종목 {len(ambigous_list)}개')
            for cd in ambigous_list:
                self.Asset_info[cd] = 'unknown'


        price_df = gdu.data.copy()

        self.b_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after+1) # 수익률로 계산하기 때문에 더하기 1 필요
        self.s_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after)[1:]
        self.ratio_df_buydate = P_w_pvt_input.copy()
        self.ratio_df_buydate.index = self.b_dts
        self.ratio_df_selldate = P_w_pvt_input.iloc[1:]
        self.ratio_df_selldate.index = self.s_dts

        _rnt_dt = price_df.pct_change()
        gr_rtn = self.get_df_grouped(_rnt_dt)
        self.period_ExPost = gr_rtn.apply(lambda x: x.add(1).prod()-1)
        try:
            self.period_ExPost.index = P_w_pvt_input.index
        except:
            self.period_ExPost.index = P_w_pvt_input.index[:-1]

        # 설정 확인
        P_w_pvt = P_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        B_w_pvt = B_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        # Portfolio는 BM 가만히 있을 때, 수시리밸런싱을 했을 수도 있으니까
        irre_rebal_date = P_w_pvt.index[~P_w_pvt.index.isin(B_w_pvt.index)]
        if len(irre_rebal_date) != 0:
            print(f'수시리밸런싱이 {len(irre_rebal_date)}회 있습니다: {list(irre_rebal_date)}')
            irre_rebal_B_w_tmp = retcnt_calculator(ratio_df=B_w_pvt).daily_account_ratio
            B_w_pvt = pd.concat([B_w_pvt, irre_rebal_B_w_tmp.loc[irre_rebal_date]], axis=0).sort_index()
        self.Bench_cls = retcnt_calculator(ratio_df=B_w_pvt, cost=cost, n_day_after=n_day_after)
        self.Port_cls = retcnt_calculator(ratio_df=P_w_pvt, cost=cost, n_day_after=n_day_after)

        self.allocation_effect, self.selection_effect, self.rP, self.rB, self.P_classweight_pvt = self.get_AA_effects(P_w_pvt, B_w_pvt)

        LATEST_Rebal = P_w_pvt.copy()
        IF_None_Rebal=self.Port_cls.before_account_ratio.copy()
        IF_None_Rebal.index=LATEST_Rebal.index
        self.Rebalancing_in_eff, self.Rebalancing_out_eff, self.rP_NonReb, self.rB_NonReb=self.get_Rebalancing_effects(LATEST_Rebal,IF_None_Rebal)

        # 확인
        # print('rP.sub(rB)',rP.sub(rB))
        # print('SUMMATION',allocation_effect_tmp.add(selection_effect_tmp).add(interaction_effect_tmp))
        # self.rP.sub(self.rB).add(1).cumprod()
        # self.allocation_effect.add(self.selection_effect).add(self.interaction_effect).add(1).cumprod()
        # return_calculator(B_w_pvt,cost=0, n_day_after=n_day_after).portfolio_cumulative_return.loc[:self.rB.index[-1]]
        # self.rB.add(1).cumprod()


        """
        # BHB(1986)와 DGTW(1997)를 기반
        TR = P_w_pvt.mul(period_ExPost).sum(1)
        SAA = B_w_pvt.mul(period_ExPost).sum(1)
        TAA = P_cw_pvt.sub(B_cw_pvt, fill_value=0).mul(B_cr_pvt).sum(1)
        AS = P_cw_pvt.mul(P_cr_pvt.sub(B_cr_pvt)).sum(1)
        TR
        SAA.add(TAA).add(AS)
        """
    def get_normed_pvt(self,_w_pvt, Ass_inf):
        # _w_pvt, Ass_inf = B_w_pvt.copy(), self.Asset_info.copy()
        # Asset_info_input DataFrame을 사용하여 P_w_pvt의 각 컬럼에 대한 클래스를 찾음
        class_for_columns = Ass_inf.loc[_w_pvt.columns].values.flatten()
        # 같은 클래스에 속하는 컬럼들의 합을 계산
        # cls=class_for_columns[0]
        class_sums = {cls: _w_pvt.loc[:, class_for_columns == cls].sum(axis=1) for cls in set(class_for_columns)}
        # P_w_pvt의 각 컬럼을 해당 클래스의 합으로 나눔
        P_w_pvt_normalized = _w_pvt.apply(lambda col: col / class_sums[class_for_columns[_w_pvt.columns.get_loc(col.name)]])
        return P_w_pvt_normalized
    def convert_pvt_wrt_class(self, w_pvt_, convert=True):
        # w_pvt_ = B_w_pvt.copy()
        # w_pvt_ = P_w_pvt_.copy()
        if convert:
            nw_pvt_ = self.get_normed_pvt(w_pvt_, self.Asset_info)
            # cr_pvt_tag = self.period_ExPost[w_pvt_.notna()].fillna(0).mul(nw_pvt_, fill_value=1).stack().rename('value').reset_index('date').merge(pd.DataFrame([self.Asset_info.to_dict()]).stack().droplevel(0).rename('class').reset_index().set_index('index'), how='left', left_index=True, right_index=True).rename_axis('code').dropna(subset='class', axis=0).reset_index().pivot(index='date', columns=['class', 'code'], values='value')  # .stack()
        else:
            nw_pvt_ = w_pvt_.copy()
            # cr_pvt_tag = self.period_ExPost[w_pvt_.notna()].fillna(0).mul(nw_pvt_, fill_value=1).rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        cr_pvt = self.period_ExPost[w_pvt_.notna()].fillna(0).mul(nw_pvt_, fill_value=1).rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        cw_pvt = w_pvt_.rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        # cw_pvt = w_pvt_.stack().rename('value').reset_index('date').merge(pd.DataFrame([self.Asset_info.to_dict()]).stack().droplevel(0).rename('class').reset_index().set_index('index'), how='left', left_index=True, right_index=True).rename_axis('code').dropna(subset='class', axis=0).reset_index().pivot(index='date', columns=['class', 'code'], values='value')
        # return cw_pvt_tag,cr_pvt_tag
        return cw_pvt,cr_pvt
    def get_AA_effects(self, P_w_pvt_, B_w_pvt_):
        # P_w_pvt_, B_w_pvt_ = P_w_pvt.copy(), B_w_pvt.copy()
        rP_ = P_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        rB_ = B_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)


        ######################## class별 변환
        P_cw_pvt_, P_cr_pvt_ = self.convert_pvt_wrt_class(P_w_pvt_)
        B_cw_pvt_, B_cr_pvt_ = self.convert_pvt_wrt_class(B_w_pvt_, convert=False)

        # P_cw_pvt_tag, P_cr_pvt_tag = self.convert_pvt_wrt_class(P_w_pvt_)
        # B_cw_pvt_tag, B_cr_pvt_tag = self.convert_pvt_wrt_class(B_w_pvt_, convert=False)

        # P_cw_pvt_ = P_cw_pvt_tag.stack().stack().groupby(['date','class']).sum().unstack()
        # P_cr_pvt_ = P_cr_pvt_tag.stack().stack().groupby(['date','class']).sum().unstack()
        # B_cw_pvt_ = B_cw_pvt_tag.copy()#.stack().stack().groupby(['date','class']).sum().unstack()
        # B_cr_pvt_ = B_cr_pvt_tag.copy()#.stack().stack().groupby(['date','class']).sum().unstack()

        # P_cw_pvt_tag, P_cr_pvt_tag = cw_pvt_tag.copy(), cr_pvt_tag.copy()
        P_classweight_pvt = P_cw_pvt_.copy()
        I_cr_pvt_ = self.Index_Daily_price_input.loc[P_cr_pvt_.index.to_list()+[self.Index_Daily_price_input.index[-1]]].pct_change().shift(-1).dropna(how='all', axis=0)

        # Allocation Effect
        wPj_minus_wBj_ = P_cw_pvt_.sub(B_cw_pvt_, fill_value=0)  # .rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        # wPj_minus_wBj_=P_cw_pvt_tag.sub(P_cw_pvt_tag.stack().stack().div(P_cw_pvt_tag.stack().stack().groupby(['date','class']).sum()).unstack(1).unstack(1).mul(B_cw_pvt_, level=0), fill_value=0)
        # P_cw_pvt_tag.sub(P_cw_pvt_tag.stack().stack().div(P_cw_pvt_tag.stack().stack().groupby(['date', 'class']).sum()).unstack(1).unstack(1).mul(B_cw_pvt_, fill_value=0,level=0), fill_value=0).stack().stack().groupby(['date','class']).sum().unstack()

        print(f'Allocation Effect: \n{wPj_minus_wBj_.mul(I_cr_pvt_, fill_value=0).shift(1).iloc[-1].dropna()}')
        allocation_effect_tmp_ = wPj_minus_wBj_.mul(I_cr_pvt_, fill_value=0).sum(1)
        wPj_minus_wBj_.mul(I_cr_pvt_, level=0, fill_value=0).sum(1)


        # Selection Effect
        rPj_minus_rIj_ = P_cr_pvt_.sub(I_cr_pvt_, fill_value=0)
        print(f'Selection Effect: \n{rPj_minus_rIj_.mul(P_cw_pvt_, fill_value=0).shift(1).iloc[-1].dropna()}')
        selection_effect_tmp_ = rPj_minus_rIj_.mul(P_cw_pvt_, fill_value=0).sum(1)

        allocation_effect = allocation_effect_tmp_#.shift(1)
        selection_effect = selection_effect_tmp_#.shift(1)
        rB = rB_#.shift(1)
        rP = rP_#.shift(1)
        # 확인
        # allocation_effect.add(selection_effect)
        # rP.sub(rB)
        return allocation_effect, selection_effect, rP, rB, P_classweight_pvt
    def get_Rebalancing_effects(self, P_w_pvt_, B_w_pvt_):
        # P_w_pvt_, B_w_pvt_ = LATEST_Rebal.copy(), IF_None_Rebal.copy()
        rP_ = P_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        rB_ = B_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)

        Port_change = P_w_pvt_.sub(B_w_pvt_, fill_value=0)
        Port_in = Port_change[Port_change>0]
        Port_out = Port_change[Port_change<0]

        Port_in_eff = Port_in.mul(self.period_ExPost).sum(1)
        Port_out_eff = Port_out.mul(self.period_ExPost).sum(1)
        print(f'Rebalancing-In:\n{Port_in.mul(self.period_ExPost).iloc[-1].dropna()}')
        print(f'Rebalancing-Out:\n{Port_out.mul(self.period_ExPost).iloc[-1].dropna()}')
        # Port_in_eff.add(Port_out_eff)
        # rP_.sub(rB_)
        return Port_in_eff, Port_out_eff, rP_, rB_





class BrinsonHoodBeebower_calculator_old(retcnt_calculator):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input,Index_Daily_price_input, cost=0.00, n_day_after=0):
        """
        Asset_info: pandas Series
        index = 종목코드
        value = class
        """
        all_list = list(set(P_w_pvt_input.columns))


        self.Asset_info = Asset_info_input.copy()
        # 시계열로 변하는 class를 고려하지 않음
        # 혹시라도 섹터가 구분되지 않는 종목이 섞여들어왔을 때
        self.Index_Daily_price_input = Index_Daily_price_input.copy()
        ambigous_list = list(set(all_list)-set(self.Asset_info.index))
        if len(ambigous_list)>0:
            print(f'class가 구분되지 않는 종목 {len(ambigous_list)}개')
            for cd in ambigous_list:
                self.Asset_info[cd] = 'unknown'


        price_df = gdu.data.copy()

        self.b_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after+1) # 수익률로 계산하기 때문에 더하기 1 필요
        self.s_dts = self.get_nday_delay(P_w_pvt_input.index, n_day_after)[1:]
        self.ratio_df_buydate = P_w_pvt_input.copy()
        self.ratio_df_buydate.index = self.b_dts
        self.ratio_df_selldate = P_w_pvt_input.iloc[1:]
        self.ratio_df_selldate.index = self.s_dts

        _rnt_dt = price_df.pct_change()
        gr_rtn = self.get_df_grouped(_rnt_dt)
        self.period_ExPost = gr_rtn.apply(lambda x: x.add(1).prod()-1)
        try:
            self.period_ExPost.index = P_w_pvt_input.index
        except:
            self.period_ExPost.index = P_w_pvt_input.index[:-1]

        # 설정 확인
        P_w_pvt = P_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        B_w_pvt = B_w_pvt_input.rename_axis('date', axis=0).rename_axis('code', axis=1)
        # Portfolio는 BM 가만히 있을 때, 수시리밸런싱을 했을 수도 있으니까
        irre_rebal_date = P_w_pvt.index[~P_w_pvt.index.isin(B_w_pvt.index)]
        if len(irre_rebal_date) != 0:
            print(f'수시리밸런싱이 {len(irre_rebal_date)}회 있습니다: {list(irre_rebal_date)}')
            irre_rebal_B_w_tmp = retcnt_calculator(ratio_df=B_w_pvt).daily_account_ratio
            B_w_pvt = pd.concat([B_w_pvt, irre_rebal_B_w_tmp.loc[irre_rebal_date]], axis=0).sort_index()
        self.Bench_cls = retcnt_calculator(ratio_df=B_w_pvt, cost=cost, n_day_after=n_day_after)
        self.Port_cls = retcnt_calculator(ratio_df=P_w_pvt, cost=cost, n_day_after=n_day_after)

        self.allocation_effect, self.selection_effect, self.rP, self.rB, self.P_classweight_pvt = self.get_AA_effects(P_w_pvt, B_w_pvt)

        LATEST_Rebal = P_w_pvt.copy()
        IF_None_Rebal=self.Port_cls.before_account_ratio.copy()
        IF_None_Rebal.index=LATEST_Rebal.index
        self.Rebalancing_in_eff, self.Rebalancing_out_eff, self.rP_NonReb, self.rB_NonReb=self.get_Rebalancing_effects(LATEST_Rebal,IF_None_Rebal)

        # 확인
        # print('rP.sub(rB)',rP.sub(rB))
        # print('SUMMATION',allocation_effect_tmp.add(selection_effect_tmp).add(interaction_effect_tmp))
        # self.rP.sub(self.rB).add(1).cumprod()
        # self.allocation_effect.add(self.selection_effect).add(self.interaction_effect).add(1).cumprod()
        # return_calculator(B_w_pvt,cost=0, n_day_after=n_day_after).portfolio_cumulative_return.loc[:self.rB.index[-1]]
        # self.rB.add(1).cumprod()


        """
        # BHB(1986)와 DGTW(1997)를 기반
        TR = P_w_pvt.mul(period_ExPost).sum(1)
        SAA = B_w_pvt.mul(period_ExPost).sum(1)
        TAA = P_cw_pvt.sub(B_cw_pvt, fill_value=0).mul(B_cr_pvt).sum(1)
        AS = P_cw_pvt.mul(P_cr_pvt.sub(B_cr_pvt)).sum(1)
        TR
        SAA.add(TAA).add(AS)
        """
    def get_normed_pvt(self,_w_pvt, Ass_inf):
        # _w_pvt, Ass_inf = B_w_pvt.copy(), self.Asset_info.copy()
        # Asset_info_input DataFrame을 사용하여 P_w_pvt의 각 컬럼에 대한 클래스를 찾음
        class_for_columns = Ass_inf.loc[_w_pvt.columns].values.flatten()
        # 같은 클래스에 속하는 컬럼들의 합을 계산
        # cls=class_for_columns[0]
        class_sums = {cls: _w_pvt.loc[:, class_for_columns == cls].sum(axis=1) for cls in set(class_for_columns)}
        # P_w_pvt의 각 컬럼을 해당 클래스의 합으로 나눔
        P_w_pvt_normalized = _w_pvt.apply(lambda col: col / class_sums[class_for_columns[_w_pvt.columns.get_loc(col.name)]])
        return P_w_pvt_normalized
    def convert_pvt_wrt_class(self, w_pvt_, convert=True):
        # w_pvt_ = B_w_pvt.copy()
        if convert:
            nw_pvt_ = self.get_normed_pvt(w_pvt_, self.Asset_info)
        else:
            nw_pvt_ = w_pvt_.copy()
        cr_pvt_ = self.period_ExPost[w_pvt_.notna()].fillna(0).mul(nw_pvt_, fill_value=1).rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        cw_pvt_ = w_pvt_.rename(columns=self.Asset_info.to_dict()).stack().groupby(level=[0, 1]).sum().unstack()
        return cw_pvt_,cr_pvt_
    def get_AA_effects(self, P_w_pvt_, B_w_pvt_):
        # P_w_pvt_, B_w_pvt_ = P_w_pvt.copy(), B_w_pvt.copy()
        rP_ = P_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        rB_ = B_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)


        ######################## class별 변환
        P_cw_pvt_, P_cr_pvt_ = self.convert_pvt_wrt_class(P_w_pvt_)
        P_classweight_pvt = P_cw_pvt_.copy()
        B_cw_pvt_, B_cr_pvt_ = self.convert_pvt_wrt_class(B_w_pvt_, convert=False)
        I_cr_pvt_ = self.Index_Daily_price_input.loc[P_cr_pvt_.index].pct_change().shift(-1)

        # Allocation Effect
        wPj_minus_wBj_ = P_cw_pvt_.sub(B_cw_pvt_, fill_value=0)  # .rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        allocation_effect_tmp_ = wPj_minus_wBj_.mul(I_cr_pvt_, fill_value=0).sum(1)

        # Selection Effect
        rPj_minus_rIj_ = P_cr_pvt_.sub(I_cr_pvt_, fill_value=0)
        selection_effect_tmp_ = rPj_minus_rIj_.mul(P_cw_pvt_, fill_value=0).sum(1)

        allocation_effect = allocation_effect_tmp_.shift(1)
        selection_effect = selection_effect_tmp_.shift(1)
        rB = rB_.shift(1)
        rP = rP_.shift(1)
        # 확인
        # allocation_effect.add(selection_effect)
        # rP.sub(rB)
        return allocation_effect, selection_effect, rP, rB, P_classweight_pvt
    def get_Rebalancing_effects(self, P_w_pvt_, B_w_pvt_):
        # P_w_pvt_, B_w_pvt_ = LATEST_Rebal.copy(), IF_None_Rebal.copy()
        rP_ = P_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)
        rB_ = B_w_pvt_.mul(self.period_ExPost).dropna(how='all', axis=0).dropna(how='all', axis=1).sum(1)

        Port_change = P_w_pvt_.sub(B_w_pvt_, fill_value=0)
        Port_in = Port_change[Port_change>0]
        Port_out = Port_change[Port_change<0]

        Port_in_eff = Port_in.mul(self.period_ExPost).sum(1)
        Port_out_eff = Port_out.mul(self.period_ExPost).sum(1)
        # Port_in_eff.add(Port_out_eff)
        # rP_.sub(rB_)
        return Port_in_eff, Port_out_eff, rP_, rB_

if __name__ == "__main__":
    from tqdm import tqdm
    B_w_pvt_input, P_w_pvt_input = pd.read_excel(f'ETF_test.xlsx', sheet_name='BM_pvt',index_col=0,parse_dates=[0]), pd.read_excel(f'ETF_test.xlsx', sheet_name='MyPort_pvt', index_col=0, parse_dates=[0])
    price_df = pd.DataFrame()

    datadate = "20240129"
    from GD_utils.general_utils import save_as_pd_parquet, read_pd_parquet

    if os.path.exists(f'./ETF_price_{datadate}.hd5'):
        price_df = read_pd_parquet(f'./ETF_price_{datadate}.hd5')
    else:

        from pykrx import stock
        ETF_tickers = stock.get_etf_ticker_list(datadate)

        for stock in tqdm(ETF_tickers):
            tmp = gdu.get_data.get_naver_close(stock).rename(columns=lambda x: "A"+stock)
            price_df = pd.concat([price_df, tmp], axis=1)
            time.sleep(0.5)
        price_df = price_df.loc[:"2022-12"]
        save_as_pd_parquet(f'./ETF_price_{datadate}.hd5', price_df)


    gdu.data = price_df.copy()
    Asset_info = pd.read_excel(f'./ETF_test.xlsx', sheet_name='표1')[['단축코드', '기초시장분류', '기초자산분류']]
    Asset_info.columns = ['code', 'class1', 'class2']
    Asset_info['code'] = Asset_info['code'].astype(str).apply(lambda x: 'A'+'0'*(6-len(x))+x)
    Asset_info['class'] = Asset_info['class1'] + Asset_info['class2']
    Asset_info_input = Asset_info.set_index('code')['class']

    Index_Daily_price_input = price_df.rename_axis('code',axis=1).stack().rename('price').reset_index().merge(Asset_info_input, how='left', on='code').groupby(['class', 'date'])['price'].mean().swaplevel().unstack().dropna(how='all', axis=0).dropna(how='all', axis=1)

    self=Modified_BrinsonFachler_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input=Asset_info_input,Index_Daily_price_input=Index_Daily_price_input, cost=0.00, n_day_after=0)



    # # # example 1
    # P_w_pvt_input = pd.DataFrame([{'1-1': 0.75*(1/3),'1-2': 0.75*(2/3), '2': 0.25}], index=[pd.to_datetime('2020-01-01')])
    # B_w_pvt_input = pd.DataFrame([{'3': 0.6, '4': 0.4}], index=[pd.to_datetime('2020-01-01')])
    # price_df = pd.DataFrame({'1-1': [1,1.12],'1-2': [1,1.135], '2': [1,1.19], '3': [1,1.1], '4':[1, 1.2]}, index=[pd.to_datetime('2020-01-01'), pd.to_datetime('2020-02-01')])
    # Asset_info_input = pd.DataFrame({'code': ['1-1','1-2', '2', '3', '4'], 'class': ['A','A', 'B', 'A', 'B']}).set_index('code')['class']
    # gdu.data = price_df.copy()
    # self=BrinsonFachler_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input=Asset_info_input, cost=0.00, n_day_after=0)
    #
    # #
    # #
    # # # example 2
    # P_w_pvt_input = pd.DataFrame([{'1-1': 0.7*(1/3),'1-2': 0.7*(2/3), '2': 0.25 ,'5':0.05}], index=[pd.to_datetime('2020-01-01')])
    # B_w_pvt_input = pd.DataFrame([{'3': 0.6, '4': 0.4}], index=[pd.to_datetime('2020-01-01')])
    # price_df = pd.DataFrame({'1-1': [1,1.12],'1-2': [1,1.135], '2': [1,1.19], '3': [1,1.1], '4':[1, 1.2], '5':[1,1.05]}, index=[pd.to_datetime('2020-01-01'), pd.to_datetime('2020-02-01')])
    # Asset_info_input = pd.DataFrame({'code': ['1-1','1-2', '2', '3', '4','5'], 'class': ['A','A', 'B', 'A', 'B','C']}).set_index('code')['class']
    # gdu.data = price_df.copy()
    # self=BrinsonFachler_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input=Asset_info_input, cost=0.00, n_day_after=0)
    #
    # #
    # # # example 3
    # P_w_pvt_input = pd.DataFrame([{'3': 0.6, '4': 0.4}], index=[pd.to_datetime('2020-01-01')])
    # B_w_pvt_input = pd.DataFrame([{'1-1': 0.7*(1/3),'1-2': 0.7*(2/3), '2': 0.25 ,'5':0.05}], index=[pd.to_datetime('2020-01-01')])
    # price_df = pd.DataFrame({'1-1': [1,1.12],'1-2': [1,1.135], '2': [1,1.19], '3': [1,1.1], '4':[1, 1.2], '5':[1,1.05]}, index=[pd.to_datetime('2020-01-01'), pd.to_datetime('2020-02-01')])
    # Asset_info_input = pd.DataFrame({'code': ['1-1','1-2', '2', '3', '4','5'], 'class': ['A','A', 'B', 'A', 'B','C']}).set_index('code')['class']
    # gdu.data = price_df.copy()
    # self=BrinsonFachler_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input=Asset_info_input, cost=0.00, n_day_after=0)
