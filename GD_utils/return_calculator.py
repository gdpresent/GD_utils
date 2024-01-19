import pandas as pd
import numpy as np
import GD_utils as gdu
import time
class return_calculator:
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


# 2024-01-18 디버그 완료
class return_calculator_v2:
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
        _rnt_dt = price_df.pct_change(fill_method=None).loc[:,ratio_df.columns]

        # 리밸런싱 날짜별로 잘 그루핑을 해놓자
        gr_rtn = self.get_df_grouped(_rnt_dt)

        # 회전율 관련
        _daily_ratio = gr_rtn.apply(self.calc_bt_daily_ratio) # 실제 일별 내 계좌 잔고의 종목별 비중[%]
        self._rb_tr_ratio_stockwise = abs(self.calc_rb_turnover(_daily_ratio)) # 실제 리밸런싱 날짜의 종목별 회전율
        self._rb_tr_ratio = self._rb_tr_ratio_stockwise.sum(1) # 실제 리밸런싱 날짜의 회전율

        # 수익률 기여도
        self.daily_ret_cntrbtn_tmp = gr_rtn.apply(self.calc_bt_compound_return).droplevel(0)
        self.daily_ret_cntrbtn = self.daily_ret_cntrbtn_tmp.loc[self.s_dts]
        # back-test daily return
        self.backtest_daily_return = self.daily_ret_cntrbtn_tmp.sum(1)
        # back-test daily cumulative return
        self.backtest_cumulative_return = self.backtest_daily_return.add(1).cumprod()

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
    def calc_compound_return(self, grouped_price):
        return grouped_price.drop('gr_idx', axis=1).pct_change().fillna(0).add(1).cumprod().sub(1)
    def calc_bt_compound_return(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][-1])
        gr_comp_rtn = grouped_return.set_index('gr_idx').fillna(0).add(1).cumprod().sub(1) # input된 것은 일별수익률임 따라서 복리수익률을 만들어 준 이후,
        daily_comp_rtn = gr_comp_rtn.mul(self.ratio_df_buydate.loc[gr_comp_rtn.index[0]]).dropna(axis=1, how='all')  # "복리수익률" * "리밸런싱때 조정한 비중" = 포트폴리오 종목별 일별 (누적)복리수익률
        daily_comp_rtn.index = grouped_return.index

        # stock-wise decomposing
        first_line = daily_comp_rtn.iloc[0] # pct_change를 할 때 첫 줄이 날아가기 때문에 남겨놓아야 하는 첫 번째날 수익률(리밸런싱 하루 이후 포트폴리오 종목별 수익률 backup)
        daily_comp_rtn = daily_comp_rtn.add(1).pct_change() # 포트폴리오 종목별 일별 수익률
        daily_comp_rtn.iloc[0] = first_line

        rb_d = daily_comp_rtn.index[0]
        tr_applied_here = self._rb_tr_ratio_stockwise.loc[rb_d].dropna() # 회전율은 하루 전으로 날짜가 잡혀있고, 여기 계산된 수익률은 하루 뒤로 밀려있음(수익률이기 때문에) / 날짜를 맞춰주기 위한 조작

        # apply trading cost0
        daily_comp_rtn.loc[rb_d] = daily_comp_rtn.loc[rb_d] - self._cost * tr_applied_here
        # output = daily_comp_rtn.sum(1)
        return daily_comp_rtn
    def calc_bt_daily_ratio(self, grouped_return):
        # grouped_return = gr_rtn.get_group([x for x in gr_rtn.groups.keys()][0])
        gr_rtn_ = grouped_return.set_index('gr_idx').dropna(how='all', axis=1).add(1).cumprod()
        output = gr_rtn_.mul(self.ratio_df_buydate.loc[gr_rtn_.index[0]])#.add(1).pct_change().sum(1).fillna(0)
        output = output.div(output.sum(1), axis=0)
        output.index = grouped_return.index
        return output
    def calc_rb_turnover(self, daily_account_ratio):
        # daily_account_ratio = _daily_ratio.copy()
        s_dt_account_ratio = daily_account_ratio.droplevel(0).loc[self.s_dts]
        b_dt_ratio_target = self.ratio_df_buydate.loc[self.b_dts]
        s_dt_account_ratio.loc[b_dt_ratio_target.index[0]]=0
        s_dt_account_ratio = s_dt_account_ratio.sort_index()
        s_dt_account_ratio.index = b_dt_ratio_target.index

        rb_ratio_diff = b_dt_ratio_target.sub(s_dt_account_ratio, fill_value=0)
        return rb_ratio_diff


if __name__ == "__main__":
    from tqdm import tqdm
    BM_w_df, Port_w_df = pd.read_excel(f'ETF_test.xlsx', sheet_name='BM_pvt',index_col=0,parse_dates=[0]), pd.read_excel(f'ETF_test.xlsx', sheet_name='MyPort_pvt', index_col=0, parse_dates=[0])
    price_df = pd.DataFrame()
    for stock in tqdm(BM_w_df.columns.to_list() + Port_w_df.columns.to_list()):
        tmp = gdu.get_data.get_naver_close(stock[1:]).rename(columns=lambda x: stock)
        price_df = pd.concat([price_df, tmp], axis=1)
    gdu.data = price_df.copy()
    ratio_df = Port_w_df.copy()
    ratio_df = ratio_df.div(100)
    self = return_calculator_v2(ratio_df)
    pass