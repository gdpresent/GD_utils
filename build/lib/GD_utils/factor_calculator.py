import pandas as pd
from GD_utils.return_calculator import return_calculator
from GD_utils.portfolio_calculator import PortfolioAnalysis
import GD_utils as gdu
import numpy as np

class FactorAnalysis:
    def __init__(self, df_with_factor, price_df, BM_data):
        self.df = df_with_factor.copy()
        self.price = price_df.copy()
        self.BM_data = BM_data.copy()
        gdu.data = self.price.copy()

    def get_Decile_result_sector(self, sector, col_name, drtion):
        df_to_calc = self.df.loc[self.df['sector']==sector,['date', '종목코드', col_name]].copy()
        rank_df = df_to_calc.assign(int_rank=df_to_calc.groupby('date')[col_name].rank(method='first', ascending=drtion, pct=True))
        rank_piv = rank_df.pivot(index='date', columns='종목코드', values='int_rank').dropna(how='all', axis=1).dropna(how='all', axis=0)

        cum_rnt_df = pd.DataFrame()
        for i in range(0, 10):
            n_q = (rank_piv > i * 0.1) & (rank_piv <= (1 + i) * 0.1)
            n_q = n_q[n_q]
            n_q = n_q.multiply(1 / n_q.sum(1), axis='index').dropna(axis=1, how='all').dropna(axis=0, how='all').fillna(0)

            port_calc = return_calculator(n_q)
            cum_rnt_df[f'P{i + 1}'] = port_calc.backtest_cumulative_return

        IC_df = cum_rnt_df.iloc[-1].rank(ascending=False, method='first').reset_index()
        IC_df['index'] = IC_df['index'].str[1:].astype(int)
        IC_df.columns = ['tile',col_name]

        metric_table = self.get_table(cum_rnt_df)
        metric_table['IC'] = IC_df.corr(method='spearman')[col_name]['tile']
        metric_table['num_data_mean'] = rank_piv.count(1).mean()
        metric_table['num_data_min'] = rank_piv.count(1).min()

        return cum_rnt_df, metric_table
    def get_Decile_result(self, col_name, drtion):
        df_to_calc = self.df[['date', '종목코드', col_name]].copy()
        rank_df = df_to_calc.assign(int_rank=df_to_calc.groupby('date')[col_name].rank(method='first', ascending=drtion, pct=True))
        rank_piv = rank_df.pivot(index='date', columns='종목코드', values='int_rank').dropna(how='all', axis=1).dropna(how='all', axis=0)


        cum_rnt_df = pd.DataFrame()
        for i in range(0, 10):
            n_q = (rank_piv > i * 0.1) & (rank_piv <= (1 + i) * 0.1)
            n_q = n_q[n_q]
            n_q = n_q.multiply(1 / n_q.sum(1), axis='index').dropna(axis=1, how='all').dropna(axis=0, how='all').fillna(0)

            port_calc = return_calculator(n_q)
            cum_rnt_df[f'P{i + 1}'] = port_calc.backtest_cumulative_return

        IC_df = cum_rnt_df.iloc[-1].rank(ascending=False, method='first').reset_index()
        IC_df['index'] = IC_df['index'].str[1:].astype(int)
        IC_df.columns = ['tile',col_name]

        metric_table = self.get_table(cum_rnt_df)
        metric_table['IC'] = IC_df.corr(method='spearman')[col_name]['tile']
        metric_table['num_data_mean'] = rank_piv.count(1).mean()
        metric_table['num_data_min'] = rank_piv.count(1).min()

        return cum_rnt_df, metric_table
    def get_ThreeFactor_Decile_result(self,col_name1,drtion1,col_name2,drtion2,col_name3,drtion3):
        tot_col_name = f'{col_name1}_{str(drtion1)[0]}_{col_name2}_{str(drtion2)[0]}_{col_name3}_{str(drtion3)[0]}'
        df_to_calc = self.df[['date', '종목코드', col_name1,col_name2,col_name3]].copy()

        rank_df = df_to_calc.assign(int_rank=df_to_calc.groupby('date')[col_name1].rank(method='first', ascending=drtion1, pct=True).\
                                             add(df_to_calc.groupby('date')[col_name2].rank(method='first', ascending=drtion2, pct=True)).\
                                             add(df_to_calc.groupby('date')[col_name3].rank(method='first', ascending=drtion3, pct=True))
                                    ).\
                             assign(re_rank=lambda x:x.groupby('date')['int_rank'].rank(method='first', ascending=True, pct=True))
        rank_piv = rank_df.pivot(index='date', columns='종목코드', values='re_rank').dropna(how='all', axis=1).dropna(how='all', axis=0)


        cum_rnt_df = pd.DataFrame()
        for i in range(0, 10):
            n_q = (rank_piv > i * 0.1) & (rank_piv <= (1 + i) * 0.1)
            n_q = n_q[n_q]
            n_q = n_q.multiply(1 / n_q.sum(1), axis='index').dropna(axis=1, how='all').dropna(axis=0, how='all').fillna(0)

            port_calc = return_calculator(n_q)
            cum_rnt_df[f'P{i + 1}'] = port_calc.backtest_cumulative_return

        IC_df = cum_rnt_df.iloc[-1].rank(ascending=False, method='first').reset_index()
        IC_df['index'] = IC_df['index'].str[1:].astype(int)
        IC_df.columns = ['tile', tot_col_name]

        metric_table = self.get_table(cum_rnt_df)
        metric_table['IC'] = IC_df.corr(method='spearman')[tot_col_name]['tile']
        metric_table['num_data_mean'] = rank_piv.count(1).mean()
        metric_table['num_data_min'] = rank_piv.count(1).min()

        return cum_rnt_df, metric_table
    def get_top_n_result(self, col_name, drtion, n):
        # col_name, drtion, n = '매출총이익_매출액', False, 10
        df_to_calc = self.df[['date', '종목코드', col_name]].copy()
        rank_df = df_to_calc.assign(int_rank=df_to_calc.groupby('date')[col_name].rank(method='first', ascending=drtion))
        slcted_df = rank_df[rank_df['int_rank']<=n]

        w_df = pd.pivot(slcted_df, index='date', columns='종목코드', values='int_rank')
        w_df = w_df.multiply(1 / w_df.sum(1), axis='index').dropna(axis=1, how='all').dropna(axis=0, how='all').fillna(0)

        port_calc = return_calculator(w_df)
        return port_calc.backtest_cumulative_return
    def get_three_factor_top_n_result(self,col_name1,drtion1,col_name2,drtion2,col_name3,drtion3,n):
        # col_name, drtion, n = '매출총이익_매출액', False, 10

        df_to_calc = self.df[['date', '종목코드', col_name1, col_name2, col_name3]].copy()

        rank_df = df_to_calc.assign(
            int_rank=df_to_calc.groupby('date')[col_name1].rank(method='first', ascending=drtion1, pct=True). \
            add(df_to_calc.groupby('date')[col_name2].rank(method='first', ascending=drtion2, pct=True)). \
            add(df_to_calc.groupby('date')[col_name3].rank(method='first', ascending=drtion3, pct=True))
            ). \
            assign(re_rank=lambda x: x.groupby('date')['int_rank'].rank(method='first', ascending=True))

        slcted_df = rank_df[rank_df['re_rank']<=n]

        w_df = pd.pivot(slcted_df, index='date', columns='종목코드', values='re_rank')
        w_df = w_df.multiply(1 / w_df.sum(1), axis='index').dropna(axis=1, how='all').dropna(axis=0, how='all').fillna(0)

        port_calc = return_calculator(w_df)
        return port_calc.backtest_cumulative_return
    def get_table(self, cum_rnt_ts):
        cum_rnt_ts['시장수익률'] = self.BM_data
        cum_rnt_ts = cum_rnt_ts / cum_rnt_ts.iloc[0]

        return_daily = cum_rnt_ts.pct_change().fillna(0)
        alpha_daily = cum_rnt_ts.pct_change().sub(cum_rnt_ts['시장수익률'].pct_change(), axis='index').fillna(0)
        cum_alpha = alpha_daily.add(1).cumprod()

        PortAnalysis_Calc = PortfolioAnalysis(return_daily,last_BM=True)
        Active_Calc = PortfolioAnalysis(alpha_daily,last_BM=True)

        # 각종 포트폴리오 성과지표
        output_df = pd.DataFrame()
        output_df['CAGR'] = PortAnalysis_Calc.cagr
        output_df['std'] = PortAnalysis_Calc.std
        output_df['Sharpe'] = PortAnalysis_Calc.sharpe
        output_df['Sortino'] = PortAnalysis_Calc.sortino
        output_df['MDD'] = PortAnalysis_Calc.mdd

        output_df['Cumulative Alpha'] = cum_alpha.iloc[-1]
        output_df['Alpha CAGR'] = Active_Calc.cagr
        output_df['Tracking Error'] = Active_Calc.std
        output_df['IR'] = Active_Calc.cagr / Active_Calc.std

        def calc_monthly_hit(df):
            return (df.add(1).groupby(pd.Grouper(freq='BM')).apply(lambda x: x.cumprod().tail(1)) > 1).agg(
                [sum, len]).T.assign(win=lambda x: x['sum'] / x['len'])['win']
        def calc_rolling1Y_hit(df):
            return (df.add(1).cumprod().pct_change(250).dropna(axis=0) > 0).agg([sum, len]).T.assign(
                win=lambda x: x['sum'] / x['len'])['win']

        output_df['Hit'] = calc_monthly_hit(return_daily)
        output_df['R-Hit'] = calc_rolling1Y_hit(return_daily)
        output_df['Hit(alpha)'] = calc_monthly_hit(alpha_daily)
        output_df['R-Hit(alpha)'] = calc_rolling1Y_hit(alpha_daily)
        return output_df

    def factor_report(self, col_name, drtion, outputname='./UnnamedReport', display=True):
        # col_name, drtion = '매출총이익_매출액', False
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column, row
        from bokeh.models import Column

        curdoc().clear()
        output_file(outputname + '.html')

        # 팩터 분위수 결과물 Bokeh Figure Objects
        cum_rnt_decile, metric_table_decile = self.get_Decile_result(col_name, drtion)
        decile_fig = PortfolioAnalysis(cum_rnt_decile.pct_change().fillna(0))

        logscale_return_TS_obj = decile_fig.get_logscale_rtn_obj('above')
        CAGR_bar_obj = decile_fig.get_CAGR_bar_obj()
        inputtable_obj = decile_fig.get_inputtable_obj(metric_table_decile)

        # 팩터 top n Portfolios
        top5 = self.get_top_n_result(col_name, drtion, 5).rename('Top 5')
        top10 = self.get_top_n_result(col_name, drtion, 10).rename('Top 10')
        top20 = self.get_top_n_result(col_name, drtion, 20).rename('Top 20')
        top_n_df = pd.concat([top5, top10, top20, self.BM_data], axis=1).dropna()
        top_n_fig = PortfolioAnalysis(top_n_df.pct_change().fillna(0), last_BM=True)

        top_n_logscale_return_TS_obj = top_n_fig.get_logscale_rtn_obj('above')
        top_n_dd_TS_obj = top_n_fig.get_dd_obj('above')
        top_n_table_obj = top_n_fig.get_table_obj()

        if display == True:
            show(column(row(logscale_return_TS_obj, CAGR_bar_obj), Column(inputtable_obj), top_n_logscale_return_TS_obj, top_n_dd_TS_obj,Column(top_n_table_obj)))
        else:
            save(column(row(logscale_return_TS_obj, CAGR_bar_obj), Column(inputtable_obj), top_n_logscale_return_TS_obj, top_n_dd_TS_obj,Column(top_n_table_obj)))
        return
    def three_factor_decompose_report(self, col_name1,drtion1,col_name2,drtion2,col_name3,drtion3, outputname='./UnnamedReport', display=True):
        # col_name1,col_name2,col_name3, drtion1,drtion2,drtion3='매출원가_매출액','매출총이익_매출액','법인세비용_매출액',False,False,False
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column, row
        from bokeh.models import Column

        curdoc().clear()
        output_file(outputname + '.html')

        # 팩터 분위수 결과물 Bokeh Figure Objects
        cum_rnt_decile_tot, metric_table_decile_tot = self.get_ThreeFactor_Decile_result(col_name1,drtion1,col_name2,drtion2,col_name3,drtion3)
        cum_rnt_decile1, metric_table_decile1 = self.get_Decile_result(col_name1, drtion1)
        cum_rnt_decile2, metric_table_decile2 = self.get_Decile_result(col_name2, drtion2)
        cum_rnt_decile3, metric_table_decile3 = self.get_Decile_result(col_name3, drtion3)

        decile_fig_tot = PortfolioAnalysis(cum_rnt_decile_tot.pct_change().fillna(0), last_BM=True)
        decile_fig1 = PortfolioAnalysis(cum_rnt_decile1.pct_change().fillna(0), last_BM=True)
        decile_fig2 = PortfolioAnalysis(cum_rnt_decile2.pct_change().fillna(0), last_BM=True)
        decile_fig3 = PortfolioAnalysis(cum_rnt_decile3.pct_change().fillna(0), last_BM=True)

        logscale_return_TS_obj_tot = decile_fig_tot.get_logscale_rtn_obj('above')
        logscale_return_TS_obj1 = decile_fig1.get_logscale_rtn_obj('above')
        logscale_return_TS_obj2 = decile_fig2.get_logscale_rtn_obj('above')
        logscale_return_TS_obj3 = decile_fig3.get_logscale_rtn_obj('above')

        CAGR_bar_obj_tot = decile_fig_tot.get_CAGR_bar_obj()
        CAGR_bar_obj1 = decile_fig1.get_CAGR_bar_obj()
        CAGR_bar_obj2 = decile_fig2.get_CAGR_bar_obj()
        CAGR_bar_obj3 = decile_fig3.get_CAGR_bar_obj()

        inputtable_obj_tot = decile_fig_tot.get_inputtable_obj(metric_table_decile_tot)
        inputtable_obj1 = decile_fig1.get_inputtable_obj(metric_table_decile1)
        inputtable_obj2 = decile_fig2.get_inputtable_obj(metric_table_decile2)
        inputtable_obj3 = decile_fig3.get_inputtable_obj(metric_table_decile3)

        # 팩터 top n Portfolios
        top5 = self.get_three_factor_top_n_result(col_name1,drtion1,col_name2,drtion2,col_name3,drtion3, n=5).rename('Top 5')
        top10 = self.get_three_factor_top_n_result(col_name1,drtion1,col_name2,drtion2,col_name3,drtion3, n=10).rename('Top 10')
        top20 = self.get_three_factor_top_n_result(col_name1,drtion1,col_name2,drtion2,col_name3,drtion3, n=20).rename('Top 20')
        top_n_df = pd.concat([top5, top10, top20, self.BM_data], axis=1).dropna()
        top_n_fig = PortfolioAnalysis(top_n_df.pct_change().fillna(0), last_BM=True)

        top_n_logscale_return_TS_obj = top_n_fig.get_logscale_rtn_obj('above')
        top_n_dd_TS_obj = top_n_fig.get_dd_obj('above')
        top_n_table_obj = top_n_fig.get_table_obj(_width=1500)

        if display == True:
            show(
                column(
                    top_n_logscale_return_TS_obj, top_n_dd_TS_obj,Column(top_n_table_obj),
                    row(logscale_return_TS_obj_tot, CAGR_bar_obj_tot), Column(inputtable_obj_tot),
                    row(logscale_return_TS_obj1, CAGR_bar_obj1), Column(inputtable_obj1),
                    row(logscale_return_TS_obj2, CAGR_bar_obj2), Column(inputtable_obj2),
                    row(logscale_return_TS_obj3, CAGR_bar_obj3), Column(inputtable_obj3)
                    )
                )
        else:
            save(
                column(
                    top_n_logscale_return_TS_obj, top_n_dd_TS_obj,Column(top_n_table_obj),
                    row(logscale_return_TS_obj_tot, CAGR_bar_obj_tot), Column(inputtable_obj_tot),
                    row(logscale_return_TS_obj1, CAGR_bar_obj1), Column(inputtable_obj1),
                    row(logscale_return_TS_obj2, CAGR_bar_obj2), Column(inputtable_obj2),
                    row(logscale_return_TS_obj3, CAGR_bar_obj3), Column(inputtable_obj3)
                    )
)
        return
    def factor_sector_report_temp(self, sectors6, col_name, drtion, outputname='./UnnamedReport', display=True):
        # col_name, drtion = '현금의증가_type_1_자산총계_ZoY', False
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column, row
        from bokeh.models import Column
        s1, s2, s3, s4, s5, s6 = sectors6
        curdoc().clear()
        output_file(outputname + '.html')
        Log_TS_obj_s1,CAGR_bar_obj_s1,table_obj_s1 = self.get_sectors_report(s1, col_name, drtion)
        Log_TS_obj_s2,CAGR_bar_obj_s2,table_obj_s2 = self.get_sectors_report(s2, col_name, drtion)
        Log_TS_obj_s3,CAGR_bar_obj_s3,table_obj_s3 = self.get_sectors_report(s3, col_name, drtion)
        Log_TS_obj_s4,CAGR_bar_obj_s4,table_obj_s4 = self.get_sectors_report(s4, col_name, drtion)
        Log_TS_obj_s5,CAGR_bar_obj_s5,table_obj_s5 = self.get_sectors_report(s5, col_name, drtion)
        Log_TS_obj_s6,CAGR_bar_obj_s6,table_obj_s6 = self.get_sectors_report(s6, col_name, drtion)
        # show(
        #     column(
        #            column(row(Log_TS_obj_s1, CAGR_bar_obj_s1), table_obj_s1),
        #            column(row(Log_TS_obj_s2, CAGR_bar_obj_s2), table_obj_s2),
        #            column(row(Log_TS_obj_s3, CAGR_bar_obj_s3), table_obj_s3),
        #            column(row(Log_TS_obj_s4, CAGR_bar_obj_s4), table_obj_s4),
        #            column(row(Log_TS_obj_s5, CAGR_bar_obj_s5), table_obj_s5),
        #            column(row(Log_TS_obj_s6, CAGR_bar_obj_s6), table_obj_s6),
        #         ))
    def get_sectors_report(self, sector, col_name, drtion):
        # sector, col_name, drtion = s1, col_name, drtion
        # 팩터 분위수 결과물 Bokeh Figure Objects
        cum_rnt_decile, metric_table_decile = self.get_Decile_result_sector(sector, col_name, drtion)
        decile_fig = PortfolioAnalysis(cum_rnt_decile.pct_change().fillna(0))

        logscale_return_TS_obj = decile_fig.get_logscale_rtn_obj('above')
        CAGR_bar_obj = decile_fig.get_CAGR_bar_obj()
        inputtable_obj = decile_fig.get_inputtable_obj(metric_table_decile)
        return logscale_return_TS_obj,CAGR_bar_obj,inputtable_obj

    def factor_result(self, col_name, drtion):
        cum_rnt_decile, metric_table_decile = self.get_Decile_result(col_name, drtion)
        return cum_rnt_decile, metric_table_decile