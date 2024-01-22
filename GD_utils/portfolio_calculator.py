import GD_utils as gdu
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

class PortfolioAnalysis:
    def __init__(self, daily_return, outputname='./Unnamed', last_BM=False, BM_name='KOSPI'):
        # 포트폴리오 일별 수익률
        self.daily_return = daily_return
        # 포트폴리오 복리수익률
        self.cum_ret_cmpd = self.daily_return.add(1).cumprod()
        self.cum_ret_cmpd.iloc[0] = 1
        # 포트폴리오 단리수익률
        self.cum_ret_smpl = self.daily_return.cumsum()
        # 분석 기간
        self.num_years = self.get_num_year(self.daily_return.index.year.unique())

        # 각종 포트폴리오 성과지표
        self.cagr = self._calculate_cagr(self.cum_ret_cmpd, self.num_years)
        self.std = self._calculate_std(self.daily_return,self.num_years)

        self.rolling_std_6M = self.daily_return.rolling(min_periods=120, window=120).apply(lambda x:self._calculate_std(x, self.num_years))
        self.rolling_CAGR_6M = self.cum_ret_cmpd.rolling(min_periods=120, window=120).apply(lambda x:self._calculate_cagr(x, self.num_years))
        self.rolling_sharpe_6M = self.rolling_CAGR_6M/self.rolling_std_6M

        self.sharpe = self.cagr/self.std
        self.sortino = self.cagr/self._calculate_downsiderisk(self.daily_return,self.num_years)
        self.drawdown = self._calculate_dd(self.cum_ret_cmpd)
        self.average_drawdown = self.drawdown.mean()
        self.mdd = self._calculate_mdd(self.drawdown)


        # BM대비 성과지표
        if last_BM == False:
            self.BM = self.get_BM(BM_name)
            print(f"BM장착=================== {BM_name}")
            self.daily_return_to_BM = self.daily_return.copy()
        else:
            self.BM = self.daily_return.iloc[:,[-1]].add(1).cumprod().fillna(1)
            self.daily_return_to_BM = self.daily_return.iloc[:, :-1]

        # BM 대비성과
        self.daily_alpha = self.daily_return_to_BM.sub(self.BM.iloc[:, 0].pct_change(), axis=0).dropna()
        self.cum_alpha_cmpd = self.daily_alpha.add(1).cumprod()

        self.alpha_cagr = self._calculate_cagr(self.cum_alpha_cmpd, self.num_years)
        self.alpha_std = self._calculate_std(self.daily_alpha,self.num_years)
        self.alpha_sharpe = self.alpha_cagr/self.alpha_std
        self.alpha_sortino = self.alpha_cagr/self._calculate_downsiderisk(self.daily_alpha,self.num_years)
        self.alpha_drawdown = self._calculate_dd(self.cum_alpha_cmpd)
        self.alpha_average_drawdown = self.alpha_drawdown.mean()
        self.alpha_mdd = self._calculate_mdd(self.alpha_drawdown)

        # Monthly & Yearly
        self.yearly_return = self.daily_return_to_BM.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
        self.yearly_alpha = self.daily_alpha.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)

        self.monthly_return = self.daily_return_to_BM.add(1).groupby(pd.Grouper(freq='BM')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
        self.monthly_alpha = self.daily_alpha.add(1).groupby(pd.Grouper(freq='BM')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
        self.monthly_return_WR = (self.monthly_return > 0).agg([sum, len]).apply(lambda x: x['sum'] / x['len']).iloc[0]
        self.monthly_alpha_WR = (self.monthly_alpha > 0).agg([sum, len]).apply(lambda x: x['sum'] / x['len']).iloc[0]

        try:
            self.R1Y_HPR, self.R1Y_HPR_WR = self._holding_period_return(self.cum_ret_cmpd, self.num_years)
            self.R1Y_HPA, self.R1Y_HPA_WR = self._holding_period_return(self.cum_alpha_cmpd, self.num_years)
            self.key_rates_3Y = self._calculate_key_rates(self.daily_return.iloc[-252*3:], self.daily_alpha.iloc[-252*3:])
            self.key_rates_5Y = self._calculate_key_rates(self.daily_return.iloc[-252*5:], self.daily_alpha.iloc[-252*5:])
        except:
            pass

        # Bokeh Plot을 위한 기본 변수 설정
        from bokeh import palettes
        # self.color_list = ['#ec008e','#0086d4', '#361b6f',  '#8c98a0'] + list(palettes.Category20_20)
        self.color_list = ['#192036','#eaa88f', '#8c98a0'] + list(palettes.Category20_20)
        self.outputname = outputname

    def basic_report(self, simple=False, display = True, toolbar_location='above'):
        from bokeh.plotting import figure, output_file, show, curdoc, save
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, Legend, Column
        from bokeh.models.widgets import DataTable, TableColumn
        from bokeh.models import NumeralTickFormatter, LogTickFormatter

        def to_source(df):
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
            return ColumnDataSource(df)

        curdoc().clear()
        output_file(self.outputname + '.html')


        try:
            static_data = pd.concat([self.cum_ret_cmpd.iloc[-1]-1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd, self.average_drawdown,self.R1Y_HPR_WR], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD',
                                   'Average Drawdown', 'HPR(1Y)']
        except:
            static_data = pd.concat([self.cum_ret_cmpd.iloc[-1]-1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd, self.average_drawdown], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation', 'MDD', 'Average Drawdown']
        for col in static_data.columns:
            if col in ['Compound_Return', 'CAGR', 'MDD', 'Average Drawdown', 'Standard Deviation','HPR(1Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))
        static_data.reset_index(inplace=True)
        static_data.rename(columns={'index': 'Portfolio'}, inplace=True)
        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_obj = DataTable(source=source, columns=columns, width=1500, height=200,index_position=None)


        if simple==True:
            # Plot 단리
            source_for_chart = to_source(self.cum_ret_smpl)
            return_TS_obj = figure(x_axis_type='datetime',
                        title='Simple Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                        width=1500, height=400, toolbar_location=toolbar_location)
        elif simple=='log':
            # Plot 로그
            source_for_chart = to_source(self.cum_ret_cmpd)
            return_TS_obj = figure(x_axis_type='datetime', y_axis_type='log', y_axis_label=r"$$\frac{P_n}{P_0}$$",
                        title='Cumulative Return(LogScaled)' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                        width=1500, height=450, toolbar_location=toolbar_location)
        else:
            # Plot 복리
            source_for_chart = to_source(self.cum_ret_cmpd-1)
            return_TS_obj = figure(x_axis_type='datetime',
                        title='Cumulative Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                        width=1500, height=450, toolbar_location=toolbar_location)

        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col, color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        # Plot drawdown
        dd_TS_obj = figure(x_axis_type='datetime',
                    title='Drawdown',
                    width=1500, height=170, toolbar_location=toolbar_location)
        source_dd_TS = to_source(self.drawdown)
        dd_TS_lgd_list = []
        for i, col in enumerate(self.drawdown.columns):
            dd_TS_line = dd_TS_obj.line(source=source_dd_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        try:
            source_R1Y_HPR = to_source(self.R1Y_HPR)
            R1Y_HPR_obj = figure(x_axis_type='datetime',
                        title='Rolling Holding Period Return',
                        width=1500, height=170, toolbar_location=toolbar_location)
            R1Y_HPR_lgd_list = []
            for i, col in enumerate(self.R1Y_HPR.columns):
                p_line = R1Y_HPR_obj.line(source=source_R1Y_HPR, x='date', y=col, color=self.color_list[i], line_width=2)
                R1Y_HPR_lgd_list.append((col, [p_line]))
            R1Y_HPR_lgd = Legend(items=R1Y_HPR_lgd_list, location='center')

            R1Y_HPR_obj.add_layout(R1Y_HPR_lgd, 'right')
            R1Y_HPR_obj.legend.click_policy = "hide"
            R1Y_HPR_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        except:
            pass

        if display == True:
            try:
                show(column(return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj)))
            except:
                show(column(return_TS_obj, dd_TS_obj, Column(data_table_obj)))
        else:
            try:
                save(column(return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj)))
            except:
                save(column(return_TS_obj, dd_TS_obj, Column(data_table_obj)))
    def report(self, display = True, toolbar_location='above'):
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column
        from bokeh.models import Column

        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj(_width=1500)
        data_alpha_table_obj = self.get_alpha_table_obj(_width=1500)
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location)

        if display == True:
            try:
                show(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj),Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
            except:
                show(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, Column(data_table_obj),Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
        else:
            try:
                save(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj),Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
            except:
                save(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, Column(data_table_obj),Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
    def single_report(self, display = True, toolbar_location='above'):
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column,row
        from bokeh.models import Column

        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj()
        data_alpha_table_obj = self.get_alpha_table_obj()
        data_table_obj_3Y = self.get_table_obj_3Y()
        data_table_obj_5Y = self.get_table_obj_5Y()
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location)

        Monthly_rtn_obj = self.get_monthly_rtn_obj(toolbar_location)
        Monthly_alpha_obj = self.get_monthly_alpha_obj(toolbar_location)
        Monthly_rtn_dist_obj = self.get_monthly_rtn_dist_obj(toolbar_location)
        Monthly_alpha_dist_obj = self.get_monthly_alpha_dist_obj(toolbar_location)

        RllnCAGR_obj = self.get_rollingCAGR_obj(toolbar_location)
        Rllnstd_obj = self.get_rollingstd_obj(toolbar_location)
        Rllnshrp_obj = self.get_rollingSharpe_obj(toolbar_location)


        if display == True:
            try:
                show(
                   row(
                       column(
                              Column(data_table_obj),
                              Column(data_alpha_table_obj),
                              Column(data_table_obj_3Y),
                              Column(data_table_obj_5Y),
                             ),
                        column(
                               cmpd_return_TS_obj,
                               logscale_return_TS_obj,
                               dd_TS_obj, R1Y_HPR_obj,
                               Yearly_rtn_obj,
                               Yearly_alpha_obj,
                               row(Monthly_rtn_obj, Monthly_alpha_obj),
                               row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                               RllnCAGR_obj,
                               Rllnstd_obj,
                               Rllnshrp_obj,
                               )
                       )
                    )
            except:
                show(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )
                )
        else:
            try:
                save(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                            Column(data_table_obj_3Y),
                            Column(data_table_obj_5Y),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj, R1Y_HPR_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )

                )
            except:
                save(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )

                )

    def to_source(self, df):
        from bokeh.models import ColumnDataSource
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return ColumnDataSource(df)
    def get_table_obj_3Y(self,_width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn
        cumrnt,cagr,std,sharpe,sortino,average_drawdown,mdd,alpha_cumrnt,alpha_cagr,alpha_std,alpha_sharpe,alpha_sortino,alpha_average_drawdown,alpha_mdd = self.key_rates_3Y
        
        static_data = pd.concat(
            [cumrnt.iloc[-1] - 1, cagr, sharpe, sortino, std, mdd, average_drawdown,
             alpha_cumrnt.iloc[-1] - 1, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown
             ], axis=1)
        static_data.columns = ['Compound Return(3Y)', 'CAGR(3Y)', 'Sharpe Ratio(3Y)', 'Sortino Ratio(3Y)', 'Standard Deviation(3Y)', 'MDD(3Y)', 'Average Drawdown(3Y)',
                               'Compound Alpha(3Y)', 'CAGR(Alpha,3Y)', 'IR(3Y)', 'Sortino Ratio(Alpha,3Y)', 'Tracking Error(3Y)', 'MDD(Alpha,3Y)', 'Average Drawdown(Alpha,3Y)']
        
        for col in static_data.columns:
            if col in ['Compound Return(3Y)','Compound Alpha(3Y)', 'CAGR(3Y)','CAGR(Alpha,3Y)', 'MDD(3Y)','MDD(Alpha,3Y)',
                       'Average Drawdown(3Y)','Average Drawdown(Alpha,3Y)', 'Standard Deviation(3Y)','Tracking Error(3Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)
        return data_table_fig
    def get_table_obj_5Y(self,_width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn
        cumrnt,cagr,std,sharpe,sortino,average_drawdown,mdd,alpha_cumrnt,alpha_cagr,alpha_std,alpha_sharpe,alpha_sortino,alpha_average_drawdown,alpha_mdd = self.key_rates_5Y
        
        static_data = pd.concat(
            [cumrnt.iloc[-1] - 1, cagr, sharpe, sortino, std, mdd, average_drawdown,
             alpha_cumrnt.iloc[-1] - 1, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown
             ], axis=1)
        static_data.columns = ['Compound Return(5Y)', 'CAGR(5Y)', 'Sharpe Ratio(5Y)', 'Sortino Ratio(5Y)', 'Standard Deviation(5Y)', 'MDD(5Y)', 'Average Drawdown(5Y)',
                               'Compound Alpha(5Y)', 'CAGR(Alpha,5Y)', 'IR(5Y)', 'Sortino Ratio(Alpha,5Y)', 'Tracking Error(5Y)', 'MDD(Alpha,5Y)', 'Average Drawdown(Alpha,5Y)']
        
        for col in static_data.columns:
            if col in ['Compound Return(5Y)','Compound Alpha(5Y)', 'CAGR(5Y)','CAGR(Alpha,5Y)', 'MDD(5Y)','MDD(Alpha,5Y)',
                       'Average Drawdown(5Y)','Average Drawdown(Alpha,5Y)', 'Standard Deviation(5Y)','Tracking Error(5Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)
        return data_table_fig
    def get_table_obj(self, _width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn

        try:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd,
                 self.average_drawdown, self.R1Y_HPR_WR], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD',
                                   'Average Drawdown', 'HPR(1Y)']
        except:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd,
                 self.average_drawdown], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD', 'Average Drawdown']
        for col in static_data.columns:
            if col in ['Compound_Return', 'CAGR', 'MDD', 'Average Drawdown', 'Standard Deviation', 'HPR(1Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=300, index_position=None)
        return data_table_fig
    def get_alpha_table_obj(self,_width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn

        try:
            static_data = pd.concat(
                [self.cum_alpha_cmpd.iloc[-1] - 1, self.alpha_cagr, self.alpha_cagr, self.alpha_sortino, self.alpha_std, self.alpha_mdd,
                 self.alpha_average_drawdown, self.R1Y_HPA_WR], axis=1)
            static_data.columns = ['Compound_alpha', 'CAGR(alpha)', 'IR', 'Sortino Ratio(alpha)', 'Tracking Error',
                                   'MDD(alpha)', 'Avg Drawdown(alpha)', 'HPA(1Y)']
        except:
            static_data = pd.concat(
                [self.cum_alpha_cmpd.iloc[-1] - 1, self.alpha_cagr, self.alpha_cagr, self.alpha_sortino, self.alpha_std, self.alpha_mdd,
                 self.alpha_average_drawdown], axis=1)
            static_data.columns = ['Compound_alpha', 'CAGR(alpha)', 'IR', 'Sortino Ratio(alpha)', 'Tracking Error',
                                   'MDD(alpha)', 'Avg Drawdown(alpha)']
        for col in static_data.columns:
            if col in ['Compound_alpha', 'CAGR(alpha)', 'MDD(alpha)', 'Avg Drawdown(alpha)', 'Tracking Error', 'HPA(1Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))
        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=300, index_position=None)


        return data_table_fig
    def get_inputtable_obj(self, input_tbl):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn
        # input_tbl = metric_table_decile.copy()

        # input_tbl.columns
        # input_tbl.filter(like='Alpha')

        pct_display = ['CAGR', 'std', 'MDD', 'Alpha CAGR', 'Tracking Error', 'Hit', 'R-Hit', 'Hit(alpha)', 'R-Hit(alpha)']
        for col in input_tbl.columns:
            if col in pct_display:
                input_tbl.loc[:, col] = input_tbl.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                input_tbl.loc[:, col] = input_tbl.loc[:, col].apply(lambda x: np.around(x, decimals=4))
        input_tbl.reset_index(inplace=True)
        input_tbl.rename(columns={'index': 'Portfolio'}, inplace=True)
        source = ColumnDataSource(input_tbl)
        columns = [TableColumn(field=col, title=col) for col in input_tbl.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=1500, height=200, index_position=None)
        return data_table_fig
    def get_smpl_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        # Plot 단리
        source_for_chart = self.to_source(self.cum_ret_smpl)
        return_TS_obj = figure(x_axis_type='datetime',
                    title='Simple Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                    width=1500, height=400, toolbar_location=toolbar_location)
        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col, color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_cmpd_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        # Plot 복리
        source_for_chart = self.to_source(self.cum_ret_cmpd - 1)
        return_TS_obj = figure(x_axis_type='datetime',
                               title='Cumulative Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=450, toolbar_location=toolbar_location)

        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col,
                                                color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_logscale_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        # Plot 로그
        source_for_chart = self.to_source(self.cum_ret_cmpd)
        return_TS_obj = figure(x_axis_type='datetime', y_axis_type='log', y_axis_label=r"$$\frac{P_n}{P_0}$$",
                               title='Cumulative Return(LogScaled)' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=450, toolbar_location=toolbar_location)
        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col,
                                                color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_dd_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        # Plot drawdown
        dd_TS_obj = figure(x_axis_type='datetime',
                    title='Drawdown',
                    width=1500, height=170, toolbar_location=toolbar_location)
        source_dd_TS = self.to_source(self.drawdown)
        dd_TS_lgd_list = []
        for i, col in enumerate(self.drawdown.columns):
            dd_TS_line = dd_TS_obj.line(source=source_dd_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return dd_TS_obj
    def get_rollingCAGR_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        RllnCAGR_TS_obj = figure(x_axis_type='datetime',
                    title='Rolling CAGR(6M)',
                    width=1500, height=200, toolbar_location=toolbar_location)
        source_RllnCAGR_TS = self.to_source(self.rolling_CAGR_6M)
        RllnCAGR_TS_lgd_list = []
        for i, col in enumerate(self.rolling_CAGR_6M.columns):
            RllnCAGR_TS_line = RllnCAGR_TS_obj.line(source=source_RllnCAGR_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            RllnCAGR_TS_lgd_list.append((col, [RllnCAGR_TS_line]))
        RllnCAGR_TS_lgd = Legend(items=RllnCAGR_TS_lgd_list, location='center')
        RllnCAGR_TS_obj.add_layout(RllnCAGR_TS_lgd, 'right')
        RllnCAGR_TS_obj.legend.click_policy = "hide"
        RllnCAGR_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0.0 %')
        return RllnCAGR_TS_obj
    def get_rollingstd_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        Rllnstd_TS_obj = figure(x_axis_type='datetime',
                    title='Rolling Standard Deviation(6M)',
                    width=1500
                                , height=200, toolbar_location=toolbar_location)
        source_Rllnstd_TS = self.to_source(self.rolling_std_6M)
        Rllnstd_TS_lgd_list = []
        for i, col in enumerate(self.rolling_std_6M.columns):
            Rllnstd_TS_line = Rllnstd_TS_obj.line(source=source_Rllnstd_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            Rllnstd_TS_lgd_list.append((col, [Rllnstd_TS_line]))
        Rllnstd_TS_lgd = Legend(items=Rllnstd_TS_lgd_list, location='center')
        Rllnstd_TS_obj.add_layout(Rllnstd_TS_lgd, 'right')
        Rllnstd_TS_obj.legend.click_policy = "hide"
        Rllnstd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0.0 %')
        return Rllnstd_TS_obj
    def get_rollingSharpe_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        Rllnshrp_TS_obj = figure(x_axis_type='datetime',
                    title='Rolling Sharpe(6M)',
                    width=1500, height=200, toolbar_location=toolbar_location)
        source_Rllnshrp_TS = self.to_source(self.rolling_sharpe_6M)
        Rllnshrp_TS_lgd_list = []
        for i, col in enumerate(self.rolling_sharpe_6M.columns):
            Rllnshrp_TS_line = Rllnshrp_TS_obj.line(source=source_Rllnshrp_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            Rllnshrp_TS_lgd_list.append((col, [Rllnshrp_TS_line]))
        Rllnshrp_TS_lgd = Legend(items=Rllnshrp_TS_lgd_list, location='center')
        Rllnshrp_TS_obj.add_layout(Rllnshrp_TS_lgd, 'right')
        Rllnshrp_TS_obj.legend.click_policy = "hide"

        return Rllnshrp_TS_obj
    def get_yearly_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend, ColumnDataSource
        from bokeh.transform import dodge
        # Plot Yearly Performance
        input_Data = self.yearly_return.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Return',
            width=1500, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range),  width=0.2 ,top=col, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        return dd_TS_obj
    def get_yearly_alpha_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend, ColumnDataSource
        from bokeh.transform import dodge
        # Plot Yearly Performance
        input_Data = self.yearly_alpha.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Alpha',
            width=1500, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range),  width=0.2 ,top=col, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        return dd_TS_obj
    def get_R1Y_HPR_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        try:
            source_R1Y_HPR = self.to_source(self.R1Y_HPR)
            R1Y_HPR_obj = figure(x_axis_type='datetime',
                                 title='Rolling Holding Period Return',
                                 width=1500, height=170, toolbar_location=toolbar_location)
            R1Y_HPR_lgd_list = []
            for i, col in enumerate(self.R1Y_HPR.columns):
                p_line = R1Y_HPR_obj.line(source=source_R1Y_HPR, x='date', y=col, color=self.color_list[i],
                                          line_width=2)
                R1Y_HPR_lgd_list.append((col, [p_line]))
            R1Y_HPR_lgd = Legend(items=R1Y_HPR_lgd_list, location='center')

            R1Y_HPR_obj.add_layout(R1Y_HPR_lgd, 'right')
            R1Y_HPR_obj.legend.click_policy = "hide"
            R1Y_HPR_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
            return R1Y_HPR_obj
        except:
            return None

    def get_monthly_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, LinearColorMapper,ColorBar,BasicTicker,PrintfTickFormatter, LabelSet
        from bokeh.transform import transform
        from bokeh.palettes import RdBu
        colors = RdBu[max(RdBu.keys())]

        # Plot Monthly Performance
        input_Data = self.monthly_return.copy()
        input_Data = input_Data.apply(lambda x:round(x*100, 2))
        input_Data = input_Data.rename(columns={input_Data.columns[0]:'value'})
        input_Data['Year'] = input_Data.index.strftime("%Y")
        input_Data['Month'] = input_Data.index.strftime("%m")
        input_Data = input_Data[['Year', 'Month', 'value']].reset_index(drop='index')

        years = sorted(list(input_Data['Year'].unique()), reverse=True)
        months = sorted(list(input_Data['Month'].unique()))

        source_for_heatmap = ColumnDataSource(input_Data)
        mapper = LinearColorMapper(palette=colors, low=-1, high=1)

        rtn_fig_obj = figure(
                             title="Monthly Return",
                             x_range=months, y_range=years,
                             x_axis_location="above",
                             width=750, height=400,
                             toolbar_location=toolbar_location,
                             tooltips=[('date', '@Year @Month'), ('value', f'@value%')]
                            )


        rtn_fig_obj.rect(
                         x="Month", y="Year", width=1, height=1,
                         source=source_for_heatmap,
                         fill_color=transform('value', mapper),
                         line_color=None
                         )

        color_bar = ColorBar(
                             color_mapper=mapper, major_label_text_font_size="10px",
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="%d%%"),
                             )

        rtn_fig_obj.add_layout(color_bar, 'right')
        rtn_fig_obj.grid.grid_line_color = None
        rtn_fig_obj.axis.axis_line_color = None
        rtn_fig_obj.axis.major_tick_line_color = None
        rtn_fig_obj.axis.major_label_text_font_size = "10px"
        rtn_fig_obj.axis.major_label_standoff = 0

        heatmap_annotation_data = input_Data.copy()
        heatmap_annotation_data['value'] = heatmap_annotation_data['value'].astype(str).apply(lambda x:x+'0'*(2-len(x.split('.')[-1]))+"%")
        source_for_heatmap_annotation = ColumnDataSource(heatmap_annotation_data)
        labels = LabelSet(x='Month', y='Year', text='value', level='overlay',
                          text_font_size={'value': '10px'},
                          text_color='#000000',
                          text_align='center',
                          text_alpha=0.75,
                          x_offset=0, y_offset=-5, source=source_for_heatmap_annotation)
        rtn_fig_obj.add_layout(labels)

        return rtn_fig_obj
    def get_monthly_alpha_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, \
            LabelSet
        from bokeh.transform import transform
        from bokeh.palettes import RdBu
        colors = RdBu[max(RdBu.keys())]

        # Plot Monthly Performance
        input_Data = self.monthly_alpha.copy()
        input_Data = input_Data.apply(lambda x: round(x * 100, 2))
        input_Data = input_Data.rename(columns={input_Data.columns[0]: 'value'})
        input_Data['Year'] = input_Data.index.strftime("%Y")
        input_Data['Month'] = input_Data.index.strftime("%m")
        input_Data = input_Data[['Year', 'Month', 'value']].reset_index(drop='index')

        years = sorted(list(input_Data['Year'].unique()), reverse=True)
        months = sorted(list(input_Data['Month'].unique()))

        source_for_heatmap = ColumnDataSource(input_Data)
        mapper = LinearColorMapper(palette=colors, low=-1, high=1)

        alpha_fig_obj = figure(
            title="Monthly Alpha",
            x_range=months, y_range=years,
            x_axis_location="above", width=750, height=400,
            toolbar_location=toolbar_location,
            tooltips=[('date', '@Year @Month'), ('value', f'@value%')]
        )

        alpha_fig_obj.rect(
            x="Month", y="Year", width=1, height=1,
            source=source_for_heatmap,
            fill_color=transform('value', mapper),
            line_color=None
        )

        color_bar = ColorBar(
            color_mapper=mapper, major_label_text_font_size="10px",
            ticker=BasicTicker(desired_num_ticks=len(colors)),
            formatter=PrintfTickFormatter(format="%d%%"),
        )

        alpha_fig_obj.add_layout(color_bar, 'right')
        alpha_fig_obj.grid.grid_line_color = None
        alpha_fig_obj.axis.axis_line_color = None
        alpha_fig_obj.axis.major_tick_line_color = None
        alpha_fig_obj.axis.major_label_text_font_size = "10px"
        alpha_fig_obj.axis.major_label_standoff = 0

        heatmap_annotation_data = input_Data.copy()
        heatmap_annotation_data['value'] = heatmap_annotation_data['value'].astype(str).apply(
            lambda x: x + '0' * (2 - len(x.split('.')[-1])) + "%")
        source_for_heatmap_annotation = ColumnDataSource(heatmap_annotation_data)
        labels = LabelSet(x='Month', y='Year', text='value', level='overlay',
                          text_font_size={'value': '10px'},
                          text_color='#000000',
                          text_align='center',
                          text_alpha=0.75,
                          x_offset=0, y_offset=-5, source=source_for_heatmap_annotation)
        alpha_fig_obj.add_layout(labels)

        return alpha_fig_obj

    def get_monthly_rtn_dist_obj(self,toolbar_location):
        from scipy import stats
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Span
        from decimal import Decimal
        monthly_data = self.monthly_return.copy()
        monthly_rtns = monthly_data.values.flatten()
        m,std = round(monthly_rtns.mean(),2), round(monthly_rtns.std(),2)
        skwnss, krts = round(stats.skew(monthly_rtns),2), round(stats.kurtosis(monthly_rtns), 2)

        hist, edges = np.histogram(monthly_rtns, density=True, bins=50)
        estmt_pdf = stats.gaussian_kde(monthly_rtns).pdf(edges)

        dist_fig_obj = figure(title=f'Monthly Return Distribution  (mean={m}, std={std}, skewness={skwnss}, kurtosis={krts})', y_axis_label=r"Density", x_axis_label=r"Monthly Return",
                              width=750, height=400,
                              toolbar_location=toolbar_location)
        dist_fig_obj.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=self.color_list[0], line_color="white", alpha=0.5)
        dist_fig_obj.line(edges, estmt_pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF", )
        dist_fig_obj.xaxis.formatter = NumeralTickFormatter(format='0.0 %')
        dist_fig_obj.legend.visible=False
        vrtc_line = Span(location=monthly_rtns.mean(),
                                    dimension='height', line_color='gray',
                                    line_dash='dashed', line_width=3)
        dist_fig_obj.add_layout(vrtc_line)

        return dist_fig_obj
    def get_monthly_alpha_dist_obj(self,toolbar_location):
        from scipy import stats
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Span

        monthly_data = self.monthly_alpha.copy()
        monthly_rtns = monthly_data.values.flatten()
        m, std = round(monthly_rtns.mean(), 2), round(monthly_rtns.std(), 2)
        skwnss, krts = round(stats.skew(monthly_rtns), 2), round(stats.kurtosis(monthly_rtns), 2)


        hist, edges = np.histogram(monthly_rtns, density=True, bins=50)
        estmt_pdf = stats.gaussian_kde(monthly_rtns).pdf(edges)

        dist_fig_obj = figure(title=f'Monthly Alpha Distribution  (mean={m}, std={std}, skewness={skwnss}, kurtosis={krts})', y_axis_label=r"Density", x_axis_label=r"Monthly Alpha",
                              width=750, height=400,
                              toolbar_location=toolbar_location)
        dist_fig_obj.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=self.color_list[0], line_color="white", alpha=0.5)
        dist_fig_obj.line(edges, estmt_pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF", )
        dist_fig_obj.xaxis.formatter = NumeralTickFormatter(format='0.0 %')
        dist_fig_obj.legend.visible=False
        vrtc_line = Span(location=monthly_rtns.mean(),
                                    dimension='height', line_color='gray',
                                    line_dash='dashed', line_width=3)
        dist_fig_obj.add_layout(vrtc_line)
        return dist_fig_obj

    def deciles_bar_color_list(self,bar_num,bench_num):
        # Spectral6 컬러 목록
        # #['#3288bd', '#99d594', '#e6f598', '#fee08b',, '#d53e4f']
        deciles_bar_color_list = []
        for i in range(0, bar_num):
            deciles_bar_color_list.append('#fc8d59')
        for i in range(0, bench_num):
            deciles_bar_color_list.append('#e6f598')
        return deciles_bar_color_list
    def get_CAGR_bar_obj(self):
        from scipy.stats import linregress
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, NumeralTickFormatter

        CAGR_values=self.cagr.values
        CAGR_index=self.cagr.index

        slp, itrct, rval, pval, stderr = linregress(range(1,len(CAGR_index)), CAGR_values[:-1])
        title_text = f'10분위 연환산 수익률(r_value:{round(rval,2)}, p_value:{round(pval,2)}, std_err:{round(stderr,2)})'

        qun_sourse = ColumnDataSource(data = dict(분위=list(CAGR_index), CAGR =CAGR_values, color=self.deciles_bar_color_list(10, 1)))
        qun = figure(x_range=list(CAGR_index), height=440, title=title_text, width=390)
        qun.vbar(x='분위', top='CAGR', width=0.9,  source=qun_sourse, color='color')
        qun.line(list(range(1,len(CAGR_index))), [x*slp + itrct for x in range(1,len(CAGR_index))], color='black', line_width=2)
        qun.xgrid.grid_line_color = None
        qun.toolbar.logo = None
        qun.toolbar_location = None
        qun.yaxis[0].formatter = NumeralTickFormatter(format="0.00%")
        return qun

    def _array_to_df(self, arr):
        try:
            return pd.DataFrame(arr,
                              index=self.daily_return.index.values,
                              columns=self.daily_return.columns.values).rename_axis("date")
        except:
            return pd.DataFrame(arr,
                                index=self.daily_alpha.index.values,
                                columns=self.daily_alpha.columns.values).rename_axis("date")
    def get_num_year(self, num_years):
        num_years = len(num_years)
        if num_years ==2 :
            # 기간이 1년 이상이면, 1년이란 길이의 기준은 데이터의 갯수로 한다.
            start_date = self.daily_return.index[0]
            end_date = start_date + pd.DateOffset(years=1)

            date_list = self.daily_return.loc[start_date:end_date].index
            num_days = len(date_list)

        elif num_years==1:
            # 기간이 1년 미만이면, 1년이란 길이의 기준은 다음해까지의 영업일 기준으로 가상으로 확장시킨다.
            start_date = self.daily_return.index[0]
            end_date = self.daily_return.index[-1]
            end_date_ = start_date + pd.DateOffset(years=1)

            # 1년이란 기준의 날짜 수 정의
            date_list = pd.date_range(start=start_date, end=end_date_, freq=BDay())
            date_list2 = pd.date_range(start=start_date, end=end_date, freq=BDay())
            num_days = len(date_list)/len(date_list2) * len(self.daily_return.index)

        else:
            # 3년 이상이면, input된 데이터의 첫해와 마지막 해를 제외하고 한 해의 날짜수의 평균으로 한다.
            num_days = self.daily_return.groupby(pd.Grouper(freq='Y')).count().iloc[1:-1].mean()[0]
        return num_days
    def get_BM(self, BM_name):
        if BM_name.lower()=="kospi":
            BM = gdu.get_data.get_naver_close('KOSPI')
        elif (BM_name.lower()=="s&p500")|(BM_name.lower()=="snp500"):
            BM = gdu.get_data.get_data_yahoo_close('^GSPC').rename(columns={'^GSPC':'S&P500'})
        elif (BM_name.lower()=="nasdaq"):
            BM = gdu.get_data.get_data_yahoo_close('^IXIC').rename(columns={'^IXIC':'NASDAQ'})
        else:
            try:
                BM = gdu.get_data.get_naver_close(BM_name)
            except:
                BM = gdu.get_data.get_data_yahoo_close(BM_name)
        return BM

    def _calculate_key_rates(self, daily_returns, daily_alpha):
        # daily_returns, daily_alpha = self.daily_return.iloc[-252 * 3:].copy(),  self.daily_alpha.iloc[-252 * 3:].copy()
        cum_ret_cmpd = daily_returns.add(1).cumprod()

        cum_ret_cmpd.iloc[0] = 1
        num_years = self.get_num_year(daily_returns.index.year.unique())

        cagr = self._calculate_cagr(cum_ret_cmpd, num_years)
        std = self._calculate_std(daily_returns,num_years)
        sharpe = cagr/std
        sortino = cagr/self._calculate_downsiderisk(daily_returns,num_years)
        drawdown = self._calculate_dd(cum_ret_cmpd)
        average_drawdown = drawdown.mean()
        mdd = self._calculate_mdd(drawdown)

        cum_alpha_cmpd = daily_alpha.add(1).cumprod()
        alpha_cagr = self._calculate_cagr(cum_alpha_cmpd, num_years)
        alpha_std = self._calculate_std(daily_alpha, num_years)
        alpha_sharpe = alpha_cagr / alpha_std
        alpha_sortino = alpha_cagr / self._calculate_downsiderisk(daily_alpha, num_years)
        alpha_drawdown = self._calculate_dd(cum_alpha_cmpd)
        alpha_average_drawdown = alpha_drawdown.mean()
        alpha_mdd = self._calculate_mdd(alpha_drawdown)

        return cum_ret_cmpd,cagr,std,sharpe,sortino,average_drawdown,mdd,cum_alpha_cmpd,alpha_cagr,alpha_std,alpha_sharpe,alpha_sortino,alpha_average_drawdown,alpha_mdd

    def _calculate_dd(self, df):
        # df = self.cum_ret_cmpd.copy()
        # df = t_df.pct_change().copy()
        # df = self.cum_alpha_cmpd.copy()
        # df = cum_ret_cmpd.copy()
        max_list = df.iloc[0].values
        out_list = [np.array([0]*len(max_list))]

        for ix in range(1, len(df.index)):
            max_list = np.max([max_list, df.iloc[ix].values], axis=0)
            out_list.append((df.iloc[ix].values - max_list) / max_list)
        try:
            out = self._array_to_df(out_list)
        except:
            out = pd.DataFrame(out_list, index=df.index, columns=df.columns)

        return out
    @staticmethod
    def _calculate_cagr(df, num_days):
        return ((df.iloc[-1]/df.iloc[0]) ** (1 / len(df.index))) ** num_days - 1
    @staticmethod
    def _calculate_std(df, num_days):
        return df.std() * np.sqrt(num_days)
    @staticmethod
    def _calculate_mdd(df):
        return df.min()
    @staticmethod
    def _calculate_downsiderisk(df,num_days):
        return df.applymap(lambda x: 0 if x >= 0 else x).std() * np.sqrt(num_days)
    @staticmethod
    def _holding_period_return(df, num_days):
        Rolling_HPR_1Y = df.pct_change(int(num_days.round())).dropna()
        HPR_1Y_mean = Rolling_HPR_1Y.mean()
        HPR_1Y_max = Rolling_HPR_1Y.max()
        HPR_1Y_min = Rolling_HPR_1Y.min()
        Rolling_HPR_1Y_WR = (Rolling_HPR_1Y > 0).sum() / Rolling_HPR_1Y.shape[0]
        return Rolling_HPR_1Y, Rolling_HPR_1Y_WR

class PortfolioAnalysis_v2:
    def __init__(self, P_w_pvt,B_w_pvt,Asset_info, BM_nm='BM', Port_nm='My Portfolio', outputname='./Unnamed'):
        """
        Asset_info:
        index = 종목코드
        column1 = 1차분류
        column2 = 2차분류
        """
        # 설정 확인
        Asset_info.columns = [f'class{i+1}' for i in range(len(Asset_info.columns))]
        P_w_pvt = P_w_pvt.rename_axis('date', axis=0).rename_axis('code', axis=1)
        B_w_pvt = B_w_pvt.rename_axis('date', axis=0).rename_axis('code', axis=1)

        # 종목별 Return Contribution이 main variable임!!
        # self = gdu.return_calculator_v2(B_w_pvt)
        B_ret_cls,P_ret_cls=gdu.return_calculator_v2(B_w_pvt),gdu.return_calculator_v2(P_w_pvt)
        B_ret_cntrbtn = B_ret_cls.ret_cntrbtn
        P_ret_cntrbtn = P_ret_cls.ret_cntrbtn
        B_daily_ret = B_ret_cls.backtest_daily_return
        P_daily_ret = P_ret_cls.backtest_daily_return
        B_ret_cntrbtn.sum(1).add(1).cumprod()
        B_daily_ret.loc[:B_ret_cntrbtn.index.max()].add(1).cumprod()


        # stacked
        P_w_df = P_w_pvt.stack().rename('w').reset_index().merge(Asset_info, how='left', on=['code'])
        B_w_df = B_w_pvt.stack().rename('w').reset_index().merge(Asset_info, how='left', on=['code'])
        
        # class 1
        cls1_diff = pd.concat([B_w_df.groupby(['date', 'class1'])['w'].sum(),P_w_df.groupby(['date', 'class1'])['w'].sum()], axis=1).fillna(0).diff(axis=1).iloc[:,-1].sort_index()
        cls2_diff = pd.concat([B_w_df.groupby(['date', 'class2'])['w'].sum(),P_w_df.groupby(['date', 'class2'])['w'].sum()], axis=1).fillna(0).diff(axis=1).iloc[:,-1].sort_index()











        # BM_p_df = gdu.calc_return(B_w_pvt, cost=0, n_days_after=2).rename(BM_nm)
        # Port_p_df = gdu.calc_return(P_w_pvt, cost=0, n_days_after=2).rename(Port_nm)
        daily_return = pd.concat([Port_p_df,BM_p_df], axis=1).pct_change()
        daily_return.iloc[0]=0


        # SAA

















        # 포트폴리오 일별 수익률
        self.daily_return = daily_return.copy()
        # 포트폴리오 복리수익률
        self.cum_ret_cmpd = self.daily_return.add(1).cumprod()
        self.cum_ret_cmpd.iloc[0] = 1
        # 포트폴리오 단리수익률
        self.cum_ret_smpl = self.daily_return.cumsum()
        # 분석 기간
        self.num_years = self.get_num_year(self.daily_return.index.year.unique())

        # 각종 포트폴리오 성과지표
        self.cagr = self._calculate_cagr(self.cum_ret_cmpd, self.num_years)
        self.std = self._calculate_std(self.daily_return,self.num_years)

        self.rolling_std_6M = self.daily_return.rolling(min_periods=120, window=120).apply(lambda x:self._calculate_std(x, self.num_years))
        self.rolling_CAGR_6M = self.cum_ret_cmpd.rolling(min_periods=120, window=120).apply(lambda x:self._calculate_cagr(x, self.num_years))
        self.rolling_sharpe_6M = self.rolling_CAGR_6M/self.rolling_std_6M

        self.sharpe = self.cagr/self.std
        self.sortino = self.cagr/self._calculate_downsiderisk(self.daily_return,self.num_years)
        self.drawdown = self._calculate_dd(self.cum_ret_cmpd)
        self.average_drawdown = self.drawdown.mean()
        self.mdd = self._calculate_mdd(self.drawdown)


        self.BM = self.daily_return.iloc[:,[-1]].add(1).cumprod().fillna(1)
        self.daily_return_to_BM = self.daily_return.iloc[:, :-1]

        # BM 대비성과
        self.daily_alpha = self.daily_return_to_BM.sub(self.BM.iloc[:, 0].pct_change(), axis=0).dropna()
        self.cum_alpha_cmpd = self.daily_alpha.add(1).cumprod()

        self.alpha_cagr = self._calculate_cagr(self.cum_alpha_cmpd, self.num_years)
        self.alpha_std = self._calculate_std(self.daily_alpha,self.num_years)
        self.alpha_sharpe = self.alpha_cagr/self.alpha_std
        self.alpha_sortino = self.alpha_cagr/self._calculate_downsiderisk(self.daily_alpha,self.num_years)
        self.alpha_drawdown = self._calculate_dd(self.cum_alpha_cmpd)
        self.alpha_average_drawdown = self.alpha_drawdown.mean()
        self.alpha_mdd = self._calculate_mdd(self.alpha_drawdown)

        # Monthly & Yearly
        self.yearly_return = self.daily_return_to_BM.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
        self.yearly_alpha = self.daily_alpha.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)

        self.monthly_return = self.daily_return_to_BM.add(1).groupby(pd.Grouper(freq='BM')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
        self.monthly_alpha = self.daily_alpha.add(1).groupby(pd.Grouper(freq='BM')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
        self.monthly_return_WR = (self.monthly_return > 0).agg([sum, len]).apply(lambda x: x['sum'] / x['len']).iloc[0]
        self.monthly_alpha_WR = (self.monthly_alpha > 0).agg([sum, len]).apply(lambda x: x['sum'] / x['len']).iloc[0]

        try:
            self.R1Y_HPR, self.R1Y_HPR_WR = self._holding_period_return(self.cum_ret_cmpd, self.num_years)
            self.R1Y_HPA, self.R1Y_HPA_WR = self._holding_period_return(self.cum_alpha_cmpd, self.num_years)
            self.key_rates_3Y = self._calculate_key_rates(self.daily_return.iloc[-252*3:], self.daily_alpha.iloc[-252*3:])
            self.key_rates_5Y = self._calculate_key_rates(self.daily_return.iloc[-252*5:], self.daily_alpha.iloc[-252*5:])
        except:
            pass

        # Bokeh Plot을 위한 기본 변수 설정
        from bokeh import palettes
        # self.color_list = ['#ec008e','#0086d4', '#361b6f',  '#8c98a0'] + list(palettes.Category20_20)
        self.color_list = ['#192036','#eaa88f', '#8c98a0'] + list(palettes.Category20_20)
        self.outputname = outputname

    def basic_report(self, simple=False, display=True, toolbar_location='above'):
        from bokeh.plotting import figure, output_file, show, curdoc, save
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, Legend, Column
        from bokeh.models.widgets import DataTable, TableColumn
        from bokeh.models import NumeralTickFormatter, LogTickFormatter

        def to_source(df):
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
            return ColumnDataSource(df)

        curdoc().clear()
        output_file(self.outputname + '.html')

        try:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd,
                 self.average_drawdown, self.R1Y_HPR_WR], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD',
                                   'Average Drawdown', 'HPR(1Y)']
        except:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd,
                 self.average_drawdown], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD', 'Average Drawdown']
        for col in static_data.columns:
            if col in ['Compound_Return', 'CAGR', 'MDD', 'Average Drawdown', 'Standard Deviation', 'HPR(1Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))
        static_data.reset_index(inplace=True)
        static_data.rename(columns={'index': 'Portfolio'}, inplace=True)
        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_obj = DataTable(source=source, columns=columns, width=1500, height=200, index_position=None)

        if simple == True:
            # Plot 단리
            source_for_chart = to_source(self.cum_ret_smpl)
            return_TS_obj = figure(x_axis_type='datetime',
                                   title='Simple Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                                   width=1500, height=400, toolbar_location=toolbar_location)
        elif simple == 'log':
            # Plot 로그
            source_for_chart = to_source(self.cum_ret_cmpd)
            return_TS_obj = figure(x_axis_type='datetime', y_axis_type='log', y_axis_label=r"$$\frac{P_n}{P_0}$$",
                                   title='Cumulative Return(LogScaled)' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                                   width=1500, height=450, toolbar_location=toolbar_location)
        else:
            # Plot 복리
            source_for_chart = to_source(self.cum_ret_cmpd - 1)
            return_TS_obj = figure(x_axis_type='datetime',
                                   title='Cumulative Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                                   width=1500, height=450, toolbar_location=toolbar_location)

        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col,
                                                color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        # Plot drawdown
        dd_TS_obj = figure(x_axis_type='datetime',
                           title='Drawdown',
                           width=1500, height=170, toolbar_location=toolbar_location)
        source_dd_TS = to_source(self.drawdown)
        dd_TS_lgd_list = []
        for i, col in enumerate(self.drawdown.columns):
            dd_TS_line = dd_TS_obj.line(source=source_dd_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        try:
            source_R1Y_HPR = to_source(self.R1Y_HPR)
            R1Y_HPR_obj = figure(x_axis_type='datetime',
                                 title='Rolling Holding Period Return',
                                 width=1500, height=170, toolbar_location=toolbar_location)
            R1Y_HPR_lgd_list = []
            for i, col in enumerate(self.R1Y_HPR.columns):
                p_line = R1Y_HPR_obj.line(source=source_R1Y_HPR, x='date', y=col, color=self.color_list[i],
                                          line_width=2)
                R1Y_HPR_lgd_list.append((col, [p_line]))
            R1Y_HPR_lgd = Legend(items=R1Y_HPR_lgd_list, location='center')

            R1Y_HPR_obj.add_layout(R1Y_HPR_lgd, 'right')
            R1Y_HPR_obj.legend.click_policy = "hide"
            R1Y_HPR_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        except:
            pass

        if display == True:
            try:
                show(column(return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj)))
            except:
                show(column(return_TS_obj, dd_TS_obj, Column(data_table_obj)))
        else:
            try:
                save(column(return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj)))
            except:
                save(column(return_TS_obj, dd_TS_obj, Column(data_table_obj)))
    def report(self, display=True, toolbar_location='above'):
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column
        from bokeh.models import Column

        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj(_width=1500)
        data_alpha_table_obj = self.get_alpha_table_obj(_width=1500)
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location)

        if display == True:
            try:
                show(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj),
                            Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
            except:
                show(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, Column(data_table_obj),
                            Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
        else:
            try:
                save(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, R1Y_HPR_obj, Column(data_table_obj),
                            Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
            except:
                save(column(cmpd_return_TS_obj, logscale_return_TS_obj, dd_TS_obj, Column(data_table_obj),
                            Column(data_alpha_table_obj), Yearly_rtn_obj, Yearly_alpha_obj))
    def single_report(self, display=True, toolbar_location='above'):
        from bokeh.plotting import output_file, show, curdoc, save
        from bokeh.layouts import column, row
        from bokeh.models import Column

        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj()
        data_alpha_table_obj = self.get_alpha_table_obj()
        data_table_obj_3Y = self.get_table_obj_3Y()
        data_table_obj_5Y = self.get_table_obj_5Y()
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location)

        Monthly_rtn_obj = self.get_monthly_rtn_obj(toolbar_location)
        Monthly_alpha_obj = self.get_monthly_alpha_obj(toolbar_location)
        Monthly_rtn_dist_obj = self.get_monthly_rtn_dist_obj(toolbar_location)
        Monthly_alpha_dist_obj = self.get_monthly_alpha_dist_obj(toolbar_location)

        RllnCAGR_obj = self.get_rollingCAGR_obj(toolbar_location)
        Rllnstd_obj = self.get_rollingstd_obj(toolbar_location)
        Rllnshrp_obj = self.get_rollingSharpe_obj(toolbar_location)

        if display == True:
            try:
                show(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                            Column(data_table_obj_3Y),
                            Column(data_table_obj_5Y),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj, R1Y_HPR_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )
                )
            except:
                show(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )
                )
        else:
            try:
                save(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                            Column(data_table_obj_3Y),
                            Column(data_table_obj_5Y),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj, R1Y_HPR_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )

                )
            except:
                save(
                    row(
                        column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                        ),
                        column(
                            cmpd_return_TS_obj,
                            logscale_return_TS_obj,
                            dd_TS_obj,
                            Yearly_rtn_obj,
                            Yearly_alpha_obj,
                            row(Monthly_rtn_obj, Monthly_alpha_obj),
                            row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                            RllnCAGR_obj,
                            Rllnstd_obj,
                            Rllnshrp_obj,
                        )
                    )

                )


    def to_source(self, df):
        from bokeh.models import ColumnDataSource
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return ColumnDataSource(df)
    def get_table_obj_3Y(self, _width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn
        cumrnt, cagr, std, sharpe, sortino, average_drawdown, mdd, alpha_cumrnt, alpha_cagr, alpha_std, alpha_sharpe, alpha_sortino, alpha_average_drawdown, alpha_mdd = self.key_rates_3Y

        static_data = pd.concat(
            [cumrnt.iloc[-1] - 1, cagr, sharpe, sortino, std, mdd, average_drawdown,
             alpha_cumrnt.iloc[-1] - 1, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd,
             alpha_average_drawdown
             ], axis=1)
        static_data.columns = ['Compound Return(3Y)', 'CAGR(3Y)', 'Sharpe Ratio(3Y)', 'Sortino Ratio(3Y)',
                               'Standard Deviation(3Y)', 'MDD(3Y)', 'Average Drawdown(3Y)',
                               'Compound Alpha(3Y)', 'CAGR(Alpha,3Y)', 'IR(3Y)', 'Sortino Ratio(Alpha,3Y)',
                               'Tracking Error(3Y)', 'MDD(Alpha,3Y)', 'Average Drawdown(Alpha,3Y)']

        for col in static_data.columns:
            if col in ['Compound Return(3Y)', 'Compound Alpha(3Y)', 'CAGR(3Y)', 'CAGR(Alpha,3Y)', 'MDD(3Y)',
                       'MDD(Alpha,3Y)',
                       'Average Drawdown(3Y)', 'Average Drawdown(Alpha,3Y)', 'Standard Deviation(3Y)',
                       'Tracking Error(3Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)
        return data_table_fig
    def get_table_obj_5Y(self, _width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn
        cumrnt, cagr, std, sharpe, sortino, average_drawdown, mdd, alpha_cumrnt, alpha_cagr, alpha_std, alpha_sharpe, alpha_sortino, alpha_average_drawdown, alpha_mdd = self.key_rates_5Y

        static_data = pd.concat(
            [cumrnt.iloc[-1] - 1, cagr, sharpe, sortino, std, mdd, average_drawdown,
             alpha_cumrnt.iloc[-1] - 1, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd,
             alpha_average_drawdown
             ], axis=1)
        static_data.columns = ['Compound Return(5Y)', 'CAGR(5Y)', 'Sharpe Ratio(5Y)', 'Sortino Ratio(5Y)',
                               'Standard Deviation(5Y)', 'MDD(5Y)', 'Average Drawdown(5Y)',
                               'Compound Alpha(5Y)', 'CAGR(Alpha,5Y)', 'IR(5Y)', 'Sortino Ratio(Alpha,5Y)',
                               'Tracking Error(5Y)', 'MDD(Alpha,5Y)', 'Average Drawdown(Alpha,5Y)']

        for col in static_data.columns:
            if col in ['Compound Return(5Y)', 'Compound Alpha(5Y)', 'CAGR(5Y)', 'CAGR(Alpha,5Y)', 'MDD(5Y)',
                       'MDD(Alpha,5Y)',
                       'Average Drawdown(5Y)', 'Average Drawdown(Alpha,5Y)', 'Standard Deviation(5Y)',
                       'Tracking Error(5Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)
        return data_table_fig
    def get_table_obj(self, _width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn

        try:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd,
                 self.average_drawdown, self.R1Y_HPR_WR], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD',
                                   'Average Drawdown', 'HPR(1Y)']
        except:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd,
                 self.average_drawdown], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation',
                                   'MDD', 'Average Drawdown']
        for col in static_data.columns:
            if col in ['Compound_Return', 'CAGR', 'MDD', 'Average Drawdown', 'Standard Deviation', 'HPR(1Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=300, index_position=None)
        return data_table_fig
    def get_alpha_table_obj(self, _width=350):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn

        try:
            static_data = pd.concat(
                [self.cum_alpha_cmpd.iloc[-1] - 1, self.alpha_cagr, self.alpha_cagr, self.alpha_sortino, self.alpha_std,
                 self.alpha_mdd,
                 self.alpha_average_drawdown, self.R1Y_HPA_WR], axis=1)
            static_data.columns = ['Compound_alpha', 'CAGR(alpha)', 'IR', 'Sortino Ratio(alpha)', 'Tracking Error',
                                   'MDD(alpha)', 'Avg Drawdown(alpha)', 'HPA(1Y)']
        except:
            static_data = pd.concat(
                [self.cum_alpha_cmpd.iloc[-1] - 1, self.alpha_cagr, self.alpha_cagr, self.alpha_sortino, self.alpha_std,
                 self.alpha_mdd,
                 self.alpha_average_drawdown], axis=1)
            static_data.columns = ['Compound_alpha', 'CAGR(alpha)', 'IR', 'Sortino Ratio(alpha)', 'Tracking Error',
                                   'MDD(alpha)', 'Avg Drawdown(alpha)']
        for col in static_data.columns:
            if col in ['Compound_alpha', 'CAGR(alpha)', 'MDD(alpha)', 'Avg Drawdown(alpha)', 'Tracking Error',
                       'HPA(1Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))
        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=300, index_position=None)

        return data_table_fig
    def get_inputtable_obj(self, input_tbl):
        from bokeh.models import ColumnDataSource
        from bokeh.models.widgets import DataTable, TableColumn
        # input_tbl = metric_table_decile.copy()

        # input_tbl.columns
        # input_tbl.filter(like='Alpha')

        pct_display = ['CAGR', 'std', 'MDD', 'Alpha CAGR', 'Tracking Error', 'Hit', 'R-Hit', 'Hit(alpha)',
                       'R-Hit(alpha)']
        for col in input_tbl.columns:
            if col in pct_display:
                input_tbl.loc[:, col] = input_tbl.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                input_tbl.loc[:, col] = input_tbl.loc[:, col].apply(lambda x: np.around(x, decimals=4))
        input_tbl.reset_index(inplace=True)
        input_tbl.rename(columns={'index': 'Portfolio'}, inplace=True)
        source = ColumnDataSource(input_tbl)
        columns = [TableColumn(field=col, title=col) for col in input_tbl.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=1500, height=200, index_position=None)
        return data_table_fig
    def get_smpl_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        # Plot 단리
        source_for_chart = self.to_source(self.cum_ret_smpl)
        return_TS_obj = figure(x_axis_type='datetime',
                               title='Simple Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=400, toolbar_location=toolbar_location)
        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col,
                                                color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_cmpd_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        # Plot 복리
        source_for_chart = self.to_source(self.cum_ret_cmpd - 1)
        return_TS_obj = figure(x_axis_type='datetime',
                               title='Cumulative Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=450, toolbar_location=toolbar_location)

        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col,
                                                color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_logscale_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        # Plot 로그
        source_for_chart = self.to_source(self.cum_ret_cmpd)
        return_TS_obj = figure(x_axis_type='datetime', y_axis_type='log', y_axis_label=r"$$\frac{P_n}{P_0}$$",
                               title='Cumulative Return(LogScaled)' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=450, toolbar_location=toolbar_location)
        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col,
                                                color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "hide"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_dd_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        # Plot drawdown
        dd_TS_obj = figure(x_axis_type='datetime',
                           title='Drawdown',
                           width=1500, height=170, toolbar_location=toolbar_location)
        source_dd_TS = self.to_source(self.drawdown)
        dd_TS_lgd_list = []
        for i, col in enumerate(self.drawdown.columns):
            dd_TS_line = dd_TS_obj.line(source=source_dd_TS, x='date', y=col, color=self.color_list[i], line_width=2)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return dd_TS_obj
    def get_rollingCAGR_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        RllnCAGR_TS_obj = figure(x_axis_type='datetime',
                                 title='Rolling CAGR(6M)',
                                 width=1500, height=200, toolbar_location=toolbar_location)
        source_RllnCAGR_TS = self.to_source(self.rolling_CAGR_6M)
        RllnCAGR_TS_lgd_list = []
        for i, col in enumerate(self.rolling_CAGR_6M.columns):
            RllnCAGR_TS_line = RllnCAGR_TS_obj.line(source=source_RllnCAGR_TS, x='date', y=col,
                                                    color=self.color_list[i], line_width=2)
            RllnCAGR_TS_lgd_list.append((col, [RllnCAGR_TS_line]))
        RllnCAGR_TS_lgd = Legend(items=RllnCAGR_TS_lgd_list, location='center')
        RllnCAGR_TS_obj.add_layout(RllnCAGR_TS_lgd, 'right')
        RllnCAGR_TS_obj.legend.click_policy = "hide"
        RllnCAGR_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0.0 %')
        return RllnCAGR_TS_obj
    def get_rollingstd_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        Rllnstd_TS_obj = figure(x_axis_type='datetime',
                                title='Rolling Standard Deviation(6M)',
                                width=1500
                                , height=200, toolbar_location=toolbar_location)
        source_Rllnstd_TS = self.to_source(self.rolling_std_6M)
        Rllnstd_TS_lgd_list = []
        for i, col in enumerate(self.rolling_std_6M.columns):
            Rllnstd_TS_line = Rllnstd_TS_obj.line(source=source_Rllnstd_TS, x='date', y=col, color=self.color_list[i],
                                                  line_width=2)
            Rllnstd_TS_lgd_list.append((col, [Rllnstd_TS_line]))
        Rllnstd_TS_lgd = Legend(items=Rllnstd_TS_lgd_list, location='center')
        Rllnstd_TS_obj.add_layout(Rllnstd_TS_lgd, 'right')
        Rllnstd_TS_obj.legend.click_policy = "hide"
        Rllnstd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0.0 %')
        return Rllnstd_TS_obj
    def get_rollingSharpe_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend

        Rllnshrp_TS_obj = figure(x_axis_type='datetime',
                                 title='Rolling Sharpe(6M)',
                                 width=1500, height=200, toolbar_location=toolbar_location)
        source_Rllnshrp_TS = self.to_source(self.rolling_sharpe_6M)
        Rllnshrp_TS_lgd_list = []
        for i, col in enumerate(self.rolling_sharpe_6M.columns):
            Rllnshrp_TS_line = Rllnshrp_TS_obj.line(source=source_Rllnshrp_TS, x='date', y=col,
                                                    color=self.color_list[i], line_width=2)
            Rllnshrp_TS_lgd_list.append((col, [Rllnshrp_TS_line]))
        Rllnshrp_TS_lgd = Legend(items=Rllnshrp_TS_lgd_list, location='center')
        Rllnshrp_TS_obj.add_layout(Rllnshrp_TS_lgd, 'right')
        Rllnshrp_TS_obj.legend.click_policy = "hide"

        return Rllnshrp_TS_obj
    def get_yearly_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend, ColumnDataSource
        from bokeh.transform import dodge
        # Plot Yearly Performance
        input_Data = self.yearly_return.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Return',
            width=1500, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2 * n_col_ord[i], range=dd_TS_obj.x_range),
                                        width=0.2, top=col, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        return dd_TS_obj
    def get_yearly_alpha_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend, ColumnDataSource
        from bokeh.transform import dodge
        # Plot Yearly Performance
        input_Data = self.yearly_alpha.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Alpha',
            width=1500, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2 * n_col_ord[i], range=dd_TS_obj.x_range),
                                        width=0.2, top=col, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "hide"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        return dd_TS_obj
    def get_R1Y_HPR_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Legend
        try:
            source_R1Y_HPR = self.to_source(self.R1Y_HPR)
            R1Y_HPR_obj = figure(x_axis_type='datetime',
                                 title='Rolling Holding Period Return',
                                 width=1500, height=170, toolbar_location=toolbar_location)
            R1Y_HPR_lgd_list = []
            for i, col in enumerate(self.R1Y_HPR.columns):
                p_line = R1Y_HPR_obj.line(source=source_R1Y_HPR, x='date', y=col, color=self.color_list[i],
                                          line_width=2)
                R1Y_HPR_lgd_list.append((col, [p_line]))
            R1Y_HPR_lgd = Legend(items=R1Y_HPR_lgd_list, location='center')

            R1Y_HPR_obj.add_layout(R1Y_HPR_lgd, 'right')
            R1Y_HPR_obj.legend.click_policy = "hide"
            R1Y_HPR_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
            return R1Y_HPR_obj
        except:
            return None
    def get_monthly_rtn_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, \
            LabelSet
        from bokeh.transform import transform
        from bokeh.palettes import RdBu
        colors = RdBu[max(RdBu.keys())]

        # Plot Monthly Performance
        input_Data = self.monthly_return.copy()
        input_Data = input_Data.apply(lambda x: round(x * 100, 2))
        input_Data = input_Data.rename(columns={input_Data.columns[0]: 'value'})
        input_Data['Year'] = input_Data.index.strftime("%Y")
        input_Data['Month'] = input_Data.index.strftime("%m")
        input_Data = input_Data[['Year', 'Month', 'value']].reset_index(drop='index')

        years = sorted(list(input_Data['Year'].unique()), reverse=True)
        months = sorted(list(input_Data['Month'].unique()))

        source_for_heatmap = ColumnDataSource(input_Data)
        mapper = LinearColorMapper(palette=colors, low=-1, high=1)

        rtn_fig_obj = figure(
            title="Monthly Return",
            x_range=months, y_range=years,
            x_axis_location="above",
            width=750, height=400,
            toolbar_location=toolbar_location,
            tooltips=[('date', '@Year @Month'), ('value', f'@value%')]
        )

        rtn_fig_obj.rect(
            x="Month", y="Year", width=1, height=1,
            source=source_for_heatmap,
            fill_color=transform('value', mapper),
            line_color=None
        )

        color_bar = ColorBar(
            color_mapper=mapper, major_label_text_font_size="10px",
            ticker=BasicTicker(desired_num_ticks=len(colors)),
            formatter=PrintfTickFormatter(format="%d%%"),
        )

        rtn_fig_obj.add_layout(color_bar, 'right')
        rtn_fig_obj.grid.grid_line_color = None
        rtn_fig_obj.axis.axis_line_color = None
        rtn_fig_obj.axis.major_tick_line_color = None
        rtn_fig_obj.axis.major_label_text_font_size = "10px"
        rtn_fig_obj.axis.major_label_standoff = 0

        heatmap_annotation_data = input_Data.copy()
        heatmap_annotation_data['value'] = heatmap_annotation_data['value'].astype(str).apply(
            lambda x: x + '0' * (2 - len(x.split('.')[-1])) + "%")
        source_for_heatmap_annotation = ColumnDataSource(heatmap_annotation_data)
        labels = LabelSet(x='Month', y='Year', text='value', level='overlay',
                          text_font_size={'value': '10px'},
                          text_color='#000000',
                          text_align='center',
                          text_alpha=0.75,
                          x_offset=0, y_offset=-5, source=source_for_heatmap_annotation)
        rtn_fig_obj.add_layout(labels)

        return rtn_fig_obj
    def get_monthly_alpha_obj(self, toolbar_location):
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, \
            LabelSet
        from bokeh.transform import transform
        from bokeh.palettes import RdBu
        colors = RdBu[max(RdBu.keys())]

        # Plot Monthly Performance
        input_Data = self.monthly_alpha.copy()
        input_Data = input_Data.apply(lambda x: round(x * 100, 2))
        input_Data = input_Data.rename(columns={input_Data.columns[0]: 'value'})
        input_Data['Year'] = input_Data.index.strftime("%Y")
        input_Data['Month'] = input_Data.index.strftime("%m")
        input_Data = input_Data[['Year', 'Month', 'value']].reset_index(drop='index')

        years = sorted(list(input_Data['Year'].unique()), reverse=True)
        months = sorted(list(input_Data['Month'].unique()))

        source_for_heatmap = ColumnDataSource(input_Data)
        mapper = LinearColorMapper(palette=colors, low=-1, high=1)

        alpha_fig_obj = figure(
            title="Monthly Alpha",
            x_range=months, y_range=years,
            x_axis_location="above", width=750, height=400,
            toolbar_location=toolbar_location,
            tooltips=[('date', '@Year @Month'), ('value', f'@value%')]
        )

        alpha_fig_obj.rect(
            x="Month", y="Year", width=1, height=1,
            source=source_for_heatmap,
            fill_color=transform('value', mapper),
            line_color=None
        )

        color_bar = ColorBar(
            color_mapper=mapper, major_label_text_font_size="10px",
            ticker=BasicTicker(desired_num_ticks=len(colors)),
            formatter=PrintfTickFormatter(format="%d%%"),
        )

        alpha_fig_obj.add_layout(color_bar, 'right')
        alpha_fig_obj.grid.grid_line_color = None
        alpha_fig_obj.axis.axis_line_color = None
        alpha_fig_obj.axis.major_tick_line_color = None
        alpha_fig_obj.axis.major_label_text_font_size = "10px"
        alpha_fig_obj.axis.major_label_standoff = 0

        heatmap_annotation_data = input_Data.copy()
        heatmap_annotation_data['value'] = heatmap_annotation_data['value'].astype(str).apply(
            lambda x: x + '0' * (2 - len(x.split('.')[-1])) + "%")
        source_for_heatmap_annotation = ColumnDataSource(heatmap_annotation_data)
        labels = LabelSet(x='Month', y='Year', text='value', level='overlay',
                          text_font_size={'value': '10px'},
                          text_color='#000000',
                          text_align='center',
                          text_alpha=0.75,
                          x_offset=0, y_offset=-5, source=source_for_heatmap_annotation)
        alpha_fig_obj.add_layout(labels)

        return alpha_fig_obj
    def get_monthly_rtn_dist_obj(self, toolbar_location):
        from scipy import stats
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Span
        from decimal import Decimal
        monthly_data = self.monthly_return.copy()
        monthly_rtns = monthly_data.values.flatten()
        m, std = round(monthly_rtns.mean(), 2), round(monthly_rtns.std(), 2)
        skwnss, krts = round(stats.skew(monthly_rtns), 2), round(stats.kurtosis(monthly_rtns), 2)

        hist, edges = np.histogram(monthly_rtns, density=True, bins=50)
        estmt_pdf = stats.gaussian_kde(monthly_rtns).pdf(edges)

        dist_fig_obj = figure(
            title=f'Monthly Return Distribution  (mean={m}, std={std}, skewness={skwnss}, kurtosis={krts})',
            y_axis_label=r"Density", x_axis_label=r"Monthly Return",
            width=750, height=400,
            toolbar_location=toolbar_location)
        dist_fig_obj.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=self.color_list[0],
                          line_color="white", alpha=0.5)
        dist_fig_obj.line(edges, estmt_pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF", )
        dist_fig_obj.xaxis.formatter = NumeralTickFormatter(format='0.0 %')
        dist_fig_obj.legend.visible = False
        vrtc_line = Span(location=monthly_rtns.mean(),
                         dimension='height', line_color='gray',
                         line_dash='dashed', line_width=3)
        dist_fig_obj.add_layout(vrtc_line)

        return dist_fig_obj
    def get_monthly_alpha_dist_obj(self, toolbar_location):
        from scipy import stats
        from bokeh.plotting import figure
        from bokeh.models import NumeralTickFormatter, Span

        monthly_data = self.monthly_alpha.copy()
        monthly_rtns = monthly_data.values.flatten()
        m, std = round(monthly_rtns.mean(), 2), round(monthly_rtns.std(), 2)
        skwnss, krts = round(stats.skew(monthly_rtns), 2), round(stats.kurtosis(monthly_rtns), 2)

        hist, edges = np.histogram(monthly_rtns, density=True, bins=50)
        estmt_pdf = stats.gaussian_kde(monthly_rtns).pdf(edges)

        dist_fig_obj = figure(
            title=f'Monthly Alpha Distribution  (mean={m}, std={std}, skewness={skwnss}, kurtosis={krts})',
            y_axis_label=r"Density", x_axis_label=r"Monthly Alpha",
            width=750, height=400,
            toolbar_location=toolbar_location)
        dist_fig_obj.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=self.color_list[0],
                          line_color="white", alpha=0.5)
        dist_fig_obj.line(edges, estmt_pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF", )
        dist_fig_obj.xaxis.formatter = NumeralTickFormatter(format='0.0 %')
        dist_fig_obj.legend.visible = False
        vrtc_line = Span(location=monthly_rtns.mean(),
                         dimension='height', line_color='gray',
                         line_dash='dashed', line_width=3)
        dist_fig_obj.add_layout(vrtc_line)
        return dist_fig_obj
    def deciles_bar_color_list(self, bar_num, bench_num):
        # Spectral6 컬러 목록
        # #['#3288bd', '#99d594', '#e6f598', '#fee08b',, '#d53e4f']
        deciles_bar_color_list = []
        for i in range(0, bar_num):
            deciles_bar_color_list.append('#fc8d59')
        for i in range(0, bench_num):
            deciles_bar_color_list.append('#e6f598')
        return deciles_bar_color_list
    def get_CAGR_bar_obj(self):
        from scipy.stats import linregress
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, NumeralTickFormatter

        CAGR_values = self.cagr.values
        CAGR_index = self.cagr.index

        slp, itrct, rval, pval, stderr = linregress(range(1, len(CAGR_index)), CAGR_values[:-1])
        title_text = f'10분위 연환산 수익률(r_value:{round(rval, 2)}, p_value:{round(pval, 2)}, std_err:{round(stderr, 2)})'

        qun_sourse = ColumnDataSource(
            data=dict(분위=list(CAGR_index), CAGR=CAGR_values, color=self.deciles_bar_color_list(10, 1)))
        qun = figure(x_range=list(CAGR_index), height=440, title=title_text, width=390)
        qun.vbar(x='분위', top='CAGR', width=0.9, source=qun_sourse, color='color')
        qun.line(list(range(1, len(CAGR_index))), [x * slp + itrct for x in range(1, len(CAGR_index))], color='black',
                 line_width=2)
        qun.xgrid.grid_line_color = None
        qun.toolbar.logo = None
        qun.toolbar_location = None
        qun.yaxis[0].formatter = NumeralTickFormatter(format="0.00%")
        return qun


    def _array_to_df(self, arr):
        try:
            return pd.DataFrame(arr,
                                index=self.daily_return.index.values,
                                columns=self.daily_return.columns.values).rename_axis("date")
        except:
            return pd.DataFrame(arr,
                                index=self.daily_alpha.index.values,
                                columns=self.daily_alpha.columns.values).rename_axis("date")
    def get_num_year(self, num_years):
        num_years = len(num_years)
        if num_years == 2:
            # 기간이 1년 이상이면, 1년이란 길이의 기준은 데이터의 갯수로 한다.
            start_date = self.daily_return.index[0]
            end_date = start_date + pd.DateOffset(years=1)

            date_list = self.daily_return.loc[start_date:end_date].index
            num_days = len(date_list)

        elif num_years == 1:
            # 기간이 1년 미만이면, 1년이란 길이의 기준은 다음해까지의 영업일 기준으로 가상으로 확장시킨다.
            start_date = self.daily_return.index[0]
            end_date = self.daily_return.index[-1]
            end_date_ = start_date + pd.DateOffset(years=1)

            # 1년이란 기준의 날짜 수 정의
            date_list = pd.date_range(start=start_date, end=end_date_, freq=BDay())
            date_list2 = pd.date_range(start=start_date, end=end_date, freq=BDay())
            num_days = len(date_list) / len(date_list2) * len(self.daily_return.index)

        else:
            # 3년 이상이면, input된 데이터의 첫해와 마지막 해를 제외하고 한 해의 날짜수의 평균으로 한다.
            num_days = self.daily_return.groupby(pd.Grouper(freq='Y')).count().iloc[1:-1].mean()[0]
        return num_days
    def get_BM(self, BM_name):
        if BM_name.lower() == "kospi":
            BM = gdu.get_data.get_naver_close('KOSPI')
        elif (BM_name.lower() == "s&p500") | (BM_name.lower() == "snp500"):
            BM = gdu.get_data.get_data_yahoo_close('^GSPC').rename(columns={'^GSPC': 'S&P500'})
        elif (BM_name.lower() == "nasdaq"):
            BM = gdu.get_data.get_data_yahoo_close('^IXIC').rename(columns={'^IXIC': 'NASDAQ'})
        else:
            try:
                BM = gdu.get_data.get_naver_close(BM_name)
            except:
                BM = gdu.get_data.get_data_yahoo_close(BM_name)
        return BM
    def _calculate_key_rates(self, daily_returns, daily_alpha):
        # daily_returns, daily_alpha = self.daily_return.iloc[-252 * 3:].copy(),  self.daily_alpha.iloc[-252 * 3:].copy()
        cum_ret_cmpd = daily_returns.add(1).cumprod()

        cum_ret_cmpd.iloc[0] = 1
        num_years = self.get_num_year(daily_returns.index.year.unique())

        cagr = self._calculate_cagr(cum_ret_cmpd, num_years)
        std = self._calculate_std(daily_returns, num_years)
        sharpe = cagr / std
        sortino = cagr / self._calculate_downsiderisk(daily_returns, num_years)
        drawdown = self._calculate_dd(cum_ret_cmpd)
        average_drawdown = drawdown.mean()
        mdd = self._calculate_mdd(drawdown)

        cum_alpha_cmpd = daily_alpha.add(1).cumprod()
        alpha_cagr = self._calculate_cagr(cum_alpha_cmpd, num_years)
        alpha_std = self._calculate_std(daily_alpha, num_years)
        alpha_sharpe = alpha_cagr / alpha_std
        alpha_sortino = alpha_cagr / self._calculate_downsiderisk(daily_alpha, num_years)
        alpha_drawdown = self._calculate_dd(cum_alpha_cmpd)
        alpha_average_drawdown = alpha_drawdown.mean()
        alpha_mdd = self._calculate_mdd(alpha_drawdown)

        return cum_ret_cmpd, cagr, std, sharpe, sortino, average_drawdown, mdd, cum_alpha_cmpd, alpha_cagr, alpha_std, alpha_sharpe, alpha_sortino, alpha_average_drawdown, alpha_mdd
    def _calculate_dd(self, df):
        # df = self.cum_ret_cmpd.copy()
        # df = t_df.pct_change().copy()
        # df = self.cum_alpha_cmpd.copy()
        # df = cum_ret_cmpd.copy()
        max_list = df.iloc[0].values
        out_list = [np.array([0] * len(max_list))]

        for ix in range(1, len(df.index)):
            max_list = np.max([max_list, df.iloc[ix].values], axis=0)
            out_list.append((df.iloc[ix].values - max_list) / max_list)
        try:
            out = self._array_to_df(out_list)
        except:
            out = pd.DataFrame(out_list, index=df.index, columns=df.columns)

        return out

    @staticmethod
    def _calculate_cagr(df, num_days):
        return ((df.iloc[-1] / df.iloc[0]) ** (1 / len(df.index))) ** num_days - 1
    @staticmethod
    def _calculate_std(df, num_days):
        return df.std() * np.sqrt(num_days)
    @staticmethod
    def _calculate_mdd(df):
        return df.min()
    @staticmethod
    def _calculate_downsiderisk(df, num_days):
        return df.applymap(lambda x: 0 if x >= 0 else x).std() * np.sqrt(num_days)
    @staticmethod
    def _holding_period_return(df, num_days):
        Rolling_HPR_1Y = df.pct_change(int(num_days.round())).dropna()
        HPR_1Y_mean = Rolling_HPR_1Y.mean()
        HPR_1Y_max = Rolling_HPR_1Y.max()
        HPR_1Y_min = Rolling_HPR_1Y.min()
        Rolling_HPR_1Y_WR = (Rolling_HPR_1Y > 0).sum() / Rolling_HPR_1Y.shape[0]
        return Rolling_HPR_1Y, Rolling_HPR_1Y_WR

if __name__ == "__main__":
    from tqdm import tqdm
    B_w_pvt, P_w_pvt = pd.read_excel(f'ETF_test.xlsx', sheet_name='BM_pvt',index_col=0,parse_dates=[0]), pd.read_excel(f'ETF_test.xlsx', sheet_name='MyPort_pvt', index_col=0, parse_dates=[0])
    price_df = pd.DataFrame()
    for stock in tqdm(B_w_pvt.columns.to_list() + P_w_pvt.columns.to_list()):
        tmp = gdu.get_data.get_naver_close(stock[1:]).rename(columns=lambda x: stock)
        price_df = pd.concat([price_df, tmp], axis=1)
    gdu.data = price_df.copy()
    Asset_info = pd.read_excel(f'./ETF_test.xlsx', sheet_name='표1')[['단축코드', '기초시장분류', '기초자산분류']]
    Asset_info.columns = ['code', 'class1', 'class2']
    Asset_info['code'] = Asset_info['code'].astype(str).apply(lambda x: 'A'+'0'*(6-len(x))+x)
    Asset_info['class2'] = Asset_info['class1'] + Asset_info['class2']
    Asset_info = Asset_info.set_index('code')
    # gdu.data.columns
    # BM_w_df.columns
    pass