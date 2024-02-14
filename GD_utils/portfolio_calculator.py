import GD_utils as gdu
from GD_utils.return_calculator import BrinsonHoodBeebower_calculator,Modified_BrinsonFachler_calculator,BrinsonFachler_calculator
from GD_utils.general_utils import save_as_pd_parquet, read_pd_parquet

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from scipy import stats
from scipy.stats import linregress

from bokeh.transform import dodge,transform,cumsum
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, LabelSet
from bokeh.models import NumeralTickFormatter, Span, HoverTool, FactorRange, Legend, Column, Dodge, HTMLTemplateFormatter,Div
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.palettes import RdBu, Category20_20, Category20c, HighContrast3,Pastel1
from bokeh.plotting import figure, output_file, show, curdoc, save
from bokeh.layouts import column, row


class PortfolioAnalysis:
    def __init__(self, daily_return, outputname='./Unnamed', last_BM=False, BM_name='KOSPI', hover=True):
        self.hover = hover
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
        self.yearly_return = self.daily_return.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
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
        # self.color_list = ['#ec008e','#0086d4', '#361b6f',  '#8c98a0'] + list(Category20_20)
        self.color_list = ['#192036','#eaa88f', '#8c98a0'] + list(Category20_20)
        self.outputname = outputname

    def basic_report(self, simple=False, display = True, toolbar_location='above'):
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
        return_TS_obj.legend.click_policy = "mute"
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
        dd_TS_obj.legend.click_policy = "mute"
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
            R1Y_HPR_obj.legend.click_policy = "mute"
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
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return ColumnDataSource(df)
    def get_table_obj_3Y(self,_width=400, all_None=False):
        if all_None:
            cumrnt=cagr=std=sharpe=sortino=average_drawdown=mdd=alpha_cumrnt=alpha_cagr=alpha_std=alpha_sharpe=alpha_sortino=alpha_average_drawdown=alpha_mdd = np.nan
            static_data = pd.concat([pd.DataFrame([cumrnt, cagr, sharpe, sortino, std, mdd, average_drawdown, alpha_cumrnt, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown], columns=['0']),
                                    pd.DataFrame([cumrnt, cagr, sharpe, sortino, std, mdd, average_drawdown, alpha_cumrnt, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown], columns=['1'])
                                    ], axis=1).T

        else:
            cumrnt,cagr,std,sharpe,sortino,average_drawdown,mdd,alpha_cumrnt,alpha_cagr,alpha_std,alpha_sharpe,alpha_sortino,alpha_average_drawdown,alpha_mdd = self.key_rates_3Y
            static_data = pd.concat([cumrnt.iloc[-1] - 1, cagr, sharpe, sortino, std, mdd, average_drawdown, alpha_cumrnt.iloc[-1] - 1, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown], axis=1)

        static_data.columns = ['Compound Return(3Y)', 'CAGR(3Y)', 'Sharpe Ratio(3Y)', 'Sortino Ratio(3Y)', 'Standard Deviation(3Y)', 'MDD(3Y)', 'Average Drawdown(3Y)', 'Compound Alpha(3Y)', 'CAGR(Alpha,3Y)', 'IR(3Y)', 'Sortino Ratio(Alpha,3Y)', 'Tracking Error(3Y)', 'MDD(Alpha,3Y)', 'Average Drawdown(Alpha,3Y)']

        for col in static_data.columns:
            if col in ['Compound Return(3Y)','Compound Alpha(3Y)', 'CAGR(3Y)','CAGR(Alpha,3Y)', 'MDD(3Y)','MDD(Alpha,3Y)', 'Average Drawdown(3Y)','Average Drawdown(Alpha,3Y)', 'Standard Deviation(3Y)','Tracking Error(3Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: str(np.around((x * 100), decimals=2)) + "%" if ~np.isnan(x) else "")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4) if ~np.isnan(x) else "")

        # static_data[static_data.isnull()]=""
        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})
        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)
        # formatter = HTMLTemplateFormatter(template=f'<div style="font-family: {self.font}; font-size: 14px;"><%= value %></div>')
        # columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]
        # data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)


        return data_table_fig
    def get_table_obj_5Y(self,_width=400, all_None=False):
        if all_None:
            cumrnt=cagr=std=sharpe=sortino=average_drawdown=mdd=alpha_cumrnt=alpha_cagr=alpha_std=alpha_sharpe=alpha_sortino=alpha_average_drawdown=alpha_mdd=np.nan
            static_data = pd.concat([pd.DataFrame([cumrnt, cagr, sharpe, sortino, std, mdd, average_drawdown, alpha_cumrnt, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown],columns=['0']),
                                    pd.DataFrame([cumrnt, cagr, sharpe, sortino, std, mdd, average_drawdown, alpha_cumrnt, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown],columns=['1'])
                                    ], axis=1).T
        else:
            cumrnt,cagr,std,sharpe,sortino,average_drawdown,mdd,alpha_cumrnt,alpha_cagr,alpha_std,alpha_sharpe,alpha_sortino,alpha_average_drawdown,alpha_mdd = self.key_rates_5Y
            static_data = pd.concat([cumrnt.iloc[-1] - 1, cagr, sharpe, sortino, std, mdd, average_drawdown, alpha_cumrnt.iloc[-1] - 1, alpha_cagr, alpha_sharpe, alpha_sortino, alpha_std, alpha_mdd, alpha_average_drawdown], axis=1)

        static_data.columns = ['Compound Return(5Y)', 'CAGR(5Y)', 'Sharpe Ratio(5Y)', 'Sortino Ratio(5Y)', 'Standard Deviation(5Y)', 'MDD(5Y)', 'Average Drawdown(5Y)',
                               'Compound Alpha(5Y)', 'CAGR(Alpha,5Y)', 'IR(5Y)', 'Sortino Ratio(Alpha,5Y)', 'Tracking Error(5Y)', 'MDD(Alpha,5Y)', 'Average Drawdown(Alpha,5Y)']
        
        for col in static_data.columns:
            if col in ['Compound Return(5Y)','Compound Alpha(5Y)', 'CAGR(5Y)','CAGR(Alpha,5Y)', 'MDD(5Y)','MDD(Alpha,5Y)', 'Average Drawdown(5Y)','Average Drawdown(Alpha,5Y)', 'Standard Deviation(5Y)','Tracking Error(5Y)']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: str(np.around((x * 100), decimals=2)) + "%" if ~np.isnan(x) else "")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4) if ~np.isnan(x) else "")

        # static_data[static_data.isnull()]=""
        static_data = static_data.T.reset_index().rename(columns={'index': 'Portfolio'})

        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)
        # formatter = HTMLTemplateFormatter(template=f'<div style="font-family: {self.font}; font-size: 14px;"><%= value %></div>')
        # columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]
        # data_table_fig = DataTable(source=source, columns=columns, width=_width, height=500, index_position=None)

        return data_table_fig
    def get_table_obj(self, _width=400):
        try:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd, self.average_drawdown, self.R1Y_HPR_WR], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation', 'MDD', 'Average Drawdown', 'HPR(1Y)']
        except:
            static_data = pd.concat(
                [self.cum_ret_cmpd.iloc[-1] - 1, self.cagr, self.sharpe, self.sortino, self.std, self.mdd, self.average_drawdown], axis=1)
            static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Standard Deviation', 'MDD', 'Average Drawdown']
        for col in static_data.columns:
            if col in ['Compound Return', 'CAGR', 'MDD', 'Average Drawdown', 'Standard Deviation', 'HPR(1Y)']:
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

        # formatter = HTMLTemplateFormatter(template=f'<div style="font-family: {self.font}; font-size: 14px;"><%= value %></div>')
        # columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]
        # data_table_fig = DataTable(source=source, columns=columns, width=_width, height=300, index_position=None)

        return data_table_fig
    def get_inputtable_obj(self, input_tbl):
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
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        return return_TS_obj
    def get_cmpd_rtn_obj(self, toolbar_location):
        # Plot 복리
        source_for_chart = self.to_source(self.cum_ret_cmpd - 1)
        return_TS_obj = figure(x_axis_type='datetime',
                               title='Cumulative Return' + f'({self.cum_ret_cmpd.index[0].strftime("%Y-%m-%d")} ~ {self.cum_ret_cmpd.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=450, toolbar_location=toolbar_location)

        return_TS_lgd_list = []
        for i, col in enumerate(self.cum_ret_cmpd.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col, color=self.color_list[i], line_width=2)
            return_TS_lgd_list.append((col, [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            V1, V2 = self.cum_ret_cmpd.columns[0],self.cum_ret_cmpd.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"),(f"{V2}", f"@{{{V2}}}{{0.2f}}")], formatters={"@date": "datetime"})
            return_TS_obj.add_tools(hover)
        return return_TS_obj
    def get_logscale_rtn_obj(self, toolbar_location):
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
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            V1, V2 = self.cum_ret_cmpd.columns[0],self.cum_ret_cmpd.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"),(f"{V2}", f"@{{{V2}}}{{0.2f}}")], formatters={"@date": "datetime"})
            return_TS_obj.add_tools(hover)
        return return_TS_obj
    def get_dd_obj(self, toolbar_location):
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
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            V1, V2 = self.drawdown.columns[0],self.drawdown.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"),(f"{V2}", f"@{{{V2}}}{{0.2f}}")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj
    def get_rollingCAGR_obj(self, toolbar_location):
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
        RllnCAGR_TS_obj.legend.click_policy = "mute"
        RllnCAGR_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0.0 %')

        if self.hover:
            V1, V2 = self.rolling_CAGR_6M.columns[0],self.rolling_CAGR_6M.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"),(f"{V2}", f"@{{{V2}}}{{0.2f}}")], formatters={"@date": "datetime"})
            RllnCAGR_TS_obj.add_tools(hover)
        return RllnCAGR_TS_obj
    def get_rollingstd_obj(self, toolbar_location):
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
        Rllnstd_TS_obj.legend.click_policy = "mute"
        Rllnstd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0.0 %')
        if self.hover:
            V1, V2 = self.rolling_std_6M.columns[0],self.rolling_std_6M.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"),(f"{V2}", f"@{{{V2}}}{{0.2f}}")], formatters={"@date": "datetime"})
            Rllnstd_TS_obj.add_tools(hover)
        return Rllnstd_TS_obj
    def get_rollingSharpe_obj(self, toolbar_location):
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
        Rllnshrp_TS_obj.legend.click_policy = "mute"

        if self.hover:
            V1, V2 = self.rolling_sharpe_6M.columns[0],self.rolling_sharpe_6M.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"),(f"{V2}", f"@{{{V2}}}{{0.2f}}")], formatters={"@date": "datetime"})
            Rllnshrp_TS_obj.add_tools(hover)
        return Rllnshrp_TS_obj
    def get_yearly_rtn_obj(self, toolbar_location, W=1):
        # Plot Yearly Performance
        input_Data = self.yearly_return.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Return',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i],range=dd_TS_obj.x_range),width=0.2,top=col,color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = self.yearly_return.columns[0], self.yearly_return.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date"), (f"{V1}", f"@{{{V1}}}{{0.00%}}"), (f"{V2}", f"@{{{V2}}}{{0.00%}}")]) # formatters={"@date": "datetime"}
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj
    def get_yearly_alpha_obj(self, toolbar_location, W=1):
        # Plot Yearly Performance
        input_Data = self.yearly_alpha.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Alpha',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range),  width=0.2 ,top=col, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1 = self.yearly_alpha.columns[0]
            hover = HoverTool(tooltips=[("Date", "@date"), (f"{V1}", f"@{{{V1}}}{{0.00%}}")]) #formatters={"@date": "datetime"}
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj
    def get_R1Y_HPR_obj(self, toolbar_location):
        try:
            source_R1Y_HPR = self.to_source(self.R1Y_HPR)
            R1Y_HPR_obj = figure(x_axis_type='datetime',
                                 title='Rolling Holding Period Return',
                                 width=1500, height=170, toolbar_location=toolbar_location)
            hover = HoverTool(tooltips=[("Date", "@date{%F}"), ("Value", "@value{0.2f}")], formatters={"@date": "datetime"})
            R1Y_HPR_obj.add_tools(hover)
            R1Y_HPR_lgd_list = []
            for i, col in enumerate(self.R1Y_HPR.columns):
                p_line = R1Y_HPR_obj.line(source=source_R1Y_HPR, x='date', y=col, color=self.color_list[i],
                                          line_width=2)
                R1Y_HPR_lgd_list.append((col, [p_line]))
            R1Y_HPR_lgd = Legend(items=R1Y_HPR_lgd_list, location='center')

            R1Y_HPR_obj.add_layout(R1Y_HPR_lgd, 'right')
            R1Y_HPR_obj.legend.click_policy = "mute"
            R1Y_HPR_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

            if self.hover:
                V1, V2 = self.R1Y_HPR.columns[0], self.R1Y_HPR.columns[1]
                hover = HoverTool(
                    tooltips=[("Date", "@date{%F}"), (f"{V1}", f"@{{{V1}}}{{0.2f}}"), (f"{V2}", f"@{{{V2}}}{{0.2f}}")],
                    formatters={"@date": "datetime"})
                R1Y_HPR_obj.add_tools(hover)
            return R1Y_HPR_obj
        except:
            return None

    def get_monthly_rtn_obj(self, toolbar_location):
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
                             formatter=PrintfTickFormatter(format="%0.2f%%"),
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
            formatter=PrintfTickFormatter(format="%0.2f%%"),
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
        monthly_data = self.monthly_return.copy()
        monthly_rtns = monthly_data.values.flatten()
        m,std = round(monthly_rtns.mean(),2), round(monthly_rtns.std(),2)
        skwnss, krts = round(stats.skew(monthly_rtns),2), round(stats.kurtosis(monthly_rtns), 2)

        hist, edges = np.histogram(monthly_rtns, density=True, bins=50)
        estmt_pdf = stats.gaussian_kde(monthly_rtns).pdf(edges)

        dist_fig_obj = figure(title=f'Monthly Return Distribution  (mean={m}, std={std}, skewness={skwnss}, kurtosis={krts})', y_axis_label=r"Density", x_axis_label=r"Monthly Return",
                              width=750, height=400,
                              toolbar_location=toolbar_location)

        source = ColumnDataSource(data=dict(top_True=hist,
                                            left_True=[l for l in edges[:-1]],
                                            right_True=[r for r in edges[1:]],
                                            pdf_True=estmt_pdf[:-1],
                                            top=hist,
                                            left=[l*100 for l in edges[:-1]],
                                            right=[r*100 for r in edges[1:]],
                                            pdf=[est for est in estmt_pdf[:-1]]
                                            ))
        # dist_fig_obj.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=self.color_list[0], line_color="white", alpha=0.5)
        dist_fig_obj.quad(source=source, top='top'+"_True", bottom=0, left='left_True', right='right_True', fill_color=self.color_list[0], line_color="white", alpha=0.5)

        # dist_fig_obj.line(edges, estmt_pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
        dist_fig_obj.line('left_True', 'pdf_True', source=source, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")

        dist_fig_obj.xaxis.formatter = NumeralTickFormatter(format='0.0 %')
        dist_fig_obj.legend.visible=False
        vrtc_line = Span(location=monthly_rtns.mean(),
                                    dimension='height', line_color='gray',
                                    line_dash='dashed', line_width=3)
        # HoverTool 추가
        if self.hover:
            hover = HoverTool(tooltips=[
                                    ("Density", "@top{0.00}%"),
                                    ("Range", "@left{0.00}% ~ @right{0.00}%"),
                                    ("pdf", "@pdf{0.00}%")  # PDF 데이터 표시
                                    ])
            dist_fig_obj.add_tools(hover)

        dist_fig_obj.add_layout(vrtc_line)
        return dist_fig_obj
    def get_monthly_alpha_dist_obj(self,toolbar_location):
        monthly_data = self.monthly_alpha.copy()
        monthly_rtns = monthly_data.values.flatten()
        m, std = round(monthly_rtns.mean(), 2), round(monthly_rtns.std(), 2)
        skwnss, krts = round(stats.skew(monthly_rtns), 2), round(stats.kurtosis(monthly_rtns), 2)

        hist, edges = np.histogram(monthly_rtns, density=True, bins=50)
        estmt_pdf = stats.gaussian_kde(monthly_rtns).pdf(edges)

        dist_fig_obj = figure(title=f'Monthly Alpha Distribution  (mean={m}, std={std}, skewness={skwnss}, kurtosis={krts})', y_axis_label=r"Density", x_axis_label=r"Monthly Alpha",
                              width=750, height=400,
                              toolbar_location=toolbar_location)
        # 2024-01-24: hover기능 추가 ##################
        # Hist가gram 데이터를 ColumnDataSource로 변환
        # dist_fig_obj.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=self.color_list[0], line_color="white", alpha=0.5)
        source = ColumnDataSource(data=dict(top_True=hist,
                                            left_True=[l for l in edges[:-1]],
                                            right_True=[r for r in edges[1:]],
                                            pdf_True=estmt_pdf[:-1],
                                            top=hist,
                                            left=[l * 100 for l in edges[:-1]],
                                            right=[r * 100 for r in edges[1:]],
                                            pdf=[est for est in estmt_pdf[:-1]]
                                            ))
        # dist_fig_obj.quad(source=source, top='top', bottom='bottom', left='left', right='right', fill_color=self.color_list[0], line_color="white", alpha=0.5)
        dist_fig_obj.quad(source=source, top='top_True', bottom=0, left='left_True', right='right_True', fill_color=self.color_list[0], line_color="white", alpha=0.5)


        # dist_fig_obj.line(edges, estmt_pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
        dist_fig_obj.line('left_True', 'pdf_True', source=source, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
        dist_fig_obj.xaxis.formatter = NumeralTickFormatter(format='0.0 %')
        dist_fig_obj.legend.visible=False
        vrtc_line = Span(location=monthly_rtns.mean(),
                                    dimension='height', line_color='gray',
                                    line_dash='dashed', line_width=3)
        dist_fig_obj.add_layout(vrtc_line)

        # HoverTool 추가
        if self.hover:
            hover = HoverTool(tooltips=[
                                    ("Density", "@top{0.00}%"),
                                    ("Range", "@left{0.00}% ~ @right{0.00}%"),
                                    ("PDF", "@pdf{0.00}%")  # PDF 데이터 표시
                                    ])
            dist_fig_obj.add_tools(hover)
        #############################################
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
class BrinsonFachler_PortfolioAnalysis(PortfolioAnalysis):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input, cost=0.00, n_day_after=0, BM_nm='BM', Port_nm='My Portfolio', outputname='./Unnamed', hover=True, yearly=True):
        """
        P_w_pvt_input = Portfolio의 weight를 담은 DataFrame(index='(rebalancing)date', columns='code', values='weight')
        B_w_pvt_input = Benchmark의 weight를 담은 DataFrame(index='(rebalancing)date', columns='code', values='weight')
        """
        self.code_to_name = Asset_info_input.set_index('종목코드')[['종목명','class']]

        Asset_info_input = Asset_info_input.set_index('종목코드')['class']
        BF_clac = BrinsonFachler_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input, cost, n_day_after)
        Port_p_df = BF_clac.Port_cls.portfolio_cumulative_return
        BM_p_df = BF_clac.Bench_cls.portfolio_cumulative_return

        self.allocation_effect = BF_clac.allocation_effect
        self.selection_effect = BF_clac.selection_effect
        self.interaction_effect = BF_clac.interaction_effect
        self.alpha = BF_clac.rP.sub(BF_clac.rB)
        self.decompose_allocation_effect = pd.concat([self.alpha.rename('alpha'),
                                                      self.allocation_effect.rename('Allocation Effect'),
                                                      self.selection_effect.rename('Selection Effect'),
                                                      self.interaction_effect.rename('Interaction Effect')], axis=1).dropna(how='all', axis=0)
        self.decompose_allocation_effect_BM_Port = self.decompose_allocation_effect.assign(BM=BF_clac.rB, Port=BF_clac.rP).dropna(how='all', axis=0)
        self.Port_portfolio_turnover_ratio = BF_clac.Port_cls.portfolio_turnover_ratio
        self.Port_stockwise_turnover_ratio = BF_clac.Port_cls.stockwise_turnover_ratio
        self.Port_stockwise_period_return_contribution = BF_clac.Port_cls.stockwise_period_return_contribution
        self.Port_daily_account_ratio = BF_clac.Port_cls.daily_account_ratio
        self.Port_daily_account_ratio_wrt_class = self.Port_daily_account_ratio.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        self.Port_daily_account_ratio_wrt_class=self.Port_daily_account_ratio_wrt_class[self.Port_daily_account_ratio_wrt_class.mean().sort_values(ascending=False).index].fillna(0)

        self.Port_rebal_date_class = BF_clac.P_classweight_pvt.copy()
        self.Port_daily_account_ratio_class_mean = self.Port_rebal_date_class.mean()
        self.Port_latest_rebalancing = pd.concat([BF_clac.Port_cls.ratio_df.iloc[-1].rename('today'), BF_clac.Port_cls.ratio_df.iloc[-2].rename('previous')], axis=1).dropna(how='all', axis=0)
        self.latest_rebal_date=BF_clac.Port_cls.ratio_df.index[-1]
        latest_return_contribution=self.Port_stockwise_period_return_contribution.loc[self.latest_rebal_date].dropna().rename('return contribution')
        self.Port_latest_rebalancing=pd.concat([self.Port_latest_rebalancing, pd.DataFrame(self.code_to_name).loc[self.Port_latest_rebalancing.index], latest_return_contribution], axis=1)
        self.Port_latest_rebalancing['delta'] = self.Port_latest_rebalancing['today'].sub(self.Port_latest_rebalancing['previous'])
        self.Port_latest_rebalancing = self.Port_latest_rebalancing.sort_values(by=['today','class','종목명'], ascending=[False,True,True])
        self.latest_decompose_allocation_effect_BM_Port = self.decompose_allocation_effect_BM_Port.loc[self.latest_rebal_date]

        try:
            self.stacked_line_color = Category20c[len(self.Port_daily_account_ratio_wrt_class.columns)]
        except:
            from bokeh.palettes import Category20b_20, Category20c_20
            self.stacked_line_color = Category20b_20+Category20c_20

        self.stacked_line_color_dict = dict(zip(self.Port_daily_account_ratio_wrt_class.columns,self.stacked_line_color))

        self.Bench_portfolio_turnover_ratio = BF_clac.Bench_cls.portfolio_turnover_ratio
        # self.Bench_stockwise_turnover_ratio = BF_clac.Bench_cls.stockwise_turnover_ratio
        # self.Bench_stockwise_period_return_contribution = BF_clac.Bench_cls.stockwise_period_return_contribution
        # self.Bench_daily_account_ratio = BF_clac.Bench_cls.daily_account_ratio
        # self.Bench_daily_account_ratio_wrt_class = self.Bench_daily_account_ratio.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()


        self.hover=hover
        self.yearly = yearly
        self.Port_nm,self.BM_nm = Port_nm, BM_nm

        # 포트폴리오 일별 수익률
        daily_return = pd.concat([Port_p_df.rename(Port_nm),BM_p_df.rename(BM_nm)], axis=1).pct_change()
        daily_return.iloc[0]=0
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
        self.yearly_return = self.daily_return.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
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
        # Shinhan Blue
        self.color_list = ['#0046ff','#8C98A0'] + list(Category20_20)

        # self.color_list = ['#192036','#eaa88f', '#8c98a0'] + list(Category20_20)
        self.outputname = outputname
    def BrinsonFachler_report(self, display = True, toolbar_location='above'):
        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj()
        data_alpha_table_obj = self.get_alpha_table_obj()
        try:
            data_table_obj_3Y = self.get_table_obj_3Y()
        except:
            data_table_obj_3Y = self.get_table_obj_3Y(all_None=True)

        try:
            data_table_obj_5Y = self.get_table_obj_5Y()
        except:
            data_table_obj_5Y = self.get_table_obj_5Y(all_None=True)
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location, W=2)
        Yearly_tottr_obj = self.get_yearly_tottr_obj(toolbar_location, W=2)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location, W=2)
        Yearly_avgtr_obj = self.get_yearly_avgtr_obj(toolbar_location, W=2)

        Monthly_rtn_obj = self.get_monthly_rtn_obj(toolbar_location)
        Monthly_alpha_obj = self.get_monthly_alpha_obj(toolbar_location)
        Monthly_rtn_dist_obj = self.get_monthly_rtn_dist_obj(toolbar_location)
        Monthly_alpha_dist_obj = self.get_monthly_alpha_dist_obj(toolbar_location)

        RllnCAGR_obj = self.get_rollingCAGR_obj(toolbar_location)
        Rllnstd_obj = self.get_rollingstd_obj(toolbar_location)
        Rllnshrp_obj = self.get_rollingSharpe_obj(toolbar_location)

        stacked_line_obj = self.get_class_stacked_line_obj(toolbar_location)
        mean_donut_obj = self.get_class_holding_mean_donut_obj(toolbar_location)
        BrinsonFachler_obj = self.BrinsonFachler_obj(toolbar_location, self.yearly)
        latest_rebalancing_tbl_obj = self.get_latest_rebalancing_tbl_obj()
        latest_rebalancing_donut_obj = self.get_latest_rebalancing_donut_obj(toolbar_location)


        report_title = Div(
            text="""
            <div style=font-size: 13px; color: #333333;">
                <h1>포트폴리오 성과 분석 리포트</h1>
            </div>
            """,
            width=800,
            height=80
        )
        if display == True:
            try:
                show(
                    column(
                           report_title,
                           row(column(
                                      Column(data_table_obj),
                                      Column(data_alpha_table_obj),
                                      Column(data_table_obj_3Y),
                                      Column(data_table_obj_5Y),
                                      ),
                               column(
                                      cmpd_return_TS_obj,
                                      logscale_return_TS_obj,
                                      dd_TS_obj, R1Y_HPR_obj,
                                      row(Yearly_rtn_obj, Yearly_tottr_obj),
                                      row(Yearly_alpha_obj, Yearly_avgtr_obj),
                                      row(Monthly_rtn_obj, Monthly_alpha_obj),
                                      row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                                      RllnCAGR_obj,
                                      Rllnstd_obj,
                                      Rllnshrp_obj,
                                      ),
                               column(
                                      BrinsonFachler_obj,
                                      row(stacked_line_obj,mean_donut_obj),
                                      row(latest_rebalancing_tbl_obj,latest_rebalancing_donut_obj),
                                      )
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



    def get_yearly_tottr_obj(self, toolbar_location, W=1):
        # Plot Yearly Turnover Ratio
        P_rebal_tr, B_rebal_tr = self.Port_portfolio_turnover_ratio.copy(), self.Bench_portfolio_turnover_ratio.copy()
        input_Data_ = pd.concat([P_rebal_tr.rename(f'{self.Port_nm}'), B_rebal_tr.rename(f'{self.BM_nm}')], axis=1).rename_axis('date')

        Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).sum()
        # Year_mean_tr = input_Data_.groupby(pd.Grouper(freq='Y')).mean().add_suffix(' Year Avg TR')
        # Year_tr = pd.concat([Year_tot_tr, Year_mean_tr],axis=1)

        Year_tot_tr.index = Year_tot_tr.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=Year_tot_tr.index.to_list(),
            title='Yearly Turnonver Ratio',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(Year_tot_tr.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(pd.concat([Year_tot_tr.add_suffix('_True'), Year_tot_tr.mul(100)], axis=1))
        for i, col in enumerate(Year_tot_tr.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, top=col+"_True",x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range), width=0.2, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')
        # show(dd_TS_obj)
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = Year_tot_tr.columns[0],Year_tot_tr.columns[1]
            hover = HoverTool(tooltips=[("Year", "@date"), (f"{V1} Turnover", f"@{{{V1}}}{{0.00}}%"), (f"{V2} Turnover", f"@{{{V2}}}{{0.00}}%")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj
    def get_yearly_avgtr_obj(self, toolbar_location, W=1):
        # Plot Yearly Turnover Ratio
        P_rebal_tr, B_rebal_tr = self.Port_portfolio_turnover_ratio.copy(), self.Bench_portfolio_turnover_ratio.copy()
        input_Data_ = pd.concat([P_rebal_tr.rename(f'{self.Port_nm}'), B_rebal_tr.rename(f'{self.BM_nm}')], axis=1).rename_axis('date')

        # Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).sum()
        Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).mean()
        # Year_tr = pd.concat([Year_tot_tr, Year_mean_tr],axis=1)

        Year_tot_tr.index = Year_tot_tr.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=Year_tot_tr.index.to_list(),
            title='Yearly Average Turnonver Ratio',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(Year_tot_tr.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(pd.concat([Year_tot_tr.add_suffix('_True'), Year_tot_tr.mul(100)], axis=1))
        for i, col in enumerate(Year_tot_tr.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, top=col+"_True",x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range), width=0.2, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')
        # show(dd_TS_obj)
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = Year_tot_tr.columns[0],Year_tot_tr.columns[1]
            hover = HoverTool(tooltips=[("Year", "@date"), (f"{V1} Turnover", f"@{{{V1}}}{{0.00}}%"), (f"{V2} Turnover", f"@{{{V2}}}{{0.00}}%")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj

    # 2024-01-25: 업데이트
    def BrinsonFachler_obj(self, toolbar_location, Yearly=True):
        input_Data=self.decompose_allocation_effect

        if Yearly:
            input_Data = input_Data.groupby(pd.Grouper(freq='Y')).mean()
            input_Data.index = input_Data.index.strftime("%Y")
        else:
            input_Data = input_Data.groupby(pd.Grouper(freq='M')).mean()
            input_Data.index = input_Data.index.strftime("%Y-%m")

        cr_list = ['#D5DBDB']+list(HighContrast3)

        BF_obj = figure(x_range=FactorRange(*input_Data.index), title="Brinson-Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)
        BF_obj.title.text_font_size = '13pt'
        # BF_obj = figure(x_range=input_Data.index.to_list(), title="Brinson Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)

        BF_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)

        # alpha 막대 너비
        alpha_width = 0.6

        # alpha 막대 그리기
        # 나머지 막대 너비 및 dodge 값
        other_width = alpha_width/3
        dodge_val = alpha_width/3

        # 나머지 막대 그리기
        for i, col in enumerate(input_Data.columns):
            if i == 0:
                BF_line = BF_obj.vbar(x='date', top=col, source=source_TS, width=alpha_width, color=cr_list[i], alpha=0.8)
            else:
                dodge = Dodge(value=(i - 1-1) * dodge_val, range=BF_obj.x_range)
                BF_line=BF_obj.vbar(x={'field': 'date', 'transform': dodge}, top=col, source=source_TS, width=other_width, color=cr_list[i], alpha=0.8)
            BF_lgd_list.append((col, [BF_line]))

        if self.hover:
            tooltips = [("Date", "@date")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in input_Data.columns]
            # hover = HoverTool(tooltips=tooltips)#, formatters={"@date": "datetime"})
            # hover = HoverTool(renderers=[BF_obj.renderers[-3], BF_obj.renderers[-2], BF_obj.renderers[-1]], tooltips=tooltips)
            hover = HoverTool(renderers=[BF_obj.renderers[0]], tooltips=tooltips)
            BF_obj.add_tools(hover)

        BF_obj.x_range.range_padding = 0.05
        BF_obj.xgrid.grid_line_color = None
        BF_lgd = Legend(items=BF_lgd_list, location='center')
        BF_obj.add_layout(BF_lgd, 'right')
        BF_obj.legend.click_policy = "mute"
        BF_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # show(BF_obj)




        return BF_obj
    def get_yearly_rtn_obj__(self, toolbar_location, W=1):
        # Plot Yearly Performance
        input_Data = self.yearly_return.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Return',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i],range=dd_TS_obj.x_range),width=0.2,top=col,color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = self.yearly_return.columns[0], self.yearly_return.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date"), (f"{V1}", f"@{{{V1}}}{{0.00%}}"), (f"{V2}", f"@{{{V2}}}{{0.00%}}")]) # formatters={"@date": "datetime"}
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj

    def get_class_stacked_line_obj(self, toolbar_location):
        staked_=self.Port_daily_account_ratio_wrt_class.copy()
        source_for_chart=ColumnDataSource(pd.concat([staked_, staked_.mul(100).add_suffix('_True')], axis=1))

        return_TS_obj = figure(x_axis_type="datetime",
                               title="Class-wise Daily Account Ratio",
                               width=1000, height=500, toolbar_location=toolbar_location)
        return_TS_obj.title.text_font_size = '13pt'

        renderers = return_TS_obj.varea_stack(stackers=staked_.columns.tolist(),
                                              x='date',
                                              source=source_for_chart,
                                              color=self.stacked_line_color)
        legend_items = [(col, [rend]) for col, rend in zip(staked_.columns, renderers)]
        legend = Legend(items=legend_items, location='center')
        return_TS_obj.add_layout(legend, 'right')
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            # HoverTool 설정을 위한 데이터 필드 목록 생성
            tooltips = [("Date", "@date{%F}")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in staked_.columns]

            hover = HoverTool(
                tooltips=tooltips,
                formatters={"@date": "datetime"}
            )
            return_TS_obj.add_tools(hover)
        # show(return_TS_obj)
        return return_TS_obj
    def get_class_holding_mean_donut_obj(self, toolbar_location):
        dounut_value = self.Port_daily_account_ratio_class_mean.copy()

        # 데이터 준비
        dounut_data = pd.Series(dounut_value).rename_axis('class').reset_index(name='value')
        dounut_data['angle'] = dounut_data['value'].div(dounut_data['value'].sum()) * 2*np.pi
        dounut_data['color'] = dounut_data['class'].map(self.stacked_line_color_dict)

        source = ColumnDataSource(dounut_data)

        ClsMean_DN_obj = figure(height=500, title="Class Holding Mean", toolbar_location=None,
                                tools="hover", tooltips="@class: @value{0.00%}", x_range=(-0.5, 1.0))
        ClsMean_DN_obj.title.text_font_size = '13pt'

        # 원형 도넛 차트 추가
        ClsMean_DN_obj.annular_wedge(x=0, y=1, outer_radius=0.4, inner_radius=0.2,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                             line_color="white", fill_color='color', source=source) # legend_field='class'

        ClsMean_DN_obj.axis.axis_label = None
        ClsMean_DN_obj.axis.visible = False
        ClsMean_DN_obj.grid.grid_line_color = None
        ClsMean_DN_obj.outline_line_color = None
        # ClsMean_DN_obj.legend.location = "center_left"  # 범례 위치 중앙으로 설정
        # ClsMean_DN_obj.legend.visible = False  # 범례 위치 중앙으로 설정
        # ClsMean_DN_obj.legend.border_line_color = None  # 범례 테두리 제거
        # show(ClsMean_DN_obj)

        return ClsMean_DN_obj
    def get_latest_rebalancing_donut_obj(self, toolbar_location):
        data_tmp = self.Port_latest_rebalancing.copy()
        dounut_value = data_tmp[['class', 'today']].groupby('class')['today'].sum().rename('value')
        dounut_data = pd.Series(dounut_value).rename_axis('class').reset_index(name='value')
        dounut_data['angle'] = dounut_data['value'].div(dounut_data['value'].sum()) * 2 * np.pi
        dounut_data['color'] = dounut_data['class'].map(self.stacked_line_color_dict)

        ClsMean_DN_obj = figure(height=500, title="Latest Class Holding", toolbar_location=None, x_range=(-0.5, 1.0))
        ClsMean_DN_obj.title.text_font_size = '13pt'


        start_angle = 0
        for idx, row in dounut_data.iterrows():
            end_angle = start_angle + row['angle']
            source = ColumnDataSource(dict(start_angle=[start_angle], end_angle=[end_angle], color=[row['color']], class_name=[row['class']], value=[row['value']]))
            wedge = ClsMean_DN_obj.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.4,
                                                 start_angle='start_angle', end_angle='end_angle',
                                                 color='color', legend_label=row['class'],
                                                 muted_color='grey', muted_alpha=0.2, source=source)
            start_angle = end_angle

        # Hover 툴 설정
        hover = HoverTool(tooltips=[("Class", "@class_name"), ("weight", "@value{0.00%}")])
        ClsMean_DN_obj.add_tools(hover)

        ClsMean_DN_obj.axis.axis_label = None
        ClsMean_DN_obj.axis.visible = False
        ClsMean_DN_obj.grid.grid_line_color = None
        ClsMean_DN_obj.outline_line_color = None

        ClsMean_DN_obj.legend.location = "center_right"
        ClsMean_DN_obj.legend.border_line_color = None
        ClsMean_DN_obj.legend.click_policy = "mute"
        # show(ClsMean_DN_obj)

        return ClsMean_DN_obj
    def get_latest_rebalancing_tbl_obj(self):
        static_data_tmp = self.Port_latest_rebalancing.reset_index().fillna(0)
        contributions = static_data_tmp['return contribution'].values
        static_data_tmp['today'] = static_data_tmp['today'].map(lambda x: str(np.int64(x*10000)/100)+"%")
        static_data_tmp['previous'] = static_data_tmp['previous'].map(lambda x: str(int(x*10000)/100)+"%")
        static_data_tmp['delta'] = static_data_tmp['delta'].map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')
        static_data_tmp.loc['sum']=""
        static_data_tmp.loc['sum', 'return contribution'] = sum(contributions)

        static_data_tmp['return contribution'] = static_data_tmp['return contribution'].map(lambda x: str(int(x*10000)/100)+"%")

        decomp_df=self.latest_decompose_allocation_effect_BM_Port.loc[['alpha','Port', 'BM', 'Allocation Effect','Selection Effect','Interaction Effect']]#.rename(index={'Port':self.Port_nm, "BM":self.BM_nm})
        decomp_df=decomp_df.map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')
        static_data_tmp=pd.concat([static_data_tmp, pd.DataFrame(decomp_df.rename('previous_performance'))], axis=0)
        static_data_tmp.loc[['alpha','Port', 'BM', 'Allocation Effect','Selection Effect','Interaction Effect'],'delta'] = ['alpha',self.Port_nm, self.BM_nm, 'Allocation Effect','Selection Effect','Interaction Effect']


        static_data_tmp[static_data_tmp.isnull()] = ""
        # static_data_tmp = static_data_tmp.reset_index()
        static_data=static_data_tmp[['code', '종목명', 'return contribution','previous', 'today', 'delta','previous_performance']].rename(
                            columns={'code':'종목코드',
                                     'today':'최근리밸런싱',
                                     'previous':'직전리밸런싱',
                                     'delta':'변화',
                                     'return contribution':'수익률 기여도',
                                     'previous_performance':'직전리밸런싱 성과'
                                     })

        source = ColumnDataSource(static_data)

        # columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        # 폰트 크기를 조정하기 위한 HTML 템플릿 포맷터 생성
        formatter = HTMLTemplateFormatter(template='<div style="font-size: 16px;"><%= value %></div>')  # 수정됨

        # 각 컬럼에 HTML 템플릿 포맷터 적용
        columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]  # 수정됨

        data_table_fig = DataTable(source=source, columns=columns, width=1000, height=750, index_position=None)

        # 제목을 위한 Div 위젯 생성
        title_div = Div(text=f"<h2>최근 리밸런싱 내역: {self.latest_rebal_date.strftime('%Y-%m-%d')}</h2>", width=1000, height=30)


        # Div와 DataTable을 column 레이아웃으로 결합
        layout = column(title_div, data_table_fig)

        # show(layout)
        return layout
class Modified_BrinsonFachler_PortfolioAnalysis(PortfolioAnalysis):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input,Index_Daily_price_input, cost=0.00, n_day_after=0, BM_nm='BM', Port_nm='My Portfolio', outputname='./Unnamed', hover=True, yearly=True):
        """
        P_w_pvt_input = Portfolio의 weight를 담은 DataFrame(index='(rebalancing)date', columns='code', values='weight')
        B_w_pvt_input = Benchmark의 weight를 담은 DataFrame(index='(rebalancing)date', columns='code', values='weight')
        """
        self.code_to_name = Asset_info_input.set_index('종목코드')[['종목명','class']]

        Asset_info_input = Asset_info_input.set_index('종목코드')['class']
        BF_clac = Modified_BrinsonFachler_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input, Index_Daily_price_input, cost, n_day_after)
        Port_p_df = BF_clac.Port_cls.portfolio_cumulative_return.loc[:P_w_pvt_input.index.max()]
        BM_p_df = BF_clac.Bench_cls.portfolio_cumulative_return.loc[:P_w_pvt_input.index.max()]

        self.allocation_effect = BF_clac.allocation_effect
        self.selection_effect = BF_clac.selection_effect
        self.interaction_effect = BF_clac.interaction_effect
        self.alpha = BF_clac.rP.sub(BF_clac.rB)
        self.decompose_allocation_effect = pd.concat([self.alpha.rename('alpha'),
                                                      self.allocation_effect.rename('Allocation Effect'),
                                                      self.selection_effect.rename('Selection Effect'),
                                                      self.interaction_effect.rename('Interaction Effect')], axis=1).dropna(how='all', axis=0)
        self.decompose_allocation_effect_BM_Port = self.decompose_allocation_effect.assign(BM=BF_clac.rB, Port=BF_clac.rP).dropna(how='all', axis=0)
        self.Port_portfolio_turnover_ratio = BF_clac.Port_cls.portfolio_turnover_ratio
        self.Port_stockwise_turnover_ratio = BF_clac.Port_cls.stockwise_turnover_ratio
        self.Port_stockwise_period_return_contribution = BF_clac.Port_cls.stockwise_period_return_contribution
        self.Port_daily_account_ratio = BF_clac.Port_cls.daily_account_ratio
        self.Port_daily_account_ratio_wrt_class = self.Port_daily_account_ratio.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        self.Port_daily_account_ratio_wrt_class=self.Port_daily_account_ratio_wrt_class[self.Port_daily_account_ratio_wrt_class.mean().sort_values(ascending=False).index].fillna(0)

        # Latest Portfolio Holdings
        self.Port_rebal_date_class = BF_clac.P_classweight_pvt.copy()
        self.Port_daily_account_ratio_class_mean = self.Port_rebal_date_class.mean()
        self.Port_latest_rebalancing = pd.concat([BF_clac.Port_cls.ratio_df.iloc[-1].rename('today'), BF_clac.Port_cls.ratio_df.iloc[-2].rename('previous')], axis=1).dropna(how='all', axis=0)
        self.second_latest_rebal_date, self.latest_rebal_date=BF_clac.Port_cls.ratio_df.index[-2],BF_clac.Port_cls.ratio_df.index[-1]
        latest_return_contribution=self.Port_stockwise_period_return_contribution.loc[self.latest_rebal_date].dropna().rename('return contribution')
        self.Port_latest_rebalancing=pd.concat([self.Port_latest_rebalancing, pd.DataFrame(self.code_to_name).loc[self.Port_latest_rebalancing.index], latest_return_contribution], axis=1)
        self.Port_latest_rebalancing['delta'] = self.Port_latest_rebalancing['today'].sub(self.Port_latest_rebalancing['previous'])
        self.Port_latest_rebalancing = self.Port_latest_rebalancing.sort_values(by=['today','class','종목명'], ascending=[False,True,True])
        self.latest_decompose_allocation_effect_BM_Port = self.decompose_allocation_effect_BM_Port.loc[self.latest_rebal_date]

        # Rebalancing Effect
        self.decompose_allocation_effect_NonReb = pd.concat([
                                                      BF_clac.rP_NonReb.rename('Rebalancing Return'),
                                                      BF_clac.rB_NonReb.rename('Non-Rebalancing Return'),
                                                      BF_clac.rP_NonReb.sub(BF_clac.rB_NonReb).rename('Rebalancing Effect'),
                                                      BF_clac.Rebalancing_in_eff.rename('Rebalancing-In Effect'),
                                                      BF_clac.Rebalancing_out_eff.rename('Rebalancing-Out Effect'),
                                                      ], axis=1).dropna(how='all', axis=0)
        self.latest_decompose_allocation_effect_NonReb = self.decompose_allocation_effect_NonReb.loc[self.latest_rebal_date]

        try:
            self.stacked_line_color = Category20c[len(self.Port_daily_account_ratio_wrt_class.columns)]
        except:
            from bokeh.palettes import Category20b_20, Category20c_20
            self.stacked_line_color = Category20b_20+Category20c_20

        self.stacked_line_color_dict = dict(zip(self.Port_daily_account_ratio_wrt_class.columns,self.stacked_line_color))

        self.Bench_portfolio_turnover_ratio = BF_clac.Bench_cls.portfolio_turnover_ratio
        # self.Bench_stockwise_turnover_ratio = BF_clac.Bench_cls.stockwise_turnover_ratio
        # self.Bench_stockwise_period_return_contribution = BF_clac.Bench_cls.stockwise_period_return_contribution
        # self.Bench_daily_account_ratio = BF_clac.Bench_cls.daily_account_ratio
        # self.Bench_daily_account_ratio_wrt_class = self.Bench_daily_account_ratio.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()


        self.hover=hover
        self.yearly = yearly
        self.Port_nm,self.BM_nm = Port_nm, BM_nm

        # 포트폴리오 일별 수익률
        daily_return = pd.concat([Port_p_df.rename(Port_nm),BM_p_df.rename(BM_nm)], axis=1).pct_change()
        daily_return.iloc[0]=0
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
        self.yearly_return = self.daily_return.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
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
        # Shinhan Blue
        self.color_list = ['#0046ff','#8C98A0'] + list(Category20_20)

        # self.color_list = ['#192036','#eaa88f', '#8c98a0'] + list(Category20_20)
        self.outputname = outputname
    def BrinsonFachler_report(self, display = True, toolbar_location='above'):
        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj()
        data_alpha_table_obj = self.get_alpha_table_obj()
        try:
            data_table_obj_3Y = self.get_table_obj_3Y()
        except:
            data_table_obj_3Y = self.get_table_obj_3Y(all_None=True)

        try:
            data_table_obj_5Y = self.get_table_obj_5Y()
        except:
            data_table_obj_5Y = self.get_table_obj_5Y(all_None=True)
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location, W=2)
        Yearly_tottr_obj = self.get_yearly_tottr_obj(toolbar_location, W=2)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location, W=2)
        Yearly_avgtr_obj = self.get_yearly_avgtr_obj(toolbar_location, W=2)

        Monthly_rtn_obj = self.get_monthly_rtn_obj(toolbar_location)
        Monthly_alpha_obj = self.get_monthly_alpha_obj(toolbar_location)
        Monthly_rtn_dist_obj = self.get_monthly_rtn_dist_obj(toolbar_location)
        Monthly_alpha_dist_obj = self.get_monthly_alpha_dist_obj(toolbar_location)

        RllnCAGR_obj = self.get_rollingCAGR_obj(toolbar_location)
        Rllnstd_obj = self.get_rollingstd_obj(toolbar_location)
        Rllnshrp_obj = self.get_rollingSharpe_obj(toolbar_location)

        stacked_line_obj = self.get_class_stacked_line_obj(toolbar_location)
        mean_donut_obj = self.get_class_holding_mean_donut_obj(toolbar_location)
        BrinsonFachler_obj = self.BrinsonFachler_obj(toolbar_location, self.yearly)
        latest_rebalancing_tbl_obj = self.get_latest_rebalancing_tbl_obj()
        latest_rebalancing_donut_obj = self.get_latest_rebalancing_donut_obj(toolbar_location)

        RebalancingEffect_obj = self.RebalancingEffect_obj(toolbar_location, self.yearly)
        latest_rebaleffect_tbl_obj = self.get_latest_rebaleffect_tbl_obj()

        report_title = Div(
            text="""
            <div style=font-size: 13px; color: #333333;">
                <h1>포트폴리오 성과 분석 리포트</h1>
            </div>
            """,
            width=800,
            height=80
        )
        if display == True:
            try:
                show(
                    column(
                           report_title,
                           row(column(
                                      Column(data_table_obj),
                                      Column(data_alpha_table_obj),
                                      Column(data_table_obj_3Y),
                                      Column(data_table_obj_5Y),
                                      ),
                               column(
                                      cmpd_return_TS_obj,
                                      logscale_return_TS_obj,
                                      dd_TS_obj, R1Y_HPR_obj,
                                      row(Yearly_rtn_obj, Yearly_tottr_obj),
                                      row(Yearly_alpha_obj, Yearly_avgtr_obj),
                                      row(Monthly_rtn_obj, Monthly_alpha_obj),
                                      row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                                      RllnCAGR_obj,
                                      Rllnstd_obj,
                                      Rllnshrp_obj,
                                      ),
                               column(
                                      BrinsonFachler_obj,
                                      row(stacked_line_obj,mean_donut_obj),
                                      RebalancingEffect_obj,
                                      latest_rebaleffect_tbl_obj,
                                      row(latest_rebalancing_tbl_obj,latest_rebalancing_donut_obj),
                                      )
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



    def get_yearly_tottr_obj(self, toolbar_location, W=1):
        # Plot Yearly Turnover Ratio
        P_rebal_tr, B_rebal_tr = self.Port_portfolio_turnover_ratio.copy(), self.Bench_portfolio_turnover_ratio.copy()
        input_Data_ = pd.concat([P_rebal_tr.rename(f'{self.Port_nm}'), B_rebal_tr.rename(f'{self.BM_nm}')], axis=1).rename_axis('date')

        Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).sum()
        # Year_mean_tr = input_Data_.groupby(pd.Grouper(freq='Y')).mean().add_suffix(' Year Avg TR')
        # Year_tr = pd.concat([Year_tot_tr, Year_mean_tr],axis=1)

        Year_tot_tr.index = Year_tot_tr.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=Year_tot_tr.index.to_list(),
            title='Yearly Turnonver Ratio',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(Year_tot_tr.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(pd.concat([Year_tot_tr.add_suffix('_True'), Year_tot_tr.mul(100)], axis=1))
        for i, col in enumerate(Year_tot_tr.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, top=col+"_True",x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range), width=0.2, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')
        # show(dd_TS_obj)
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = Year_tot_tr.columns[0],Year_tot_tr.columns[1]
            hover = HoverTool(tooltips=[("Year", "@date"), (f"{V1} Turnover", f"@{{{V1}}}{{0.00}}%"), (f"{V2} Turnover", f"@{{{V2}}}{{0.00}}%")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj
    def get_yearly_avgtr_obj(self, toolbar_location, W=1):
        # Plot Yearly Turnover Ratio
        P_rebal_tr, B_rebal_tr = self.Port_portfolio_turnover_ratio.copy(), self.Bench_portfolio_turnover_ratio.copy()
        input_Data_ = pd.concat([P_rebal_tr.rename(f'{self.Port_nm}'), B_rebal_tr.rename(f'{self.BM_nm}')], axis=1).rename_axis('date')

        # Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).sum()
        Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).mean()
        # Year_tr = pd.concat([Year_tot_tr, Year_mean_tr],axis=1)

        Year_tot_tr.index = Year_tot_tr.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=Year_tot_tr.index.to_list(),
            title='Yearly Average Turnonver Ratio',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(Year_tot_tr.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(pd.concat([Year_tot_tr.add_suffix('_True'), Year_tot_tr.mul(100)], axis=1))
        for i, col in enumerate(Year_tot_tr.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, top=col+"_True",x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range), width=0.2, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')
        # show(dd_TS_obj)
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = Year_tot_tr.columns[0],Year_tot_tr.columns[1]
            hover = HoverTool(tooltips=[("Year", "@date"), (f"{V1} Turnover", f"@{{{V1}}}{{0.00}}%"), (f"{V2} Turnover", f"@{{{V2}}}{{0.00}}%")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj

    # 2024-01-25: 업데이트
    def BrinsonFachler_obj(self, toolbar_location, Yearly=True):
        input_Data=self.decompose_allocation_effect

        if Yearly:
            input_Data = input_Data.groupby(pd.Grouper(freq='Y')).mean()
            input_Data.index = input_Data.index.strftime("%Y")
        else:
            input_Data = input_Data.groupby(pd.Grouper(freq='M')).mean()
            input_Data.index = input_Data.index.strftime("%Y-%m")

        cr_list = ['#D5DBDB']+list(HighContrast3)

        BF_obj = figure(x_range=FactorRange(*input_Data.index), title="Brinson-Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)
        BF_obj.title.text_font_size = '13pt'
        # BF_obj = figure(x_range=input_Data.index.to_list(), title="Brinson Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)

        BF_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)

        # alpha 막대 너비
        alpha_width = 0.6

        # alpha 막대 그리기
        # 나머지 막대 너비 및 dodge 값
        other_width = alpha_width/3
        dodge_val = alpha_width/3

        # 나머지 막대 그리기
        for i, col in enumerate(input_Data.columns):
            if i == 0:
                # BF_line = BF_obj.vbar(x='date', top=col, source=source_TS, width=alpha_width, color=cr_list[i], alpha=0.8)
                BF_line = BF_obj.circle(x='date', y=col, source=source_TS, size=7, color='red') #alpha=0.8
            else:
                dodge = Dodge(value=(i - 1-1) * dodge_val, range=BF_obj.x_range)
                BF_line=BF_obj.vbar(x={'field': 'date', 'transform': dodge}, top=col, source=source_TS, width=other_width, color=cr_list[i], alpha=0.8)
            BF_lgd_list.append((col, [BF_line]))

        if self.hover:
            tooltips = [("Date", "@date")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in input_Data.columns]
            # hover = HoverTool(tooltips=tooltips)#, formatters={"@date": "datetime"})
            hover = HoverTool(renderers=[BF_obj.renderers[-3], BF_obj.renderers[-2], BF_obj.renderers[-1]], tooltips=tooltips)
            # hover = HoverTool(renderers=[BF_obj.renderers[0]], tooltips=tooltips)
            BF_obj.add_tools(hover)

        BF_obj.x_range.range_padding = 0.05
        BF_obj.xgrid.grid_line_color = None
        BF_lgd = Legend(items=BF_lgd_list, location='center')
        BF_obj.add_layout(BF_lgd, 'right')
        BF_obj.legend.click_policy = "mute"
        BF_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # show(BF_obj)




        return BF_obj
    def RebalancingEffect_obj(self, toolbar_location, Yearly=True):
        input_Data=self.decompose_allocation_effect_NonReb.iloc[:,-3:]

        if Yearly:
            input_Data = input_Data.groupby(pd.Grouper(freq='Y')).mean()
            input_Data.index = input_Data.index.strftime("%Y")
        else:
            input_Data = input_Data.groupby(pd.Grouper(freq='M')).mean()
            input_Data.index = input_Data.index.strftime("%Y-%m")

        cr_list = ['#D5DBDB']+list(HighContrast3)

        BF_obj = figure(x_range=FactorRange(*input_Data.index), title="Rebalancing Effects", width=1500, height=500, toolbar_location=toolbar_location)
        BF_obj.title.text_font_size = '13pt'
        # BF_obj = figure(x_range=input_Data.index.to_list(), title="Brinson Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)

        BF_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)

        # alpha 막대 너비
        alpha_width = 0.6

        # alpha 막대 그리기
        # 나머지 막대 너비 및 dodge 값
        other_width = alpha_width/2
        dodge_val = alpha_width/2

        # 나머지 막대 그리기
        for i, col in enumerate(input_Data.columns):
            if i == 0:
                # BF_line = BF_obj.vbar(x='date', top=col, source=source_TS, width=alpha_width, color=cr_list[i], alpha=0.8)
                BF_line = BF_obj.circle(x='date', y=col, source=source_TS, size=7, color='red') #alpha=0.8
            else:
                dodge = Dodge(value=(i - 1-0.5) * dodge_val, range=BF_obj.x_range)
                BF_line=BF_obj.vbar(x={'field': 'date', 'transform': dodge}, top=col, source=source_TS, width=other_width, color=cr_list[i], alpha=0.8)
            BF_lgd_list.append((col, [BF_line]))

        if self.hover:
            tooltips = [("Date", "@date")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in input_Data.columns]
            # hover = HoverTool(tooltips=tooltips)#, formatters={"@date": "datetime"})
            hover = HoverTool(renderers=[BF_obj.renderers[-2], BF_obj.renderers[-1]], tooltips=tooltips)
            # hover = HoverTool(renderers=[BF_obj.renderers[0]], tooltips=tooltips)
            BF_obj.add_tools(hover)

        BF_obj.x_range.range_padding = 0.05
        BF_obj.xgrid.grid_line_color = None
        BF_lgd = Legend(items=BF_lgd_list, location='center')
        BF_obj.add_layout(BF_lgd, 'right')
        BF_obj.legend.click_policy = "mute"
        BF_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # show(BF_obj)
        return BF_obj
    def get_yearly_rtn_obj__(self, toolbar_location, W=1):
        # Plot Yearly Performance
        input_Data = self.yearly_return.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Return',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i],range=dd_TS_obj.x_range),width=0.2,top=col,color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = self.yearly_return.columns[0], self.yearly_return.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date"), (f"{V1}", f"@{{{V1}}}{{0.00%}}"), (f"{V2}", f"@{{{V2}}}{{0.00%}}")]) # formatters={"@date": "datetime"}
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj

    def get_class_stacked_line_obj(self, toolbar_location):
        staked_=self.Port_daily_account_ratio_wrt_class.copy()
        source_for_chart=ColumnDataSource(pd.concat([staked_, staked_.mul(100).add_suffix('_True')], axis=1))

        return_TS_obj = figure(x_axis_type="datetime",
                               title="Class-wise Daily Account Ratio",
                               width=1000, height=500, toolbar_location=toolbar_location)
        return_TS_obj.title.text_font_size = '13pt'

        renderers = return_TS_obj.varea_stack(stackers=staked_.columns.tolist(),
                                              x='date',
                                              source=source_for_chart,
                                              color=self.stacked_line_color)
        legend_items = [(col, [rend]) for col, rend in zip(staked_.columns, renderers)]
        legend = Legend(items=legend_items, location='center')
        return_TS_obj.add_layout(legend, 'right')
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            # HoverTool 설정을 위한 데이터 필드 목록 생성
            tooltips = [("Date", "@date{%F}")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in staked_.columns]

            hover = HoverTool(
                tooltips=tooltips,
                formatters={"@date": "datetime"}
            )
            return_TS_obj.add_tools(hover)
        # show(return_TS_obj)
        return return_TS_obj
    def get_class_holding_mean_donut_obj(self, toolbar_location):
        dounut_value = self.Port_daily_account_ratio_class_mean.copy()

        # 데이터 준비
        dounut_data = pd.Series(dounut_value).rename_axis('class').reset_index(name='value')
        dounut_data['angle'] = dounut_data['value'].div(dounut_data['value'].sum()) * 2*np.pi
        dounut_data['color'] = dounut_data['class'].map(self.stacked_line_color_dict)

        source = ColumnDataSource(dounut_data)

        ClsMean_DN_obj = figure(height=500, title="Class Holding Mean", toolbar_location=None,
                                tools="hover", tooltips="@class: @value{0.00%}", x_range=(-0.5, 1.0))
        ClsMean_DN_obj.title.text_font_size = '13pt'

        # 원형 도넛 차트 추가
        ClsMean_DN_obj.annular_wedge(x=0, y=1, outer_radius=0.4, inner_radius=0.2,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                             line_color="white", fill_color='color', source=source) # legend_field='class'

        ClsMean_DN_obj.axis.axis_label = None
        ClsMean_DN_obj.axis.visible = False
        ClsMean_DN_obj.grid.grid_line_color = None
        ClsMean_DN_obj.outline_line_color = None
        # ClsMean_DN_obj.legend.location = "center_left"  # 범례 위치 중앙으로 설정
        # ClsMean_DN_obj.legend.visible = False  # 범례 위치 중앙으로 설정
        # ClsMean_DN_obj.legend.border_line_color = None  # 범례 테두리 제거
        # show(ClsMean_DN_obj)

        return ClsMean_DN_obj
    def get_latest_rebalancing_donut_obj(self, toolbar_location):
        data_tmp = self.Port_latest_rebalancing.copy()
        dounut_value = data_tmp[['class', 'today']].groupby('class')['today'].sum().rename('value')
        dounut_data = pd.Series(dounut_value).rename_axis('class').reset_index(name='value')
        dounut_data['angle'] = dounut_data['value'].div(dounut_data['value'].sum()) * 2 * np.pi
        dounut_data['color'] = dounut_data['class'].map(self.stacked_line_color_dict)

        ClsMean_DN_obj = figure(height=500, title="Latest Class Holding", toolbar_location=None, x_range=(-0.5, 1.0))
        ClsMean_DN_obj.title.text_font_size = '13pt'


        start_angle = 0
        for idx, row in dounut_data.iterrows():
            end_angle = start_angle + row['angle']
            source = ColumnDataSource(dict(start_angle=[start_angle], end_angle=[end_angle], color=[row['color']], class_name=[row['class']], value=[row['value']]))
            wedge = ClsMean_DN_obj.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.4,
                                                 start_angle='start_angle', end_angle='end_angle',
                                                 color='color', legend_label=row['class'],
                                                 muted_color='grey', muted_alpha=0.2, source=source)
            start_angle = end_angle

        # Hover 툴 설정
        hover = HoverTool(tooltips=[("Class", "@class_name"), ("weight", "@value{0.00%}")])
        ClsMean_DN_obj.add_tools(hover)

        ClsMean_DN_obj.axis.axis_label = None
        ClsMean_DN_obj.axis.visible = False
        ClsMean_DN_obj.grid.grid_line_color = None
        ClsMean_DN_obj.outline_line_color = None

        ClsMean_DN_obj.legend.location = "center_right"
        ClsMean_DN_obj.legend.border_line_color = None
        ClsMean_DN_obj.legend.click_policy = "mute"
        # show(ClsMean_DN_obj)

        return ClsMean_DN_obj
    def get_latest_rebalancing_tbl_obj(self):
        static_data_tmp = self.Port_latest_rebalancing.reset_index().fillna(0)
        contributions = static_data_tmp['return contribution'].values
        static_data_tmp['today'] = static_data_tmp['today'].map(lambda x: str(np.int64(x*10000)/100)+"%")
        static_data_tmp['previous'] = static_data_tmp['previous'].map(lambda x: str(int(x*10000)/100)+"%")
        static_data_tmp['delta'] = static_data_tmp['delta'].map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')
        static_data_tmp.loc['sum']=""
        static_data_tmp.loc['sum', 'return contribution'] = sum(contributions)

        static_data_tmp['return contribution'] = static_data_tmp['return contribution'].map(lambda x: str(int(x*10000)/100)+"%")

        decomp_df=self.latest_decompose_allocation_effect_BM_Port.loc[['alpha','Port', 'BM', 'Allocation Effect','Selection Effect','Interaction Effect']]#.rename(index={'Port':self.Port_nm, "BM":self.BM_nm})
        decomp_df=decomp_df.map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')
        static_data_tmp=pd.concat([static_data_tmp, pd.DataFrame(decomp_df.rename('previous_performance'))], axis=0)
        static_data_tmp.loc[['alpha','Port', 'BM', 'Allocation Effect','Selection Effect','Interaction Effect'],'delta'] = ['alpha',self.Port_nm, self.BM_nm, 'Allocation Effect','Selection Effect','Interaction Effect']


        static_data_tmp[static_data_tmp.isnull()] = ""
        # static_data_tmp = static_data_tmp.reset_index()
        static_data=static_data_tmp[['code', '종목명', 'return contribution','previous', 'today', 'delta','previous_performance']].rename(
                            columns={'code':'종목코드',
                                     'today':'최근리밸런싱',
                                     'previous':'직전리밸런싱',
                                     'delta':'변화',
                                     'return contribution':'수익률 기여도',
                                     'previous_performance':'직전리밸런싱 성과'
                                     })

        source = ColumnDataSource(static_data)

        # columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        # 폰트 크기를 조정하기 위한 HTML 템플릿 포맷터 생성
        formatter = HTMLTemplateFormatter(template='<div style="font-size: 16px;"><%= value %></div>')  # 수정됨

        # 각 컬럼에 HTML 템플릿 포맷터 적용
        columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]  # 수정됨

        data_table_fig = DataTable(source=source, columns=columns, width=1000, height=750, index_position=None)

        # 제목을 위한 Div 위젯 생성
        title_div = Div(text=f"<h2>최근 리밸런싱 내역: {self.latest_rebal_date.strftime('%Y-%m-%d')}</h2>", width=1000, height=30)


        # Div와 DataTable을 column 레이아웃으로 결합
        layout = column(title_div, data_table_fig)

        # show(layout)
        return layout
    def get_latest_rebaleffect_tbl_obj(self):
        static_data_tmp = self.latest_decompose_allocation_effect_NonReb#.reset_index().fillna(0)
        static_data_tmp = static_data_tmp.map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')

        static_data_tmp[static_data_tmp.isnull()] = ""
        # static_data_tmp = static_data_tmp.reset_index()
        # static_data=static_data_tmp[['code', '종목명', 'return contribution','previous', 'today', 'delta','previous_performance']].rename(
        #                     columns={'code':'종목코드', 'today':'최근리밸런싱', 'previous':'직전리밸런싱', 'delta':'변화', 'return contribution':'수익률 기여도', 'previous_performance':'직전리밸런싱 성과'})
        static_data=static_data_tmp.rename_axis("").rename('최근리밸런싱효과').reset_index()
        source = ColumnDataSource(static_data)

        # columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        # 폰트 크기를 조정하기 위한 HTML 템플릿 포맷터 생성
        formatter = HTMLTemplateFormatter(template='<div style="font-size: 16px;"><%= value %></div>')  # 수정됨

        # 각 컬럼에 HTML 템플릿 포맷터 적용
        columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]  # 수정됨

        data_table_fig = DataTable(source=source, columns=columns, width=int(1500*(1/3)), height=200, index_position=None)

        # 제목을 위한 Div 위젯 생성
        title_div = Div(text=f"<h2>최근 리밸런싱 효과: {self.second_latest_rebal_date.strftime('%Y-%m-%d')}~{self.latest_rebal_date.strftime('%Y-%m-%d')}</h2>", width=1000, height=30)


        # Div와 DataTable을 column 레이아웃으로 결합
        layout = column(title_div, data_table_fig)

        # show(layout)
        return layout



class BrinsonHoodBeebower_PortfolioAnalysis(PortfolioAnalysis):
    def __init__(self, P_w_pvt_input,B_w_pvt_input,Asset_info_input,Stock_Daily_price_input,Index_Daily_price_input, cost=0.00, n_day_after=0, BM_nm='BM', Port_nm='My Portfolio', outputname='./Unnamed', hover=True, yearly=True):
        """
        P_w_pvt_input: Portfolio의 weight를 담은 DataFrame(index='(rebalancing)date', columns='(종목,ETF)code', values='weight')
        B_w_pvt_input: Benchmark의 weight를 담은 DataFrame(index='(rebalancing)date', columns='(지수)code', values='weight')
        Asset_info_input: colums=[(종목,ETF)code, 종목명, (지수)code]      <----- (지수)code: "(종목,ETF)code가 어느 지수code에 mapping이 되는지"
                                                                                 ex. KODEX200, ARIRANG200, TIGER200 --> KOSPI
        Index_Daily_price_input: index='(daily)date', columns='(지수)code', values='base-price'
        """
        gdu.data = pd.concat([Stock_Daily_price_input,Index_Daily_price_input], axis=1)
        self.code_to_name = Asset_info_input.set_index('종목코드')[['종목명','class']]

        Asset_info_input = Asset_info_input.set_index('종목코드')['class']
        # P_w_pvt_input,B_w_pvt_input=Lrisk_w_pvt_input.div(100).copy(), LB_w_pvt_input.copy()
        BF_clac = BrinsonHoodBeebower_calculator(P_w_pvt_input,B_w_pvt_input,Asset_info_input, Index_Daily_price_input, cost, n_day_after)
        Port_p_df = BF_clac.Port_cls.portfolio_cumulative_return#.loc[:P_w_pvt_input.index.max()]
        BM_p_df = BF_clac.Bench_cls.portfolio_cumulative_return#.loc[:P_w_pvt_input.index.max()]

        self.allocation_effect = BF_clac.allocation_effect
        self.selection_effect = BF_clac.selection_effect
        self.alpha = BF_clac.rP.sub(BF_clac.rB)
        self.decompose_allocation_effect = pd.concat([self.alpha.rename('alpha'),
                                                      self.allocation_effect.rename('Allocation Effect'),
                                                      self.selection_effect.rename('Selection Effect'),
                                                      ], axis=1).dropna(how='all', axis=0)
        self.decompose_allocation_effect_BM_Port = self.decompose_allocation_effect.assign(BM=BF_clac.rB, Port=BF_clac.rP).dropna(how='all', axis=0)
        self.Port_portfolio_turnover_ratio = BF_clac.Port_cls.portfolio_turnover_ratio
        self.Port_stockwise_turnover_ratio = BF_clac.Port_cls.stockwise_turnover_ratio
        self.Port_stockwise_period_return_contribution = BF_clac.Port_cls.stockwise_period_return_contribution
        self.Port_daily_account_ratio = BF_clac.Port_cls.daily_account_ratio
        self.Port_daily_account_ratio_wrt_class = self.Port_daily_account_ratio.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()
        self.Port_daily_account_ratio_wrt_class=self.Port_daily_account_ratio_wrt_class[self.Port_daily_account_ratio_wrt_class.mean().sort_values(ascending=False).index].fillna(0)

        # Latest Portfolio Holdings
        self.Port_rebal_date_class = BF_clac.P_classweight_pvt.copy()
        self.Port_daily_account_ratio_class_mean = self.Port_rebal_date_class.mean()
        self.Port_latest_rebalancing = pd.concat([BF_clac.Port_cls.ratio_df.iloc[-1].rename('today'), BF_clac.Port_cls.ratio_df.iloc[-2].rename('previous')], axis=1).dropna(how='all', axis=0)
        self.second_latest_rebal_date, self.latest_rebal_date=BF_clac.Port_cls.ratio_df.index[-2],BF_clac.Port_cls.ratio_df.index[-1]
        latest_return_contribution=self.Port_stockwise_period_return_contribution.loc[self.latest_rebal_date].dropna().rename('return contribution')
        self.Port_latest_rebalancing=pd.concat([self.Port_latest_rebalancing, pd.DataFrame(self.code_to_name).loc[self.Port_latest_rebalancing.index], latest_return_contribution], axis=1)
        self.Port_latest_rebalancing['delta'] = self.Port_latest_rebalancing['today'].sub(self.Port_latest_rebalancing['previous'], fill_value=0)
        self.Port_latest_rebalancing = self.Port_latest_rebalancing.sort_values(by=['today','class','종목명'], ascending=[False,True,True])
        self.latest_decompose_allocation_effect_BM_Port = self.decompose_allocation_effect_BM_Port.loc[self.latest_rebal_date]

        # Rebalancing Effect
        self.decompose_allocation_effect_NonReb = pd.concat([
                                                      BF_clac.rP_NonReb.rename('Rebalancing Return'),
                                                      BF_clac.rB_NonReb.rename('Non-Rebalancing Return'),
                                                      BF_clac.rP_NonReb.sub(BF_clac.rB_NonReb).rename('Rebalancing Effect'),
                                                      BF_clac.Rebalancing_in_eff.rename('Rebalancing-In Effect'),
                                                      BF_clac.Rebalancing_out_eff.rename('Rebalancing-Out Effect'),
                                                      ], axis=1).dropna(how='all', axis=0)
        self.latest_decompose_allocation_effect_NonReb = self.decompose_allocation_effect_NonReb.loc[self.latest_rebal_date]

        try:
            self.stacked_line_color = Category20c[len(self.Port_daily_account_ratio_wrt_class.columns)]
        except:
            from bokeh.palettes import Category20b_20, Category20c_20
            self.stacked_line_color = Category20b_20+Category20c_20

        self.stacked_line_color_dict = dict(zip(self.Port_daily_account_ratio_wrt_class.columns,self.stacked_line_color))

        self.Bench_portfolio_turnover_ratio = BF_clac.Bench_cls.portfolio_turnover_ratio
        # self.Bench_stockwise_turnover_ratio = BF_clac.Bench_cls.stockwise_turnover_ratio
        # self.Bench_stockwise_period_return_contribution = BF_clac.Bench_cls.stockwise_period_return_contribution
        # self.Bench_daily_account_ratio = BF_clac.Bench_cls.daily_account_ratio
        # self.Bench_daily_account_ratio_wrt_class = self.Bench_daily_account_ratio.rename(columns=Asset_info_input.to_dict()).stack().groupby(level=[0,1]).sum().unstack()


        self.hover=hover
        self.yearly = yearly
        self.Port_nm,self.BM_nm = Port_nm, BM_nm

        # 포트폴리오 일별 수익률
        daily_return = pd.concat([Port_p_df.rename(Port_nm),BM_p_df.rename(BM_nm)], axis=1).pct_change()
        daily_return.iloc[0]=0
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
        self.yearly_return = self.daily_return.add(1).groupby(pd.Grouper(freq='BA')).apply(lambda x: x.cumprod().tail(1)).sub(1).droplevel(0)
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
        # Shinhan Blue
        self.color_list = ['#0046ff','#8C98A0'] + list(Category20_20)

        # self.color_list = ['#192036','#eaa88f', '#8c98a0'] + list(Category20_20)
        self.outputname = outputname
    def BrinsonHoodBeebower_report(self, display = True, toolbar_location='above'):
        curdoc().clear()
        output_file(self.outputname + '.html')

        data_table_obj = self.get_table_obj()
        data_alpha_table_obj = self.get_alpha_table_obj()
        try:
            data_table_obj_3Y = self.get_table_obj_3Y()
        except:
            data_table_obj_3Y = self.get_table_obj_3Y(all_None=True)

        try:
            data_table_obj_5Y = self.get_table_obj_5Y()
        except:
            data_table_obj_5Y = self.get_table_obj_5Y(all_None=True)
        cmpd_return_TS_obj = self.get_cmpd_rtn_obj(toolbar_location)
        logscale_return_TS_obj = self.get_logscale_rtn_obj(toolbar_location)
        dd_TS_obj = self.get_dd_obj(toolbar_location)
        R1Y_HPR_obj = self.get_R1Y_HPR_obj(toolbar_location)
        Yearly_rtn_obj = self.get_yearly_rtn_obj(toolbar_location, W=2)
        Yearly_tottr_obj = self.get_yearly_tottr_obj(toolbar_location, W=2)
        Yearly_alpha_obj = self.get_yearly_alpha_obj(toolbar_location, W=2)
        Yearly_avgtr_obj = self.get_yearly_avgtr_obj(toolbar_location, W=2)

        Monthly_rtn_obj = self.get_monthly_rtn_obj(toolbar_location)
        Monthly_alpha_obj = self.get_monthly_alpha_obj(toolbar_location)
        Monthly_rtn_dist_obj = self.get_monthly_rtn_dist_obj(toolbar_location)
        Monthly_alpha_dist_obj = self.get_monthly_alpha_dist_obj(toolbar_location)

        RllnCAGR_obj = self.get_rollingCAGR_obj(toolbar_location)
        Rllnstd_obj = self.get_rollingstd_obj(toolbar_location)
        Rllnshrp_obj = self.get_rollingSharpe_obj(toolbar_location)

        stacked_line_obj = self.get_class_stacked_line_obj(toolbar_location)
        mean_donut_obj = self.get_class_holding_mean_donut_obj(toolbar_location)
        BrinsonHoodBeebower_obj = self.BrinsonHoodBeebower_obj(toolbar_location, self.yearly)
        latest_rebalancing_tbl_obj = self.get_latest_rebalancing_tbl_obj()
        latest_rebalancing_donut_obj = self.get_latest_rebalancing_donut_obj(toolbar_location)

        RebalancingEffect_obj = self.RebalancingEffect_obj(toolbar_location, self.yearly)
        latest_rebaleffect_tbl_obj = self.get_latest_rebaleffect_tbl_obj()
        MTD_rtn_obj = self.get_MTD_rtn_obj(toolbar_location)
        MTD_rtn_tbl_obj = self.get_MTD_rtn_tbl_obj()

        report_title = Div(
            text="""
            <div style=font-size: 13px; color: #333333;">
                <h1>포트폴리오 성과 분석 리포트</h1>
            </div>
            """,
            width=800,
            height=80
        )
        if display == True:
            try:
                show(
                    column(
                           report_title,
                           row(column(
                                      Column(data_table_obj),
                                      Column(data_alpha_table_obj),
                                      Column(data_table_obj_3Y),
                                      Column(data_table_obj_5Y),
                                      ),
                               column(
                                      cmpd_return_TS_obj,
                                      logscale_return_TS_obj,
                                      dd_TS_obj, R1Y_HPR_obj,
                                      row(Yearly_rtn_obj, Yearly_tottr_obj),
                                      row(Yearly_alpha_obj, Yearly_avgtr_obj),
                                      row(Monthly_rtn_obj, Monthly_alpha_obj),
                                      row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                                      RllnCAGR_obj,
                                      Rllnstd_obj,
                                      Rllnshrp_obj,
                                      ),
                               column(
                                      BrinsonHoodBeebower_obj,
                                      row(stacked_line_obj,mean_donut_obj),
                                      RebalancingEffect_obj,
                                      latest_rebaleffect_tbl_obj,
                                      row(latest_rebalancing_tbl_obj,latest_rebalancing_donut_obj),
                                      MTD_rtn_obj,
                                      MTD_rtn_tbl_obj,
                                      )
                               )
                           )
                    )
            except:
                show(
                    column(
                        report_title,
                        row(column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                            # Column(data_table_obj_3Y),
                            # Column(data_table_obj_5Y),
                        ),
                            column(
                                cmpd_return_TS_obj,
                                logscale_return_TS_obj,
                                dd_TS_obj, #R1Y_HPR_obj,
                                row(Yearly_rtn_obj, Yearly_tottr_obj),
                                row(Yearly_alpha_obj, Yearly_avgtr_obj),
                                row(Monthly_rtn_obj, Monthly_alpha_obj),
                                row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                                RllnCAGR_obj,
                                Rllnstd_obj,
                                Rllnshrp_obj,
                            ),
                            column(
                                BrinsonHoodBeebower_obj,
                                row(stacked_line_obj, mean_donut_obj),
                                RebalancingEffect_obj,
                                latest_rebaleffect_tbl_obj,
                                row(latest_rebalancing_tbl_obj, latest_rebalancing_donut_obj),
                                MTD_rtn_obj,
                                MTD_rtn_tbl_obj,
                            )
                        )
                    )
                )
        else:
            try:
                save(
                    column(
                           report_title,
                           row(column(
                                      Column(data_table_obj),
                                      Column(data_alpha_table_obj),
                                      Column(data_table_obj_3Y),
                                      Column(data_table_obj_5Y),
                                      ),
                               column(
                                      cmpd_return_TS_obj,
                                      logscale_return_TS_obj,
                                      dd_TS_obj, R1Y_HPR_obj,
                                      row(Yearly_rtn_obj, Yearly_tottr_obj),
                                      row(Yearly_alpha_obj, Yearly_avgtr_obj),
                                      row(Monthly_rtn_obj, Monthly_alpha_obj),
                                      row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                                      RllnCAGR_obj,
                                      Rllnstd_obj,
                                      Rllnshrp_obj,
                                      ),
                               column(
                                      BrinsonHoodBeebower_obj,
                                      row(stacked_line_obj,mean_donut_obj),
                                      RebalancingEffect_obj,
                                      latest_rebaleffect_tbl_obj,
                                      row(latest_rebalancing_tbl_obj,latest_rebalancing_donut_obj),
                                      MTD_rtn_obj,
                                      MTD_rtn_tbl_obj,
                                      )
                               )
                           )
                )
            except:
                save(
                    column(
                        report_title,
                        row(column(
                            Column(data_table_obj),
                            Column(data_alpha_table_obj),
                            # Column(data_table_obj_3Y),
                            # Column(data_table_obj_5Y),
                        ),
                            column(
                                cmpd_return_TS_obj,
                                logscale_return_TS_obj,
                                dd_TS_obj, #R1Y_HPR_obj,
                                row(Yearly_rtn_obj, Yearly_tottr_obj),
                                row(Yearly_alpha_obj, Yearly_avgtr_obj),
                                row(Monthly_rtn_obj, Monthly_alpha_obj),
                                row(Monthly_rtn_dist_obj, Monthly_alpha_dist_obj),
                                RllnCAGR_obj,
                                Rllnstd_obj,
                                Rllnshrp_obj,
                            ),
                            column(
                                BrinsonHoodBeebower_obj,
                                row(stacked_line_obj, mean_donut_obj),
                                RebalancingEffect_obj,
                                latest_rebaleffect_tbl_obj,
                                row(latest_rebalancing_tbl_obj, latest_rebalancing_donut_obj),
                                MTD_rtn_obj,
                                MTD_rtn_tbl_obj,
                            )
                        )
                    )

                )



    def get_yearly_tottr_obj(self, toolbar_location, W=1):
        # Plot Yearly Turnover Ratio
        P_rebal_tr, B_rebal_tr = self.Port_portfolio_turnover_ratio.copy(), self.Bench_portfolio_turnover_ratio.copy()
        input_Data_ = pd.concat([P_rebal_tr.rename(f'{self.Port_nm}'), B_rebal_tr.rename(f'{self.BM_nm}')], axis=1).rename_axis('date')

        Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).sum()
        # Year_mean_tr = input_Data_.groupby(pd.Grouper(freq='Y')).mean().add_suffix(' Year Avg TR')
        # Year_tr = pd.concat([Year_tot_tr, Year_mean_tr],axis=1)

        Year_tot_tr.index = Year_tot_tr.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=Year_tot_tr.index.to_list(),
            title='Yearly Turnonver Ratio',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(Year_tot_tr.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(pd.concat([Year_tot_tr.add_suffix('_True'), Year_tot_tr.mul(100)], axis=1))
        for i, col in enumerate(Year_tot_tr.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, top=col+"_True",x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range), width=0.2, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')
        # show(dd_TS_obj)
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = Year_tot_tr.columns[0],Year_tot_tr.columns[1]
            hover = HoverTool(tooltips=[("Year", "@date"), (f"{V1} Turnover", f"@{{{V1}}}{{0.00}}%"), (f"{V2} Turnover", f"@{{{V2}}}{{0.00}}%")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj
    def get_yearly_avgtr_obj(self, toolbar_location, W=1):
        # Plot Yearly Turnover Ratio
        P_rebal_tr, B_rebal_tr = self.Port_portfolio_turnover_ratio.copy(), self.Bench_portfolio_turnover_ratio.copy()
        input_Data_ = pd.concat([P_rebal_tr.rename(f'{self.Port_nm}'), B_rebal_tr.rename(f'{self.BM_nm}')], axis=1).rename_axis('date')

        # Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).sum()
        Year_tot_tr = input_Data_.groupby(pd.Grouper(freq='Y')).mean()
        # Year_tr = pd.concat([Year_tot_tr, Year_mean_tr],axis=1)

        Year_tot_tr.index = Year_tot_tr.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=Year_tot_tr.index.to_list(),
            title='Yearly Average Turnonver Ratio',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(Year_tot_tr.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(pd.concat([Year_tot_tr.add_suffix('_True'), Year_tot_tr.mul(100)], axis=1))
        for i, col in enumerate(Year_tot_tr.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, top=col+"_True",x=dodge('date', 0.2*n_col_ord[i], range=dd_TS_obj.x_range), width=0.2, color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')
        # show(dd_TS_obj)
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = Year_tot_tr.columns[0],Year_tot_tr.columns[1]
            hover = HoverTool(tooltips=[("Year", "@date"), (f"{V1} Turnover", f"@{{{V1}}}{{0.00}}%"), (f"{V2} Turnover", f"@{{{V2}}}{{0.00}}%")], formatters={"@date": "datetime"})
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj

    # 2024-01-25: 업데이트
    def BrinsonHoodBeebower_obj(self, toolbar_location, Yearly=True):
        input_Data=self.decompose_allocation_effect

        if Yearly:
            input_Data = input_Data.groupby(pd.Grouper(freq='Y')).mean()
            input_Data.index = input_Data.index.strftime("%Y")
        else:
            input_Data = input_Data.groupby(pd.Grouper(freq='M')).mean()
            input_Data.index = input_Data.index.strftime("%Y-%m")

        cr_list = ['#D5DBDB']+list(HighContrast3)

        BF_obj = figure(x_range=FactorRange(*input_Data.index), title="Brinson-Hood-Beebower Analysis", width=1500, height=500, toolbar_location=toolbar_location)
        BF_obj.title.text_font_size = '13pt'
        # BF_obj = figure(x_range=input_Data.index.to_list(), title="Brinson Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)

        source_TS = ColumnDataSource(data=input_Data)

        # alpha 막대 너비
        alpha_width = 0.6

        # alpha 막대 그리기
        # 나머지 막대 너비 및 dodge 값
        other_width = alpha_width/2
        dodge_val = alpha_width/2

        # 나머지 막대 그리기
        BF_lgd_list = []

        for i, col in enumerate(input_Data.columns):
            if i == 0:
                # BF_line = BF_obj.vbar(x='date', top=col, source=source_TS, width=alpha_width, color=cr_list[i], alpha=0.8)
                BF_line = BF_obj.circle(x='date', y=col, source=source_TS, size=7, color='red') #alpha=0.8
            else:
                dodge = Dodge(value=(i - 1-0.5) * dodge_val, range=BF_obj.x_range)
                BF_line=BF_obj.vbar(x={'field': 'date', 'transform': dodge}, top=col, source=source_TS, width=other_width, color=cr_list[i], alpha=0.8)
            BF_lgd_list.append((col, [BF_line]))

        if self.hover:
            tooltips = [("Date", "@date")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in input_Data.columns]
            # hover = HoverTool(tooltips=tooltips)#, formatters={"@date": "datetime"})
            hover = HoverTool(renderers=[BF_obj.renderers[-2], BF_obj.renderers[-1]], tooltips=tooltips)
            # hover = HoverTool(renderers=[BF_obj.renderers[0]], tooltips=tooltips)
            BF_obj.add_tools(hover)

        BF_obj.x_range.range_padding = 0.05
        BF_obj.xgrid.grid_line_color = None
        BF_lgd = Legend(items=BF_lgd_list, location='center')
        BF_obj.add_layout(BF_lgd, 'right')
        BF_obj.legend.click_policy = "mute"
        BF_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # show(BF_obj)




        return BF_obj
    def RebalancingEffect_obj(self, toolbar_location, Yearly=True):
        input_Data=self.decompose_allocation_effect_NonReb.iloc[:,-3:]

        if Yearly:
            input_Data = input_Data.groupby(pd.Grouper(freq='Y')).mean()
            input_Data.index = input_Data.index.strftime("%Y")
        else:
            input_Data = input_Data.groupby(pd.Grouper(freq='M')).mean()
            input_Data.index = input_Data.index.strftime("%Y-%m")

        cr_list = ['#D5DBDB']+list(Pastel1[len(input_Data.columns)])

        BF_obj = figure(x_range=FactorRange(*input_Data.index), title="Rebalancing Effects", width=1500, height=500, toolbar_location=toolbar_location)
        BF_obj.title.text_font_size = '13pt'
        # BF_obj = figure(x_range=input_Data.index.to_list(), title="Brinson Fachler Analysis", width=1500, height=500, toolbar_location=toolbar_location)

        BF_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)

        # alpha 막대 너비
        alpha_width = 0.6

        # alpha 막대 그리기
        # 나머지 막대 너비 및 dodge 값
        other_width = alpha_width/2
        dodge_val = alpha_width/2

        # 나머지 막대 그리기
        for i, col in enumerate(input_Data.columns):
            if i == 0:
                # BF_line = BF_obj.vbar(x='date', top=col, source=source_TS, width=alpha_width, color=cr_list[i], alpha=0.8)
                BF_line = BF_obj.circle(x='date', y=col, source=source_TS, size=7, color='red') #alpha=0.8
            else:
                dodge = Dodge(value=(i - 1-0.5) * dodge_val, range=BF_obj.x_range)
                BF_line=BF_obj.vbar(x={'field': 'date', 'transform': dodge}, top=col, source=source_TS, width=other_width, color=cr_list[i], alpha=0.8)
            BF_lgd_list.append((col, [BF_line]))

        if self.hover:
            tooltips = [("Date", "@date")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in input_Data.columns]
            # hover = HoverTool(tooltips=tooltips)#, formatters={"@date": "datetime"})
            hover = HoverTool(renderers=[BF_obj.renderers[-2], BF_obj.renderers[-1]], tooltips=tooltips)
            # hover = HoverTool(renderers=[BF_obj.renderers[0]], tooltips=tooltips)
            BF_obj.add_tools(hover)

        BF_obj.x_range.range_padding = 0.05
        BF_obj.xgrid.grid_line_color = None
        BF_lgd = Legend(items=BF_lgd_list, location='center')
        BF_obj.add_layout(BF_lgd, 'right')
        BF_obj.legend.click_policy = "mute"
        BF_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # show(BF_obj)
        return BF_obj
    def get_yearly_rtn_obj__(self, toolbar_location, W=1):
        # Plot Yearly Performance
        input_Data = self.yearly_return.copy()
        input_Data.index = input_Data.index.strftime("%Y")
        dd_TS_obj = figure(
            # x_axis_type='datetime',
            x_range=input_Data.index.to_list(),
            title='Yearly Return',
            width=1500//W, height=200, toolbar_location=toolbar_location)

        n_col = len(input_Data.columns)
        n_col_ord = list(range(-n_col // 2 + 1, n_col // 2 + 1))
        dd_TS_lgd_list = []
        source_TS = ColumnDataSource(data=input_Data)
        for i, col in enumerate(input_Data.columns):
            dd_TS_line = dd_TS_obj.vbar(source=source_TS, x=dodge('date', 0.2*n_col_ord[i],range=dd_TS_obj.x_range),width=0.2,top=col,color=self.color_list[i], alpha=0.8)
            dd_TS_lgd_list.append((col, [dd_TS_line]))
        dd_TS_lgd = Legend(items=dd_TS_lgd_list, location='center')
        dd_TS_obj.add_layout(dd_TS_lgd, 'right')
        dd_TS_obj.legend.click_policy = "mute"
        dd_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')
        # dd_TS_obj.y_range.start = 0
        if self.hover:
            V1, V2 = self.yearly_return.columns[0], self.yearly_return.columns[1]
            hover = HoverTool(tooltips=[("Date", "@date"), (f"{V1}", f"@{{{V1}}}{{0.00%}}"), (f"{V2}", f"@{{{V2}}}{{0.00%}}")]) # formatters={"@date": "datetime"}
            dd_TS_obj.add_tools(hover)
        return dd_TS_obj

    def get_class_stacked_line_obj(self, toolbar_location):
        staked_=self.Port_daily_account_ratio_wrt_class.copy()
        source_for_chart=ColumnDataSource(pd.concat([staked_, staked_.mul(100).add_suffix('_True')], axis=1))

        return_TS_obj = figure(x_axis_type="datetime",
                               title="Class-wise Daily Account Ratio",
                               width=1000, height=500, toolbar_location=toolbar_location)
        return_TS_obj.title.text_font_size = '13pt'

        renderers = return_TS_obj.varea_stack(stackers=staked_.columns.tolist(),
                                              x='date',
                                              source=source_for_chart,
                                              color=self.stacked_line_color)
        legend_items = [(col, [rend]) for col, rend in zip(staked_.columns, renderers)]
        legend = Legend(items=legend_items, location='center')
        return_TS_obj.add_layout(legend, 'right')
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0%')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            # HoverTool 설정을 위한 데이터 필드 목록 생성
            tooltips = [("Date", "@date{%F}")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in staked_.columns]

            hover = HoverTool(
                tooltips=tooltips,
                formatters={"@date": "datetime"}
            )
            return_TS_obj.add_tools(hover)
        # show(return_TS_obj)
        return return_TS_obj
    def get_class_holding_mean_donut_obj(self, toolbar_location):
        dounut_value = self.Port_daily_account_ratio_class_mean.copy()

        # 데이터 준비
        dounut_data = pd.Series(dounut_value).rename_axis('class').reset_index(name='value')
        dounut_data['angle'] = dounut_data['value'].div(dounut_data['value'].sum()) * 2*np.pi
        dounut_data['color'] = dounut_data['class'].map(self.stacked_line_color_dict)

        source = ColumnDataSource(dounut_data)

        ClsMean_DN_obj = figure(height=500, title="Class Holding Mean", toolbar_location=None,
                                tools="hover", tooltips="@class: @value{0.00%}", x_range=(-0.5, 1.0))
        ClsMean_DN_obj.title.text_font_size = '13pt'

        # 원형 도넛 차트 추가
        ClsMean_DN_obj.annular_wedge(x=0, y=1, outer_radius=0.4, inner_radius=0.2,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                             line_color="white", fill_color='color', source=source) # legend_field='class'

        ClsMean_DN_obj.axis.axis_label = None
        ClsMean_DN_obj.axis.visible = False
        ClsMean_DN_obj.grid.grid_line_color = None
        ClsMean_DN_obj.outline_line_color = None
        # ClsMean_DN_obj.legend.location = "center_left"  # 범례 위치 중앙으로 설정
        # ClsMean_DN_obj.legend.visible = False  # 범례 위치 중앙으로 설정
        # ClsMean_DN_obj.legend.border_line_color = None  # 범례 테두리 제거
        # show(ClsMean_DN_obj)

        return ClsMean_DN_obj
    def get_latest_rebalancing_donut_obj(self, toolbar_location):
        data_tmp = self.Port_latest_rebalancing.copy()
        dounut_value = data_tmp[['class', 'today']].groupby('class')['today'].sum().rename('value')
        dounut_data = pd.Series(dounut_value).rename_axis('class').reset_index(name='value')
        dounut_data['angle'] = dounut_data['value'].div(dounut_data['value'].sum()) * 2 * np.pi
        dounut_data['color'] = dounut_data['class'].map(self.stacked_line_color_dict)

        ClsMean_DN_obj = figure(height=500, title="Latest Class Holding", toolbar_location=None, x_range=(-0.5, 1.0))
        ClsMean_DN_obj.title.text_font_size = '13pt'


        start_angle = 0
        for idx, row in dounut_data.iterrows():
            end_angle = start_angle + row['angle']
            source = ColumnDataSource(dict(start_angle=[start_angle], end_angle=[end_angle], color=[row['color']], class_name=[row['class']], value=[row['value']]))
            wedge = ClsMean_DN_obj.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.4,
                                                 start_angle='start_angle', end_angle='end_angle',
                                                 color='color', legend_label=row['class'],
                                                 muted_color='grey', muted_alpha=0.2, source=source)
            start_angle = end_angle

        # Hover 툴 설정
        hover = HoverTool(tooltips=[("Class", "@class_name"), ("weight", "@value{0.00%}")])
        ClsMean_DN_obj.add_tools(hover)

        ClsMean_DN_obj.axis.axis_label = None
        ClsMean_DN_obj.axis.visible = False
        ClsMean_DN_obj.grid.grid_line_color = None
        ClsMean_DN_obj.outline_line_color = None

        ClsMean_DN_obj.legend.location = "center_right"
        ClsMean_DN_obj.legend.border_line_color = None
        ClsMean_DN_obj.legend.click_policy = "mute"
        # show(ClsMean_DN_obj)

        return ClsMean_DN_obj
    def get_latest_rebalancing_tbl_obj(self):
        static_data_tmp = self.Port_latest_rebalancing.reset_index().fillna(0)
        contributions = static_data_tmp['return contribution'].values
        static_data_tmp['today'] = static_data_tmp['today'].map(lambda x: str(np.int64(x*10000)/100)+"%")
        static_data_tmp['previous'] = static_data_tmp['previous'].map(lambda x: str(int(x*10000)/100)+"%")
        static_data_tmp['delta'] = static_data_tmp['delta'].map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')
        static_data_tmp.loc['sum']=""
        static_data_tmp.loc['sum', 'return contribution'] = sum(contributions)

        static_data_tmp['return contribution'] = static_data_tmp['return contribution'].map(lambda x: str(int(x*10000)/100)+"%")

        decomp_df=self.latest_decompose_allocation_effect_BM_Port.loc[['alpha','Port', 'BM', 'Allocation Effect','Selection Effect']]#.rename(index={'Port':self.Port_nm, "BM":self.BM_nm})
        decomp_df=decomp_df.map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')
        static_data_tmp=pd.concat([static_data_tmp, pd.DataFrame(decomp_df.rename('previous_performance'))], axis=0)
        static_data_tmp.loc[['alpha','Port', 'BM', 'Allocation Effect','Selection Effect'],'delta'] = ['alpha',self.Port_nm, self.BM_nm, 'Allocation Effect','Selection Effect']


        static_data_tmp[static_data_tmp.isnull()] = ""
        # static_data_tmp = static_data_tmp.reset_index()
        static_data=static_data_tmp[['code', '종목명', 'return contribution','previous', 'today', 'delta','previous_performance']].rename(
                            columns={'code':'종목코드',
                                     'today':'최근리밸런싱',
                                     'previous':'직전리밸런싱',
                                     'delta':'변화',
                                     'return contribution':'수익률 기여도',
                                     'previous_performance':'직전리밸런싱 성과'
                                     })

        source = ColumnDataSource(static_data)

        # columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        # 폰트 크기를 조정하기 위한 HTML 템플릿 포맷터 생성
        formatter = HTMLTemplateFormatter(template='<div style="font-size: 16px;"><%= value %></div>')  # 수정됨

        # 각 컬럼에 HTML 템플릿 포맷터 적용
        columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]  # 수정됨

        data_table_fig = DataTable(source=source, columns=columns, width=1000, height=750, index_position=None)

        # 제목을 위한 Div 위젯 생성
        title_div = Div(text=f"<h2>최근 리밸런싱 내역: {self.latest_rebal_date.strftime('%Y-%m-%d')}</h2>", width=1000, height=30)


        # Div와 DataTable을 column 레이아웃으로 결합
        layout = column(title_div, data_table_fig)

        # show(layout)
        return layout
    def get_latest_rebaleffect_tbl_obj(self):
        static_data_tmp = self.latest_decompose_allocation_effect_NonReb#.reset_index().fillna(0)
        static_data_tmp = static_data_tmp.map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')

        static_data_tmp[static_data_tmp.isnull()] = ""
        # static_data_tmp = static_data_tmp.reset_index()
        # static_data=static_data_tmp[['code', '종목명', 'return contribution','previous', 'today', 'delta','previous_performance']].rename(
        #                     columns={'code':'종목코드', 'today':'최근리밸런싱', 'previous':'직전리밸런싱', 'delta':'변화', 'return contribution':'수익률 기여도', 'previous_performance':'직전리밸런싱 성과'})
        static_data=static_data_tmp.rename_axis("").rename('최근리밸런싱효과').reset_index()
        source = ColumnDataSource(static_data)

        # columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        # 폰트 크기를 조정하기 위한 HTML 템플릿 포맷터 생성
        formatter = HTMLTemplateFormatter(template='<div style="font-size: 16px;"><%= value %></div>')  # 수정됨

        # 각 컬럼에 HTML 템플릿 포맷터 적용
        columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]  # 수정됨

        data_table_fig = DataTable(source=source, columns=columns, width=int(1500*(1/3)), height=200, index_position=None)

        # 제목을 위한 Div 위젯 생성
        title_div = Div(text=f"<h2>최근 리밸런싱 효과: {self.second_latest_rebal_date.strftime('%Y-%m-%d')} ~ {self.latest_rebal_date.strftime('%Y-%m-%d')}</h2>", width=1000, height=30)


        # Div와 DataTable을 column 레이아웃으로 결합
        layout = column(title_div, data_table_fig)

        # show(layout)
        return layout

    def get_MTD_rtn_obj(self, toolbar_location):
        # Plot 복리
        MTD_price=gdu.data[self.Port_latest_rebalancing['today'].dropna().index].loc[self.Port_rebal_date_class.index[-1]:].dropna(how='all', axis=0)
        MTD_return = MTD_price.pct_change().add(1).cumprod().sub(1)
        MTD_return.iloc[0]=0
        source_for_chart = self.to_source(MTD_return)
        return_TS_obj = figure(x_axis_type='datetime',
                               title='MTD' + f'({MTD_return.index[0].strftime("%Y-%m-%d")} ~ {MTD_return.index[-1].strftime("%Y-%m-%d")})',
                               width=1500, height=450, toolbar_location=toolbar_location)

        return_TS_lgd_list = []
        for i, col in enumerate(MTD_return.columns):
            return_TS_line = return_TS_obj.line(source=source_for_chart, x=self.cum_ret_cmpd.index.name, y=col, color=self.color_list[i], line_width=2)
            today_w=self.Port_latest_rebalancing['today'].loc[col]
            return_TS_lgd_list.append((f'{col}[{self.code_to_name.loc[col,"종목명"]}] - ({int(today_w*10000)/100}%)', [return_TS_line]))
        return_TS_lgd = Legend(items=return_TS_lgd_list, location='center')
        return_TS_obj.add_layout(return_TS_lgd, 'right')
        return_TS_obj.legend.click_policy = "mute"
        return_TS_obj.yaxis.formatter = NumeralTickFormatter(format='0 %')

        # 마우스 올렸을 때 값 표시
        if self.hover:
            # HoverTool 설정을 위한 데이터 필드 목록 생성
            tooltips = [("Date", "@date{%F}")]
            tooltips += [(col, f"@{{{col}}}{{0.00%}}") for col in MTD_return.columns]

            hover = HoverTool(
                tooltips=tooltips,
                formatters={"@date": "datetime"}
            )
            return_TS_obj.add_tools(hover)
        return return_TS_obj
    def get_MTD_rtn_tbl_obj(self):
        MTD_price=gdu.data[self.Port_latest_rebalancing['today'].dropna().index].loc[self.Port_rebal_date_class.index[-1]:].dropna(how='all', axis=0)
        MTD_return = MTD_price.pct_change().add(1).cumprod().sub(1).iloc[-1]
        MTD_contribution = MTD_return.mul(self.Port_latest_rebalancing['today'].dropna())
        MTD_contribution.index = [self.code_to_name.loc[x,'종목명'] for x in MTD_contribution.index]


        static_data_tmp = MTD_contribution.copy()
        static_data_tmp = static_data_tmp.map(lambda x: "+"+str(int(x*10000)/100)+"%" if x>0 else f'({str(int(x * 10000) / 100)})%')

        static_data_tmp[static_data_tmp.isnull()] = ""
        # static_data_tmp = static_data_tmp.reset_index()
        # static_data=static_data_tmp[['code', '종목명', 'return contribution','previous', 'today', 'delta','previous_performance']].rename(
        #                     columns={'code':'종목코드', 'today':'최근리밸런싱', 'previous':'직전리밸런싱', 'delta':'변화', 'return contribution':'수익률 기여도', 'previous_performance':'직전리밸런싱 성과'})
        static_data=static_data_tmp.rename_axis("").rename('MTD기여도').reset_index()
        source = ColumnDataSource(static_data)

        # columns = [TableColumn(field=col, title=col) for col in static_data.columns]

        # 폰트 크기를 조정하기 위한 HTML 템플릿 포맷터 생성
        formatter = HTMLTemplateFormatter(template='<div style="font-size: 16px;"><%= value %></div>')  # 수정됨

        # 각 컬럼에 HTML 템플릿 포맷터 적용
        columns = [TableColumn(field=col, title=col, formatter=formatter) for col in static_data.columns]  # 수정됨

        data_table_fig = DataTable(source=source, columns=columns, width=int(1500*(1/3)), height=200, index_position=None)

        # 제목을 위한 Div 위젯 생성
        title_div = Div(text=f"<h2>MTD기여도: {MTD_price.index[0].strftime('%Y-%m-%d')} ~ {MTD_price.index[-1].strftime('%Y-%m-%d')}</h2>", width=1000, height=30)


        # Div와 DataTable을 column 레이아웃으로 결합
        layout = column(title_div, data_table_fig)

        # show(layout)
        return layout


if __name__ == "__main__":
    from tqdm import tqdm
    Hrisk_w_pvt_input = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='고위험',index_col=0,parse_dates=[0]).rename_axis('date', axis=0).rename_axis('code', axis=1)
    Mrisk_w_pvt_input = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='중위험',index_col=0,parse_dates=[0]).rename_axis('date', axis=0).rename_axis('code', axis=1)
    Lrisk_w_pvt_input = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='저위험',index_col=0,parse_dates=[0]).rename_axis('date', axis=0).rename_axis('code', axis=1)
    HB_w_pvt_input = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='H_BM_pvt', index_col=0, parse_dates=[0]).rename_axis('date', axis=0).rename_axis('code', axis=1)
    MB_w_pvt_input = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='M_BM_pvt', index_col=0, parse_dates=[0]).rename_axis('date', axis=0).rename_axis('code', axis=1)
    LB_w_pvt_input = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='L_BM_pvt', index_col=0, parse_dates=[0]).rename_axis('date', axis=0).rename_axis('code', axis=1)
    HB_w_pvt_input = HB_w_pvt_input[HB_w_pvt_input>0].dropna(how='all', axis=1)
    MB_w_pvt_input = MB_w_pvt_input[MB_w_pvt_input>0].dropna(how='all', axis=1)
    LB_w_pvt_input = LB_w_pvt_input[LB_w_pvt_input>0].dropna(how='all', axis=1)

    # Hrisk_w_pvt_input.loc[pd.to_datetime("2024-02-01")] = Hrisk_w_pvt_input.iloc[-1]
    # Mrisk_w_pvt_input.loc[pd.to_datetime("2024-02-01")] = Mrisk_w_pvt_input.iloc[-1]
    # Lrisk_w_pvt_input.loc[pd.to_datetime("2024-02-01")] = Lrisk_w_pvt_input.iloc[-1]
    # HB_w_pvt_input.loc[pd.to_datetime("2024-02-01")] = HB_w_pvt_input.iloc[-1]
    # MB_w_pvt_input.loc[pd.to_datetime("2024-02-01")] = MB_w_pvt_input.iloc[-1]
    # LB_w_pvt_input.loc[pd.to_datetime("2024-02-01")] = LB_w_pvt_input.iloc[-1]


    BM_code_to_name={'A069500': 'KODEX 200',
                    'A148070': 'KOSEF 국고채10년',
                   'A379800': 'KODEX 미국S&P500TR',
                  'A308620': 'KODEX 미국채10년선물'}
    # BM_code_to_index={'A069500': '국내주식',
    #                 'A148070': '국내채권',
    #                'A379800': '해외주식',
    #               'A308620': '해외채권'}
    BM_code_to_index={'KBPMABIN': '국내채권',
                      "KOSPI Index":'국내주식', 'LEGATRUU Index': '해외채권',
                      'RUGL Index':'리츠', 'SPGSGC Index':'금',
                      'NDUEEGF Index':'신흥주식','NDDUWI Index':'선진주식'} # NDUEACWF:해외주식

    index_price = pd.read_excel(f'Shinhan_Robost.xlsx', sheet_name='Index', index_col=0, parse_dates=[0]).rename(columns=BM_code_to_index)
    index_price['대안자산'] = index_price[['금','리츠']].pct_change().mean(1).fillna(0).add(1).cumprod()
    index_price = index_price.pct_change().add(1).cumprod()
    index_price.iloc[0]=1

    price_df = pd.DataFrame()
    for stock in tqdm(set(Mrisk_w_pvt_input.columns.to_list())):
        tmp = gdu.get_data.get_naver_close(stock[1:]).rename(columns=lambda x: stock)
        price_df = pd.concat([price_df, tmp], axis=1)
    # price_df.loc[pd.to_datetime("2024-02-01")] = price_df.iloc[-1]
    # price_df.loc[pd.to_datetime("2024-02-02")] = price_df.iloc[-1]
    # index_price.loc[pd.to_datetime("2024-02-01")] = index_price.iloc[-1]
    # index_price.loc[pd.to_datetime("2024-02-02")] = index_price.iloc[-1]
    Stock_Daily_price_input = price_df.loc[:"2024-02-13"].copy()
    Index_Daily_price_input = index_price.copy()
    # Index_Daily_price_input = Stock_Daily_price_input[['A069500','A148070','A379800','A308620']].rename(columns=BM_code_to_index)


    Asset_info = pd.read_excel(f'./Shinhan_Robost.xlsx', sheet_name='표1')[['단축코드', '한글종목약명','기초시장분류', '기초자산분류']]
    Asset_info.columns = ['종목코드', '종목명','class1', 'class2']
    Asset_info['종목코드'] = Asset_info['종목코드'].astype(str).apply(lambda x: 'A'+'0'*(6-len(x))+x)
    Asset_info['class'] = Asset_info['class1'] + Asset_info['class2']
    # Asset_info_input = Asset_info[['종목코드', '종목명','class']]
    Asset_info_input = Asset_info.set_index('종목코드').loc[list(set(Hrisk_w_pvt_input.columns)|set(Mrisk_w_pvt_input.columns)|set(Lrisk_w_pvt_input.columns))]
    Asset_info_input.loc['A153130', 'class'] = '국내채권'
    Asset_info_input.loc['A195920', 'class'] = '선진주식'
    Asset_info_input.loc['A200250', 'class'] = '신흥주식'
    Asset_info_input.loc['A272560', 'class'] = '국내채권'
    Asset_info_input.loc['A273130', 'class'] = '국내채권'
    Asset_info_input.loc['A329750', 'class'] = '해외채권'
    Asset_info_input.loc['A411060', 'class'] = '대안자산'
    Asset_info_input.loc['A437080', 'class'] = '해외채권'
    Asset_info_input.loc['A452360', 'class'] = '선진주식'
    Asset_info_input.loc['A455850', 'class'] = '선진주식'
    Asset_info_input.loc['A329200', 'class'] = '대안자산'
    Asset_info_input.loc['A302190', 'class'] = '국내채권'
    Asset_info_input = Asset_info_input.reset_index()[['종목코드', '종목명', 'class']]


    self=BrinsonHoodBeebower_PortfolioAnalysis(Lrisk_w_pvt_input.div(100),LB_w_pvt_input,Asset_info_input,Stock_Daily_price_input,Index_Daily_price_input,yearly=False,outputname='./LL_risk')
    # self.BrinsonHoodBeebower_report()

    BrinsonHoodBeebower_PortfolioAnalysis(Hrisk_w_pvt_input.div(100),HB_w_pvt_input,Asset_info_input,Stock_Daily_price_input,Index_Daily_price_input,yearly=False,outputname='./H_risk_20240213').BrinsonHoodBeebower_report()
    BrinsonHoodBeebower_PortfolioAnalysis(Mrisk_w_pvt_input.div(100),MB_w_pvt_input,Asset_info_input,Stock_Daily_price_input,Index_Daily_price_input,yearly=False,outputname='./M_risk_20240213').BrinsonHoodBeebower_report()
    BrinsonHoodBeebower_PortfolioAnalysis(Lrisk_w_pvt_input.div(100),LB_w_pvt_input,Asset_info_input,Stock_Daily_price_input,Index_Daily_price_input,yearly=False,outputname='./L_risk_20240213').BrinsonHoodBeebower_report()


    # ####################################################################### 섹터 배분 예시
    # # test_df = pd.read_pickle("./../test_df_top_all_sector.pickle")
    # # test_df_day = pd.read_pickle("./../test_df_sector_day.pickle")
    # # test_300df_day = pd.read_pickle("./../test_df_top_300_sector.pickle")
    # # save_as_pd_parquet("./../test_df_top_all_sector.hd5", test_df)
    # # save_as_pd_parquet("./../test_df_sector_day.hd5", test_df_day)
    # # save_as_pd_parquet("./../test_df_top_300_sector.hd5", test_300df_day)
    # # test_w = pd.read_excel('./../test_w_df_20220404.xlsx', index_col='date', parse_dates=['date'])
    # test_df = read_pd_parquet("./../test_df_top_all_sector.hd5")
    # test_df_day = read_pd_parquet("./../test_df_sector_day.hd5")
    # test_w = read_pd_parquet("./../test_w_20220404.hd5")
    #
    # stt_date = "2004-01-01"
    # test_df=test_df[test_df['date']>=stt_date]
    # test_df_day=test_df_day[test_df_day['date']>=stt_date]
    # test_w=test_w.loc[stt_date:]
    #
    #
    #
    # drop_list = ['A008080', 'A037030', 'A013890', 'A005560', 'A022220', 'A001780','A003260', 'A064520', 'A000030']
    # test_df_day = test_df_day[~test_df_day['종목코드'].isin(drop_list)]
    # test_df = test_df[~test_df['종목코드'].isin(drop_list)]
    # test_df = test_df[~test_df['종목코드'].isin(drop_list)]
    # daily_cap = test_df_day.pivot(index='date', columns='종목코드', values='시가총액')
    # daily_cap = daily_cap.loc[daily_cap.index[daily_cap.index.isin(test_w.index)]]
    # daily_cap300 = daily_cap[daily_cap.rank(ascending=False, axis=1, method='first') <= 300].dropna(how='all', axis=1).dropna(how='all', axis=0)
    # price_pvt = test_df_day.pivot(index='date', columns='종목코드', values='수정주가').assign(CASH=1)
    #
    # Asset_info_tmp = test_df[['종목코드', '종목명', 'sector']].rename(columns={'sector':'class'}).drop_duplicates()
    # Asset_info_tmp = Asset_info_tmp[~Asset_info_tmp['종목코드'].duplicated()].reset_index(drop='index')
    # Asset_info_tmp.loc[len(Asset_info_tmp)] = ['CASH', '현금', '무위험자산']
    # Asset_info_input = Asset_info_tmp.copy()
    #
    # # B_w_pvt_input = daily_cap300.div(daily_cap300.sum(axis=1), axis=0)
    # B_w_pvt_input = read_pd_parquet('./B_w_pvt_input.hd5')
    # P_w_pvt_input = test_w[test_w>0].dropna(how='all', axis=1).dropna(how='all', axis=0).loc[:B_w_pvt_input.index.max()]
    #
    # Stock_Daily_price_input = price_pvt.copy()
    # Index_Daily_price_input = price_pvt.pct_change().rename_axis('종목코드', axis=1).stack()\
    #     .rename('return').reset_index().merge(Asset_info_input, how='left', on='종목코드').fillna({'class':'unknown'})\
    #     .groupby(['class', 'date'])['return'].mean().swaplevel().unstack().add(1).cumprod()\
    #     .dropna(how='all', axis=0).dropna(how='all', axis=1)




    # 매우 간다
    # B_w_pvt_input, P_w_pvt_input = pd.read_excel(f'ETF_test.xlsx', sheet_name='BM_pvt',index_col=0,parse_dates=[0]), pd.read_excel(f'ETF_test.xlsx', sheet_name='MyPort_pvt', index_col=0, parse_dates=[0])
    # price_df = pd.DataFrame()
    # for stock in tqdm(B_w_pvt_input.columns.to_list() + P_w_pvt_input.columns.to_list()):
    #     tmp = gdu.get_data.get_naver_close(stock[1:]).rename(columns=lambda x: stock)
    #     price_df = pd.concat([price_df, tmp], axis=1)
    # price_df = price_df.loc[:"2022-12"]
    # gdu.data = price_df.copy()
    #
    # Asset_info = pd.read_excel(f'./ETF_test.xlsx', sheet_name='표1')[['단축코드', '기초시장분류', '기초자산분류']]
    # Asset_info.columns = ['code', 'class1', 'class2']
    # Asset_info['code'] = Asset_info['code'].astype(str).apply(lambda x: 'A'+'0'*(6-len(x))+x)
    # Asset_info['class'] = Asset_info['class1'] + Asset_info['class2']
    # Asset_info_input = Asset_info.set_index('code')['class']