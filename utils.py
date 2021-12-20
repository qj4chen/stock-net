import os
import time
from typing import List

import numpy as np
import requests
import tushare as ts
import pandas as pd
from datetime import datetime


def dt2str(*args,
                 datetime_format='%Y%m%d'):
    tmp = pd.to_datetime(args).strftime(datetime_format).values.astype(str).tolist()
    if len(tmp) == 1:
        return tmp[0]
    else:
        return tmp

def fetch_stock_list_and_embedding_from_w2v_model(model):
    stock_ids = model.wv.index_to_key
    stock_embeddings = model.wv.get_normed_vectors()
    return stock_ids, stock_embeddings



def compose_filename_from_ts_code_and_trade_date(prefix, ts_code, trade_date):
    return prefix + '_' + '_'.join((ts_code, trade_date)) + '.pkl'


def save_to_local(result: pd.DataFrame, prefix: str):
    if 'ts_code' in result.columns and 'trade_date' in result.columns:
        for ind, _df in iter(result.groupby(['ts_code', 'trade_date'])):
            filename = compose_filename_from_ts_code_and_trade_date(prefix, *ind)
            _df.to_pickle(filename)


def is_trade_date():
    pass


def load_from_local(ts_code, start_date, end_date, prefix):
    if isinstance(ts_code, str):
        ts_code = ts_code.split(',')
    trade_dates = pd.date_range(start_date, end_date)


    trade_dates = [t.strftime('%Y%m%d') for t in trade_dates]
    trade_dates = [t for t in trade_dates if is_trade_date(t)]

    from itertools import product
    result = []
    to_fetch_from_api = []
    for tc, td in product(ts_code, trade_dates):
        filename = compose_filename_from_ts_code_and_trade_date(ts_code=tc, trade_date=td, prefix=prefix)
        if os.path.exists(filename):
            result.append(pd.read_pickle(filename))
        else:
            to_fetch_from_api.append({'ts_code': tc, 'start_date': td, 'end_date': td})
    if result:
        result = pd.concat(result)
    else:
        result = pd.DataFrame()
    return result, to_fetch_from_api


def load_local_content(func):
    def decorated_func(ts_code, start_date, end_date, *args, **kwargs):
        prefix = func.__name__
        result, to_fetch_from_api = load_from_local(ts_code=ts_code,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    prefix=prefix)
        if not to_fetch_from_api:
            return result
        else:
            new_result = pd.DataFrame()
            for param in to_fetch_from_api:
                new_result = new_result.append(func(*args, **param, **kwargs))
            save_to_local(new_result, prefix=prefix)
            result = result.append(new_result)
        return result
    return decorated_func


class Portfolio(pd.Series):
    def __init__(self,
                 stocks_to_hold: List or np.ndarray = None,
                 quantities_to_hold: List or np.ndarray = None,
                 portfolio: pd.Series = None,
                 buying_date: datetime or str or pd.Timestamp = None,
                 ):
        """
        :param stocks_to_hold:
        :param quantities_to_hold:
        :param portfolio: 每只股票持有的数量
        :param buying_date:
        """

        if portfolio is None:
            super().__init__(index=stocks_to_hold, data=quantities_to_hold)
        else:
            super().__init__(data=portfolio)
        self.buying_date = dt2str(buying_date)


def is_trx_day(date_to_verify):
    date_to_verify = dt2str(date_to_verify)
    return date_to_verify in BackTestApiStatic.__trx_dates__


def next_trx_day(current_trx_day):
    current_trx_day = dt2str(current_trx_day)
    return BackTestApiStatic.__trx_dates__[
        np.argwhere(BackTestApiStatic.__trx_dates__ == current_trx_day).flatten() + 1]


class BackTestApiStatic:
    __dataframe_columns_mapping__ = {
        'ts_code': 'stock_code',
        'trade_date': 'trade_date',
        'close': 'close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'vol': 'volume',
        'amount': 'amount',
    }

    __token__ = '6a721053ea3e70bb52605d6c0972caeda9ff080d3671f69bd8b6b434'
    __batch_size__ = 100  # restriction from tushare api
    __today__ = datetime.today().strftime('%Y%m%d')
    ts_api = ts.pro_api(token=__token__)
    try:
        __trx_dates__ = np.load(file=f'./trx_calendar_{__today__}.npy')
    except FileNotFoundError:
        __trx_dates__ = ts_api.query('trade_cal', end_date=__today__)
        __trx_dates__ = __trx_dates__.loc[__trx_dates__['is_open'] == 1, 'cal_date'].values.tolist()
        __trx_dates__ = np.array(__trx_dates__)
        np.save(arr=__trx_dates__, file=f'./trx_calendar_{__today__}.npy')

    def __init__(self,
                 start_date: datetime or str or pd.Timestamp,
                 end_date: datetime or str or pd.Timestamp,
                 stock_pool: List or np.ndarray,
                 porforlio: Portfolio):
        """
        初始化回测框架, 在 start date 到 end date 这段时间内, 不调仓
        :param start_date: 资产组合的买入日
        :param end_date: 资产组合的卖出日
        :param stock_pool: 股票池
        """
        self.start_date = dt2str(start_date)
        self.end_date = dt2str(end_date)
        self.portfolio = porforlio
        self.stock_pool = self.portfolio.index.values
        self.num_stocks = len(self.stock_pool)
        self.stock_market_data = self.get_stock_market_data()
        self.date_range = pd.date_range(self.start_date, self.end_date, freq='1D')
        self.nav = self.get_daily_nav()


    def get_stock_market_data(self):
        if self.num_stocks > BackTestApiStatic.__batch_size__:
            l_idx = 0
            r_idx = BackTestApiStatic.__batch_size__
            response = pd.DataFrame()
            while True:
                response = response.append(
                    BackTestApiStatic.ts_api.daily(ts_code=','.join(self.stock_pool[l_idx:r_idx]),
                                                   start_date=self.start_date,
                                                   end_date=self.end_date),
                    ignore_index=True
                )
                l_idx = r_idx
                r_idx += BackTestApiStatic.__batch_size__
                if l_idx >= self.num_stocks:
                    break
                if r_idx > self.num_stocks:
                    r_idx = self.num_stocks
                time.sleep(0.1)
        else:
            response = BackTestApiStatic.ts_api.daily(ts_code=','.join(self.stock_pool),
                                                      start_date=self.start_date,
                                                      end_date=self.end_date)

        response = response.rename(BackTestApiStatic.__dataframe_columns_mapping__,
                               axis='columns')
        response['trade_date'] = pd.to_datetime(response['trade_date'])
        return response

    def get_daily_nav(self):
        for dt in self.date_range:



    def __repr__(self):
        return self

    @classmethod
    def init_by_class_method(cls, start_date, end_date, stock_pool, num_s):
        pass


    def __add__(self, other):
        assert self.end_date == other.start_date
        tmp = BackTestApiStatic(start_date=self.start_date,
                                end_date=other.end_date)




