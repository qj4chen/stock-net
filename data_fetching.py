import time
import os
from abc import abstractmethod
from typing import List, Any

import numpy as np
import pandas as pd
import tushare as ts
pro = ts.pro_api(token='6a721053ea3e70bb52605d6c0972caeda9ff080d3671f69bd8b6b434')


def retry(times: int, exceptions: Any):
    def decorator(func):
        def new_func(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(
                        'Exception thrown when attempting to run %s, attempt '
                        '%d of %d' % (func, attempt, times)
                    )
                    attempt += 1
            return func(*args, **kwargs)

        return new_func

    return decorator


# def compose_filename_from_ts_code_and_trade_date(prefix, ts_code, trade_date):
#     return prefix + '_' + '_'.join((ts_code, trade_date)) + '.pkl'
#
#
# def save_to_local(result: pd.DataFrame, prefix: str):
#     if 'ts_code' in result.columns and 'trade_date' in result.columns:
#         for ind, _df in iter(result.groupby(['ts_code', 'trade_date'])):
#             filename = compose_filename_from_ts_code_and_trade_date(prefix, *ind)
#             _df.to_pickle(filename)
#
#
# def is_trade_date():
#     pass
#
#
# def load_from_local(ts_code, start_date, end_date, prefix):
#     if isinstance(ts_code, str):
#         ts_code = ts_code.split(',')
#     trade_dates = pd.date_range(start_date, end_date)
#
#
#     trade_dates = [t.strftime('%Y%m%d') for t in trade_dates]
#     trade_dates = [t for t in trade_dates if is_trade_date(t)]
#
#     from itertools import product
#     result = []
#     to_fetch_from_api = []
#     for tc, td in product(ts_code, trade_dates):
#         filename = compose_filename_from_ts_code_and_trade_date(ts_code=tc, trade_date=td, prefix=prefix)
#         if os.path.exists(filename):
#             result.append(pd.read_pickle(filename))
#         else:
#             to_fetch_from_api.append({'ts_code':tc, 'start_date': td, 'end_date': td})
#     if result:
#         result = pd.concat(result)
#     else:
#         result = pd.DataFrame()
#     return result, to_fetch_from_api
#
#
# def load_local_content(func):
#     def decorated_func(ts_code, start_date, end_date, *args, **kwargs):
#         prefix = func.__name__
#         result, to_fetch_from_api = load_from_local(ts_code=ts_code,
#                                                     start_date=start_date,
#                                                     end_date=end_date,
#                                                     prefix=prefix)
#         if not to_fetch_from_api:
#             return result
#         else:
#             new_result = pd.DataFrame()
#             for param in to_fetch_from_api:
#                 new_result = new_result.append(func(*args, **param, **kwargs))
#             save_to_local(new_result, prefix=prefix)
#             result = result.append(new_result)
#         return result
#     return decorated_func


def datetime2str(*args,
                 datetime_format='%Y%m%d'):
    tmp = pd.to_datetime(args).strftime(datetime_format).values.astype(str).tolist()
    if len(tmp) == 1:
        return tmp[0]
    else:
        return tmp


def get_trx_dates(start, end):
    start, end = datetime2str(start, end)
    response = pro.trade_cal(exchange='', start_date=start, end_date=end)
    return pd.to_datetime(response.loc[response['is_open'] == 1, 'cal_date'].values)


# todo: 应该是格林尼治时间，转化为 local time或者 +8小时
historic_trx_days = get_trx_dates(start='2001-01-01', end=datetime2str(pd.to_datetime(time.time(), unit='s')))
historic_trx_days_dataframe = pd.DataFrame({'date': historic_trx_days})
historic_trx_days_dataframe['year'] = historic_trx_days_dataframe['date'].dt.year
historic_trx_days_dataframe['month'] = historic_trx_days_dataframe['date'].dt.month
historic_last_trx_day_in_each_month = historic_trx_days_dataframe.groupby(['year', 'month'])['date'].last().values


def n_trx_days_before(dt, n=20):
    """
    返回dt的前n个交易日
    :param n:
    :param dt:
    :return:
    """
    dt = datetime2str(dt)
    return historic_trx_days[np.argwhere(historic_trx_days == dt).flatten()[0] - n]


def get_historic_return(stock_ids,
                        start=None,
                        end=None,
                        batch_size=100
                        ) -> np.ndarray:
    """
    返回指定股票过去20个交易日的累计涨跌幅,
    Note: tushare的api接口限制一次只能查询100个stock_id
    :param batch_size:
    :param start:
    :param end:
    :param stock_ids:
    :return:
    """
    response = fetch_stock_market_data(batch_size, end, start, stock_ids)

    return response.groupby('ts_code').apply(get_cumulated_pct_chg).reindex(stock_ids).fillna(value=-100000).values


def fetch_stock_market_data(batch_size, end, start, stock_ids):
    start = datetime2str(start)
    end = datetime2str(end)
    if isinstance(stock_ids, list):
        num_stock_ids = len(stock_ids)
        if num_stock_ids > batch_size:
            l_idx = 0
            r_idx = batch_size
            response = pd.DataFrame()
            while True:
                response = response.append(
                    pro.daily(ts_code=','.join(stock_ids[l_idx:r_idx]), start_date=start, end_date=end),
                    ignore_index=True
                )
                l_idx = r_idx
                r_idx += batch_size
                if l_idx >= num_stock_ids:
                    break
                if r_idx > num_stock_ids:
                    r_idx = num_stock_ids
                time.sleep(0.1)
        else:
            response = pro.daily(ts_code=','.join(stock_ids), start_date=start, end_date=end)
    else:
        response = pro.daily(ts_code=stock_ids, start_date=start, end_date=end)
    return response


def get_cumulated_pct_chg(x):
    x['trade_date'] = pd.to_datetime(x['trade_date'])
    x.sort_values('trade_date', inplace=True)
    return x['close'].iloc[-1] / x['close'].iloc[0] - 1


@abstractmethod
def is_trx_date(dt):
    pass