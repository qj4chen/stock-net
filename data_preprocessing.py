import os
import json
import numpy as np
import pandas as pd

from data_fetching import get_trx_dates


# 导入原始的基金持仓数据
fundHoldData = pd.read_pickle('./data/fund_holding_data.pkl')

# 筛选基金持仓数据
if not os.path.exists('./output'):
    os.makedirs('./output')

# default value
FILTERED_DATA_PATH = './output/fund_holding_data_filtered.pkl'
RESTART_PREPROCESSING = True  # 是否重新开始数据预处理过程， 即不从硬盘加载之前预处理过的数据
FILL_BY_MONTH = False  # 是否根据季度报或者年报半年报填充到月
USE_ANNUAL_REPORT = False
STOCK_WEIGHT_THRESHOLD = 0.6

with open('./config.json', 'r') as f:
    globals().update(json.load(f)['data_preprocessing'])

if not RESTART_PREPROCESSING:
    if os.path.exists(FILTERED_DATA_PATH):
        fundHoldData = pd.read_pickle(FILTERED_DATA_PATH)
else:
    # 选取半年报或者年报披露的持仓数据
    if USE_ANNUAL_REPORT:
        fundHoldData = fundHoldData.loc[fundHoldData['replace'] == 0]
    else:
        fundHoldData = fundHoldData.loc[fundHoldData['replace'] == 1]

    # 选取连续四期持股比例超过阈值的数据
    criteria = fundHoldData.groupby(['fund_id', 'dt'])['nav_w'].sum().to_frame()
    criteria.reset_index(inplace=True)
    criteria = criteria.groupby(['fund_id']).rolling(window=4, on='dt').min().pipe(
        lambda x: x[x['nav_w'] >= STOCK_WEIGHT_THRESHOLD]
    ).reset_index()

    fundHoldData = fundHoldData.loc[pd.MultiIndex.from_frame(fundHoldData[['dt', 'fund_id']]).isin(
        pd.MultiIndex.from_frame(criteria[['dt', 'fund_id']])), ]

    del criteria

    # 剔除港股
    fundHoldData = fundHoldData[~fundHoldData['wind_code'].str.endswith('.HK')]

    # 选取每只基金的前10大持仓股
    fundHoldData = fundHoldData.sort_values(by='sz', ascending=False).groupby(['dt', 'fund_id']).head(10)

    if FILL_BY_MONTH:
        # 获取每个月的最后一个交易日
        start_date, end_date = fundHoldData['dt'].agg(['min', 'max']).to_list()
        td = pd.DataFrame(index=get_trx_dates(start_date, end_date),
                          columns=['year', 'month', 'date'])
        td['year'] = td.index.year
        td['month'] = td.index.month
        td['date'] = td.index
        td = td.groupby(['year', 'month'])['date'].last().values

        # 填充数据, 至每个月末都有数据
        raw_td = fundHoldData.dt.unique()
        assert np.intersect1d(td, raw_td).shape == raw_td.shape, '数据的时间戳可能存在问题，请检查！'

        raw_td.sort()
        td = td[td >= raw_td[0]]
        for d in td:
            if d in raw_td:
                tmp = fundHoldData.loc[fundHoldData['dt'] == d, :]
                continue
            tmp['dt'] = d
            fundHoldData = fundHoldData.append(tmp)
        del tmp

    fundHoldData.to_pickle(FILTERED_DATA_PATH)


key_dts = fundHoldData.dt.unique()
key_dts.sort()
num_periods = len(key_dts)