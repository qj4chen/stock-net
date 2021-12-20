"""
根据基金持仓数据, 二部图, 随机游走, node2vec, 生成stock embedding
"""
import os.path

import networkx as nx
import json
from data_fetching import *
from node2vec.main import learn_stock_embedding
from data_preprocessing import fundHoldData

# default value
LOAD_EMBEDDING_MODEL_FROM_DISK = True
STOCK2VEC_MODEL_PATH = './output/model/'
NUM_STOCKS_TO_HOLD = 10
NUM_WALKS = 100
P = 5
Q = 0.1
EXP_PORTFOLIO_RETURN = 0.5

with open('./config.json', 'r') as f:
    globals().update(json.load(f)['stock_embedding'])


if not os.path.exists(STOCK2VEC_MODEL_PATH):
    os.makedirs(STOCK2VEC_MODEL_PATH)


def compose_model_filename(dt):
    return os.path.join(STOCK2VEC_MODEL_PATH, f'stock2vec_{datetime2str(dt)}.model')


def build_and_train_stock_embedding_model(dt,
                                          load_from_disk=LOAD_EMBEDDING_MODEL_FROM_DISK,
                                          edge='nav_w'):
    model_path = compose_model_filename(dt)
    if load_from_disk:
        try:
            from gensim.models import Word2Vec
            return Word2Vec.load(model_path)
        except FileNotFoundError:
            pass

    sample = fundHoldData.loc[fundHoldData['dt'] == dt, ['fund_id', 'wind_code', edge]].rename({edge: 'weight'},
                                                                                               axis=1)
    G = nx.from_pandas_edgelist(sample, 'fund_id', 'wind_code', 'weight')
    model = learn_stock_embedding(G, num_walks=NUM_WALKS, p=P, q=Q)
    model.save(model_path)
    return model


def fetch_stock_list_and_embedding_from_w2v_model(model):
    stock_ids = model.wv.index_to_key
    stock_embeddings = model.wv.get_normed_vectors()
    return stock_ids, stock_embeddings


def build_portfolio(dt,
                    model=None,
                    principal: int or float = 1_000_000,
                    exp_portfolio_return=EXP_PORTFOLIO_RETURN
                    ) -> tuple[pd.Series, float, str]:
    """
    :param dt:
    :param model:
    :param principal:
    :param exp_portfolio_return:
    :return:
    """
    dt = datetime2str(dt)
    if model is None:
        model = build_and_train_stock_embedding_model(dt)
    # now we have the stocks' vector representation. next, use the stock embedding to do portfolio management
    # stock embedding -> similarity matrix by node2vec -> as the risk matrix
    stock_ids, stock_embeddings = fetch_stock_list_and_embedding_from_w2v_model(model)
    risk_mat = np.dot(stock_embeddings, stock_embeddings.T)
    # what about add the term involving the raw covariance matrix to the objective func
    historic_return = get_historic_return(stock_ids=stock_ids, start=n_trx_days_before(dt, 20), end=dt)

    def obj_func(w):
        return np.dot(w.T, risk_mat).dot(w)

    from scipy.optimize import Bounds, minimize

    bounds = Bounds(lb=np.zeros(len(stock_ids)), ub=np.ones(len(stock_ids)))
    constraints = [
        {'type': 'eq',
         'fun': lambda x: np.dot(x, np.ones_like(x)) - 1},
        {'type': 'ineq',
         'fun': lambda x: np.dot(x, historic_return) - exp_portfolio_return}
    ]
    # 如果是ineq的constraint, 在scipy里表示>=0
    res = minimize(obj_func,
                   x0=np.ones_like(stock_ids, dtype=float) / len(stock_ids),
                   bounds=bounds,
                   constraints=constraints)

    # 选取份额在前十的股票持有
    index = np.argsort(res.x)
    index = index[-NUM_STOCKS_TO_HOLD:]
    stocks = np.array(stock_ids)[index]
    weights = res.x[index] / res.x[index].sum()
    holding_val = pd.Series(index=stocks, data=weights * principal)
    buying_dt = get_next_valid_trx_day(dt)
    buying_dt = datetime2str(buying_dt)

    holding_price = pro.daily(ts_code=','.join(stocks),
                              trade_date=buying_dt).set_index('ts_code')['close'].reindex(stocks)
    holding_quantities = holding_val / holding_price
    cash = holding_val[holding_quantities.isna()].sum()
    holding_quantities = holding_quantities[~holding_quantities.isna()]
    return holding_quantities, cash, buying_dt


def get_next_valid_trx_day(dt):
    return n_trx_days_before(dt, -1)


def calculate_daily_nav(holding_quantities: pd.Series,
                        buying_dt,
                        selling_dt=None,
                        cash=0.0) -> pd.Series:
    if selling_dt is None:
        selling_dt = pd.to_datetime(buying_dt) + pd.to_timedelta('30D')
        # todo: change the default selling_dt to the most recent trx_day
    buying_dt, selling_dt = datetime2str(buying_dt, selling_dt)
    stocks = holding_quantities.index.to_list()
    daily_k_lines = pro.daily(ts_code=','.join(stocks), start_date=buying_dt, end_date=selling_dt)
    daily_k_lines['trade_date'] = pd.to_datetime(daily_k_lines['trade_date'])
    # todo: 可能有股票在当天没有价格, 需要用前一天的价格填充 (在确保该股票还没有退市的基础上)
    # 填充价格, 每个股票, 每个交易日, 都应该有数据了, 只要该股票在portfolio建成的那天
    daily_k_lines = daily_k_lines.set_index(['ts_code', 'trade_date']).reindex(
        pd.MultiIndex.from_product([stocks, daily_k_lines['trade_date'].unique()], names=['ts_code', 'trade_date'])
    ).sort_index().groupby(level=0).fillna(method='ffill').reset_index()

    def calculate_asset_for_each_day(x):
        x.set_index('ts_code', inplace=True)
        return (x['close'] * holding_quantities).sum()

    daily_nav = daily_k_lines.groupby('trade_date').apply(calculate_asset_for_each_day)
    daily_nav = fill_non_trx_days(daily_nav)
    return daily_nav + cash


def fill_non_trx_days(series: pd.Series) -> pd.Series:
    start = series.index.min()
    end = series.index.max()
    return series.reindex(pd.date_range(start, end, freq='1D')).fillna(method='ffill')



