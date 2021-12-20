from data_preprocessing import key_dts
from stock_embedding import *

PRINCIPAL = 1_000_000
PORTFOLIO_PATH = './output/portfolio'
if not os.path.exists(PORTFOLIO_PATH):
    os.makedirs(PORTFOLIO_PATH)
assert np.intersect1d(key_dts, historic_last_trx_day_in_each_month).shape == key_dts.shape

daily_nav = pd.Series(dtype=float)
# for idx, dt in enumerate(key_dts[:-1]):
#     sub_loop = historic_last_trx_day_in_each_month[(key_dts[idx] <= historic_last_trx_day_in_each_month) & (
#             historic_last_trx_day_in_each_month < key_dts[idx + 1])]
#     dt = datetime2str(dt)
#     model = build_and_train_stock_embedding_model(dt)
#
#     for idx_, ddt in enumerate(sub_loop):
#         portfolio, cash, buying_dt = build_portfolio(dt=ddt, model=model, principal=PRINCIPAL)
#         try:
#             selling_dt = sub_loop[idx_ + 1]
#         except IndexError:
#             selling_dt = key_dts[idx + 1]
#         selling_dt = datetime2str(selling_dt)
#         daily_nav = daily_nav.append(
#             calculate_daily_nav(holding_quantities=portfolio,
#                                 buying_dt=buying_dt,
#                                 selling_dt=selling_dt,
#                                 cash=cash),
#         )
#         portfolio.to_csv(os.path.join(PORTFOLIO_PATH, f'{buying_dt}_{selling_dt}.csv'))
#         print(selling_dt[:-2],
#               ', 当月盈亏:', daily_nav[-1] - PRINCIPAL,
#               ', 剩余金额: ', daily_nav[-1],
#               '\n')
#         PRINCIPAL = daily_nav[-1]

for idx, dt in enumerate(key_dts[:-1]):
    dt = datetime2str(dt)
    model = build_and_train_stock_embedding_model(dt)

    portfolio, cash, buying_dt = build_portfolio(dt=dt, model=model, principal=PRINCIPAL)

    selling_dt = key_dts[idx + 1]
    selling_dt = datetime2str(selling_dt)
    daily_nav = daily_nav.append(
        calculate_daily_nav(holding_quantities=portfolio,
                            buying_dt=buying_dt,
                            selling_dt=selling_dt,
                            cash=cash),
    )
    portfolio.to_csv(os.path.join(PORTFOLIO_PATH, f'{buying_dt}_{selling_dt}.csv'))
    print(selling_dt[:-2],
          ', 当期盈亏:', daily_nav[-1] - PRINCIPAL,
          ', 剩余金额: ', daily_nav[-1],
          '\n')
    PRINCIPAL = daily_nav[-1]

daily_nav = fill_non_trx_days(daily_nav)
