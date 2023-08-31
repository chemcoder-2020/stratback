import pandas as pd
from stratback.utils.TOS_API_wrapper import (
    client,
)
import tda
from stratback.strategy import VWAPBounceStrategy
import numpy as np
import optuna
from stratback.walkforward import WalkforwardOptimization, objective_generator
import pandas_ta as pt
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.max_columns = 200

# constants
TOKEN_PATH = "tdtoken.json"
ACC_NUMBER = os.getenv("ACC_NUMBER")
API_KEY = os.getenv("API_KEY")
REDIRECT_URI = "https://localhost"

# Setting up the client
c = tda.auth.easy_client(
    token_path=TOKEN_PATH,  # follow this: https://developer.tdameritrade.com/content/simple-auth-local-apps
    api_key=API_KEY,
    redirect_uri=REDIRECT_URI,
)
c = client(session=c.session, api_key=c.api_key)

# ticker = "SPY"
# data = pd.read_csv(
#     rf"/Users/traderHuy/Documents/AITrading/TOS Bot 2/bigcaps_data/{ticker}_5m.csv"
# )
# data = data.drop_duplicates()
# data["date"] = pd.DatetimeIndex(data.date)

# Alpaca data
ticker = "QQQ"
data = pd.read_csv(
    rf"/Users/traderHuy/Documents/AITrading/TOS Bot 2/bigcaps_data/alpaca_{ticker}_5m.csv"
)
data = data.drop_duplicates()
data["date"] = [
    pd.to_datetime(dt, unit="ns", utc=True).tz_convert("US/Pacific") for dt in data.date
]
data["date"] = pd.DatetimeIndex(data.date)
data.dropna(inplace=True)
data.drop_duplicates(subset="date", inplace=True)


# data = c.get_five_minute_data_long(ticker)
# data.set_index("date", inplace=True)
# data = data[np.unique(data.index.date)[-60].strftime("%Y-%m-%d"):]

# optimized params
optimization_params = dict(
    # ntouch=(1, 5, 1, "int"),
    # HTF1=(
    #     [
    #         "1H",
    #         "2H",
    #         "4H",
    #         "D",
    #     ],
    #     "categorical",
    # ),
    # vwap_diff_n=(1, 10, 1, "int"),
    stop_pct = (0.005,0.05,0.001, "float"),
    pl_pct_tp = (0.005,0.05,0.001, "float"),
)
# optimization_targets = {"Avg. Trade WL Ratio": "maximize", "Calmar Ratio": "maximize"}
# optimization_targets = {"Avg. Trade WL Ratio": "maximize", "Calmar Ratio": "maximize", "Win Rate [%]": "maximize"}
optimization_targets = {"Avg. Trade WL Ratio": "maximize", "Win Rate [%]": "maximize"}
# optimization_targets = {"Avg. Trade WL Ratio": "maximize"}
# optimization_targets = {"Win Rate [%]": "maximize"}
# optimization_targets = {"Return (Ann.) [%]": "maximize"}

def best_trial_function(t):
    try:
        return np.round((2 * t.values[0] - 2.8) * np.abs(t.values[1]), 2)
    except Exception:
        return np.round((2 * t[0] - 2.8) * np.abs(t[1]), 2)


# data.set_index("date", inplace=True)
# base strategy_kwargs
strategy_kwargs = dict(
    size=None,
    long_only=False,
    short_only=False,
    HTF1="4H",
    HTF2="W",
    crossing_count_reset="1H",
    eod_time="12:50",
    support_rejection=True,
    ignore_vwap_crossabove=False,
    rolling_tf=False,
    vwap_diff_n=1,
    # filter_by_secondary_timeframe=True,
    # resistance_rejection=True,
    ntouch=1,
    # consider_wicks=True
    # entry_zone="('6:30','10:30')",
)  # initialize base strategy_kwargs

# settings
lookback = 178
wfo = WalkforwardOptimization(
    VWAPBounceStrategy,
    data.set_index("date"),
    lookback=lookback,
    optimization_params=optimization_params,
    optimization_targets=optimization_targets,
    strategy_kwargs=strategy_kwargs,
    best_trial_function=best_trial_function,
    max_trials=60,
    ticker=ticker,
    min_trials=50,
    patience=10,
    objective_generator=objective_generator,
    sampler=optuna.samplers.TPESampler(
        n_ei_candidates=24,
        n_startup_trials=10,
        multivariate=True,
        group=True,
    ),
    # sampler=optuna.samplers.GridSampler(
    #     {
    #         'ntouch': [1,2,3,4,5],
    #         'vwap_diff_n': [1,2,3,4,5,6,7,8,9,10],
    #     }
    # ),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3),
    storage="sqlite:///db.sqlite3",
    # walk_length=1200,
    # detailed_optimization=True,
    walk_step=1,
)
# wfo.optimize()
print(wfo.profit_surface_summary)

# wfo.walk()
