import pandas as pd
from stratback.utils.TOS_API_wrapper import (
    client,
)
import tda
from strategy.singleCloudMAStrategy import SingleCloudMAStrategy
import numpy as np
import optuna
from walkforward.walkforward import WalkforwardOptimization, objective_generator

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.max_columns = 200

# constants
TOKEN_PATH = "tdtoken.json"
ACC_NUMBER = "270199869"
API_KEY = "MTPZYEHVAXFY9Y3MPO1RF56RXG2Q7EP1"
REDIRECT_URI = "https://localhost"

# Setting up the client
c = tda.auth.easy_client(
    token_path=TOKEN_PATH,  # follow this: https://developer.tdameritrade.com/content/simple-auth-local-apps
    api_key=API_KEY,
    redirect_uri=REDIRECT_URI,
)
c = client(session=c.session, api_key=c.api_key)

ticker = "SPY"
data = pd.read_csv(
    rf"/Users/traderHuy/Documents/AITrading/TOS Bot 2/bigcaps_data/{ticker}_5m.csv"
)
data = data.drop_duplicates()
data["date"] = pd.DatetimeIndex(data.date)
# data.set_index("date", inplace=True)
# logic = {
#     "open": "first",
#     "close": "last",
#     "high":"max",
#     "low":"min",
#     "volume":"sum"
# }
# data = data.resample("15min").agg(logic).dropna()

# optimized params

optimization_params = dict(
    n1=(10, 20, 1, "int"),
    n2=(21, 40, 1, "int"),
    target_pct=(0.65, 0.75, 0.005, "float"),
)
optimization_targets = {"Avg. Trade WL Ratio": "maximize", "Sortino Ratio": "maximize"}

# base strategy_kwargs
strategy_kwargs = dict(
    size=None,
    long_only=False,
    short_only=False,
    with_longX=False,
    with_shortX=False,
    assume_downtrend_follows_uptrend=False,
    daytrade=True,
)  # initialize base strategy_kwargs


def best_trial_function(t):
    try:
        return np.round((2 * t.values[0] - 2.8) * t.values[1], 2)
    except Exception:
        return np.round((2 * t[0] - 2.8) * t[1], 2)


# Walk settings
lookback = 90

wfo = WalkforwardOptimization(
    SingleCloudMAStrategy,
    data,
    lookback=lookback,
    optimization_params=optimization_params,
    optimization_targets=optimization_targets,
    strategy_kwargs=strategy_kwargs,
    best_trial_function=best_trial_function,
    max_trials=1000,
    ticker=ticker,
    min_trials=30,
    patience=30,
    objective_generator=objective_generator,
)
wfo.walk()
