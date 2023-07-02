import pandas as pd
from stratback.utils.TOS_API_wrapper import (
    client,
)
import tda
from stratback.strategy import DoubleCloudMAStrategy
import numpy as np
import optuna
from stratback.walkforward import WalkforwardOptimization, objective_generator
import os
import dotenv

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.max_columns = 200

# constants
TOKEN_PATH = "tdtoken.json"
dotenv.load_dotenv("/.env")
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

ticker = "SPY"
data = pd.read_csv(
    rf"/data/{ticker}_5m.csv"
)
data = data.drop_duplicates()
data["date"] = pd.DatetimeIndex(data.date)


# optimized params
optimization_params = dict(
    # n1=(11, 17, 1, "int"),
    # n2=(23, 29, 1, "int"),
    # n3=(65, 71, 1, "int"),
    n1=(10, 20, 1, "int"),
    n2=(21, 40, 1, "int"),
    n3=(41, 80, 1, "int"),
    target_pct=(0.65, 0.75, 0.005, "float"),
)
optimization_targets = {"Avg. Trade WL Ratio": "maximize", "Sortino Ratio": "maximize"}

# base strategy_kwargs
strategy_kwargs = dict(
    size=None,
    long_only=False,
    short_only=False,
    assume_downtrend_follows_uptrend=False,
    with_longX=True,
    with_shortX=True,
)  # initialize base strategy_kwargs


def best_trial_function(t):
    try:
        return np.round((2 * t.values[0] - 2.8) * t.values[1], 2)
    except Exception:
        return np.round((2 * t[0] - 2.8) * t[1], 2)


# settings
lookback = 90

wfo = WalkforwardOptimization(
    DoubleCloudMAStrategy,
    data,
    lookback=lookback,
    optimization_params=optimization_params,
    optimization_targets=optimization_targets,
    strategy_kwargs=strategy_kwargs,
    best_trial_function=best_trial_function,
    max_trials=1000,
    ticker=ticker,
    min_trials=50,
    patience=30,
    objective_generator=objective_generator,
    sampler=optuna.samplers.TPESampler(
        n_ei_candidates=24,
        n_startup_trials=10,
        multivariate=True,
        group=True,
    ),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3),
)
# wfo.profit_surface()
wfo.walk()
