import pandas as pd
import datetime
import numpy as np
import optuna
import plotly.graph_objects as go
from backtesting import Backtest
from os import mkdir
from os.path import exists, join
import json

optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.display.max_columns = 200


class WalkforwardOptimization:
    """Walkforward Optimization for any strategy from backtesting, given the data"""

    def __init__(
        self,
        strategy,
        data,
        lookback,
        optimization_params,
        optimization_targets,
        strategy_kwargs,
        best_trial_function,
        **kwargs,
    ) -> None:
        self.strategy = strategy

        # data
        self.data = data.reset_index()
        self.data["day"] = pd.DatetimeIndex(self.data["date"]).date
        self.data.set_index("date", inplace=True)
        # self.data.drop(columns=["date"], inplace=True)
        self.days = self.data.day.unique()

        # other params
        self.lookback = lookback
        self.optimization_params = optimization_params
        self.optimization_targets = optimization_targets
        self.strategy_kwargs = strategy_kwargs
        self.best_trial_function = best_trial_function
        self.kwargs = kwargs
        self.walk_length = min(
            self.kwargs.get("walk_length", len(data) - self.lookback),
            len(self.days) - self.lookback,
        )
        self.sampler = self.kwargs.get(
            "sampler",
            optuna.samplers.TPESampler(
                n_ei_candidates=self.kwargs.get("n_ei_candidates", 24),
                n_startup_trials=self.kwargs.get("n_startup_trials", 10),
            ),
        )

    def backtest(data, strategy, **kwargs):
        data = data.copy()
        data.columns = data.columns.str.capitalize()
        try:
            data.set_index("Date", inplace=True)
        except Exception:
            pass

        bt = Backtest(
            data, strategy, cash=30000, commission=0.000, exclusive_orders=True
        )
        output = bt.run(**kwargs)
        return output

    def profit_surface(self):
        objective = objective_generator(
            self.data,
            self.strategy,
            strategy_kwargs=self.strategy_kwargs,
            optimization_params=self.optimization_params,
            optimization_targets=self.optimization_targets,
            best_trial_function=self.best_trial_function,
            **self.kwargs,
        )
        now = datetime.datetime.now()

        if len(self.optimization_targets) > 1:
            study = optuna.create_study(
                study_name=self.strategy.__name__ + str(now),
                sampler=self.sampler,
                directions=list(self.optimization_targets.values()),
                storage="sqlite:///db.sqlite3",
                load_if_exists=True,
                pruner=self.kwargs.get("pruner", None),
            )
        else:
            study = optuna.create_study(
                study_name=self.strategy.__name__ + str(now),
                sampler=self.sampler,
                direction=list(self.optimization_targets.values())[0],
                storage="sqlite:///db.sqlite3",
                load_if_exists=True,
                pruner=self.kwargs.get("pruner", None),
            )
        study.optimize(
            objective,
            n_trials=self.kwargs.get("max_trials", 1000),
            n_jobs=-1,
            show_progress_bar=True,
        )
        print(
            "Survey analysis done. run `optuna-dashboard sqlite:///db.sqlite3` in terminal to see the results"
        )

    def walk(self):
        trades = []
        params = []
        now = datetime.datetime.now()
        ticker = self.kwargs.get("ticker", "")
        if len(self.optimization_targets) > 1:
            s = "-".join(list(self.optimization_targets.keys()))
            nobj = f"MultiObj_{s}"
        else:
            nobj = f"SingleObj_{list(self.optimization_targets.keys())[0]}"
        folder_name = f"{self.strategy.__name__}_{ticker}_{self.lookback}_{nobj}_{self.walk_length}-{str(now).replace(':', '-')}"

        filename = join(folder_name, "settings.json")

        if not exists(folder_name):
            mkdir(folder_name)
        # if not exists(filename):
        with open(filename, "w") as outfile:
            subset_kwargs = {k: v for k, v in self.kwargs.items() if not callable(v)}
            subset_kwargs.update(self.optimization_params)
            subset_kwargs.update(self.optimization_targets)
            subset_kwargs.update(self.strategy_kwargs)
            # known non-serializable objects
            subset_kwargs["sampler"] = str(subset_kwargs["sampler"])
            subset_kwargs["pruner"] = str(subset_kwargs["pruner"])
            json.dump(subset_kwargs, outfile)

        for i in reversed(
            range(self.lookback + 1, self.walk_length + self.lookback + 1)
        ):
            print("Days left", i - self.lookback)

            if i == 1:
                j = None
            else:
                j = i - self.lookback

            self.in_sample_data = self.data[
                self.data.day.ge(self.days[-i]) & self.data.day.lt(self.days[-j])
            ].reset_index()  # Update in_sample_data for optim
            assert (
                len(self.in_sample_data["day"].unique()) == self.lookback
            ), f"in-sample data not equal to lookback {self.lookback}"

            objective = objective_generator(
                self.in_sample_data,
                self.strategy,
                strategy_kwargs=self.strategy_kwargs,
                optimization_params=self.optimization_params,
                optimization_targets=self.optimization_targets,
                best_trial_function=self.best_trial_function,
                **self.kwargs,
            )

            if len(self.optimization_targets) > 1:
                study = optuna.create_study(
                    study_name=self.strategy.__name__,
                    sampler=self.sampler,
                    directions=list(self.optimization_targets.values()),
                    pruner=self.kwargs.get("pruner",None)
                )
            else:
                study = optuna.create_study(
                    study_name=self.strategy.__name__,
                    sampler=self.sampler,
                    direction=list(self.optimization_targets.values())[0],
                    pruner=self.kwargs.get("pruner",None)
                )

            try:
                # print(best_params)
                additional_trial = best_params
                study.enqueue_trial(additional_trial)
            except Exception:
                print("best_params does not exist yet")

            # print(self.strategy_kwargs)

            study.optimize(
                objective,
                n_trials=self.kwargs.get("max_trials", 1000),
                n_jobs=-1,
                show_progress_bar=True,
            )
            if len(self.optimization_targets) > 1:
                best_trial = max(study.best_trials, key=self.best_trial_function)
                best_params = best_trial.params

            else:
                best_params = study.best_params

            backtest_params = self.strategy_kwargs.copy()

            for k, v in best_params.items():
                if int(v) == float(v):
                    backtest_params[k] = v
                else:
                    backtest_params[k] = np.floor(v * 1000) / 1000

            print(best_params)
            print(backtest_params)

            self.oos_data = self.data[
                self.data.day.gt(self.days[-i]) & self.data.day.le(self.days[-j])
            ].reset_index()  # Update out of sample data (next day walk)
            assert (
                len(self.oos_data["day"].unique()) == self.lookback
            ), f"out-of-sample data not equal to lookback {self.lookback}"

            apply_forward_output = WalkforwardOptimization.backtest(
                self.oos_data, self.strategy, **backtest_params
            )
            eq_curve = apply_forward_output._trades
            eq_curve["date"] = pd.DatetimeIndex(
                apply_forward_output._trades.EntryTime
            ).date
            todays_trades = eq_curve[
                pd.DatetimeIndex(eq_curve.date).date == self.oos_data.iloc[-1].day
            ]
            print(self.in_sample_data.iloc[-1].day)
            print(self.oos_data.iloc[-1].day)

            last_pl = todays_trades["PnL"].sum()
            print(f"Last PL: ${last_pl:.2f}")
            if len(todays_trades) > 0:
                assert np.all(
                    (todays_trades["ExitPrice"] - todays_trades["EntryPrice"])
                    * todays_trades["Size"]
                    == todays_trades["PnL"]
                ), f"PnL aggregations has a problem"
                todays_trades["Direction"] = todays_trades["Size"].gt(0)
                todays_trades["Direction"].replace(True, "Long", inplace=True)
                todays_trades["Direction"].replace(False, "Short", inplace=True)
                todays_trades["PnL_Low"] = (
                    todays_trades.apply(
                        lambda x: self.oos_data.set_index("date")
                        .loc[x["EntryTime"] : x["ExitTime"]]
                        .low.min()
                        if x["Size"] > 0
                        else self.oos_data.set_index("date")
                        .loc[x["EntryTime"] : x["ExitTime"]]
                        .high.max(),
                        axis=1,
                    )
                    - todays_trades["EntryPrice"]
                ) * todays_trades["Size"]

                todays_trades["PnL_High"] = (
                    todays_trades.apply(
                        lambda x: self.oos_data.set_index("date")
                        .loc[x["EntryTime"] : x["ExitTime"]]
                        .high.max()
                        if x["Size"] > 0
                        else self.oos_data.set_index("date")
                        .loc[x["EntryTime"] : x["ExitTime"]]
                        .low.min(),
                        axis=1,
                    )
                    - todays_trades["EntryPrice"]
                ) * todays_trades["Size"]

                trades.append(todays_trades)
                for i in range(len(todays_trades)):
                    params.append(best_params)
            else:
                continue

            print(todays_trades)
            trade_df = pd.concat(trades, ignore_index=True)
            trade_df["PnL_Low_perShare"] = trade_df["PnL_Low"] / trade_df["Size"].abs()
            trade_df["PnL_High_perShare"] = (
                trade_df["PnL_High"] / trade_df["Size"].abs()
            )
            trade_df["PnL_perShare"] = trade_df["PnL"] / trade_df["Size"].abs()
            # Typical SL
            typ_sl = (
                trade_df["PnL_Low_perShare"][trade_df["PnL"].gt(0)]
                .rolling(10)
                .apply(lambda x: np.percentile(x, 20))
            )
            trade_df.loc[typ_sl.index, "Typical SL"] = typ_sl
            trade_df["Typical SL"].ffill(inplace=True)

            # Typical TP
            typ_tp = (
                trade_df["PnL_High_perShare"][trade_df["PnL"].gt(0)]
                .rolling(10)
                .apply(lambda x: np.percentile(x, 20))
            )
            trade_df.loc[typ_tp.index, "Typical TP"] = typ_tp
            trade_df["Typical TP"].ffill(inplace=True)
            # Trade analysis
            trade_df = trade_df.set_index("EntryTime")
            walkforward_equity_diffs = trade_df["PnL"]
            trade_df["Direction"] = trade_df.apply(
                lambda x: "Long" if x["Size"] > 0 else "Short", axis=1
            )
            walkforward_equity_curve = walkforward_equity_diffs.cumsum() + 30000

            winrate_avg = walkforward_equity_diffs.replace(0, np.nan).dropna().gt(
                0
            ).sum() / len(walkforward_equity_diffs.replace(0, np.nan).dropna())
            profit_factor_avg = (
                walkforward_equity_diffs[walkforward_equity_diffs > 0].sum()
                / walkforward_equity_diffs[walkforward_equity_diffs < 0].abs().sum()
            )
            ehlers_criterion = (2 * winrate_avg - 1) * profit_factor_avg
            avg_win = walkforward_equity_diffs[walkforward_equity_diffs.gt(0)].mean()
            avg_loss = -walkforward_equity_diffs[walkforward_equity_diffs.lt(0)].mean()
            avg_wl_ratio = avg_win / avg_loss

            print(
                f"""
Optimized Output:

Winrate: {apply_forward_output["Win Rate [%]"]:.1f},
W/L Ratio: {apply_forward_output["Avg. Trade WL Ratio"]:.1f},
Profit Factor: {apply_forward_output["Profit Factor"]:.2f},

"""
            )
            print(
                "Average Daily Walk Forward Winrate:", np.round(winrate_avg * 100), "%"
            )
            print(
                "Average Daily Walk Forward Profit Factor:",
                np.round(profit_factor_avg, 1),
            )
            print(
                "Average Daily Walk Forward Ehlers Criterion:",
                np.round(ehlers_criterion, 2),
            )
            print(
                f"Average Win/Loss: {avg_win:.2f}/{avg_loss:.2f} = {avg_wl_ratio:.2f} for 1 share"
            )
            try:
                X = list(range(len(walkforward_equity_curve.dropna().index)))
                Y = walkforward_equity_curve.dropna().values
                fit = np.polyfit(X, Y, deg=1, full=True)
                z = fit[0]
                res = fit[1][0]
                res_from_mean = ((Y - Y.mean()) ** 2).sum()
                R2 = 1 - res / res_from_mean
                print(z)
                print("R-squared", R2)
                print("Daily profit: $", round(z[0]))
            except Exception as e:
                print(e)
                pass
            print("Current equity", walkforward_equity_curve.iloc[-1])

            winrates = walkforward_equity_diffs.replace(0, np.nan).dropna().gt(
                0
            ).cumsum() / np.array(
                np.arange(
                    1, len(walkforward_equity_diffs.replace(0, np.nan).dropna()) + 1
                )
            )

            walkforward_equity_diffs_pos = walkforward_equity_diffs.copy()
            walkforward_equity_diffs_neg = walkforward_equity_diffs.copy()

            walkforward_equity_diffs_pos[walkforward_equity_diffs < 0] = 0

            walkforward_equity_diffs_neg[walkforward_equity_diffs > 0] = 0
            profit_factors = (
                walkforward_equity_diffs_pos.cumsum()
                / walkforward_equity_diffs_neg.cumsum().abs()
            )

            ehlers_criterions = (2 * winrates - 1) * profit_factors
            plotly_df = pd.DataFrame()
            plotly_df["PnL"] = walkforward_equity_curve
            plotly_df["PnL_High"] = trade_df["PnL_High"]
            plotly_df["PnL_Low"] = trade_df["PnL_Low"].abs()
            plotly_df["date"] = walkforward_equity_curve.index
            plotly_df["Win Rate"] = winrates
            plotly_df["Profit Factor"] = profit_factors
            plotly_df["Ehlers"] = ehlers_criterions
            plotly_df["Direction"] = trade_df["Direction"]
            plotly_df["End"] = trade_df["ExitTime"]
            plotly_df["Week"] = (
                walkforward_equity_curve.index.isocalendar().week
                - walkforward_equity_curve.index.isocalendar().week.min()
                + 1
            )
            plotly_df["Trades/Week"] = (plotly_df.reset_index().index + 1) / (
                plotly_df["Week"].diff().fillna(0).cumsum() + 1
            )

            plotly_df["Typical SL"] = trade_df["Typical SL"].copy()
            plotly_df["Typical TP"] = trade_df["Typical TP"].copy()

            for e, idx in zip(params, plotly_df.index):
                for k, v in e.items():
                    plotly_df.loc[idx, k] = v

            plotly_df["PnL_Open"] = plotly_df["PnL"].shift().fillna(30000)
            plotly_df["PnL_High"] += plotly_df["PnL_Open"]
            plotly_df["PnL_Low"] = plotly_df["PnL_Open"] - plotly_df["PnL_Low"]

            text_columns = [
                "End",
                "Win Rate",
                "Profit Factor",
                "Ehlers",
                "Week",
                "Trades/Week",
                "Typical SL",
                "Typical TP",
                "Direction",
            ] + list(params[0].keys())

            text = plotly_df[text_columns].to_numpy()

            txt = []
            for t in text:
                st = ""
                for it, tc in enumerate(text_columns):
                    if type(t[it]) in [float, int]:
                        st += f"{tc}: {t[it]:.3f}<br>"
                    else:
                        st += f"{tc}: {t[it]}<br>"
                txt.append(st)

            candlestick = go.Candlestick(
                x=plotly_df["date"],
                open=plotly_df["PnL_Open"],
                high=plotly_df["PnL_High"],
                low=plotly_df["PnL_Low"],
                close=plotly_df["PnL"],
                text=txt,
                hoverinfo="all",
            )
            plotly_fig = go.Figure(data=[candlestick])
            plotly_fig.update_layout(xaxis_rangeslider_visible=False)

            plotly_fig.update_layout(
                title=f"{ticker} - WR {winrate_avg * 100:.0f}%, PF {profit_factor_avg:.1f}, Ehlers {ehlers_criterion:.2f} WL {avg_wl_ratio:.2f}. Trade Exposure: {plotly_df['Trades/Week'].mean():.1f} trades/week",
                yaxis_title="PnL",
                hovermode="x",
            )

            plotly_fig.write_html(
                join(
                    folder_name,
                    f"{self.strategy.__name__}_{ticker}_{self.lookback}_{nobj}_{self.walk_length}.html",
                )
            )


def objective_generator(
    in_sample_data,
    strategy,
    strategy_kwargs,
    optimization_params,
    optimization_targets,
    best_trial_function,
    **kwargs,
):
    """Generate objective function based on the current in_sample_data for optimization

    Args:
        in_sample_data (pd.DataFrame): current dataframe in the walkforward task
    """

    keys_to_remove = [k for k in strategy_kwargs.keys() if k not in dir(strategy)]
    [strategy_kwargs.pop(k) for k in keys_to_remove]

    def objective(trial):
        optimization_dict = {}
        for k, v in optimization_params.items():
            if v[-1] == "int":
                optimization_dict[k] = trial.suggest_int(
                    k, low=v[0], high=v[1], step=v[2]
                )
            elif v[-1] == "float":
                optimization_dict[k] = trial.suggest_float(
                    k, low=v[0], high=v[1], step=v[2]
                )

        backtest_params = strategy_kwargs.copy()
        backtest_params.update(optimization_dict)

        backtest_output = WalkforwardOptimization.backtest(
            in_sample_data, strategy, **backtest_params
        )
        obj = tuple(backtest_output[optimization_targets.keys()].values)

        if trial.number > kwargs.get("min_trials", 50):
            if len(optimization_targets) > 1:
                obj_value = np.round(best_trial_function(obj), 2)
                best_trial = max(
                    trial.study.best_trials,
                    key=best_trial_function,
                )

                if obj_value < np.round(
                    best_trial_function(best_trial.values), 2
                ) and trial.number > best_trial.number + kwargs.get("patience", 30):
                    print(f"Exiting optimization at {trial.number}")
                    trial.study.stop()  # stop if no improvements after n trials
            else:
                best_obj = trial.study.best_value
                best_trial = trial.study.best_trial
                if obj[0] < best_obj and trial.number > best_trial.number + kwargs.get(
                    "patience", 30
                ):
                    print(f"Exiting optimization at {trial.number}")
                    trial.study.stop()  # stop if no improvements after n trials
        return tuple(np.round(obj, 2))

    return objective
