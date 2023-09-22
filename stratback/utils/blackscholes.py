# -----------------------------------------------------------------------
# blackscholes.py - Princeton CS
# -----------------------------------------------------------------------

from datetime import date, datetime, timedelta
import math
from time import time

import numpy as np
import yfinance as yf

np.seterr(all="warn")
import warnings
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
from functools import cached_property
import pandas as pd
import pandas_ta as pt


# -----------------------------------------------------------------------

# Return the value of the Gaussian probability function with mean 0.0
# and standard deviation 1.0 at the given x value.


def phi(x):
    return np.exp(-x * x / 2.0) / np.sqrt(2.0 * np.pi)


# -----------------------------------------------------------------------

# Return the value of the Gaussian probability function with mean mu
# and standard deviation sigma at the given x value.


def pdf(x, mu=0.0, sigma=1.0):
    return phi((x - mu) / sigma) / sigma


# -----------------------------------------------------------------------

# Return the value of the cumulative Gaussian distribution function
# with mean 0.0 and standard deviation 1.0 at the given z value.


def Phi(z):
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    total = 0.0
    term = z
    i = 3
    while total != total + term:
        total += term
        term *= z * z / float(i)
        i += 2
    return 0.5 + total * phi(z)


# -----------------------------------------------------------------------

# Return standard Gaussian cdf with mean mu and stddev sigma.
# Use Taylor approximation.


def cdf(z, mu=0.0, sigma=1.0):
    return Phi((z - mu) / sigma)


# -----------------------------------------------------------------------

# Black-Scholes formula.


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


prepost = False


def custom_round(x, base=5):
    return np.round(base * np.round(np.float64(x) / base), 2)


def round_to_1(x):
    return np.round(x, -np.int64(np.floor(np.log10(np.abs(x)))))


class Alerter:
    def __init__(
        self,
        ticker,
        support=None,
        resistance=None,
        prepost=False,
        close="Close",
        high="High",
        low="Low",
        timeframes=["max:5d", "2y:1d", "30d:60m"],
        sup_res_thresholds=(90, 10),
        sup_res_timeframe="180d:60m",
        mfi_length=None,
    ) -> None:
        self.ticker = ticker
        self.timeframes = timeframes
        self.timeframes.append("2d:5m")  # default timeframe
        self.prepost = prepost
        self.dfs = None
        self.df_setter()
        self.sup_res_thresholds = list(sup_res_thresholds)
        self.sup_res_timeframe = sup_res_timeframe
        self.close = close
        self.high = high
        self.low = low
        self.mfis = {}
        # self.mfi_signals = None
        self.mfi_angles = {}
        self.mfi_diffs = {}
        self.mfi_length = mfi_length

        self._five_min_data = None

        self.support = support
        self.resistance = resistance

        self.mfi_codes = {True: "IN", False: "OUT", None: "Neutral"}

        self.price_trend_codes = {True: "UP", False: "DOWN", None: "Neutral"}

        self.alert_codes = {
            0: r"📈",
            1: r"⤻",
            2: r"↷",
            3: r"📉",
        }

        self.cci_codes = {
            2: "Strong Bullish Weekly Trend",
            1: "Bullish Weekly Trend",
            0: "Flat Weekly Trend",
            -1: "Bearish Weekly Trend",
            -2: "Strong Bearish Weekly Trend",
            None: None,
        }

        self.cci_daily_codes = {
            2: "Strong Bullish 3-Day Trend",
            1: "Bullish 3-Day Trend",
            0: "Flat 3-Day Trend",
            -1: "Bearish 3-Day Trend",
            -2: "Strong Bearish 3-Day Trend",
            None: None,
        }

        if support is None or resistance is None:
            # self.get_nearest_levels(sup_res_thresholds=self.sup_res_thresholds)
            self.support, self.resistance = self.nearest_levels

        if self.resistance is None:
            self.resistance = self.dfs[self.timeframes[1]][
                self.dfs[self.timeframes[1]].columns[1]
            ][-150:].max()

        if self.support is None:
            self.support = self.dfs[self.timeframes[1]][
                self.dfs[self.timeframes[1]].columns[2]
            ][-150:].min()

        assert self.support is not None, "Support is None"
        assert self.resistance is not None, "Resistance is None"

    def data_getter(self, ticker, interval, period, prepost=False):
        intrvl = interval

        if intrvl in ["4h", "240m", "90m", "2h", "120m"]:
            data = (
                yf.Ticker(ticker)
                .history(period=period, interval="60m", prepost=prepost)
                .drop(columns=["Dividends", "Stock Splits"])
            )

            if intrvl == "90m":
                intrvl += "in"

            logic = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
            data = data.resample(intrvl, origin="end").apply(logic)
            data = data.dropna()
            # print(data.tail(5))
        elif intrvl in ["2d", "3d", "4d", "5d"]:
            data = (
                yf.Ticker(ticker)
                .history(period=period, interval="1d", prepost=prepost)
                .drop(columns=["Dividends", "Stock Splits"])
            )

            logic = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
            data = data.resample(intrvl, origin="end").apply(logic)
            data = data.dropna()
            # print(data.tail(5))
        else:
            data = (
                yf.Ticker(ticker)
                .history(period=period, interval=intrvl, prepost=prepost)
                .drop(columns=["Dividends", "Stock Splits"])
            )

        data = data.reset_index().rename(columns={"Datetime": "Date"}).set_index("Date")
        return data

    def df_setter(self):

        dfs = {}
        for tf in self.timeframes:
            period, intrvl = tf.split(":")
            data = self.data_getter(
                self.ticker,
                intrvl,
                period,
                prepost=self.prepost,
            )
            dfs[tf] = data
        self.dfs = dfs
        self.cprice = data["Close"][-1]

    @property
    def mfi_signals(self):
        dfs = self.dfs.copy()
        mfi_signals = {}
        for tf in self.timeframes:
            df = dfs[tf]
            mfi = (
                df.ta.mfi(fill_method="ffill", length=self.mfi_length)
                .replace(0, np.nan)
                .fillna(method="pad")
            )

            self.mfis[tf] = mfi[-1]
            mfi_diff = mfi.diff()
            self.mfi_diffs[tf] = mfi_diff[-1]

            self.mfi_angles[tf] = mfi_diff[-1]

            # mfi_ttmtrend = pt.ttm_trend(mfi, mfi, mfi, length=6)
            mfi_ttmtrend = self.trend_painter(mfi)

            # if mfi_diff[-1] > 0:
            #     signal = True
            # elif mfi_diff[-1] < 0:
            #     signal = False
            # else:
            #     signal = None

            if mfi_ttmtrend.values[-1] == 1:  # and mfi.values[-1] <= 80:
                signal = True
            elif mfi_ttmtrend.values[-1] == -1:  # and mfi.values[-1] >= 20:
                signal = False
            else:
                signal = None

            mfi_signals[tf] = signal

        return mfi_signals

    @property
    def cmfs(self):
        dfs = self.dfs.copy()
        cmfs = {}
        for tf in self.timeframes:
            df = dfs[tf]
            cmf = (
                df.ta.cmf(fill_method="ffill", length=14)
                .replace(0, np.nan)
                .fillna(method="pad")
            )

            cmfs[tf] = np.logical_and(
                cmf[-1] > 0.05, self.trend_painter(cmf).values[-1] == 1
            )
        return cmfs

    @property
    def atrs(self):
        dfs = self.dfs.copy()
        atrs = {}
        for tf in self.timeframes:
            df = dfs[tf]
            atr = (
                df.ta.atr(fill_method="ffill", length=14)
                .replace(0, np.nan)
                .fillna(method="pad")
            )

            atrs[tf] = atr[-1]
        return atrs

    @property
    def nvis(self):
        dfs = self.dfs.copy()
        nvis = {}
        for tf in self.timeframes:
            df = dfs[tf]
            # nvi = (
            #     df.ta.nvi(fill_method="ffill", length=5)
            #     .replace(0, np.nan)
            #     .fillna(method="pad")
            # )
            nvi = df.ta.nvi(length=5)
            nvis[tf] = self.trend_painter(nvi).values[-1]
        return nvis

    @property
    def overbought(self):
        dfs = self.dfs.copy()
        overbought = {}
        for tf in self.timeframes:
            df = dfs[tf]
            mfi = (
                df.ta.mfi(fill_method="ffill", length=self.mfi_length)
                .replace(0, np.nan)
                .fillna(method="pad")
            )
            if mfi[-1] > 70:
                overbought[tf] = True
            else:
                overbought[tf] = False
        return overbought

    @property
    def oversold(self):
        dfs = self.dfs.copy()
        oversold = {}
        for tf in self.timeframes:
            df = dfs[tf]
            mfi = (
                df.ta.mfi(fill_method="ffill", length=self.mfi_length)
                .replace(0, np.nan)
                .fillna(method="pad")
            )
            if mfi[-1] < 30:
                oversold[tf] = True
            else:
                oversold[tf] = False
        return oversold

    @property
    def price_trending_signals(self):
        dfs = self.dfs.copy()
        price_trending_signals = {}
        for tf in self.timeframes:
            df = dfs[tf]
            # trends = df.ta.ttm_trend(length=6)
            trends = self.trend_painter(df.ta.ohlc4())
            if trends.values[-1] == 1:
                signal = True
            elif trends.values[-1] == -1:
                signal = False
            else:
                signal = None
            price_trending_signals[tf] = signal
        return price_trending_signals

    @property
    def squeeze_signals(self):
        dfs = self.dfs.copy()
        squeeze_signals = {}
        for tf in self.timeframes:
            df = dfs[tf]
            # print(df)
            sqz = df.ta.squeeze()
            sqz_on = sqz.SQZ_ON
            sqz_off = sqz.SQZ_OFF

            squeeze_signals[tf] = (sqz_on.values[-1], sqz_off.values[-1])
        return squeeze_signals

    @property
    def cci_signal(self):
        data = self.data_getter(
            self.ticker,
            interval="5d",
            period="max",
            prepost=self.prepost,
        )
        cci_wk = data.ta.cci(length=14)
        self.cci_wk = cci_wk[-1]
        if cci_wk[-1] >= 100:
            return 2
        elif 70 <= cci_wk[-1] < 100:
            return 1
        elif -100 < cci_wk[-1] <= -70:
            return -1
        elif cci_wk[-1] <= -100:
            return -2
        else:
            return 0

    @property
    def cci_signal_daily(self):
        data = self.data_getter(
            self.ticker,
            interval="3d",
            period="180d",
            prepost=self.prepost,
        )
        cci_d = data.ta.cci(length=14)
        self.cci_d = cci_d[-1]
        if cci_d[-1] >= 100:
            return 2
        elif 70 <= cci_d[-1] < 100:
            return 1
        elif -100 < cci_d[-1] <= -70:
            return -1
        elif cci_d[-1] <= -100:
            return -2
        else:
            return 0

    @property
    def near_support(self):
        if (
            0
            < (self.cprice - self.support) / (self.resistance - self.support)
            <= 1 / 10
        ):
            return True
        else:
            return False

    @property
    def near_resistance(self):
        if (
            9 / 10
            <= (self.cprice - self.support) / (self.resistance - self.support)
            < 1
        ):
            return True
        else:
            return False

    @property
    def alert_signals(self):

        alert_signals = {}
        mfi_signals = self.mfi_signals
        price_trending_signals = self.price_trending_signals

        for tf in self.timeframes:
            mfi_signal = mfi_signals[tf]
            price_trending_signal = price_trending_signals[tf]

            if (price_trending_signal, mfi_signal) == (1, 1):
                out = 0
            elif (price_trending_signal, mfi_signal) == (0, 1):
                out = 1
            elif (price_trending_signal, mfi_signal) == (1, 0):
                out = 2
            else:
                out = 3

            alert_signals[tf] = out
        return alert_signals

    @property
    def alert_message(self):
        self.df_setter()  # refresh new df

        signals = self.alert_signals
        if self.cprice < self.support or self.cprice > self.resistance:
            # del self.nearest_levels
            # del self.strong_supports_resistances

            self.nearest_levels = self.get_nearest_levels(
                sup_res_thresholds=self.sup_res_thresholds
            )
            self.support, self.resistance = self.nearest_levels
            signals = self.alert_signals
        self.support, self.resistance = self.nearest_levels

        fib_pct = (self.cprice - self.support) / (self.resistance - self.support) * 100

        cci_signal = self.cci_signal
        cci_message = self.cci_codes[cci_signal]

        cci_daily_signal = self.cci_signal_daily
        cci_daily_message = self.cci_daily_codes[cci_daily_signal]

        overbought = [["", "X"][ob] for ob in self.overbought.values()]
        oversold = [["", "X"][osld] for osld in self.oversold.values()]
        squeeze_on = [["", "X"][sqz[0]] for sqz in self.squeeze_signals.values()]
        squeeze_off = [["", "X"][sqz[1]] for sqz in self.squeeze_signals.values()]

        if (
            overbought[-3] == "X"
            or overbought[-2] == "X"
            or oversold[-3] == "X"
            or oversold[-2] == "X"
        ):
            fontsize = 20
            underline_open = "<u>"
            underline_close = "</u>"
        else:
            fontsize = 12
            underline_open = ""
            underline_close = ""

        msg = (
            f'<b style="font-size:{fontsize}px">{underline_open}{self.ticker}{underline_close}</b>:\n\t'
            + f"Price: ${self.cprice:4f} | Fib.: {fib_pct:.1f}% | S{self.support:.3f} ({self.sup_res_thresholds[0]}%) | R{self.resistance:.3f} ({100 - self.sup_res_thresholds[1]}%) | CCI(W): {self.cci_wk:.1f} ({cci_message}) | CCI(3D): {self.cci_d:1f} ({cci_daily_message}) | Breakout: {str(self.incoming_breakout)}.<br><br>"
        )

        alert_msg = [self.alert_codes[sig] for sig in signals.values()]

        msg_df = pd.DataFrame(
            [
                [
                    self.price_trend_codes[c]
                    for c in list(self.price_trending_signals.values())
                ],
                [self.mfi_codes[c] for c in list(self.mfi_signals.values())],
                alert_msg,
                overbought,
                oversold,
                squeeze_on,
                squeeze_off,
            ],
            index=[
                "Price trend",
                "Money Flow Trend",
                "Alert Signal",
                "Overbought (MFI>70)",
                "Oversold (MFI<30)",
                "Squeeze On",
                "Squeeze Off",
            ],
            columns=self.timeframes,
        )  # .to_string()

        html = """\
        <html>
        <head></head>
        <body>
            <div>
                {a}
            </div>
            {b}
        </body>
        </html>
        """.format(
            a=msg, b=msg_df.to_html(justify="center")
        )

        # for tf, signal in signals.items():

        #     if signal is not None:
        #         alert_code = self.alert_codes[signal]
        #         # print(self.mfi_angles[tf])

        #         out = f"\n\t {alert_code} | {tf} | Price: ${self.cprice:4f} (Fib. pct: {fib_pct:.0f}%) | S{self.support:.3f} ({self.sup_res_thresholds[0]}%) | R{self.resistance:.3f} ({100 - self.sup_res_thresholds[1]}%) | MFI: {self.mfis[tf]:.1f} | MFI Diff: {self.mfi_angles[tf]:.0f} | CCI(W): {self.cci_wk:.1f} ({cci_message}) | CCI(3D): {self.cci_d:1f} ({cci_daily_message}) | Breakout: {str(self.incoming_breakout)} | Price trending (5 min): {trending} | Money Flow trending (5 min): {mfi_trending}.\n"  # | Vol/Avg. Vol: {(self.df.volume[-1]/self.df.volume[-20:].mean()):.2f}

        #         # MFI Bigtrend Angle: {self.mfi_angle:.0f} deg |  Dev. R. {self.mfi_deviation_ratio:.2f} |

        #     else:
        #         if cci_message is not None:
        #             out = f"\n\t CCI(W): {self.cci_wk:.1f} ({cci_message}) | CCI(3D): {self.cci_d:1f} ({cci_daily_message}) | Price trending: {trending} | Money Flow trending: {mfi_trending}.\n"
        #         else:
        #             out = None

        #     msg += out

        # return msg + msg_df
        return html

    @cached_property
    def strong_supports_resistances(self):
        return self.get_strong_support_and_resistance_lines(
            thresholds=self.sup_res_thresholds, timeframe=self.sup_res_timeframe
        )

    @cached_property
    def nearest_levels(self):
        return self.get_nearest_levels(sup_res_thresholds=self.sup_res_thresholds)

    @property
    def current_inflow_pct(self):
        inflow_pct = {}
        for tf in self.timeframes:
            inflow_pct[tf] = self.get_current_inflow_pct(self.dfs[tf])

        return inflow_pct

    def trend_painter(self, close, length=20, threshold=0.75, influence=0.15):
        arr = close[:length].values
        detector = real_time_peak_detection(
            arr, lag=length, threshold=threshold, influence=influence
        )
        ttmtrend = close.apply(lambda x: detector.thresholding_algo(x))
        return ttmtrend

    def get_current_inflow_pct(self, df):
        """Return % of inflow/total vol"""
        df = df.copy()
        df_cp = df.copy()

        idx = [cc.lower() for cc in df.columns].index("close")
        vol_idx = [cc.lower() for cc in df.columns].index("volume")
        atr = df_cp.ta.atr(length=14)
        atr = atr.fillna(atr.mean())
        close = df.columns[idx]
        vol = df.columns[vol_idx]
        df["Change"] = df[close].diff() > 0
        df["ATR"] = atr
        df["Key Levels"] = df.apply(
            lambda x: custom_round(x[close], base=round_to_1(x["ATR"] / 2)), axis=1
        )
        current_kl = df["Key Levels"].values[-1]

        df_kl = df.groupby(["Key Levels", "Change"]).sum()
        df_kl = df_kl.unstack()
        df_kl["Level Strength"] = (
            100 * df_kl[(vol, True)] / (df_kl[(vol, False)] + df_kl[(vol, True)])
        )
        return df_kl.loc[(current_kl, "Level Strength")].values[0]

    def get_strong_support_and_resistance_lines(
        self, df=None, close="Close", thresholds=(85, 15), timeframe="180d:60m"
    ):
        # t0 = time()
        thresholds = (np.clip(thresholds[0], 5, 95), np.clip(thresholds[1], 5, 95))
        if df is None:
            per, interval = timeframe.split(":")
            df = yf.Ticker(self.ticker).history(per, interval, prepost=False)
            # logic = {
            #     "Open": "first",
            #     "High": "max",
            #     "Low": "min",
            #     "Close": "last",
            #     "Volume": "sum",
            # }
            # df = df.resample("4h").apply(logic)
            df = df.dropna()

        # print(time() - t0)

        assert close.lower() in [
            cc.lower() for cc in df.columns
        ], f"Attribute {close} is not in df."
        df = df.copy()

        idx = [cc.lower() for cc in df.columns].index(close.lower())
        df_cp = df.copy()
        atr = df_cp.ta.atr(length=14)
        atr = atr.fillna(atr.mean())
        close = df.columns[idx]
        hlc3 = df_cp.ta.hlc3()
        df["HLC3"] = hlc3
        df["Change"] = df["HLC3"].diff() > 0
        df["ATR"] = atr
        # df['STD'] = df['Change'].rolling(50).std()
        # df["ATR.2"] = atr / 2

        # df["Key Levels"] = df.apply(
        #     lambda x: custom_round(x["HLC3"], base=round_to_1(x["ATR"]/2)), axis=1
        # )
        df["Key Levels"] = df.apply(
            lambda x: custom_round(x[close], base=round_to_1(x["ATR"] / 2)), axis=1
        )
        # print(df['STD'])
        # print(df['ATR'])
        # df["Key Levels"] = df.apply(
        #     lambda x: custom_round(x["HLC3"], base=round_to_1(x["STD"])), axis=1
        # )
        # df["Key Levels 2"] = df.apply(
        #     lambda x: custom_round(x["HLC3"] + x["ATR.2"], round_to_1(x["ATR"])), axis=1
        # )
        # print(df)

        df_kl = df.groupby(["Key Levels", "Change"]).sum()
        df_kl = df_kl.unstack()
        df_kl["Level Strength"] = (
            100
            * df_kl[("Volume", True)]
            / (df_kl[("Volume", False)] + df_kl[("Volume", True)])
        )
        df_kl["Total Vol"] = df_kl[("Volume", False)] + df_kl[("Volume", True)]
        df_kl["Std Vol"] = (df_kl["Total Vol"] - df_kl["Total Vol"].mean()) / df_kl[
            "Total Vol"
        ].std()

        # df_kl["Std Vol"] = (
        #     df_kl["Total Vol"] - df_kl["Total Vol"].rolling(5).mean()
        # ) / df_kl["Total Vol"].rolling(5).std()

        # df_kl["Std Vol"] = df_kl["Std Vol"] - df_kl["Std Vol"].min() + 1
        # df_kl["color"] = [cm.hot(v) for v in df_kl["Std Vol"] / df_kl["Std Vol"].max()]

        df_kl = df_kl.dropna()
        # print(df_kl)
        # plt.hist2d(df_kl.index, df_kl[('Volume', True)], bins=(40,3))

        try:

            # df_kl["Strong Support"] = np.logical_and(
            #     df_kl["Level Strength"]
            #     >= np.nanpercentile(df_kl["Level Strength"], thresholds[0]),
            #     df_kl["Volume"][True] > df_kl["Volume"][True].rolling(5).mean(),
            # )

            # df_kl["Strong Resistance"] = np.logical_and(
            #     df_kl["Level Strength"]
            #     <= np.nanpercentile(df_kl["Level Strength"], thresholds[1]),
            #     df_kl["Volume"][False] > df_kl["Volume"][False].rolling(5).mean(),
            # )
            # df_kl["Strong Support"] = np.logical_and(
            #     df_kl["Level Strength"] >= thresholds[0],
            #     df_kl["Volume"][True] > df_kl["Volume"][True].rolling(5).mean(),
            # )

            # df_kl["Strong Resistance"] = np.logical_and(
            #     df_kl["Level Strength"] <= thresholds[1],
            #     df_kl["Volume"][False] > df_kl["Volume"][False].rolling(5).mean(),
            # )

            df_kl["Strong Support"] = np.logical_and.reduce(
                [
                    df_kl["Level Strength"] >= thresholds[0],
                    # df_kl["Volume"][True] > df_kl["Volume"][True].rolling(10).mean(),
                    df_kl["Std Vol"] > 0.5,
                ]
            )

            df_kl["Strong Resistance"] = np.logical_and.reduce(
                [
                    df_kl["Level Strength"] <= thresholds[1],
                    # df_kl["Volume"][False] > df_kl["Volume"][False].rolling(10).mean(),
                    df_kl["Std Vol"] > 0.5,
                ]
            )

            # df_kl["Strong Support"] = df_kl["Level Strength"] >= thresholds[0]

            # df_kl["Strong Resistance"] = df_kl["Level Strength"] <= thresholds[1]

        except KeyError:

            # df_kl["Strong Support"] = np.logical_and(
            #     df_kl["Level Strength"] >= thresholds[0],
            #     df_kl["volume"][True] > df_kl["volume"][True].rolling(5).mean(),
            # )

            # df_kl["Strong Resistance"] = np.logical_and(
            #     df_kl["Level Strength"] <= thresholds[1],
            #     df_kl["volume"][False] > df_kl["volume"][False].rolling(5).mean(),
            # )

            df_kl["Strong Support"] = np.logical_and.reduce(
                [
                    df_kl["Level Strength"] >= thresholds[0],
                    # df_kl["Volume"][True] > df_kl["Volume"][True].rolling(10).mean(),
                    df_kl["Std Vol"] > 0.5,
                ]
            )

            df_kl["Strong Resistance"] = np.logical_and.reduce(
                [
                    df_kl["Level Strength"] <= thresholds[1],
                    # df_kl["Volume"][False] > df_kl["Volume"][False].rolling(10).mean(),
                    df_kl["Std Vol"] > 0.5,
                ]
            )

            # df_kl["Strong Support"] = df_kl["Level Strength"] >= thresholds[0]

            # df_kl["Strong Resistance"] = df_kl["Level Strength"] <= thresholds[1]

        ####
        ss = df_kl["Strong Support"].index[df_kl["Strong Support"]]

        sr = df_kl["Strong Resistance"].index[df_kl["Strong Resistance"]]

        sups = pd.DataFrame(index=range(len(ss)), columns=["Date", "Price"])
        sups["Date"] = [df.index] * len(ss)
        sups["Price"] = [[s] * len(df.index) for s in ss]

        res = pd.DataFrame(index=range(len(sr)), columns=["Date", "Price"])
        res["Date"] = [df.index] * len(sr)
        res["Price"] = [[s] * len(df.index) for s in sr]

        # Testing renko

        # print(sups)
        return sups, res

    def get_nearest_levels(self, sup_res_thresholds=[85, 15]):
        # (
        #     supports_df,
        #     resistances_df,
        # ) = self.strong_supports_resistances
        (supports_df, resistances_df,) = self.get_strong_support_and_resistance_lines(
            thresholds=sup_res_thresholds, timeframe=self.sup_res_timeframe
        )

        support_levels = [idx[0] for idx in supports_df["Price"]]
        levels = pd.DataFrame(columns=["labels"])

        for lev in support_levels:
            if lev <= self.cprice:
                levels.loc[lev, "labels"] = "support"
            else:
                levels.loc[lev, "labels"] = "resistance"

        resistance_levels = [idx[0] for idx in resistances_df["Price"]]
        for lev in resistance_levels:
            if lev <= self.cprice:
                levels.loc[lev, "labels"] = "support"
            else:
                levels.loc[lev, "labels"] = "resistance"
        levels = levels.sort_index()

        try:
            support = levels[levels == "support"].dropna().index[-1]
        except IndexError:
            try:
                support = self.dfs[self.timeframes[-1]].Low[-50:].min()
            except AttributeError:
                support = self.dfs[self.timeframes[-1]].low[-50:].min()
        try:
            resistance = levels[levels == "resistance"].dropna().index[0]
        except IndexError:  # no resistance
            try:
                resistance = self.dfs[self.timeframes[-1]].High[-50:].max()
            except AttributeError:
                resistance = self.dfs[self.timeframes[-1]].high[-50:].max()

        try:
            if (
                abs(resistance - support) / self.cprice < 0.03
            ):  # narrow support and resistance (in range)
                self.incoming_breakout = True
            else:
                self.incoming_breakout = False
        except TypeError:
            self.incoming_breakout = False
        return support, resistance


def callPrice(s, x, r, sigma, t):
    """Call Option Pricing

    Args:
        s (float): Current share price
        x (float): Strike Price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        t (float): Time to expiration, years > 0

    Returns:
        [type]: [description]
    """
    # print(math.log(s / x))
    warnings.filterwarnings("error")

    try:
        a = (np.log(s / x) + (r + sigma * sigma / 2.0) * t) / (sigma * np.sqrt(t))

        b = a - sigma * np.sqrt(t)

        out = s * cdf(a) - x * np.exp(-r * t) * cdf(b)
    except (ValueError, Warning) as e:
        out = np.nan
    # print(out)
    warnings.filterwarnings("default")
    return out


def putPrice(s, x, r, sigma, t):
    """Call Option Pricing

    Args:
        s (float): Current share price
        x (float): Strike Price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        t (float): Time to expiration, years > 0

    Returns:
        [type]: [description]
    """
    # print(math.log(s / x))
    warnings.filterwarnings("error")

    try:
        a = (np.log(s / x) + (r + sigma * sigma / 2.0) * t) / (sigma * np.sqrt(t))

        b = a - sigma * np.sqrt(t)

        out = -s * cdf(-a) + x * np.exp(-r * t) * cdf(-b)
    except (ValueError, Warning) as e:
        out = np.nan
    # print(out)
    warnings.filterwarnings("default")
    return out

# def callPrice2(s, x=1, r=1, sigma=1, t=1):
# callPrice(s, x, r, sigma, t)


# np.apply_along_axis(callPrice2, -1, [25, 30], 10, 0.01, 0.35, 0.00001)

# t0 = time()
# callPrice(21.22, 10, 0.01, 0.35, 0.00001)


def get_naked_call_fn(strike, IV, riskfree_rate=0.02, max_days=14):

    """Returns an interp2d function of (time (days), price (dollars)) for a call option, given the strike price and implied volatility of the option

    Returns:
        np.interp2d: function of (time (days), price (dollars))
    """

    t = np.arange(0, max_days, 0.25)
    p = np.arange(strike - min(strike, 100), strike + 100, 0.25)
    z = np.array(
        [[callPrice(pr, strike, riskfree_rate, IV, tm / 365) for pr in p] for tm in t]
    ).flatten()

    f = interp2d(t, p, z, kind="linear")
    return f


def get_naked_put_fn(strike, IV, riskfree_rate=0.02, max_days=14):

    """Returns an interp2d function of (time (days), price (dollars)) for a put option, given the strike price and implied volatility of the option

    Returns:
        np.interp2d: function of (time (days), price (dollars))
    """

    t = np.arange(0, max_days, 0.25)
    p = np.arange(strike - min(strike, 100), strike + 100, 0.25)
    z = np.array(
        [[putPrice(pr, strike, riskfree_rate, IV, tm / 365) for pr in p] for tm in t]
    ).flatten()

    f = interp2d(t, p, z, kind="linear")
    return f


def NakedCall(
    ticker,
    strike,
    next_expected_price,
    strike_date=None,
    lower_expected_price=None,
    current_price=None,
    tp_price=None,
    IV=None,
    cur_cp=None,
    time_offset=None,
    show_plot=True,
    return_rr=False,
    call_cost=None,
):
    alerter = Alerter(
        ticker=ticker,
        timeframes=["5y:1d", "180d:60m", "30d:30m"],
        sup_res_thresholds=(60, 40),
        prepost=False,
    )
    if strike_date is None:
        strike_date = yf.Ticker(ticker).options[0]
    yr, mo, d = np.array(strike_date.split("-")).astype(int)
    now = datetime.now()
    ctime = (datetime(yr, mo, d, hour=13) - now).total_seconds() / (
        365.25 * 24 * 3600
    )  #  - 1 / 365.25  # - (2/24/365.25)
    if time_offset is not None:
        ctime -= time_offset
        now += timedelta(days=time_offset * 365.25)

    if current_price is None:
        current_price = yf.Ticker(ticker).history("1d", "60m", prepost=False).Close[-1]

    # X = np.linspace(
    #     current_price - 4 * np.abs(strike - current_price),
    #     current_price + 4 * np.abs(strike - current_price),
    #     300,
    # )
    X = np.linspace(strike * 0.5, strike * 1.5, 300)
    print(
        color.BLUE
        + color.BOLD
        + color.UNDERLINE
        + ticker
        + color.END
        + color.END
        + color.END
        + ":"
    )

    if IV is None:
        try:
            ops = yf.Ticker(ticker).option_chain(strike_date).calls
            IV = ops[ops.strike == strike].impliedVolatility.values[0]
            if cur_cp is None:
                current_callPrice_market = ops[ops.strike == strike].lastPrice.values[0]
            else:
                current_callPrice_market = cur_cp
        except (ConnectionError, ValueError, IndexError):
            print("Connection Error")
            # IV = 0.497
            IV = 0.5

    print(
        f"{color.UNDERLINE}CALL{color.END} IV: {IV}; Strike: {strike}; Exp.: {strike_date}\n"
    )
    print("Money Inflow pct.", alerter.current_inflow_pct)
    current_callPrice = callPrice(current_price, strike, 0.01, IV, ctime)

    try:
        ops = yf.Ticker(ticker).option_chain(strike_date).calls
        if cur_cp is None:
            current_callPrice_market = ops[ops.strike == strike].lastPrice.values[0]
        else:
            current_callPrice_market = cur_cp
    except (ConnectionError, ValueError, IndexError):
        print("Connection Error")
        current_callPrice_market = current_callPrice
    market_vs_model_ratio = current_callPrice_market / current_callPrice

    print(
        f"Call Price on {now.date()} with stock price at ${current_price:.4f}: {current_callPrice:.4f} | market: {current_callPrice_market:.4f}",
        "\n",
    )

    fns = []

    if 80 > ctime * 365.25 > 20:
        step = 1
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "days"
    elif ctime * 365.25 > 80:
        step = 5
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "weeks"
    else:
        step = 0.25
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "quarter day"

    buyprice = None
    overall_cop = []
    overall_rr = []

    break_evens = []
    for i, t in enumerate(num_days):

        op_price_4c = np.array([callPrice(x, strike, 0.01, IV, ctime - t) for x in X])
        f = interp1d(X, op_price_4c, fill_value="extrapolate")
        fns.append(f)

        print(color.BOLD + f"{(now+timedelta(days=t*365.25)).date()}" + color.END)
        if i == 0:
            buyprice = f(next_expected_price)
        print(
            f"Call price at next expected price ${next_expected_price} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(next_expected_price):.4f} | expected market: {(f(next_expected_price)*market_vs_model_ratio):.4f}",
        )

        print(
            f"Call price at strike price ${strike} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(strike):.4f} | expected market: {(f(strike)*market_vs_model_ratio):.4f}",
        )

        if tp_price is not None:
            print(
                f"Call price at TP price ${tp_price} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(tp_price):.4f} | expected market: {(f(tp_price)*market_vs_model_ratio):.4f}",
            )

        if call_cost is None:
            break_even = np.abs(f(X) - current_callPrice).argmin()
        else:
            break_even = np.abs(f(X) - call_cost).argmin()
        break_evens.append(X[break_even])
        print(
            f"Break even price on {(now+timedelta(days=t*365.25)).date()} with ({(len(num_days)-i)} {unit} left):",
            X[break_even],
        )
        if lower_expected_price is None:
            rr = (f(strike) - buyprice) / 0.25 * buyprice
            print(
                "Reward/Risk:",
                rr,
                "at 75% of buy price",
            )
        else:
            rr = (f(strike) - buyprice) / (
                buyprice
                - f(
                    lower_expected_price
                    + 0.75 * (next_expected_price - lower_expected_price)
                )
            )
            print(
                "Reward/Risk:",
                rr,
                f"if price falls to ${lower_expected_price+ 0.75 * (next_expected_price - lower_expected_price)}",
            )
        overall_rr.append(rr)
        if tp_price is None:
            print(
                "Max profit:",
                f"${((f(strike) - buyprice)*100* market_vs_model_ratio):.0f}/call",
            )
            print(
                "Max profit %:",
                f"{((f(strike) / buyprice - 1)*100):.0f}%",
            )
        else:
            print(
                "Max profit:",
                f"${((f(tp_price) - buyprice)*100* market_vs_model_ratio):.0f}/call",
            )
            print(
                "Max profit %:",
                f"{((f(tp_price) / buyprice - 1)*100):.0f}%",
            )
        f_diff = interp1d(
            X[1:], (f(X[1:]) - f(X[:-1])) / (X[1] - X[0]), fill_value="extrapolate"
        )
        cop = f_diff(current_price) / f_diff(X[-1]) * 100  # chance of profit, %
        overall_cop.append(cop)
        print(f"Chance of profit: {cop:.1f}%")
        if show_plot:
            if t != 0.00001:
                plt.plot(
                    X,
                    op_price_4c - current_callPrice,
                    ls="--",
                    ms=0,
                    label=f"{(now+timedelta(days=t*365.25)).date()}",
                )
                plt.scatter(
                    [current_price, X[break_even]],
                    [
                        f(current_price) - current_callPrice,
                        f(X[break_even]) - current_callPrice,
                    ],
                )
            else:
                plt.plot(
                    X,
                    op_price_4c - current_callPrice,
                    ms=0,
                    lw=2,
                    label=f"{(now+timedelta(days=t*365.25)).date()}",
                )
                plt.scatter(
                    [current_price, X[break_even]],
                    [
                        f(current_price) - current_callPrice,
                        f(X[break_even]) - current_callPrice,
                    ],
                )
            print("\n")

    plt.show()
    plt.close()
    print(break_evens)
    plt.plot(break_evens)
    plt.axhline(current_price)

    overall_cop = np.array(overall_cop)[
        np.cumsum(overall_cop / np.sum(overall_cop)) < 0.9
    ].mean()  # overall chance of profit, excluding expiry

    overall_rr = np.array(overall_rr).mean()  # overall reward/risk
    print(f"Overall chance of profit: {overall_cop:.1f}%")
    if return_rr:
        return overall_cop, overall_rr
    else:
        return overall_cop


def NakedPut(
    ticker,
    strike,
    next_expected_price,
    strike_date=None,
    higher_expected_price=None,
    current_price=None,
    IV=None,
    tp_price=None,
    time_offset=None,
    show_plot=True,
    return_rr=False,
):
    alerter = Alerter(
        ticker=ticker,
        timeframes=["5y:1d", "180d:60m", "30d:30m"],
        sup_res_thresholds=(60, 40),
        prepost=False,
    )
    if strike_date is None:
        strike_date = yf.Ticker(ticker).options[0]
    yr, mo, d = np.array(strike_date.split("-")).astype(int)
    now = datetime.now()
    ctime = (datetime(yr, mo, d, hour=13) - now).total_seconds() / (
        365.25 * 24 * 3600
    )  #  - 1 / 365.25  # - (2/24/365.25)
    if time_offset is not None:
        ctime -= time_offset
        now += timedelta(days=time_offset * 365.25)

    if current_price is None:
        current_price = yf.Ticker(ticker).history("1d", "60m", prepost=False).Close[-1]

    X = np.linspace(
        strike * 0.5,
        strike * 1.5,
        300,
    )
    print(
        color.BLUE
        + color.BOLD
        + color.UNDERLINE
        + ticker
        + color.END
        + color.END
        + color.END
        + ":"
    )
    # print(X[0], X[-1])

    if IV is None:
        try:
            ops = yf.Ticker(ticker).option_chain(strike_date).puts
            IV = ops[ops.strike == strike].impliedVolatility.values[0]
        except (ConnectionError, ValueError, IndexError):
            print("Connection Error")
            # IV = 0.497
            IV = 0.5

    print(
        f"{color.UNDERLINE}PUT{color.END} IV: {IV}; Strike: {strike}; Exp.: {strike_date}\n"
    )
    print("Money Inflow pct.", alerter.current_inflow_pct)
    current_putPrice = putPrice(current_price, strike, 0.01, IV, ctime)
    try:
        ops = yf.Ticker(ticker).option_chain(strike_date).puts
        current_putPrice_market = ops[ops.strike == strike].lastPrice.values[0]

    except (ConnectionError, ValueError, IndexError):
        print("Connection Error")
        current_putPrice_market = current_putPrice
    market_vs_model_ratio = current_putPrice_market / current_putPrice
    print(
        f"Put Price on {now.date()} with stock price at ${current_price:.4f}: {current_putPrice:.4f} | market: {current_putPrice_market:.4f}",
        "\n",
    )

    fns = []
    if 80 > ctime * 365.25 > 20:
        step = 1
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "days"
    elif ctime * 365.25 > 80:
        step = 5
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "weeks"
    else:
        step = 0.25
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "quarter day"

    buyprice = None
    overall_cop = []
    overall_rr = []
    for i, t in enumerate(num_days):

        op_price_4p = np.array([putPrice(x, strike, 0.01, IV, ctime - t) for x in X])
        f = interp1d(X, op_price_4p, fill_value="extrapolate")
        fns.append(f)

        print(color.BOLD + f"{(now+timedelta(days=t*365.25)).date()}" + color.END)
        if i == 0:
            buyprice = f(next_expected_price)
        print(
            f"Put price at next expected price ${next_expected_price} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(next_expected_price):.4f} | expected market: {(f(next_expected_price)*market_vs_model_ratio):.4f}",
        )

        print(
            f"Put price at strike price ${strike} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(strike):.4f} | expected market: {(f(strike)*market_vs_model_ratio):.4f}",
        )

        if tp_price is not None:
            print(
                f"Put price at TP price ${tp_price} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(tp_price):.4f} | expected market: {(f(tp_price)*market_vs_model_ratio):.4f}",
            )
        break_even = np.abs(f(X) - current_putPrice).argmin()
        print(
            f"Break even price on {(now+timedelta(days=t*365.25)).date()} with ({(len(num_days)-i)} {unit} left):",
            X[break_even],
        )
        if higher_expected_price is None:
            rr = (f(strike) - buyprice) / 0.25 * buyprice
            print(
                "Reward/Risk:",
                rr,
                "at 75% of buy price",
            )
        else:
            # print(f(strike))
            # print(buyprice)
            rr = (f(strike) - buyprice) / (
                buyprice
                - f(
                    higher_expected_price
                    - 0.75 * np.abs(next_expected_price - higher_expected_price)
                )
            )
            print(
                "Reward/Risk:",
                rr,
                f"if price rises to ${higher_expected_price - 0.75 * np.abs(next_expected_price - higher_expected_price)}",
            )
        overall_rr.append(rr)
        if tp_price is None:
            print(
                "Max profit:",
                f"${((f(strike) - buyprice)*100* market_vs_model_ratio):.0f}/put",
            )
            print(
                "Max profit %:",
                f"{((f(strike) / buyprice -1)*100):.0f}%",
            )
        else:
            print(
                "Max profit:",
                f"${((f(tp_price) - buyprice)*100 * market_vs_model_ratio):.0f}/put",
            )
            print(
                "Max profit %:",
                f"{((f(tp_price) / buyprice -1)*100):.0f}%",
            )
        f_diff = interp1d(
            X[1:], (f(X[1:]) - f(X[:-1])) / (X[1] - X[0]), fill_value="extrapolate"
        )
        cop = f_diff(current_price) / f_diff(X[1]) * 100  # chance of profit, %
        overall_cop.append(cop)
        print(f"Chance of profit: {cop:.1f}%")

        if show_plot:
            if t != 0.00001:
                plt.plot(
                    X,
                    op_price_4p - current_putPrice,
                    ls="--",
                    ms=0,
                    label=f"{(now+timedelta(days=t*365.25)).date()}",
                )
                plt.scatter(
                    [current_price, X[break_even]],
                    [
                        f(current_price) - current_putPrice,
                        f(X[break_even]) - current_putPrice,
                    ],
                )
            else:
                plt.plot(
                    X,
                    op_price_4p - current_putPrice,
                    ms=0,
                    lw=2,
                    label=f"{(now+timedelta(days=t*365.25)).date()}",
                )
                plt.scatter(
                    [current_price, X[break_even]],
                    [
                        f(current_price) - current_putPrice,
                        f(X[break_even]) - current_putPrice,
                    ],
                )

            print("\n")

    overall_cop = np.array(overall_cop)[
        np.cumsum(overall_cop / np.sum(overall_cop)) < 0.9
    ].mean()  # overall chance of profit, excluding expiry
    overall_rr = np.array(overall_rr).mean()  # overall reward/risk
    print(f"Overall chance of profit: {overall_cop:.1f}%")
    if return_rr:
        return overall_cop, overall_rr
    else:
        return overall_cop


def SellPut(
    ticker,
    strike,
    strike_date,
    next_expected_price,
    lower_expected_price=None,
    current_price=None,
    IV=None,
    tp_price=None,
    time_offset=None,
):
    yr, mo, d = np.array(strike_date.split("-")).astype(int)
    now = datetime.now()
    ctime = (datetime(yr, mo, d, hour=14) - now).total_seconds() / (
        365.25 * 24 * 3600
    )  #  - 1 / 365.25  # - (2/24/365.25)
    if time_offset is not None:
        ctime -= time_offset
        now += timedelta(days=time_offset * 365.25)

    if current_price is None:
        current_price = yf.Ticker(ticker).history("1d", "60m", prepost=False).Close[-1]

    X = np.linspace(
        strike * 0.5,
        strike * 1.5,
        300,
    )
    print(
        color.BLUE
        + color.BOLD
        + color.UNDERLINE
        + ticker
        + color.END
        + color.END
        + color.END
        + ":"
    )
    # print(X[0], X[-1])

    if IV is None:
        try:
            ops = yf.Ticker(ticker).option_chain(strike_date).puts
            IV = ops[ops.strike == strike].impliedVolatility.values[0]
        except (ConnectionError, ValueError, IndexError):
            print("Connection Error")
            # IV = 0.497
            IV = 0.5

    print(
        f"{color.UNDERLINE}PUT{color.END} IV: {IV}; Strike: {strike}; Exp.: {strike_date}\n"
    )
    current_putPrice = putPrice(current_price, strike, 0.01, IV, ctime)
    try:
        ops = yf.Ticker(ticker).option_chain(strike_date).puts
        current_putPrice_market = ops[ops.strike == strike].lastPrice.values[0]

    except (ConnectionError, ValueError, IndexError):
        print("Connection Error")
        current_putPrice_market = current_putPrice
    market_vs_model_ratio = current_putPrice_market / current_putPrice
    print(
        f"Put Price on {now.date()} with stock price at ${current_price:.4f}: {current_putPrice:.4f} | market: {current_putPrice_market:.4f}",
        "\n",
    )

    fns = []
    if ctime * 365.25 > 20:
        step = 1
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "days"
    elif ctime * 365.25 > 80:
        step = 5
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "weeks"
    else:
        step = 0.25
        num_days = np.arange(0.00001, ctime, step / 365.25)
        unit = "quarter day"
    buyprice = None
    for i, t in enumerate(num_days):

        op_price_4p = np.array([putPrice(x, strike, 0.01, IV, ctime - t) for x in X])

        if i == 0:
            f = interp1d(X, op_price_4p, fill_value="extrapolate")
            buyprice = f(next_expected_price)
        # else:
        f = interp1d(X, op_price_4p, fill_value="extrapolate")
        fns.append(f)

        print(color.BOLD + f"{(now+timedelta(days=t*365.25)).date()}" + color.END)
        # if i == 0:
        #     buyprice = f(next_expected_price)
        print(
            f"Put price at next expected price ${next_expected_price} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(next_expected_price):.4f} | expected market: {(f(next_expected_price)*market_vs_model_ratio):.4f}",
        )

        print(
            f"Put price at strike price ${strike} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(strike):.4f} | expected market: {(f(strike)*market_vs_model_ratio):.4f}",
        )

        if tp_price is not None:
            print(
                f"Put price at TP price ${tp_price} on {(now+timedelta(days=t*365.25)).date()} ({(len(num_days) - i)} {unit} left): {f(tp_price):.4f} | expected market: {(f(tp_price)*market_vs_model_ratio):.4f}",
            )
        break_even = np.abs(f(X) - current_putPrice).argmin()
        print(
            f"Break even price on {(now+timedelta(days=t*365.25)).date()} with ({(len(num_days)-i)} {unit} left):",
            X[break_even],
        )
        if lower_expected_price is None:
            print(
                "Reward/Risk:",
                (buyprice / (buyprice - f(X[break_even]))),
            )
        else:
            print(
                "Reward/Risk:",
                (buyprice / (buyprice - f(lower_expected_price))),
            )

        if t != 0.00001:
            plt.plot(
                X,
                -(op_price_4p - current_putPrice),
                ls="--",
                ms=0,
                label=f"{(now+timedelta(days=t*365.25)).date()}",
            )
        else:
            plt.plot(
                X,
                -(op_price_4p - current_putPrice),
                ms=0,
                lw=2,
                label=f"{(now+timedelta(days=t*365.25)).date()}",
            )

        print("\n")


def optimize_naked_call(
    ticker,
    strike_min,
    strike_max,
    strike_step,
    next_expected_price,
    expiry=None,
    lower_expected_price=None,
    current_price=None,
    tp_price=None,
    IV_min=0.5,
    time_offset=None,
    cutoff=0.7,
    **kwargs,
):
    cops = []
    rrs = []
    strikes = []
    for i, strike in enumerate(np.arange(strike_min, strike_max, strike_step)):
        fn = get_naked_call_fn(strike, IV_min * 1.02 ** i, **kwargs)
        exp = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
        rr = fn(exp, strike) / fn(exp, next_expected_price) - 1
        cop = (
            (fn(exp, strike) - fn(exp, next_expected_price))
            / (strike - next_expected_price)
            * 100
        )
        # cop, rr = NakedCall(
        #     ticker,
        #     strike,
        #     next_expected_price=next_expected_price,
        #     strike_date=expiry,
        #     lower_expected_price=lower_expected_price,
        #     current_price=current_price,
        #     tp_price=tp_price,
        #     IV=IV_min * 1.02 ** i,
        #     # cur_cp=2.96,
        #     time_offset=time_offset,
        #     return_rr=True,
        #     show_plot=False,
        # )
        cops.append(cop)
        rrs.append(rr)
        strikes.append(strike)
    plt.plot(cops, rrs, ms=10)
    plt.xlabel("Chance of Profit, %")
    plt.ylabel("Reward/Risk")
    plt.twiny()
    plt.plot(strikes, rrs, "+", color="orange", ls="--", ms=10)
    plt.xlabel("Strike")
    plt.gca().invert_xaxis()
    optim_strike = np.where(
        np.cumsum(sorted(rrs, reverse=True)) / np.sum(rrs) > cutoff
    )[0][0]
    optim_strike = sorted(strikes, reverse=True)[optim_strike]
    print("Optimal strike:", optim_strike)

    return cops, rrs, strikes, optim_strike


def optimize_naked_put(
    ticker,
    strike_min,
    strike_max,
    strike_step,
    next_expected_price,
    expiry=None,
    higher_expected_price=None,
    current_price=None,
    tp_price=None,
    IV_min=0.5,
    time_offset=None,
    cutoff=0.7,
    **kwargs,
):
    cops = []
    rrs = []
    strikes = []
    for i, strike in enumerate(np.arange(strike_min, strike_max, strike_step)):
        fn = get_naked_put_fn(strike, IV_min * 1.02 ** i, **kwargs)
        exp = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
        rr = fn(exp, strike) / fn(exp, next_expected_price) - 1
        cop = (
            (fn(exp, strike) - fn(exp, next_expected_price))
            / (next_expected_price - strike)
            * 100
        )
        # cop, rr = NakedPut(
        #     ticker,
        #     strike,
        #     next_expected_price=next_expected_price,
        #     strike_date=expiry,
        #     higher_expected_price=higher_expected_price,
        #     current_price=current_price,
        #     tp_price=tp_price,
        #     IV=IV_min * 1.02 ** i,
        #     # cur_cp=2.96,
        #     time_offset=time_offset,
        #     return_rr=True,
        #     show_plot=False,
        # )
        cops.append(cop)
        rrs.append(rr)
        strikes.append(strike)

    plt.plot(cops, rrs, ms=10)
    plt.xlabel("Chance of Profit, %")
    plt.ylabel("Reward/Risk")
    plt.twiny()
    plt.plot(strikes, rrs, "+", color="orange", ls="--", ms=10)
    plt.xlabel("Strike")

    optim_strike = np.where(
        np.cumsum(sorted(rrs, reverse=True)) / np.sum(rrs) > cutoff
    )[0][0]
    optim_strike = sorted(strikes)[optim_strike]
    print("Optimal strike:", optim_strike)

    return cops, rrs, strikes, optim_strike


if __name__ == "__main__":

    NakedPut(
        "BABA",
        115,
        166,
        "2021-10-01",
        higher_expected_price=170,
        current_price=None,
        tp_price=115,
        IV=0.78,
        time_offset=None,
    )

    NakedPut(
        "BABA",
        155,
        165,
        "2021-09-10",
        higher_expected_price=171,
        current_price=None,
        tp_price=155,
        IV=0.484,
        time_offset=None,
    )

    NakedPut(
        "ROKU",
        310,
        340,
        "2021-09-17",
        higher_expected_price=344,
        current_price=None,
        tp_price=315,
        IV=0.5,
        time_offset=None,
    )

    NakedPut(
        "NVDA",
        220,
        224,
        "2021-09-17",
        higher_expected_price=226,
        current_price=None,
        tp_price=215,
        # IV=0.35,
        time_offset=None,
    )

    NakedPut(
        "NVDA",
        215,
        222,
        "2021-10-08",
        higher_expected_price=226,
        current_price=None,
        tp_price=210,
        IV=0.36,
        time_offset=None,
    )

    NakedPut(
        "TQQQ",
        110,
        140,
        "2021-10-22",
        higher_expected_price=144,
        current_price=None,
        tp_price=110,
        IV=0.846,
        time_offset=None,
    )

    NakedPut(
        "LCID",
        23,
        26,
        "2021-10-08",
        higher_expected_price=27,
        current_price=None,
        tp_price=23,
        IV=0.93,
        time_offset=None,
    )

    NakedPut(
        "ROKU",
        290,
        320,
        "2021-10-08",
        higher_expected_price=322,
        current_price=None,
        tp_price=290,
        IV=0.57,
        time_offset=None,
    )

    NakedPut(
        "MRNA",
        275,
        310,
        "2021-10-22",
        higher_expected_price=330,
        current_price=None,
        tp_price=254,
        IV=0.7,
        time_offset=None,
    )

    NakedPut(
        "UPST",
        290,
        320,
        "2021-10-15",
        higher_expected_price=330,
        current_price=None,
        tp_price=305,
        IV=0.8,
        time_offset=None,
    )

    NakedCall(
        "LRCX",
        640,
        600,
        "2021-09-03",
        lower_expected_price=590,
        current_price=None,
        tp_price=620,
        IV=0.365,
        # cur_cp=2.96,
        time_offset=None,
    )

    NakedCall(
        "HD",
        342.5,
        336,
        "2021-09-24",
        lower_expected_price=329,
        current_price=None,
        tp_price=340,
        IV=0.17,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=1.45,
    )

    NakedCall(
        "ROKU",
        345,
        320,
        "2021-09-24",
        lower_expected_price=310,
        current_price=None,
        tp_price=330,
        IV=0.46,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=1.89,
    )

    NakedCall(
        "LRCX",
        640,
        610,
        "2021-09-24",
        lower_expected_price=600,
        current_price=None,
        tp_price=640,
        IV=0.30,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=2.67,
    )

    NakedCall(
        "NVDA",
        225,
        220,
        "2021-09-24",
        lower_expected_price=216,
        current_price=None,
        tp_price=228,
        IV=0.3,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=2.67,
    )

    NakedCall(
        "AVGO",
        520,
        505,
        "2021-09-24",
        lower_expected_price=500,
        current_price=None,
        tp_price=515,
        IV=0.18,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=1,
    )

    NakedCall(
        "TSLA",
        800,
        740,
        "2021-10-15",
        lower_expected_price=730,
        current_price=None,
        tp_price=800,
        IV=0.46,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=2.6
    )

    NakedCall(
        "QS",
        22.5,
        20,
        "2023-01-23",
        lower_expected_price=15,
        current_price=None,
        tp_price=49,
        IV=0.766,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=2.6
    )

    NakedCall(
        "NTLA",
        200,
        168,
        "2021-09-17",
        lower_expected_price=160,
        current_price=None,
        tp_price=200,
        IV=0.7,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=0.96,
    )

    NakedCall(
        "NTLA",
        190,
        160,
        "2021-09-17",
        lower_expected_price=153,
        current_price=None,
        tp_price=200,
        IV=0.69,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=1,
    )

    NakedCall(
        "TSLA",
        780,
        740,
        "2021-09-17",
        lower_expected_price=720,
        current_price=None,
        tp_price=780,
        IV=0.32,
        # cur_cp=2.96,
        time_offset=None,
        call_cost=4.35,
    )

    NakedCall(
        "TSLA",
        780,
        753,
        "2021-09-10",
        lower_expected_price=720,
        current_price=None,
        tp_price=780,
        IV=0.338,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=2.6
    )

    NakedCall(
        "TSLA",
        770,
        760,
        "2021-09-10",
        lower_expected_price=720,
        current_price=None,
        tp_price=770,
        IV=0.44,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=2.6
    )

    NakedCall(
        "NIO",
        46,
        40,
        "2021-09-17",
        lower_expected_price=36,
        current_price=None,
        tp_price=46,
        IV=0.54,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=0.96
    )

    NakedCall(
        "PLUG",
        30,
        28,
        "2021-10-29",
        lower_expected_price=25,
        current_price=None,
        tp_price=33,
        IV=0.64,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=0.96
    )

    NakedCall(
        "ZM",
        275,
        256,
        "2021-10-22",
        lower_expected_price=253,
        current_price=None,
        tp_price=275,
        IV=0.38,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=0.96
    )

    NakedCall(
        "TSLA",
        830,
        256,
        "2021-10-22",
        lower_expected_price=253,
        current_price=None,
        tp_price=275,
        IV=0.38,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=0.96
    )

    NakedCall(
        "LCID",
        24,
        22.5,
        "2021-10-22",
        lower_expected_price=21,
        current_price=None,
        tp_price=25,
        IV=0.77,
        # cur_cp=2.96,
        time_offset=None,
        # call_cost=0.96
    )

    cops, rrs, strikes, optim_strike = optimize_naked_call(
        ticker="ROKU",
        strike_min=360,
        strike_max=400,
        strike_step=2.5,
        next_expected_price=352,
        expiry="2021-09-10",
        lower_expected_price=350,
        current_price=None,
        tp_price=385,
        IV_min=0.477,
        time_offset=None,
    )
    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_call(
        ticker="LRCX",
        strike_min=590,
        strike_max=630,
        strike_step=2.5,
        next_expected_price=580,
        expiry="2021-08-27",
        lower_expected_price=575,
        current_price=None,
        tp_price=610,
        IV_min=0.339,
        time_offset=None,
        cutoff=0.7,
    )
    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_call(
        ticker="BABA",
        strike_min=170,
        strike_max=190,
        strike_step=2.5,
        next_expected_price=170,
        expiry="2021-08-27",
        lower_expected_price=165,
        current_price=None,
        tp_price=185,
        IV_min=0.6,
        time_offset=None,
        cutoff=0.7,
    )
    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_call(
        ticker="TSLA",
        strike_min=707.5,
        strike_max=750,
        strike_step=2.5,
        next_expected_price=705,
        expiry="2021-08-27",
        lower_expected_price=700,
        current_price=None,
        tp_price=730,
        IV_min=0.389,
        time_offset=None,
        cutoff=0.7,
    )
    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_call(
        ticker="BA",
        strike_min=222.5,
        strike_max=235,
        strike_step=2.5,
        next_expected_price=220,
        expiry="2021-08-27",
        lower_expected_price=216,
        current_price=None,
        tp_price=225,
        IV_min=0.29,
        time_offset=None,
        cutoff=0.7,
    )
    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_put(
        ticker="LRCX",
        strike_min=550,
        strike_max=580,
        strike_step=2.5,
        next_expected_price=590,
        expiry="2021-08-27",
        higher_expected_price=600,
        current_price=None,
        tp_price=550,
        IV_min=0.339,
        time_offset=None,
        cutoff=0.8,
    )

    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_put(
        ticker="BABA",
        strike_min=150,
        strike_max=165,
        strike_step=2.5,
        next_expected_price=167.5,
        expiry="2021-09-03",
        higher_expected_price=172.5,
        current_price=None,
        tp_price=155,
        IV_min=0.475,
        time_offset=None,
        cutoff=0.8,
    )

    print("Optimal strike:", optim_strike)

    cops, rrs, strikes, optim_strike = optimize_naked_put(
        ticker="TQQQ",
        strike_min=100,
        strike_max=120,
        strike_step=5,
        next_expected_price=138,
        expiry="2021-10-22",
        higher_expected_price=140,
        current_price=None,
        tp_price=110,
        IV_min=0.75,
        time_offset=None,
        cutoff=0.8,
    )

    print("Optimal strike:", optim_strike)

    # tsla_call = get_naked_call_fn(25, 0.75, max_days=600)

    # tsla_call((datetime.strptime('2023-01-20', '%Y-%m-%d') - datetime.now()).days, 21.5)

    # (tsla_call(6, 780) - tsla_call(1, 780)) / 5
    # (tsla_call(3, 800) - tsla_call(3, 780)) / (tsla_call(3, 780) - tsla_call(0.5, 780))

    # tsla_call(4, 782)

    optimize_naked_call(
        "TSLA", 900, 1100, 5, 780, "2022-01-21", IV_min=0.44, max_days=150
    )
    optimize_naked_call(
        "LRCX",
        strike_min=570,
        strike_max=600,
        strike_step=2.5,
        next_expected_price=546,
        expiry="2021-10-29",
        IV_min=0.3,
        max_days=14,
    )

    # ROKU long call
    thecall = get_naked_call_fn(strike=400, IV=0.55, max_days=60)
    min_rr = (
        thecall(1, 400)
        / thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 320
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 1.1
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 320
        ),
    )  # 2.2

    # LRCX long call
    thecall = get_naked_call_fn(strike=620, IV=0.3, max_days=60)
    min_rr = (
        thecall(1, 620)
        / thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 550
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 0.76
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 550
        ),
    )  # 2.2

    # BABA long call
    thecall = get_naked_call_fn(strike=210, IV=0.42, max_days=60)
    min_rr = (
        thecall(1, 210)
        / thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 170
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 3.2
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 170
        ),
    )  # 2.2

    # LCID long call
    thecall = get_naked_call_fn(strike=29, IV=1.024, max_days=60)
    min_rr = (
        thecall(1, 29)
        / thecall(
            (datetime.strptime("2021-11-05", "%Y-%m-%d") - datetime.now()).days, 23.5
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 1.7
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-05", "%Y-%m-%d") - datetime.now()).days, 23.5
        ),
    )  # 2.2

    # PLUG long call
    thecall = get_naked_call_fn(strike=38, IV=0.77, max_days=60)
    min_rr = (
        thecall(1, 38)
        / thecall(
            (datetime.strptime("2021-11-05", "%Y-%m-%d") - datetime.now()).days, 34
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 1.7
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-05", "%Y-%m-%d") - datetime.now()).days, 34
        ),
    )

    # ZM long call
    thecall = get_naked_call_fn(strike=350, IV=0.43, max_days=60)
    min_rr = (
        thecall(1, 350)
        / thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 280
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 4.2
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 280
        ),
    )

    # UPST long call
    thecall = get_naked_call_fn(strike=550, IV=1.1173, max_days=60)
    min_rr = (
        thecall(15, 500)
        / thecall(
            (datetime.strptime("2021-11-19", "%Y-%m-%d") - datetime.now()).days, 320
        )
        - 1
    )
    thecall(14, 400)
    print("Minimum Reward/Risk:", min_rr[0])  # 4.2
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 280
        ),
    )

    # AAPL long call
    thecall = get_naked_call_fn(strike=160, IV=0.21, max_days=60)
    min_rr = (
        thecall(1, 160)
        / thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 144.6
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  # 0.76
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2021-11-26", "%Y-%m-%d") - datetime.now()).days, 144.6
        ),
    )  # 2.2

    # FB long call
    thecall = get_naked_call_fn(strike=400, IV=0.335, max_days=120)
    min_rr = (
        thecall(1, 400)
        / thecall(
            (datetime.strptime("2022-01-21", "%Y-%m-%d") - datetime.now()).days, 323
        )
        - 1
    )
    print("Minimum Reward/Risk:", min_rr[0])  #
    print(
        "Buy price:",
        thecall(
            (datetime.strptime("2022-01-21", "%Y-%m-%d") - datetime.now()).days, 323
        ),
    )  # 3.19

    optimize_naked_put(
        "UPST",
        strike_min=260,
        strike_max=300,
        strike_step=2.5,
        next_expected_price=315,
        expiry="2021-10-15",
        IV_min=0.85,
        max_days=14,
    )

    the_put = get_naked_put_fn(strike=32, IV=0.76, max_days=14)
    the_put((datetime.strptime("2021-10-29", "%Y-%m-%d") - datetime.now()).days + 1, 33)
    the_put((datetime.strptime("2021-10-29", "%Y-%m-%d") - datetime.now()).days + 1, 33)
    min_rr = (
        the_put(1, 32)
        / the_put(
            (datetime.strptime("2021-10-29", "%Y-%m-%d") - datetime.now()).days + 1, 33
        )
        - 1
    )
    print("Reward/Risk:", min_rr)

    optimize_naked_put(
        "LRCX",
        strike_min=510,
        strike_max=550,
        strike_step=5,
        next_expected_price=560,
        expiry="2021-10-15",
        IV_min=0.33,
        max_days=14,
    )

    the_put = get_naked_put_fn(strike=520, IV=0.379, max_days=14)
    the_put(
        (datetime.strptime("2021-10-15", "%Y-%m-%d") - datetime.now()).days + 1, 560
    )

# pd.Series([1,4,3,2]).sort_values().index


# theput1 = get_naked_call_fn(strike=77, IV=0.335, max_days=120)
# theput2 = get_naked_put_fn(strike=97, IV=0.335, max_days=120)

# plt.plot(
#     list(range(60, 110)),
#     [-theput1(0.25, x) + theput2(0.25, x) for x in range(60, 110)],
# )
# plt.plot(list(range(60, 110)), [theput2(0.25, x) for x in range(60, 110)])
# plt.plot(list(range(60, 110)), [-theput1(0.25, x) for x in range(60, 110)])
