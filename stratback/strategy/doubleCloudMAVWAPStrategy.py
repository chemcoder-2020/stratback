from stratback.backtesting import Strategy
from stratback.backtesting.lib import crossover
import numpy as np
import pandas as pd
import pandas_ta as pt
from stratback.utils.TALibrary import (
    vwap,
)
import datetime
import re


class DoubleCloudMAVWAPStrategy(Strategy):
    n1 = 11
    n2 = 23
    n3 = 70
    size = 1
    target_pct = 0.7
    HTF1 = "D"
    HTF2 = "M"
    long_only = True
    short_only = False
    assume_downtrend_follows_uptrend = False
    daytrade = True
    with_longX = False
    with_shortX = False
    pl_pct_tp = None
    limit_pct = None
    stop_pct = None

    def ma_double_cloud_signal(
        self,
        data,
        target_pct=0.7,
        price_for_ma="Open",
        long_only=True,
        short_only=False,
        assume_downtrend_follows_uptrend=False,
        with_longX=False,
        with_shortX=False,
    ):
        data = data.copy()
        ma_length1 = self.n1
        ma_length2 = self.n2
        ma_length3 = self.n3

        if "date" in data.columns:
            data.set_index("date", inplace=True)
        data["day"] = pd.DatetimeIndex(data.index).date
        data["isFirstBar"] = data["day"].diff() >= "1 days"
        def calc_vwap(df, tf):
            if re.split(r"\d", tf)[-1] in ["H", "min"] or re.split(r"\D", tf)[0] != "":
                return (df.ta.hlc3() * df.Volume).groupby(
                    df.index.floor(tf)
                ).cumsum() / df.Volume.groupby(df.index.floor(tf)).cumsum()
            else:
                return df.ta.vwap(anchor=tf)
        avwap_htf1 = calc_vwap(data, self.HTF1)
        avwap_htf2 = calc_vwap(data, self.HTF2)
        avwap_up = avwap_htf1.gt(avwap_htf2)
        avwap_dn = avwap_htf1.lt(avwap_htf2)
        price = data[price_for_ma]
        volume = data.Volume
        high = data.High
        low = data.Low
        ma1 = vwap(price, volume, ma_length1)
        ma2 = vwap(price, volume, ma_length2)
        ma3 = vwap(price, volume, ma_length3)
        ma4 = pt.sma(price, ma_length3)

        ma_mid = (ma1 + ma2) / 2
        ma_bandwidth = ma1 - ma2
        longTarget = (target_pct / 100 + 1) * ma_mid
        shortTarget = ma_mid / (target_pct / 100 + 1)
        if with_longX:
            longExit = high > longTarget
        else:
            longExit = pd.Series([False] * len(data), index=data.index)

        if with_shortX:
            shortExit = low < shortTarget
        else:
            shortExit = pd.Series([False] * len(data), index=data.index)
        trend_up = ma1.gt(ma2)
        trend_dn = ma1.lt(ma2)
        rsi = pt.rsi(data.ta.hlc3(), 10)
        strong_trend = rsi.gt(20) & rsi.lt(80)

        momentum_condition1 = (rsi.gt(rsi.shift())).fillna(False)
        momentum_condition = momentum_condition1
        newday_trendup_continuation = (
            data["isFirstBar"].fillna(False)
            & strong_trend
            & (trend_up.fillna(False) & trend_up.shift().fillna(False) & ~longExit & ma3.gt(ma4))
            & ma_bandwidth.pct_change().gt(0)
            & momentum_condition
        )
        newday_trenddn_continuation = (
            data["isFirstBar"].fillna(False)
            & strong_trend
            & (trend_dn.fillna(False) & trend_dn.shift().fillna(False) & ~shortExit & ma3.lt(ma4))
            & ma_bandwidth.pct_change().gt(0)
            & ~momentum_condition
        )

        longCondition = ((
            (trend_up & ~trend_up.shift().fillna(False))
            & momentum_condition
            & ma3.gt(ma4)
        ) | newday_trendup_continuation) & avwap_up

        shortCondition = ((
            (trend_dn & ~trend_dn.shift().fillna(False))
            & ~momentum_condition
            & ma3.lt(ma4)
        ) | newday_trenddn_continuation) & avwap_dn

        if self.daytrade:
            in_session = pd.Series(
                np.logical_and(
                    pd.DatetimeIndex(data.index).time < datetime.time(12, 25),
                    pd.DatetimeIndex(data.index).time >= datetime.time(6, 30),
                ),
                index=data.index,
            )
        else:
            in_session = (
                pd.Series(
                    [True] * len(data),
                    index=data.index,
                ),
            )

        long = in_session & longCondition & (not short_only)

        if assume_downtrend_follows_uptrend:
            short = in_session & (longExit & (not long_only))
        else:
            short = in_session & shortCondition & (not long_only)

        longX = longExit
        shortX = shortExit

        if self.daytrade:
            eod = pd.DatetimeIndex(data.index).time >= datetime.time(12, 25)
        else:
            eod = (
                pd.Series(
                    [False] * len(data),
                    index=data.index,
                ),
            )
        return long, short, longX, shortX, longTarget, shortTarget, ma_mid, rsi, eod

    def init(self):
        (
            self.longs,
            self.shorts,
            self.longXs,
            self.shortXs,
            self.longTarget,
            self.shortTarget,
            self.ma_mid,
            self.rsi,
            self.eod,
        ) = self.I(
            self.ma_double_cloud_signal,
            self.data.df,
            self.target_pct,
            long_only=self.long_only,
            short_only=self.short_only,
            assume_downtrend_follows_uptrend=self.assume_downtrend_follows_uptrend,
            with_longX=self.with_longX,
            with_shortX=self.with_shortX,
            plot=False,
        )
        self.in_session = self.I(
            lambda x: x,
            pd.Series(
                np.logical_and(
                    pd.DatetimeIndex(self.data.df.index).time < datetime.time(12, 25),
                    pd.DatetimeIndex(self.data.df.index).time >= datetime.time(6, 30),
                ),
                index=self.data.df.index,
            ),
            plot=False,
        )
        self._signals = pd.Series(index=self.data.df.index, dtype=int)
        self.bars = np.unique(self.data.df.index.strftime("%H:%M:%S"))
        self.close = self.I(lambda x: x, self.data.Close, plot=False)
        self.high = self.I(lambda x: x, self.data.High, plot=False)
        self.low = self.I(lambda x: x, self.data.Low, plot=False)

    def next(self):
        if self.eod:
            if self.position.is_long:
                self.position.close()
                self._signals[self.data.index[-1]] = 2
            elif self.position.is_short:
                self.position.close()
                self._signals[self.data.index[-1]] = -2
            else:
                self._signals[self.data.index[-1]] = np.nan
        elif self.in_session:
            if crossover(self.longs, 0.5) and not self.short_only:
                limit = (
                    (1 - self.limit_pct) * self.close[-1]
                    if self.limit_pct is not None
                    else None
                )
                if self.size is None:
                    self.buy(limit=limit)
                else:
                    self.buy(size=self.size, limit=limit)
                self._signals[self.data.index[-1]] = 1
            elif crossover(self.shorts, 0.5) and not self.long_only:
                limit = (
                    (1 + self.limit_pct) * self.close[-1]
                    if self.limit_pct is not None
                    else None
                )
                if self.size is None:
                    self.sell(limit=limit)
                else:
                    self.sell(size=self.size, limit=limit)
                self._signals[self.data.index[-1]] = -1

            elif (
                self.position.is_long
                and crossover(self.longXs, 0.5)
                and self.with_longX
            ):
                self.position.close()
                self._signals[self.data.index[-1]] = 2

            elif (
                self.position.is_short
                and crossover(self.shortXs, 0.5)
                and self.with_shortX
            ):
                self.position.close()
                self._signals[self.data.index[-1]] = -2
            else:
                self._signals[self.data.index[-1]] = np.nan

            if self.stop_pct is not None:
                if (
                    self.position.is_long or self.position.is_short
                ) and self.position.pl_pct < -self.stop_pct:
                    self.position.close()

    def signals(self):
        return self._signals.shift()

    def cloud_signals(self):
        df = pd.concat(
            [
                pd.Series(list(self.longs), index=self.data.df.index),
                pd.Series(list(self.shorts), index=self.data.df.index),
                pd.Series(list(self.longXs), index=self.data.df.index),
                pd.Series(list(self.shortXs), index=self.data.df.index),
                pd.Series(list(self.longTarget), index=self.data.df.index),
                pd.Series(list(self.shortTarget), index=self.data.df.index),
                pd.Series(list(self.ma_mid), index=self.data.df.index),
                pd.Series(list(self.rsi), index=self.data.df.index),
                pd.Series(list(self.eod), index=self.data.df.index),
            ],
            axis=1,
        )
        df.columns = [
            "Long",
            "LongX",
            "Short",
            "ShortX",
            "longTarget",
            "shortTarget",
            "maMID",
            "RSI",
            "EOD",
        ]
        df = df.astype(
            {"Long": bool, "LongX": bool, "Short": bool, "ShortX": bool, "EOD": bool}
        )
        return df