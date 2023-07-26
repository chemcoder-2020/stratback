from stratback.backtesting import Strategy
from stratback.backtesting.lib import crossover
import numpy as np
import pandas as pd
import pandas_ta as pt
from stratback.utils.TALibrary import vwap, price_position_by_pivots, calc_vwap
import datetime
import re


class MTFVWAPStrategy(Strategy):
    HTF1 = "D"
    HTF2 = "W"
    HTF3 = "M"
    use_three_TF = False
    size = None
    long_only = True
    short_only = False
    daytrade = True
    pl_pct_tp = None
    limit_pct = None
    stop_pct = None
    use_rsi = False
    use_avwap_cross = True
    pivot_shift = 1
    price_move_tp = None

    def mtfvwap_signal(
        self,
        data,
        long_only=False,
        short_only=False,
    ):
        data = data.copy()

        if "date" in data.columns:
            data.set_index("date", inplace=True)
        data["day"] = pd.DatetimeIndex(data.index).date
        data["isFirstBar"] = data["day"].diff() >= "1 days"


        rsi = pt.rsi(data.ta.hlc3(), 10)
        rsi_up = rsi.gt(rsi.shift())
        use_rsi_cond = {True: (rsi_up, ~rsi_up), False: (True, True)}

        avwap_htf1 = calc_vwap(data, self.HTF1)
        avwap_htf2 = calc_vwap(data, self.HTF2)
        if self.use_three_TF:
            avwap_htf3 = calc_vwap(data, self.HTF3)
            avwap_uptrend = avwap_htf1.gt(avwap_htf2) & avwap_htf2.gt(avwap_htf3)
            avwap_dntrend = avwap_htf1.lt(avwap_htf2) & avwap_htf2.lt(avwap_htf3)
        else:
            avwap_uptrend = avwap_htf1.gt(avwap_htf2)
            avwap_dntrend = avwap_htf1.lt(avwap_htf2)
        avwap_crossover = avwap_htf1.gt(avwap_htf2) & avwap_htf1.shift().lt(
            avwap_htf2.shift()
        )
        avwap_crossunder = avwap_htf1.lt(avwap_htf2) & avwap_htf1.shift().gt(
            avwap_htf2.shift()
        )

        if self.price_move_tp is not None:
            price_position = price_position_by_pivots(
                data, pivot_data_shift=self.pivot_shift
            )
            pexp = price_position.groupby(
                price_position.index.to_period("D")
            ).expanding()
            price_move = pexp.apply(lambda x: x[-1]) - pexp.apply(lambda x: x[0])
            price_move = price_move.droplevel(0)

        buy_call_open = data["isFirstBar"].fillna(False) & avwap_uptrend
        buy_put_open = data["isFirstBar"].fillna(False) & avwap_dntrend

        longCondition = (
            buy_call_open | avwap_crossover if self.use_avwap_cross else buy_call_open
        ) & use_rsi_cond[self.use_rsi][0]

        shortCondition = (
            buy_put_open | avwap_crossunder if self.use_avwap_cross else buy_put_open
        ) & use_rsi_cond[self.use_rsi][1]

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
        short = in_session & shortCondition & (not long_only)

        longX = (
            price_move.eq(self.price_move_tp)
            & price_move.shift().ne(self.price_move_tp)
            if self.price_move_tp is not None
            else pd.Series([False] * len(data), index=data.index)
        )
        shortX = (
            price_move.eq(-self.price_move_tp)
            & price_move.shift().ne(-self.price_move_tp)
            if self.price_move_tp is not None
            else pd.Series([False] * len(data), index=data.index)
        )
        # print(longX)

        if self.daytrade:
            eod = pd.DatetimeIndex(data.index).time >= datetime.time(12, 25)
        else:
            eod = (
                pd.Series(
                    [False] * len(data),
                    index=data.index,
                ),
            )
        return long, short, longX, shortX, avwap_htf1, rsi, eod

    def init(self):
        (
            self.longs,
            self.shorts,
            self.longXs,
            self.shortXs,
            self.ma_mid,
            self.rsi,
            self.eod,
        ) = self.I(
            self.mtfvwap_signal,
            self.data.df,
            long_only=self.long_only,
            short_only=self.short_only,
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
        self.bars = np.unique(self.data.df.index.strftime("%H:%M"))
        self.close = self.I(lambda x: x, self.data.Close, plot=False)

        self.vwap_htf1 = self.I(calc_vwap, self.data.df, self.HTF1, overlay=True)
        self.vwap_htf2 = self.I(calc_vwap, self.data.df, self.HTF2, overlay=True)
        if self.use_three_TF:
            self.vwap_htf3 = self.I(calc_vwap, self.data.df, self.HTF3, overlay=True)

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

            elif self.position.is_long and crossover(self.longXs, 0.5):
                self.position.close()
                self._signals[self.data.index[-1]] = 2

            elif self.position.is_short and crossover(self.shortXs, 0.5):
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
