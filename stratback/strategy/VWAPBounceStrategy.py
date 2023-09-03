from stratback.backtesting import Strategy
from stratback.backtesting.lib import crossover
import numpy as np
import pandas as pd
import pandas_ta as pt
from stratback.utils.TALibrary import (
    price_position_by_pivots,
    calc_vwap,
    crossabove,
    crossbelow,
)
import datetime


class VWAPBounceStrategy(Strategy):
    HTF1 = "4H"
    HTF2 = "W"
    ntouch = 1
    crossing_count_reset = "1H"
    rolling_tf = False
    entry_zone = "('6:30', '7:30')"
    sod_time = "6:30"
    eod_time = "12:50"
    size = None
    long_only = False
    short_only = False
    daytrade = True
    pl_pct_tp = None
    limit_pct = None
    stop_pct = None
    tsl_pct = None
    use_rsi = False
    pivot_shift = 1
    price_move_tp = None
    restrict_entry_zone = False
    filter_by_secondary_timeframe = False
    consider_wicks = False
    support_rejection = True
    ignore_vwap_crossabove = False
    resistance_rejection = False
    exit_on_level_rejection = False
    vwap_diff_n = 1
    ignore_firstbar_vwap_diff = False
    trade_limit = None

    def vwapbounce_signal(
        self,
        data,
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
        vwap_crossabove_htf1 = crossabove(
            data,
            avwap_htf1,
            self.crossing_count_reset,
            consider_wicks=self.consider_wicks,
            rolling_tf=self.rolling_tf,
        )

        avwap_htf2 = calc_vwap(data, self.HTF2)

        if self.price_move_tp is not None:
            price_position = price_position_by_pivots(
                data, pivot_data_shift=self.pivot_shift
            )
            pexp = price_position.groupby(
                price_position.index.to_period(self.HTF1)
            ).expanding()
            price_move = pexp.apply(lambda x: x[-1]) - pexp.apply(lambda x: x[0])
            price_move = price_move.droplevel(0)

        _, weekly_levels = price_position_by_pivots(
            data, secondary_tf=self.HTF2, return_all_levels=True
        )

        R_nreject = pd.concat(
            [
                crossbelow(
                    data,
                    weekly_levels[label],
                    self.crossing_count_reset,
                    rolling_tf=self.rolling_tf,
                )
                for label in weekly_levels.columns
            ],
            axis=1,
        ).sum(axis=1)
        S_nreject = pd.concat(
            [
                crossabove(
                    data,
                    weekly_levels[label],
                    self.crossing_count_reset,
                    rolling_tf=self.rolling_tf,
                )
                for label in weekly_levels.columns
            ],
            axis=1,
        ).sum(axis=1)

        entry_hr_left = int(eval(self.entry_zone)[0].split(":")[0])
        entry_min_left = int(eval(self.entry_zone)[0].split(":")[1])
        entry_hr_right = int(eval(self.entry_zone)[1].split(":")[0])
        entry_min_right = int(eval(self.entry_zone)[1].split(":")[1])

        if self.support_rejection:
            if self.ignore_vwap_crossabove:
                longCondition = (S_nreject.gt(0)) & use_rsi_cond[self.use_rsi][0]
            else:
                longCondition = (
                    (
                        vwap_crossabove_htf1.eq(self.ntouch)
                        & vwap_crossabove_htf1.shift().eq(self.ntouch - 1)
                    )
                    | S_nreject.gt(0)
                ) & use_rsi_cond[self.use_rsi][0]
        else:
            longCondition = (
                vwap_crossabove_htf1.eq(self.ntouch)
                & vwap_crossabove_htf1.shift().eq(self.ntouch - 1)
            ) & use_rsi_cond[self.use_rsi][0]

        if self.vwap_diff_n:
            vwap_mom = (avwap_htf1 - avwap_htf2).diff(self.vwap_diff_n)
            if self.ignore_firstbar_vwap_diff:
                vwap_mom[data["isFirstBar"]] = 0
            longCondition = longCondition & vwap_mom.gt(0)

        if self.filter_by_secondary_timeframe:
            longCondition = longCondition & avwap_htf1.gt(avwap_htf2)

        if self.restrict_entry_zone:
            longCondition = longCondition & pd.Series(
                np.logical_and(
                    pd.DatetimeIndex(data.index).time
                    < datetime.time(entry_hr_right, entry_min_right),
                    pd.DatetimeIndex(data.index).time
                    >= datetime.time(entry_hr_left, entry_min_left),
                ),
                index=data.index,
            )

        if self.resistance_rejection:
            shortCondition = (
                avwap_htf1.lt(avwap_htf2) & avwap_htf1.shift().ge(avwap_htf2.shift())
                | R_nreject.gt(0)
            ) & use_rsi_cond[self.use_rsi][1]
        else:
            shortCondition = (
                avwap_htf1.lt(avwap_htf2)
                & avwap_htf1.shift().ge(avwap_htf2.shift())
                & use_rsi_cond[self.use_rsi][1]
            )
        if self.filter_by_secondary_timeframe:
            shortCondition = shortCondition & avwap_htf1.lt(avwap_htf2)
        if self.restrict_entry_zone:
            shortCondition = shortCondition & pd.Series(
                np.logical_and(
                    pd.DatetimeIndex(data.index).time
                    < datetime.time(entry_hr_right, entry_min_right),
                    pd.DatetimeIndex(data.index).time
                    >= datetime.time(entry_hr_left, entry_min_left),
                ),
                index=data.index,
            )

        if self.daytrade:
            in_session = pd.Series(
                np.logical_and(
                    pd.DatetimeIndex(data.index).time
                    < datetime.time(
                        int(self.eod_time.split(":")[0]),
                        int(self.eod_time.split(":")[1]),
                    ),
                    pd.DatetimeIndex(data.index).time
                    >= datetime.time(
                        int(self.sod_time.split(":")[0]),
                        int(self.sod_time.split(":")[1]),
                    ),
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

        long = in_session & longCondition & (not self.short_only)
        short = in_session & shortCondition & (not self.long_only)
        group = data.index.floor("D")
        long_num = long.groupby(group).cumsum()
        short_num = short.groupby(group).cumsum()
        signal_num = long_num + short_num

        if self.exit_on_level_rejection:
            longX = R_nreject.gt(0) & R_nreject.shift().eq(0)
            shortX = S_nreject.gt(0) & S_nreject.shift().eq(0)
        else:
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

        if self.daytrade:
            eod = pd.DatetimeIndex(data.index).time >= datetime.time(
                int(self.eod_time.split(":")[0]), int(self.eod_time.split(":")[1])
            )
        else:
            eod = (
                pd.Series(
                    [False] * len(data),
                    index=data.index,
                ),
            )
        return long, short, longX, shortX, avwap_htf1, rsi, eod, signal_num

    def init(self):
        (
            self.longs,
            self.shorts,
            self.longXs,
            self.shortXs,
            self.ma_mid,
            self.rsi,
            self.eod,
            self.signal_num,
        ) = self.I(
            self.vwapbounce_signal,
            self.data.df,
            plot=False,
        )
        self.in_session = self.I(
            lambda x: x,
            pd.Series(
                np.logical_and(
                    pd.DatetimeIndex(self.data.df.index).time
                    < datetime.time(
                        int(self.eod_time.split(":")[0]),
                        int(self.eod_time.split(":")[1]),
                    ),
                    pd.DatetimeIndex(self.data.df.index).time
                    >= datetime.time(
                        int(self.sod_time.split(":")[0]),
                        int(self.sod_time.split(":")[1]),
                    ),
                ),
                index=self.data.df.index,
            ),
            plot=False,
        )
        self._signals = pd.Series(index=self.data.df.index, dtype=int)
        self.bars = np.unique(self.data.df.index.strftime("%H:%M"))
        self.close = self.I(lambda x: x, self.data.Close, plot=False)
        self.open = self.I(lambda x: x, self.data.Open, plot=False)
        self.low = self.I(lambda x: x, self.data.Low, plot=False)
        self.high = self.I(lambda x: x, self.data.High, plot=False)

        self.vwap_htf1 = self.I(calc_vwap, self.data.df, self.HTF1, overlay=True)
        self.vwap_htf2 = self.I(calc_vwap, self.data.df, self.HTF2, overlay=True)

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
            if self.trade_limit is None:
                if (
                    crossover(self.longs, 0.5)
                    and not self.short_only
                    and not self.position
                ):
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
                elif (
                    crossover(self.shorts, 0.5)
                    and not self.long_only
                    and not self.position
                ):
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

                if self.pl_pct_tp is not None:
                    if (
                        self.position.is_long or self.position.is_short
                    ) and self.position.pl_pct > self.pl_pct_tp:
                        self.position.close()

                # tsl
                if self.tsl_pct is not None:
                    for trade in self.trades:
                        entry_price = trade.entry_price
                        tsl = entry_price * self.tsl_pct
                        if trade.is_long:
                            trade.sl = max(
                                trade.sl or -np.inf, self.data["Close"][-1] - tsl
                            )
                        else:  # short
                            trade.sl = min(
                                trade.sl or np.inf,
                                self.data["Close"][-1] + tsl,
                            )
            else:
                if (
                    crossover(self.longs, 0.5)
                    and not self.short_only
                    and not self.position
                    and self.signal_num <= self.trade_limit
                ):
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
                elif (
                    crossover(self.shorts, 0.5)
                    and not self.long_only
                    and not self.position
                    and self.signal_num <= self.trade_limit
                ):
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

                if self.pl_pct_tp is not None:
                    if (
                        self.position.is_long or self.position.is_short
                    ) and self.position.pl_pct > self.pl_pct_tp:
                        self.position.close()

                # tsl
                if self.tsl_pct is not None:
                    for trade in self.trades:
                        entry_price = trade.entry_price
                        tsl = entry_price * self.tsl_pct
                        if trade.is_long:
                            trade.sl = max(
                                trade.sl or -np.inf, self.data["Close"][-1] - tsl
                            )
                        else:  # short
                            trade.sl = min(
                                trade.sl or np.inf,
                                self.data["Close"][-1] + tsl,
                            )

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
