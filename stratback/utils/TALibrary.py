import pandas as pd
import pandas_ta as pt
import numpy as np
import vectorbt as vbt
import datetime
from scipy.stats import linregress
import re

pd.options.display.max_rows = 999


def EhlersSuperSmoother(close, length):
    a1 = np.exp(-np.sqrt(2) * np.pi / length)
    coeff2 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / length)
    coeff3 = -(a1**2)
    coeff1 = 1 - coeff2 - coeff3
    filt = close.copy()
    n = filt.size
    if type(close) == pd.Series:
        for i in range(0, n):
            filt.iloc[i] = (
                coeff1 * (close.iloc[i] + close.iloc[i - 1]) / 2
                + coeff2 * filt.iloc[i - 1]
                + coeff3 * filt.iloc[i - 2]
            )
    else:
        for i in range(0, n):
            filt[i] = (
                coeff1 * (close[i] + close[i - 1]) / 2
                + coeff2 * filt[i - 1]
                + coeff3 * filt[i - 2]
            )
    return filt


def EhlersHighpass(close, length):
    a1 = np.exp(-np.sqrt(2) * np.pi / length)
    coeff2 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / length)
    coeff3 = -(a1**2)
    coeff1 = (1 + coeff2 - coeff3) / 4
    filt = close.copy()
    n = filt.size
    for i in range(0, n):
        filt.iloc[i] = (
            coeff1 * (close.iloc[i] - (2 * close.iloc[i - 1] + close.iloc[i - 2]))
            + coeff2 * filt.iloc[i - 1]
            + coeff3 * filt.iloc[i - 2]
        )
    return filt


def vwap(price, volume, length):
    return (price * volume).rolling(length).sum() / volume.rolling(length).sum()


def dsma_alpha(price, length):
    price_change = price - price.shift(2)
    filt = EhlersSuperSmoother(price_change.bfill(), length)

    def rolling_rms(x):
        r = 1.0
        a = np.sqrt((len(x) * r**2) / (x**2).sum())
        x = x * a
        return x.iloc[-1]

    ScaledFilt = filt.rolling(length).apply(rolling_rms).bfill()

    alpha = ScaledFilt.abs() * 5 / length
    return alpha


def dsma(price, length):
    alpha = dsma_alpha(price, length)
    dsma = price.copy()
    n = dsma.size
    for i in range(1, n):
        dsma.iloc[i] = alpha[i] * price[i] + (1 - alpha[i]) * dsma[i - 1]
    return dsma


def ma_cloud_signal(
    data,
    ma_length1=11,
    ma_length2=24,
    target_pct=0.65,
    price_for_ma="open",
    shift=False,
):
    data = data.copy()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    data["day"] = pd.DatetimeIndex(data.index).date
    data["isFirstBar"] = data["day"].diff() >= "1 days"
    price = data[price_for_ma]
    volume = data.volume
    high = data.high
    low = data.low
    ma1 = vwap(price, volume, ma_length1)
    ma2 = vwap(price, volume, ma_length2)
    ma_mid = (ma1 + ma2) / 2
    longTarget = (target_pct / 100 + 1) * ma_mid
    shortTarget = ma_mid / (target_pct / 100 + 1)
    longExit = high > longTarget
    shortExit = low < shortTarget
    trend_up = ma1.gt(ma2)
    trend_down = ma1.lt(ma2)
    rsi = pt.rsi(data.open, 10)
    strong_trend = rsi.gt(40) & rsi.lt(80)
    newday_trendup_continuation = (
        data["isFirstBar"].fillna(False)
        & strong_trend
        & (trend_up.fillna(False) & ~longExit)
    )
    long = (trend_up & ~trend_up.shift().fillna(False)) | newday_trendup_continuation
    longX = longExit
    short = longExit
    shortX = shortExit | long
    signal = pd.DataFrame(
        columns=["Long", "LongX", "Short", "ShortX"], index=data.index
    )
    signal["Long"] = long
    signal["LongX"] = longX
    signal["Short"] = short
    signal["ShortX"] = shortX
    signal["longTarget"] = longTarget
    signal["shortTarget"] = shortTarget
    signal["maMID"] = ma_mid
    signal["RSI"] = rsi
    signal["EOD"] = pd.DatetimeIndex(signal.index).time == datetime.time(12, 50)
    if shift:
        return signal.shift()
    else:
        return signal


def ma_cloud_ichimoku_signal(
    data,
    ma_length1=15,
    ma_length2=24,
    target_pct=0.67,
    ichimoku_conversionLine_length=9,
    ichimoku_baseLine_length=26,
    price_for_ma="open",
    shift=False,
):
    data = data.copy()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    data["day"] = pd.DatetimeIndex(data.index).date
    data["isFirstBar"] = data["day"].diff() >= "1 days"
    price = data[price_for_ma]
    volume = data.volume
    high = data.high
    low = data.low
    ma1 = vwap(price, volume, ma_length1)
    ma2 = vwap(price, volume, ma_length2)
    ma_mid = (ma1 + ma2) / 2
    ma_bandwidth = ma1 - ma2
    longTarget = (target_pct / 100 + 1) * ma_mid
    shortTarget = ma_mid / (target_pct / 100 + 1)
    longExit = high > longTarget
    shortExit = low < shortTarget

    # Ichimoku
    ma_max = pd.concat([ma1, ma2], axis=1).max(axis=1).bfill()
    conversionLine = (
        low.rolling(ichimoku_conversionLine_length).min()
        + high.rolling(ichimoku_conversionLine_length).max()
    ) / 2

    baseLine = (
        low.rolling(ichimoku_baseLine_length).min()
        + high.rolling(ichimoku_baseLine_length).max()
    ) / 2
    leadline1 = (conversionLine + baseLine) / 2
    leadline2 = (
        low.rolling(int(2 * ichimoku_baseLine_length)).min()
        + high.rolling(int(2 * ichimoku_baseLine_length)).max()
    ) / 2
    close_under_ichimoku = data.close.lt(
        pd.concat([leadline1, leadline2], axis=1)
        .min(axis=1)
        .shift(ichimoku_baseLine_length - 1)
    )
    trend_up = ma1.gt(ma2)
    trend_down = ma1.lt(ma2)
    # rsi = pt.rsi(data.open, 10)
    rsi = pt.rsi(data.ta.hlc3(), 10)
    strong_trend = rsi.gt(20) & rsi.lt(80)
    newday_trendup_continuation = (
        data["isFirstBar"].fillna(False)
        & strong_trend
        & (trend_up.fillna(False) & ~longExit)
        & ma_bandwidth.pct_change().gt(0)
    )
    shortCondition = (
        (close_under_ichimoku & ~close_under_ichimoku.shift().fillna(False))
        & trend_down.fillna(False)
    ) | (
        close_under_ichimoku.fillna(False)
        & trend_down.fillna(False)
        & data["isFirstBar"].fillna(False)
        # & ~shortExit
    )
    momentum_condition1 = (rsi.gt(rsi.shift())).fillna(False)
    # momentum_condition2 = (close - op).gt(0)
    momentum_condition = momentum_condition1
    longCondition = (
        (trend_up & ~trend_up.shift().fillna(False)) & momentum_condition
    ) | newday_trendup_continuation
    long = longCondition
    longX = longExit
    # short = longExit | shortCondition
    short = shortCondition
    shortX = shortExit | long
    # long = trend_up & ~trend_up.shift().fillna(False)
    # longX = longExit
    # short = longExit
    # shortX = shortExit | long
    signal = pd.DataFrame(
        columns=["Long", "LongX", "Short", "ShortX"], index=data.index
    )
    signal["Long"] = long
    signal["LongX"] = longX & ~longX.shift().fillna(False)
    signal["Short"] = short
    signal["ShortX"] = shortX & ~shortX.shift().fillna(False)
    signal["longTarget"] = longTarget
    signal["shortTarget"] = shortTarget
    signal["maMID"] = ma_mid
    signal["RSI"] = rsi
    signal["EOD"] = pd.DatetimeIndex(signal.index).time == datetime.time(12, 50)
    if shift:
        return signal.shift()
    else:
        return signal


def linreg_signal(data, lookback, shift=False):
    data = data.copy()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    lo = data.low.copy()
    hi = data.high.copy()
    linreg_lo = pt.linreg(lo, lookback, tsf=True)
    linreg_hi = pt.linreg(hi, lookback, tsf=True)
    trendlines = pd.DataFrame(
        index=data.index, columns=["Hi", "Lo", "Hi_Breakage", "Lo_Breakage"]
    )
    trendlines["Hi"] = linreg_hi
    trendlines["Lo"] = linreg_lo
    trendlines["Hi_Breakage"] = hi.gt(linreg_hi) & hi.shift().lt(linreg_hi.shift())
    trendlines["Lo_Breakage"] = lo.lt(linreg_lo) & lo.shift().gt(linreg_lo.shift())
    if shift:
        return trendlines.shift()
    else:
        return trendlines


def ma_double_cloud_signal(
    data,
    ma_length1=18,
    ma_length2=47,
    ma_length3=70,
    target_pct=0.665,
    price_for_ma="open",
    shift=False,
    long_only=True,
    short_only=False,
    assume_downtrend_follows_uptrend=False,
    with_longX=False,
    with_shortX=False,
    daytrade=True,
):
    data = data.copy()

    if "date" in data.columns:
        data.set_index("date", inplace=True)
    data["day"] = pd.DatetimeIndex(data.index).date
    data["isFirstBar"] = data["day"].diff() >= "1 days"
    price = data[price_for_ma]
    volume = data.volume
    high = data.high
    low = data.low
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
        & (
            trend_up.fillna(False)
            & trend_up.shift().fillna(False)
            & ~longExit
            & ma3.gt(ma4)
        )
        & ma_bandwidth.pct_change().gt(0)
        & momentum_condition
    )
    newday_trenddn_continuation = (
        data["isFirstBar"].fillna(False)
        & strong_trend
        & (
            trend_dn.fillna(False)
            & trend_dn.shift().fillna(False)
            & ~shortExit
            & ma3.lt(ma4)
        )
        & ma_bandwidth.pct_change().gt(0)
        & ~momentum_condition
    )

    longCondition = (
        (trend_up & ~trend_up.shift().fillna(False)) & momentum_condition & ma3.gt(ma4)
    ) | newday_trendup_continuation

    shortCondition = (
        (trend_dn & ~trend_dn.shift().fillna(False)) & ~momentum_condition & ma3.lt(ma4)
    ) | newday_trenddn_continuation

    if daytrade:
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

    signal = pd.DataFrame(
        columns=["Long", "LongX", "Short", "ShortX"], index=data.index
    )
    signal["Long"] = long
    signal["LongX"] = longX & ~longX.shift().fillna(False)
    signal["Short"] = short
    signal["ShortX"] = shortX & ~shortX.shift().fillna(False)
    signal["longTarget"] = longTarget
    signal["shortTarget"] = shortTarget
    signal["maMID"] = ma_mid
    signal["RSI"] = rsi
    signal["EOD"] = (pd.DatetimeIndex(signal.index).time >= datetime.time(12, 25)) & (
        pd.DatetimeIndex(signal.index).time <= datetime.time(12, 55)
    )
    if shift:
        return signal.shift()
    else:
        return signal


def price_position_by_pivots(
    data,
    secondary_tf="D",
    pivot_data_shift=1,
    return_nearest_levels=False,
    return_all_levels=False,
):
    data = data.copy().reset_index()
    data.columns = data.columns.str.lower()
    if "date" in data.columns:
        data = data.set_index("date")
    pivot_data = data.shift(pivot_data_shift)

    high = (
        pivot_data.high.groupby(pivot_data.index.to_period(secondary_tf))
        .max()
        .shift(pivot_data_shift)
    )
    high.index = (
        pd.Series(pivot_data.index)
        .groupby(pivot_data.index.to_period(secondary_tf))
        .last()
    )
    high = pivot_data.align(high, axis=0)[1].bfill()

    low = (
        pivot_data.low.groupby(pivot_data.index.to_period(secondary_tf))
        .min()
        .shift(pivot_data_shift)
    )
    low.index = (
        pd.Series(pivot_data.index)
        .groupby(pivot_data.index.to_period(secondary_tf))
        .last()
    )
    low = pivot_data.align(low, axis=0)[1].bfill()
    close = (
        pivot_data.close.groupby(pivot_data.index.to_period(secondary_tf))
        .last()
        .shift(1)
    )
    close.index = (
        pd.Series(pivot_data.index)
        .groupby(pivot_data.index.to_period(secondary_tf))
        .last()
    )
    close = pivot_data.align(close, axis=0)[1].bfill()

    hlc3 = (high + low + close) / 3
    xLow = low
    xHigh = high
    vPP = hlc3
    vR1 = vPP + (vPP - xLow)
    vS1 = vPP - (xHigh - vPP)
    vR2 = vPP + (xHigh - xLow)
    vS2 = vPP - (xHigh - xLow)
    vR3 = vR1 + (xHigh - xLow)
    vS3 = vS1 - (xHigh - xLow)
    vR4 = vR1 + 2 * (xHigh - xLow)
    vS4 = vS1 - 2 * (xHigh - xLow)
    vR5 = vR1 + 3 * (xHigh - xLow)
    vS5 = vS1 - 3 * (xHigh - xLow)

    p5 = data.close.between(vR5, np.inf)
    p4 = data.close.between(vR4, vR5)
    p3 = data.close.between(vR3, vR4)
    p2 = data.close.between(vR2, vR3)
    p1 = data.close.between(vR1, vR2)
    p0 = data.close.between(vPP, vR1)
    p_1 = data.close.between(vS1, vPP)
    p_2 = data.close.between(vS2, vS1)
    p_3 = data.close.between(vS3, vS2)
    p_4 = data.close.between(vS4, vS3)
    p_5 = data.close.between(vS5, vS4)
    p_6 = data.close.between(-np.inf, vS5)

    price_position = pd.concat(
        [p_6, p_5, p_4, p_3, p_2, p_1, p0, p1, p2, p3, p4, p5], axis=1
    ).apply(np.argmax, axis=1)
    levels = pd.concat(
        [
            pd.Series([0] * len(price_position), index=data.index),
            vS5,
            vS4,
            vS3,
            vS2,
            vS1,
            vPP,
            vR1,
            vR2,
            vR3,
            vR4,
            vR5,
            pd.Series([np.inf] * len(price_position), index=data.index),
        ],
        axis=1,
    )
    nearest_levels = pd.concat(
        [
            pd.Series(
                [row[1][p] for row, p in zip(levels.iterrows(), price_position)],
                index=data.index,
            ),
            pd.Series(
                [row[1][p + 1] for row, p in zip(levels.iterrows(), price_position)],
                index=data.index,
            ),
        ],
        axis=1,
    )
    nearest_levels.columns = ["S", "R"]
    nearest_levels["MID"] = (nearest_levels["S"] + nearest_levels["R"]) / 2
    nearest_levels["38.2"] = (
        nearest_levels["S"] + (nearest_levels["R"] - nearest_levels["S"]) * 0.382
    )
    nearest_levels["61.8"] = (
        nearest_levels["S"] + (nearest_levels["R"] - nearest_levels["S"]) * 0.618
    )

    if return_nearest_levels:
        return price_position, nearest_levels
    elif return_all_levels:
        return price_position, levels
    else:
        return price_position


def dailySpreadProbabilityBrackets(data, pivot_tf="W", pivot_data_shift=1):
    price_position, levels = price_position_by_pivots(
        data, pivot_tf, return_all_levels=True, pivot_data_shift=pivot_data_shift
    )
    p5 = data.close.between(levels[11], levels[12])
    p4 = data.close.between(levels[10], levels[11])
    p3 = data.close.between(levels[9], levels[10])
    p2 = data.close.between(levels[8], levels[9])
    p1 = data.close.between(levels[7], levels[8])
    p0 = data.close.between(levels[6], levels[7])
    p_1 = data.close.between(levels[5], levels[6])
    p_2 = data.close.between(levels[4], levels[5])
    p_3 = data.close.between(levels[3], levels[4])
    p_4 = data.close.between(levels[2], levels[3])
    p_5 = data.close.between(levels[1], levels[2])
    p_6 = data.close.between(levels[0], levels[1])

    close_pp = pd.concat(
        [p_6, p_5, p_4, p_3, p_2, p_1, p0, p1, p2, p3, p4, p5], axis=1
    ).apply(np.argmax, axis=1)

    p5 = data.open.between(levels[11], levels[12])
    p4 = data.open.between(levels[10], levels[11])
    p3 = data.open.between(levels[9], levels[10])
    p2 = data.open.between(levels[8], levels[9])
    p1 = data.open.between(levels[7], levels[8])
    p0 = data.open.between(levels[6], levels[7])
    p_1 = data.open.between(levels[5], levels[6])
    p_2 = data.open.between(levels[4], levels[5])
    p_3 = data.open.between(levels[3], levels[4])
    p_4 = data.open.between(levels[2], levels[3])
    p_5 = data.open.between(levels[1], levels[2])
    p_6 = data.open.between(levels[0], levels[1])

    open_pp = pd.concat(
        [p_6, p_5, p_4, p_3, p_2, p_1, p0, p1, p2, p3, p4, p5], axis=1
    ).apply(np.argmax, axis=1)

    mapping = {
        "0": "S6",
        "1": "S5",
        "2": "S4",
        "3": "S3",
        "4": "S2",
        "5": "S1",
        "6": "P",
        "7": "R1",
        "8": "R2",
        "9": "R3",
        "10": "R4",
        "11": "R5",
    }
    df = pd.DataFrame({"from": open_pp, "to": close_pp})

    df2 = df.groupby("from")["to"].value_counts(normalize=True)  # .loc[7]
    out = df2.reset_index().astype({"from": str, "to": str}).replace(mapping)
    return out


def calc_vwap(df, tf):
    df = df.copy()
    df.columns = df.columns.str.lower()
    if re.split(r"\d", tf)[-1] in ["H", "min"] or re.split(r"\D", tf)[0] != "":
        first_time = df.index.unique().time[0].strftime("%H:%M:%S")
        group = (df.index - pd.Timedelta(first_time)).floor(tf) + pd.Timedelta(
            first_time
        )
        return (df.ta.hlc3() * df.volume).groupby(group).cumsum() / df.volume.groupby(
            group
        ).cumsum()
    else:
        return df.ta.vwap(anchor=tf)


def crossabove(data, line, tf, consider_wicks=False, rolling_tf=False):
    data = data.copy()
    data.columns = data.columns.str.lower()
    if "date" in data.columns:
        data.set_index("date", inplace=True)

    if consider_wicks:
        crossabove = (data.close.gt(line) & data.open.lt(line)) | (
            data.open.gt(line) & data.close.gt(line) & data.low.lt(line)
        )
    else:
        crossabove = data.close.gt(line) & data.open.lt(line)
    if rolling_tf:
        crossabove = crossabove.rolling(tf).sum()
    else:
        group = data.index.floor(tf)
        crossabove = crossabove.groupby(group).cumsum()
    return crossabove


def crossbelow(data, line, tf, consider_wicks=False, rolling_tf=False):
    data = data.copy()
    data.columns = data.columns.str.lower()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    if consider_wicks:
        crossbelow = (data.close.lt(line) & data.open.gt(line)) | (
            data.open.lt(line) & data.close.lt(line) & data.high.gt(line)
        )
    else:
        crossbelow = data.close.lt(line) & data.open.gt(line)
    if rolling_tf:
        crossbelow = crossbelow.rolling(tf).sum()
    else:
        group = data.index.floor(tf)
        crossbelow = crossbelow.groupby(group).cumsum()
    return crossbelow


def MTFVWAP(
    data,
    HTF1="D",
    HTF2="W",
    HTF3="M",
    use_three_TF=False,
    use_rsi=False,
    use_avwap_cross=True,
    pivot_shift=78,
    price_move_tp=None,
    long_only=False,
    short_only=False,
    shift=False,
    daytrade=True,
):
    data = data.copy()
    data.columns = data.columns.str.lower()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    data["day"] = pd.DatetimeIndex(data.index).date
    data["isFirstBar"] = data["day"].diff() >= "1 days"

    rsi = pt.rsi(data.ta.hlc3(), 10)
    rsi_up = rsi.gt(rsi.shift())
    use_rsi_cond = {True: (rsi_up, ~rsi_up), False: (True, True)}

    avwap_htf1 = calc_vwap(data, HTF1)
    avwap_htf2 = calc_vwap(data, HTF2)
    avwap_htf3 = calc_vwap(data, HTF3)

    if use_three_TF:
        avwap_htf3 = calc_vwap(data, HTF3)
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

    if price_move_tp is not None:
        price_position = price_position_by_pivots(data, pivot_data_shift=pivot_shift)
        pexp = price_position.groupby(price_position.index.to_period("D")).expanding()
        price_move = pexp.apply(lambda x: x[-1]) - pexp.apply(lambda x: x[0])
        price_move = price_move.droplevel(0)

    buy_call_open = data["isFirstBar"].fillna(False) & avwap_uptrend
    buy_put_open = data["isFirstBar"].fillna(False) & avwap_dntrend

    longCondition = (
        buy_call_open | avwap_crossover if use_avwap_cross else buy_call_open
    ) & use_rsi_cond[use_rsi][0]

    shortCondition = (
        buy_put_open | avwap_crossunder if use_avwap_cross else buy_put_open
    ) & use_rsi_cond[use_rsi][1]

    if daytrade:
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
        price_move.eq(price_move_tp) & price_move.shift().ne(price_move_tp)
        if price_move_tp is not None
        else pd.Series([False] * len(data), index=data.index)
    )
    shortX = (
        price_move.eq(-price_move_tp) & price_move.shift().ne(-price_move_tp)
        if price_move_tp is not None
        else pd.Series([False] * len(data), index=data.index)
    )
    signal = pd.DataFrame(
        columns=["Long", "LongX", "Short", "ShortX"], index=data.index
    )
    signal["isFirstBar"] = data["isFirstBar"]
    signal["Long"] = long
    signal["LongX"] = longX
    signal["Short"] = short
    signal["ShortX"] = shortX
    signal["RSI"] = rsi
    signal["EOD"] = (pd.DatetimeIndex(signal.index).time >= datetime.time(12, 25)) & (
        pd.DatetimeIndex(signal.index).time <= datetime.time(12, 55)
    )
    if shift:
        return signal.shift()
    else:
        return signal


def intraday_dynamic_level_breaks(data, return_levels=False):
    data = data.copy()
    data.columns = data.columns.str.lower()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    dynamic_support = data.low.groupby(data.index.to_period("D")).cummin()
    dynamic_resistance = data.high.groupby(data.index.to_period("D")).cummax()
    support_nbreaks = (
        dynamic_support.ne(dynamic_support.shift())
        .groupby(dynamic_support.index.to_period("D"))
        .cumsum()
        - 1
    )
    resistance_nbreaks = (
        dynamic_resistance.ne(dynamic_resistance.shift())
        .groupby(dynamic_resistance.index.to_period("D"))
        .cumsum()
        - 1
    )
    if return_levels:
        return dynamic_support, dynamic_resistance
    else:
        return support_nbreaks, resistance_nbreaks


def vwapbounce_signal(
    data,
    HTF1="4H",
    HTF2="W",
    crossing_count_reset="1H",
    ntouch=1,
    entry_zone="('6:30', '7:30')",
    sod_time="6:30",
    eod_time="12:50",
    daytrade=True,
    use_rsi=False,
    pivot_shift=1,
    price_move_tp=None,
    long_only=False,
    short_only=False,
    shift=True,
    restrict_entry_zone=False,
    filter_by_secondary_timeframe=False,
    consider_wicks=False,
    rolling_tf=False,
    support_rejection=True,
    resistance_rejection=False,
    ignore_vwap_crossabove=False,
    exit_on_level_rejection=False,
    vwap_diff_n=1,
    ignore_firstbar_vwap_diff = False,
):
    data = data.copy()
    data.columns = data.columns.str.lower()
    if "date" in data.columns:
        data.set_index("date", inplace=True)
    data["day"] = pd.DatetimeIndex(data.index).date
    data["isFirstBar"] = data["day"].diff() >= "1 days"

    rsi = pt.rsi(data.ta.hlc3(), 10)
    rsi_up = rsi.gt(rsi.shift())
    use_rsi_cond = {True: (rsi_up, ~rsi_up), False: (True, True)}

    avwap_htf1 = calc_vwap(data, HTF1)
    vwap_crossabove_htf1 = crossabove(
        data,
        avwap_htf1,
        crossing_count_reset,
        consider_wicks=consider_wicks,
        rolling_tf=rolling_tf,
    )

    avwap_htf2 = calc_vwap(data, HTF2)

    if price_move_tp is not None:
        price_position = price_position_by_pivots(data, pivot_data_shift=pivot_shift)
        pexp = price_position.groupby(price_position.index.to_period(HTF1)).expanding()
        price_move = pexp.apply(lambda x: x[-1]) - pexp.apply(lambda x: x[0])
        price_move = price_move.droplevel(0)

    _, weekly_levels = price_position_by_pivots(
        data, secondary_tf=HTF2, return_all_levels=True
    )

    R_nreject = pd.concat(
        [
            crossbelow(
                data,
                weekly_levels[label],
                crossing_count_reset,
                rolling_tf=rolling_tf,
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
                crossing_count_reset,
                rolling_tf=rolling_tf,
            )
            for label in weekly_levels.columns
        ],
        axis=1,
    ).sum(axis=1)

    entry_hr_left = int(eval(entry_zone)[0].split(":")[0])
    entry_min_left = int(eval(entry_zone)[0].split(":")[1])
    entry_hr_right = int(eval(entry_zone)[1].split(":")[0])
    entry_min_right = int(eval(entry_zone)[1].split(":")[1])

    if support_rejection:
        if ignore_vwap_crossabove:
            longCondition = (S_nreject.gt(0)) & use_rsi_cond[use_rsi][0]
        else:
            longCondition = (
                (
                    vwap_crossabove_htf1.eq(ntouch)
                    & vwap_crossabove_htf1.shift().eq(ntouch - 1)
                )
                | S_nreject.gt(0)
            ) & use_rsi_cond[use_rsi][0]
    else:
        longCondition = (
            vwap_crossabove_htf1.eq(ntouch)
            & vwap_crossabove_htf1.shift().eq(ntouch - 1)
        ) & use_rsi_cond[use_rsi][0]

    if vwap_diff_n:
        vwap_mom = (avwap_htf1 - avwap_htf2).diff(vwap_diff_n)
        if ignore_firstbar_vwap_diff:
            vwap_mom[data['isFirstBar']] = 0
        longCondition = longCondition & vwap_mom.gt(0)

    if filter_by_secondary_timeframe:
        longCondition = longCondition & avwap_htf1.gt(avwap_htf2)

    if restrict_entry_zone:
        longCondition = longCondition & pd.Series(
            np.logical_and(
                pd.DatetimeIndex(data.index).time
                < datetime.time(entry_hr_right, entry_min_right),
                pd.DatetimeIndex(data.index).time
                >= datetime.time(entry_hr_left, entry_min_left),
            ),
            index=data.index,
        )

    if resistance_rejection:
        shortCondition = (
            avwap_htf1.lt(avwap_htf2) & avwap_htf1.shift().ge(avwap_htf2.shift())
            | R_nreject.gt(0)
        ) & use_rsi_cond[use_rsi][1]
    else:
        shortCondition = (
            avwap_htf1.lt(avwap_htf2)
            & avwap_htf1.shift().ge(avwap_htf2.shift())
            & use_rsi_cond[use_rsi][1]
        )
    if filter_by_secondary_timeframe:
        shortCondition = shortCondition & avwap_htf1.lt(avwap_htf2)
    if restrict_entry_zone:
        shortCondition = shortCondition & pd.Series(
            np.logical_and(
                pd.DatetimeIndex(data.index).time
                < datetime.time(entry_hr_right, entry_min_right),
                pd.DatetimeIndex(data.index).time
                >= datetime.time(entry_hr_left, entry_min_left),
            ),
            index=data.index,
        )

    if daytrade:
        in_session = pd.Series(
            np.logical_and(
                pd.DatetimeIndex(data.index).time
                < datetime.time(
                    int(eod_time.split(":")[0]),
                    int(eod_time.split(":")[1]),
                ),
                pd.DatetimeIndex(data.index).time
                >= datetime.time(
                    int(sod_time.split(":")[0]),
                    int(sod_time.split(":")[1]),
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

    long = in_session & longCondition & (not short_only)
    short = in_session & shortCondition & (not long_only)

    if exit_on_level_rejection:
        longX = R_nreject.gt(0) & R_nreject.shift().eq(0)
        shortX = S_nreject.gt(0) & S_nreject.shift().eq(0)
    else:
        longX = (
            price_move.eq(price_move_tp) & price_move.shift().ne(price_move_tp)
            if price_move_tp is not None
            else pd.Series([False] * len(data), index=data.index)
        )

        shortX = (
            price_move.eq(-price_move_tp) & price_move.shift().ne(-price_move_tp)
            if price_move_tp is not None
            else pd.Series([False] * len(data), index=data.index)
        )
    signal = pd.DataFrame(
        columns=["Long", "LongX", "Short", "ShortX"], index=data.index
    )
    signal["isFirstBar"] = data["isFirstBar"]
    signal["Long"] = long
    signal["LongX"] = longX
    signal["Short"] = short
    signal["ShortX"] = shortX
    signal["RSI"] = rsi
    signal["EOD"] = (
        pd.DatetimeIndex(signal.index).time
        >= datetime.time(int(eod_time.split(":")[0]), int(eod_time.split(":")[1]))
    ) & (pd.DatetimeIndex(signal.index).time <= datetime.time(13, 0))
    if shift:
        return signal.shift()
    else:
        return signal


def EhlersRoofing(close, fast_length, slow_length):
    return EhlersSuperSmoother(EhlersHighpass(close, slow_length), fast_length)


def fit_trendline(data, trend_type="low"):
    data = data.copy()
    data["dateint"] = data.index.astype("int64") - data.index.astype("int64").min()
    trendline_data = data.copy()
    # recursively fit until only 3 points or fewer are above the trendline
    while len(trendline_data) > 3:
        fit = linregress(x=trendline_data["dateint"], y=trendline_data[trend_type])
        if trend_type == "high":
            trendline_data = trendline_data.loc[
                trendline_data[trend_type].gt(
                    fit[0] * trendline_data["dateint"] + fit[1]
                )
            ]
        else:
            trendline_data = trendline_data.loc[
                trendline_data[trend_type].lt(
                    fit[0] * trendline_data["dateint"] + fit[1]
                )
            ]

    fit = linregress(
        x=trendline_data["dateint"],
        y=trendline_data[trend_type],
    )
    return fit[0] * data["dateint"] + fit[1]


def buyingpressure(
    data,
    length=15,
    normalization_period=200,
    buyingpressure_definition="3",
    how_to_treat_inside_bar=np.nan,
    with_volume=1,
):
    data = data.copy()
    data.columns = data.columns.str.lower()
    close = data.close
    open = data.open
    low = data.low
    high = data.high
    volume = data.volume
    inside = (high.le(high.shift()) & low.ge(low.shift())).fillna(False)
    # inside = (high.le(high.shift()) & low.ge(low.shift())) | (high.le(high.shift(2)) & low.ge(low.shift(2))).fillna(False)
    first_bar = data.index.strftime("%H:%M") == "06:30"
    atr = data.ta.atr(5)
    intraday_move = close - open  # / ATR
    intraday_move[
        intraday_move.abs().gt(2 * atr)
    ] = np.nan  # filter out knee jerk reactions due to news
    intraday_move.ffill(inplace=True)
    interday_move = close - close.shift()
    interday_move[first_bar] = 0
    interday_move[
        interday_move.abs().gt(2 * atr)
    ] = np.nan  # filter out knee jerk reactions due to news
    interday_move.ffill(inplace=True)
    hWick = high - np.max([close, open], axis=0)
    hWick[intraday_move.abs().gt(2 * atr)] = np.nan
    hWick.ffill(inplace=True)
    lWick = np.min([close, open], axis=0) - low
    lWick[intraday_move.abs().gt(2 * atr)] = np.nan
    lWick.ffill(inplace=True)
    candle_body = np.abs(intraday_move)
    oc = pd.concat([close, open], axis=1)
    min_oc = oc.min(axis=1)
    max_oc = oc.max(axis=1)
    touch_low = pd.Series(0, index=close.index)
    touch_low[min_oc - low <= 0.05 * candle_body] = 1
    touch_low *= np.sign(intraday_move)
    touch_high = pd.Series(0, index=close.index)
    touch_high[high - max_oc <= 0.05 * candle_body] = 1
    touch_high *= np.sign(intraday_move)

    pressure1 = (
        intraday_move
        + (lWick - hWick)
        + interday_move
        + candle_body * (touch_high + touch_low)
    ) * [1, volume][with_volume]
    pressure1[inside] = how_to_treat_inside_bar
    pressure1.ffill(inplace=True)

    pressure2 = ((close - low) / (high - low) - (high - close) / (high - low)) * [
        1,
        volume,
    ][with_volume]
    pressure2[inside] = how_to_treat_inside_bar
    pressure2.ffill(inplace=True)

    pressure3 = (intraday_move + (lWick - hWick) + interday_move) * [1, volume][
        with_volume
    ]
    pressure3[inside] = how_to_treat_inside_bar
    pressure3.ffill(inplace=True)

    if buyingpressure_definition.lower() == "1":
        pressure = pressure1
    elif buyingpressure_definition.lower() == "2":
        pressure = pressure2
    elif buyingpressure_definition.lower() == "3":
        pressure = pressure3
    elif buyingpressure_definition.lower() == "combined":
        pressure = (pressure1 + pressure2 + pressure3) / 3
    elif buyingpressure_definition.lower() == "combined2":
        pressure = (
            pressure1 / pressure1.rolling(normalization_period).std()
            + pressure2 / pressure2.rolling(normalization_period).std()
            + pressure3 / pressure3.rolling(normalization_period).std()
        ) / 3

    # pressure = pressure.fillna(0)

    pressure = pressure.ffill()

    def rolling_rms(x):
        r = 1.0
        a = np.sqrt((len(x) * r**2) / (x**2).sum())
        x = x * a
        return x[-1]

    pressure = pressure.rolling(normalization_period).apply(rolling_rms)

    # pressure = pressure / pressure.rolling(normalization_period).std()
    if length > 1:
        pressure = EhlersSuperSmoother(pressure.fillna(0), length=length)

    return pressure


def rma(close, length):
    alpha = 1 / length
    filt = close.copy()
    n = filt.size
    for i in range(0, n):
        filt.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * filt.iloc[i - 1]
    return filt


def bandpass_filter(close, low_cutoff=None, high_cutoff=None, **kwargs):
    """Bandpass filter signal

    Args:
        close (pd.Series): pandas Series
        low_cutoff (float, optional): low frequency cutoff. Defaults to 0.01. Should be 0.5 > low_cutoff > 0
        high_cutoff (float, optional): high frequency cutoff. Defaults to 0.1. Should be 0.5 > high_cutoff

    Returns:
        pd.Series: Bandpass-filtered Series
    """
    # Configuration.
    fL = (
        low_cutoff if low_cutoff else 0.01
    )  # Cutoff frequency as a fraction of the sampling rate.
    fH = (
        high_cutoff if high_cutoff else 0.1
    )  # Cutoff frequency as a fraction of the sampling rate.
    NL = kwargs.get(
        "low_cutoff_rolloff_length", 29
    )  # Filter length for roll-off at fL, must be odd.
    NH = kwargs.get(
        "high_cutoff_rolloff_length", 25
    )  # Filter length for roll-off at fH, must be odd.
    beta = kwargs.get("beta", 3.395)  # Kaiser window beta.

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (np.arange(NH) - (NH - 1) / 2))
    hlpf *= np.kaiser(NH, beta)
    hlpf /= np.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (np.arange(NL) - (NL - 1) / 2))
    hhpf *= np.kaiser(NL, beta)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[(NL - 1) // 2] += 1

    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)

    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(close, h)

    return pd.Series(s[NL + NH - 2 :], index=close.index)


def get_ohlc4(open, high, low, close):
    return 1 / 4 * (open + high + low + close)


def get_atr(high, low, close, period, MAtype="Ehlers", convert_series=False):
    if type(high) == pd.Series:
        idx = high.index
    else:
        idx = pd.RangeIndex(1, len(high) + 1)
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    shifted_close = vbt.nb.fshift_1d_nb(close)
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - shifted_close)
    tr2 = np.abs(low - shifted_close)
    tr = np.column_stack((tr0, tr1, tr2)).max(axis=1)
    if MAtype == "ema":
        atr = vbt.nb.ewm_mean_1d_nb(tr, period)
    elif MAtype == "sma":
        atr = pt.sma(pd.Series(tr), period).values
    elif MAtype == "wma":
        atr = pt.wma(pd.Series(tr), period).values
    elif MAtype == "Ehlers":
        atr = EhlersSuperSmoother(vbt.nb.fillna_1d_nb(tr, 0), period)
    elif MAtype == "rma":
        atr = rma(pd.Series(tr).fillna(0), period).values
    atr = vbt.nb.ffill_1d_nb(atr)
    if convert_series:
        atr = pd.Series(atr, index=idx)
        atr.name = "ATR"
    return atr


class TALib:
    def __init__(self, **kwargs) -> None:
        self.data = kwargs.get("data", None)

    def didiIndex(
        self,
        data=None,
        ma_fast_length=10,
        ma_mid_length=20,
        ma_slow_length=30,
        ssf_length=15,
    ):
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        hlc3 = data.ta.hlc3().ffill()
        ma_fast = EhlersSuperSmoother(hlc3, ma_fast_length)
        ma_mid = EhlersSuperSmoother(hlc3, ma_mid_length)
        ma_slow = EhlersSuperSmoother(hlc3, ma_slow_length)

        fast = (ma_fast / ma_mid - 1) * 100
        slow = (ma_slow / ma_mid - 1) * 100
        didi = EhlersSuperSmoother((fast - slow).fillna(0), ssf_length)
        return didi

    def VFI(self, data=None, length=130, maxVolumeCutOff=2.5):
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        hlc3 = data.ta.hlc3().ffill()
        cutOff = (
            0.2 * (np.log(hlc3) - np.log(hlc3.shift(1))).rolling(30).std() * data.close
        )
        hlcChange = hlc3 - hlc3.shift(1)
        avgVolume = data.volume.rolling(length).mean().shift(1)
        minVolume = pd.concat([data.volume, avgVolume * maxVolumeCutOff], axis=1).apply(
            np.min, axis=1
        )
        dirVolume = pd.Series(0, index=hlcChange.index)
        dirVolume[hlcChange.gt(cutOff)] = minVolume[hlcChange.gt(cutOff)]
        dirVolume[hlcChange.lt(-cutOff)] = -minVolume[hlcChange.lt(-cutOff)]

        VFI = EhlersSuperSmoother(
            (dirVolume.rolling(length).sum() / avgVolume).fillna(0), 50
        )
        return VFI

    def buyingpressure(
        self,
        data=None,
        length=15,
        normalization_period=200,
        buyingpressure_definition="3",
        how_to_treat_inside_bar=np.nan,
        with_volume=1,
    ):
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        close = data.close
        open = data.open
        low = data.low
        high = data.high
        volume = data.volume
        inside = (high.le(high.shift()) & low.ge(low.shift())).fillna(False)
        # inside = (high.le(high.shift()) & low.ge(low.shift())) | (high.le(high.shift(2)) & low.ge(low.shift(2))).fillna(False)
        first_bar = data.index.strftime("%H:%M") == "06:30"
        atr = data.ta.atr(5)
        intraday_move = close - open  # / ATR
        intraday_move[
            intraday_move.abs().gt(2 * atr)
        ] = np.nan  # filter out knee jerk reactions due to news
        intraday_move.ffill(inplace=True)
        interday_move = close - close.shift()
        interday_move[first_bar] = 0
        interday_move[
            interday_move.abs().gt(2 * atr)
        ] = np.nan  # filter out knee jerk reactions due to news
        interday_move.ffill(inplace=True)
        hWick = high - np.max([close, open], axis=0)
        hWick[intraday_move.abs().gt(2 * atr)] = np.nan
        hWick.ffill(inplace=True)
        lWick = np.min([close, open], axis=0) - low
        lWick[intraday_move.abs().gt(2 * atr)] = np.nan
        lWick.ffill(inplace=True)
        candle_body = np.abs(intraday_move)
        oc = pd.concat([close, open], axis=1)
        min_oc = oc.min(axis=1)
        max_oc = oc.max(axis=1)
        touch_low = pd.Series(0, index=close.index)
        touch_low[min_oc - low <= 0.05 * candle_body] = 1
        touch_low *= np.sign(intraday_move)
        touch_high = pd.Series(0, index=close.index)
        touch_high[high - max_oc <= 0.05 * candle_body] = 1
        touch_high *= np.sign(intraday_move)

        pressure1 = (
            intraday_move
            + (lWick - hWick)
            + interday_move
            + candle_body * (touch_high + touch_low)
        ) * [1, volume][with_volume]
        pressure1[inside] = how_to_treat_inside_bar
        pressure1.ffill(inplace=True)

        pressure2 = ((close - low) / (high - low) - (high - close) / (high - low)) * [
            1,
            volume,
        ][with_volume]
        pressure2[inside] = how_to_treat_inside_bar
        pressure2.ffill(inplace=True)

        pressure3 = (intraday_move + (lWick - hWick) + interday_move) * [1, volume][
            with_volume
        ]
        pressure3[inside] = how_to_treat_inside_bar
        pressure3.ffill(inplace=True)

        if buyingpressure_definition.lower() == "1":
            pressure = pressure1
        elif buyingpressure_definition.lower() == "2":
            pressure = pressure2
        elif buyingpressure_definition.lower() == "3":
            pressure = pressure3
        elif buyingpressure_definition.lower() == "combined":
            pressure = (pressure1 + pressure2 + pressure3) / 3
        elif buyingpressure_definition.lower() == "combined2":
            pressure = (
                pressure1 / pressure1.rolling(normalization_period).std()
                + pressure2 / pressure2.rolling(normalization_period).std()
                + pressure3 / pressure3.rolling(normalization_period).std()
            ) / 3

        # pressure = pressure.fillna(0)

        pressure = pressure.ffill()

        def rolling_rms(x):
            r = 1.0
            a = np.sqrt((len(x) * r**2) / (x**2).sum())
            x = x * a
            return x[-1]

        pressure = pressure.rolling(normalization_period).apply(rolling_rms)

        # pressure = pressure / pressure.rolling(normalization_period).std()
        if length > 1:
            pressure = EhlersSuperSmoother(pressure.fillna(0), length=length)

        return pressure

    def supply_demand(
        self,
        data=None,
        resample=None,
        return_sorted_levels=False,
        significant_threshold=1.8,
        prefer_recent_zones=False,
        **bp_kwargs,
    ):
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        original_data = data.copy()
        levels = pd.DataFrame(columns=["Demand", "Supply"], index=original_data.index)
        data = data.copy()
        if resample is not None:
            logic = {
                "close": "last",
                "open": "first",
                "volume": "sum",
                "low": "min",
                "high": "max",
            }
            data = data.resample(resample).apply(logic).dropna()

        def rolling_rms(x):
            r = 1.0
            a = np.sqrt((len(x) * r**2) / (x**2).sum())
            x = x * a
            return x[-1]

        volume = original_data.volume.rolling(
            bp_kwargs.get("normalization_period", 200)
        ).apply(rolling_rms)

        bp = self.buyingpressure(data, **bp_kwargs)
        original_bp = self.buyingpressure(original_data, **bp_kwargs)

        demands = data.shift(2).low[
            bp.shift(2).gt(significant_threshold)
            & data.shift(1).high.gt(data.shift(2).high)
        ]

        demands_old1 = demands.shift(1)
        demands_old2 = demands.shift(2)
        demands_old3 = demands.shift(3)

        demands = demands.align(original_data.close)[0].ffill()
        demands_old1 = demands_old1.align(original_data.close)[0].ffill()
        demands_old2 = demands_old2.align(original_data.close)[0].ffill()
        demands_old3 = demands_old3.align(original_data.close)[0].ffill()

        sorted_demands = pd.concat(
            [demands, demands_old1, demands_old2, demands_old3], axis=1
        )
        sorted_demands.columns = ["D1", "D2", "D3", "D4"]

        sorted_demands = sorted_demands.align(original_data)[0].ffill()[
            ["D1", "D2", "D3", "D4"]
        ]

        cl = pd.concat([original_data.close.shift()] * 4, axis=1)
        cl.columns = ["D1", "D2", "D3", "D4"]

        def first_nonNan(x):
            try:
                return x.dropna()[0]
            except:
                return np.nan

        if prefer_recent_zones:
            levels["Demand"] = sorted_demands[sorted_demands.lt(cl)].apply(
                first_nonNan, axis=1
            )
            levels["DemandTurnedSupply"] = sorted_demands[sorted_demands.gt(cl)].apply(
                first_nonNan, axis=1
            )
        else:
            levels["Demand"] = sorted_demands[sorted_demands.lt(cl)].max(axis=1)
            levels["DemandTurnedSupply"] = sorted_demands[sorted_demands.gt(cl)].min(
                axis=1
            )

        supplies = data.shift(2).high[
            bp.shift(2).lt(-significant_threshold)
            & data.shift(1).low.lt(data.shift(2).low)
        ]

        supplies_old1 = supplies.shift(1)
        supplies_old2 = supplies.shift(2)
        supplies_old3 = supplies.shift(3)

        supplies = supplies.align(original_data.close)[0].ffill()
        supplies_old1 = supplies_old1.align(original_data.close)[0].ffill()
        supplies_old2 = supplies_old2.align(original_data.close)[0].ffill()
        supplies_old3 = supplies_old3.align(original_data.close)[0].ffill()
        sorted_supplies = pd.concat(
            [supplies, supplies_old1, supplies_old2, supplies_old3], axis=1
        )
        sorted_supplies.columns = ["S1", "S2", "S3", "S4"]

        sorted_supplies = sorted_supplies.align(original_data)[0].ffill()[
            ["S1", "S2", "S3", "S4"]
        ]
        cl.columns = ["S1", "S2", "S3", "S4"]
        if prefer_recent_zones:
            levels["Supply"] = sorted_supplies[sorted_supplies.gt(cl)].apply(
                first_nonNan, axis=1
            )
            levels["SupplyTurnedDemand"] = sorted_supplies[
                sorted_supplies.lt(cl)
            ].apply(first_nonNan, axis=1)
        else:
            levels["Supply"] = sorted_supplies[sorted_supplies.gt(cl)].min(axis=1)
            levels["SupplyTurnedDemand"] = sorted_supplies[sorted_supplies.lt(cl)].max(
                axis=1
            )

        price_pct = (original_data.close - levels["Demand"]) / (
            levels["Supply"] - levels["Demand"]
        )

        levels["Near Demand"] = price_pct.lt(0.05)
        levels["Near Supply"] = price_pct.gt(0.95)
        levels["Touched Demand"] = (
            original_data.low.le(levels["Demand"])
            & original_data.low.shift().gt(levels["Demand"])
            & original_data.close.gt(levels["Demand"])
            & ~(
                original_data.high.ge(levels["Supply"])
                & original_data.high.shift().lt(levels["Supply"])
                & original_data.close.lt(levels["Supply"])
            )
        )
        levels["Touched Supply"] = (
            original_data.high.ge(levels["Supply"])
            & original_data.high.shift().lt(levels["Supply"])
            & original_data.close.lt(levels["Supply"])
            # & original_data.open.lt(levels["Supply"])
            & ~(
                original_data.low.le(levels["Demand"])
                & original_data.low.shift().gt(levels["Demand"])
                & original_data.close.gt(levels["Demand"])
            )
        )

        levels["Demand Touch Pressure"] = original_bp[levels["Touched Demand"]]
        levels["Demand Touch Volume"] = volume[levels["Touched Demand"]]

        levels["Touched Demand and Bounced"] = (
            levels["Touched Demand"].shift()
            & original_bp.gt(original_bp.shift())
            & original_data.close.gt(levels["Demand"].shift())
        )

        levels["Touched Demand and Failed"] = levels["Demand Touch Pressure"].lt(-0.2)

        levels["Supply Touch Pressure"] = original_bp[levels["Touched Supply"]]
        levels["Supply Touch Volume"] = volume[levels["Touched Supply"]]

        levels["Touched Supply and Bounced"] = (
            levels["Touched Supply"].shift()
            & original_bp.lt(original_bp.shift())
            & original_data.close.lt(levels["Supply"].shift())
        )

        levels["Touched Supply and Failed"] = levels["Supply Touch Pressure"].gt(0.2)

        for _, sup_df in list(levels.groupby("Supply")):
            levels.loc[sup_df.index, "Supply Touch No."] = sup_df[
                "Touched Supply"
            ].cumsum()
            levels.loc[sup_df.index, "Touched Supply and Bounced No."] = sup_df[
                "Touched Supply and Bounced"
            ].cumsum()
            levels.loc[sup_df.index, "Cumulative Supply Touch Pressure"] = sup_df[
                "Supply Touch Pressure"
            ].cumsum() / np.arange(1, len(sup_df) + 1)

        for _, dem_df in list(levels.groupby("Demand")):
            levels.loc[dem_df.index, "Demand Touch No."] = dem_df[
                "Touched Demand"
            ].cumsum()
            levels.loc[dem_df.index, "Touched Demand and Bounced No."] = dem_df[
                "Touched Demand and Bounced"
            ].cumsum()
            levels.loc[dem_df.index, "Cumulative Demand Touch Pressure"] = dem_df[
                "Demand Touch Pressure"
            ].cumsum() / np.arange(1, len(dem_df) + 1)

        levels["close"] = original_data.close
        levels["close_shift"] = original_data.close.shift()
        levels["high"] = original_data.high
        levels["high_shift"] = original_data.high.shift()

        if return_sorted_levels:
            return levels, sorted_demands, sorted_supplies
        else:
            return levels
