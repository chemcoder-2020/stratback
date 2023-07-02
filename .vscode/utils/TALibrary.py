import pandas as pd
import pandas_ta as pt
import numpy as np
import vectorbt as vbt
import datetime
from scipy.stats import linregress

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
    long = trend_up & ~trend_up.shift().fillna(False)
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
    ma_length1=11,
    ma_length2=23,
    ma_length3=41,
    ma_length4=64,
    target_pct=0.7,
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
    ma3 = vwap(price, volume, ma_length3)
    # ma4 = dsma(price, ma_length4)
    ma4 = pt.sma(price, ma_length4)
    ma_mid = (ma1 + ma2) / 2
    longTarget = (target_pct / 100 + 1) * ma_mid
    shortTarget = ma_mid / (target_pct / 100 + 1)
    longExit = high > longTarget
    shortExit = low < shortTarget
    trend_up = ma1.gt(ma2) & ma3.gt(ma4)
    rsi = pt.rsi(data.close, 10)
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
    signal["isFirstBar"] = data["isFirstBar"]
    signal["TrendUp"] = trend_up
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
