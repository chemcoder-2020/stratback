from tda.client.synchronous import Client
import datetime, httpx
import pandas as pd
import numpy as np
import pandas_ta as pt
import plotly.express as px
import os
import dotenv


def EhlersSuperSmoother(close, length):
    a1 = np.exp(-np.sqrt(2) * np.pi / length)
    coeff2 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / length)
    coeff3 = -(a1**2)
    coeff1 = 1 - coeff2 - coeff3
    filt = close.copy()
    n = filt.size
    for i in range(0, n):
        filt.iloc[i] = (
            coeff1 * (close.iloc[i] + close.iloc[i - 1]) / 2
            + coeff2 * filt.iloc[i - 1]
            + coeff3 * filt.iloc[i - 2]
        )
    return filt


def buyingpressure(
    data,
    length=15,
    normalization_period=200,
    buyingpressure_definition="3",
    how_to_treat_inside_bar=np.nan,
    with_volume=1,
):
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


def didiIndex(
    data, ma_fast_length=10, ma_mid_length=20, ma_slow_length=30, ssf_length=15
):
    hlc3 = data.ta.hlc3().ffill()
    ma_fast = EhlersSuperSmoother(hlc3, ma_fast_length)
    ma_mid = EhlersSuperSmoother(hlc3, ma_mid_length)
    ma_slow = EhlersSuperSmoother(hlc3, ma_slow_length)
    fast = (ma_fast / ma_mid - 1) * 100
    slow = (ma_slow / ma_mid - 1) * 100
    didi = EhlersSuperSmoother((fast - slow).fillna(0), ssf_length)
    return didi


def volumeFlowIndex(data, length=130, maxVolumeCutOff=2.5):
    hlc3 = data.ta.hlc3().ffill()
    cutOff = 0.2 * (np.log(hlc3) - np.log(hlc3.shift(1))).rolling(30).std() * data.close
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


def next_friday(dt: datetime.datetime, num_friday=0):
    date = dt.date()
    weekday = dt.weekday()
    delta = 4 - weekday + 7 * num_friday
    return datetime.timedelta(days=delta) + date


def max_pain(df):
    # Calculate the total open interest for each strike price
    df["total_interest"] = (
        df["openInterest"].groupby(df["strikePrice"]).transform("sum")
    )

    puts = df[df.putCall == "PUT"].sort_values(by="strikePrice", ignore_index=True)
    calls = df[df.putCall == "CALL"].sort_values(by="strikePrice", ignore_index=True)
    pain = pd.DataFrame(
        columns=["Put Pain", "Call Pain"], index=puts.strikePrice.unique()
    )
    for hypothetical_close in puts.strikePrice.unique():

        affected_put_sellers = puts[puts.strikePrice.gt(hypothetical_close)]
        pain.loc[hypothetical_close, "Put Pain"] = (
            affected_put_sellers["total_interest"]
            * (puts.strikePrice - hypothetical_close)
        ).sum()

        affected_call_sellers = calls[calls.strikePrice.lt(hypothetical_close)]
        pain.loc[hypothetical_close, "Call Pain"] = (
            affected_call_sellers["total_interest"]
            * (hypothetical_close - calls.strikePrice)
        ).sum()

    pain["Total Pain"] = pain["Put Pain"] + pain["Call Pain"]

    pain = pain.astype(float)
    # print(pain)
    maxpain = pain["Total Pain"].idxmin()
    return maxpain


class client(Client):
    def __init__(self, api_key, session, *, enforce_enums=True, token_metadata=None):
        super().__init__(
            api_key, session, enforce_enums=enforce_enums, token_metadata=token_metadata
        )
        self.tz = datetime.datetime.now().astimezone().tzinfo
        import yfinance as yf

        self.yf = yf

    def get_five_minute_data_long(self, symbol, ext=False):
        try:
            resp = self.get_price_history_every_five_minutes(
                symbol, need_extended_hours_data=ext
            )
            assert resp.status_code == httpx.codes.OK
            history = resp.json()
            data = pd.json_normalize(history, record_path=["candles"])
            # data["date"] = (
            #     pd.DatetimeIndex(
            #         [datetime.datetime.fromtimestamp(f / 1e3) for f in data["datetime"]]
            #     )
            #     .tz_localize(self.tz)
            #     .tz_convert("US/Pacific")
            # )
            data["date"] = pd.to_datetime(data["datetime"], utc=True, unit='ms').dt.tz_convert("US/Pacific")
        except Exception as e:
            print(e)
            print("Fall back to yahoo data")
            data = self.yf.Ticker(symbol).history("60d", "5m").tz_convert("US/Pacific")
            data = data.reset_index()
            data["date"] = data["Datetime"]
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()

        return data.drop_duplicates(subset="date")

    def get_five_minute_data(self, symbol, ext=False):
        try:
            resp = self.get_price_history(
                symbol,
                end_datetime=datetime.datetime.now(),
                period_type=Client.PriceHistory.PeriodType.DAY,
                period=Client.PriceHistory.Period.TEN_DAYS,
                frequency_type=Client.PriceHistory.FrequencyType.MINUTE,
                frequency=Client.PriceHistory.Frequency.EVERY_FIVE_MINUTES,
                need_extended_hours_data=ext,
            )
            assert resp.status_code == httpx.codes.OK
            history = resp.json()
            data = pd.json_normalize(history, record_path=["candles"])
            # data["date"] = (
            #     pd.DatetimeIndex(
            #         [datetime.datetime.fromtimestamp(f / 1e3) for f in data["datetime"]]
            #     )
            #     .tz_localize(self.tz)
            #     .tz_convert("US/Pacific")
            # )
            data["date"] = pd.to_datetime(data["datetime"], utc=True, unit='ms').dt.tz_convert("US/Pacific")
        except Exception as e:
            print(e)
            print("Fall back to yahoo data")
            data = self.yf.Ticker(symbol).history("1d", "5m").tz_convert("US/Pacific")
            data = data.reset_index()
            data["date"] = data["Datetime"]
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
        return data.drop_duplicates(subset="date")

    def get_ten_minute_data(self, symbol, ext=False, short=False):
        if short:
            try:
                resp = self.get_price_history(
                    symbol,
                    end_datetime=datetime.datetime.now(),
                    period_type=Client.PriceHistory.PeriodType.DAY,
                    period=Client.PriceHistory.Period.TEN_DAYS,
                    frequency_type=Client.PriceHistory.FrequencyType.MINUTE,
                    frequency=Client.PriceHistory.Frequency.EVERY_TEN_MINUTES,
                    need_extended_hours_data=ext,
                )
                assert resp.status_code == httpx.codes.OK
                history = resp.json()
                data = pd.json_normalize(history, record_path=["candles"])
                # data["date"] = (
                #     pd.DatetimeIndex(
                #         [
                #             datetime.datetime.fromtimestamp(f / 1e3)
                #             for f in data["datetime"]
                #         ]
                #     )
                #     .tz_localize(self.tz)
                #     .tz_convert("US/Pacific")
                # )
                data["date"] = pd.to_datetime(data["datetime"], utc=True, unit='ms').dt.tz_convert("US/Pacific")
            except Exception as e:
                print(e)
                print("Fall back to yahoo data")
                data = (
                    self.yf.Ticker(symbol).history("1d", "5m").tz_convert("US/Pacific")
                )
                data = (
                    data.resample("10m")
                    .agg(
                        {
                            "Open": "first",
                            "Close": "last",
                            "Low": "min",
                            "High": "max",
                            "Volume": "sum",
                        }
                    )
                    .dropna()
                )
                data = data.reset_index()
                data["date"] = data["Datetime"]
                if "Dividends" in data.columns:
                    data.drop(columns=["Dividends"], inplace=True)
                if "Stock Splits" in data.columns:
                    data.drop(columns=["Stock Splits"], inplace=True)
                if "Datetime" in data.columns:
                    data.drop(columns=["Datetime"], inplace=True)
                data.columns = data.columns.str.lower()
            return data.drop_duplicates(subset="date")
        else:
            try:
                data = self.get_five_minute_data_long(symbol, ext=ext)
                data.set_index("date", inplace=True)
                logic = {
                    "close": "last",
                    "open": "first",
                    "volume": "sum",
                    "low": "min",
                    "high": "max",
                }
                return data.resample("10min").agg(logic).dropna().reset_index()
            except Exception as e:
                print(e)
                print("Fall back to yahoo data")
                data = (
                    self.yf.Ticker(symbol).history("60d", "5m").tz_convert("US/Pacific")
                )
                data = (
                    data.resample("10m")
                    .agg(
                        {
                            "Open": "first",
                            "Close": "last",
                            "Low": "min",
                            "High": "max",
                            "Volume": "sum",
                        }
                    )
                    .dropna()
                )
                data = data.reset_index()
                data["date"] = data["Datetime"]
                if "Dividends" in data.columns:
                    data.drop(columns=["Dividends"], inplace=True)
                if "Stock Splits" in data.columns:
                    data.drop(columns=["Stock Splits"], inplace=True)
                if "Datetime" in data.columns:
                    data.drop(columns=["Datetime"], inplace=True)
                data.columns = data.columns.str.lower()
                return data.drop_duplicates(subset="date")

    def get_fifteen_minute_data(self, symbol, ext=False, short=False):
        try:
            if short:
                resp = self.get_price_history(
                    symbol,
                    end_datetime=datetime.datetime.now(),
                    period_type=Client.PriceHistory.PeriodType.DAY,
                    period=Client.PriceHistory.Period.TEN_DAYS,
                    frequency_type=Client.PriceHistory.FrequencyType.MINUTE,
                    frequency=Client.PriceHistory.Frequency.EVERY_FIFTEEN_MINUTES,
                    need_extended_hours_data=ext,
                )
                assert resp.status_code == httpx.codes.OK
                history = resp.json()
                data = pd.json_normalize(history, record_path=["candles"])
                # data["date"] = (
                #     pd.DatetimeIndex(
                #         [
                #             datetime.datetime.fromtimestamp(f / 1e3)
                #             for f in data["datetime"]
                #         ]
                #     )
                #     .tz_localize(self.tz)
                #     .tz_convert("US/Pacific")
                # )
                data["date"] = pd.to_datetime(data["datetime"], utc=True, unit='ms').dt.tz_convert("US/Pacific")
                return data.drop_duplicates(subset="date")
            else:
                data = self.get_five_minute_data_long(symbol, ext=ext)
                data.set_index("date", inplace=True)
                logic = {
                    "close": "last",
                    "open": "first",
                    "volume": "sum",
                    "low": "min",
                    "high": "max",
                }
                return data.resample("15min").agg(logic).dropna().reset_index()
        except Exception as e:
            print(e)
            print("Fall back to yahoo data")
            if short:
                period = "1d"
            else:
                period = "60d"
            data = (
                self.yf.Ticker(symbol).history(period, "15m").tz_convert("US/Pacific")
            )
            data = data.reset_index()
            data["date"] = data["Datetime"]
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
            return data.drop_duplicates(subset="date")

    def get_thirty_minute_data(self, symbol, ext=False, short=False):
        try:
            data = self.get_five_minute_data_long(symbol, ext=ext)
            data.set_index("date", inplace=True)
            logic = {
                "close": "last",
                "open": "first",
                "volume": "sum",
                "low": "min",
                "high": "max",
            }
            return data.resample("30min").agg(logic).dropna().reset_index()
        except Exception as e:
            print(e)
            print("Fall back to yahoo data")
            data = self.yf.Ticker(symbol).history("2d", "30m").tz_convert("US/Pacific")
            data = data.reset_index()
            data["date"] = data["Datetime"]
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
            return data.drop_duplicates(subset="date")

    def get_hourly_data(self, symbol, ext=False, short=False):
        try:
            data = self.get_five_minute_data_long(symbol, ext=ext)
            data.set_index("date", inplace=True)
            logic = {
                "close": "last",
                "open": "first",
                "volume": "sum",
                "low": "min",
                "high": "max",
            }
            return (
                data.resample("1H", offset="00:30:00").agg(logic).dropna().reset_index()
            )

        except Exception as e:
            print(e)
            print("Fall back to yahoo data")
            data = self.yf.Ticker(symbol).history("5d", "60m").tz_convert("US/Pacific")
            data = data.reset_index()
            data["date"] = data["Datetime"]
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
            return data.drop_duplicates(subset="date")

    def get_2H_data(self, symbol, ext=False, short=False):
        data = self.get_five_minute_data_long(symbol, ext=ext)
        data.set_index("date", inplace=True)
        logic = {
            "close": "last",
            "open": "first",
            "volume": "sum",
            "low": "min",
            "high": "max",
        }
        return data.resample("2H", offset="-1H").agg(logic).dropna().reset_index()

    def get_4H_data(self, symbol, ext=False, short=False):
        data = self.get_five_minute_data_long(symbol, ext=ext)
        data.set_index("date", inplace=True)
        logic = {
            "close": "last",
            "open": "first",
            "volume": "sum",
            "low": "min",
            "high": "max",
        }
        return data.resample("4H", offset="-1H").agg(logic).dropna().reset_index()

    def get_daily_data(self, symbol, ext=False, short=False, years=1):
        # data = self.get_five_minute_data_long(symbol, ext=ext)
        # data.set_index("date", inplace=True)
        # logic = {
        #     "close": "last",
        #     "open": "first",
        #     "volume": "sum",
        #     "low": "min",
        #     "high": "max",
        # }
        # return data.resample("B").agg(logic).dropna().reset_index()
        try:
            start_dt = datetime.datetime.now() - datetime.timedelta(weeks=years * 52)
            resp = self.get_price_history_every_day(symbol, start_datetime=start_dt)
            data = resp.json()
            data = pd.json_normalize(data, record_path=["candles"])
            data["date"] = pd.DatetimeIndex(
                (
                    pd.DatetimeIndex(
                        [
                            datetime.datetime.fromtimestamp(f / 1e3)
                            for f in data["datetime"]
                        ]
                    )
                    .tz_localize("US/Central")
                    .tz_convert("US/Pacific")
                    + datetime.timedelta(days=1)
                ).strftime("%Y-%m-%d")
            )
        except Exception as e:
            print(e)
            print("Fall back to Yahoo data")
            data = self.yf.Ticker(symbol).history(f"5y", "1d").tz_convert("US/Pacific")
            data = data.reset_index()
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
        return data.drop_duplicates(subset="date")

    def get_daily_data_long(self, symbol, ext=False, short=False):
        try:
            resp = self.get_price_history_every_day(symbol)
            assert resp.status_code == httpx.codes.OK
            history = resp.json()
            data = pd.json_normalize(history, record_path=["candles"])
            data["date"] = (
                pd.DatetimeIndex(
                    [
                        datetime.datetime.fromtimestamp(
                            f / 1e3, datetime.timezone(datetime.timedelta(0))
                        )
                        for f in data["datetime"]
                    ]
                )
                # .tz_localize(self.tz)
                # .tz_convert("US/Pacific")
            ).round("D")
        except Exception as e:
            print(e)
            print("Fall back to Yahoo data")
            data = self.yf.Ticker(symbol).history("max", "1d").tz_convert("US/Pacific")
            data = data.reset_index()
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
        return data.drop_duplicates(subset="date")

    def get_weekly_data(self, symbol, ext=False, short=False, years=3):
        try:
            start_dt = datetime.datetime.now() - datetime.timedelta(weeks=years * 52)
            resp = self.get_price_history_every_week(symbol, start_datetime=start_dt)
            data = resp.json()
            data = pd.json_normalize(data, record_path=["candles"])
            data["date"] = pd.DatetimeIndex(
                (
                    pd.DatetimeIndex(
                        [
                            datetime.datetime.fromtimestamp(f / 1e3)
                            for f in data["datetime"]
                        ]
                    )
                    .tz_localize("US/Central")
                    .tz_convert("US/Pacific")
                    + datetime.timedelta(days=1)
                ).strftime("%Y-%m-%d")
            )
        except Exception as e:
            print(e)
            print("Fall back to Yahoo data")
            data = (
                self.yf.Ticker(symbol)
                .history(f"{years}y", "1wk")
                .tz_convert("US/Pacific")
            )
            data = data.reset_index()
            if "Dividends" in data.columns:
                data.drop(columns=["Dividends"], inplace=True)
            if "Stock Splits" in data.columns:
                data.drop(columns=["Stock Splits"], inplace=True)
            if "Datetime" in data.columns:
                data.drop(columns=["Datetime"], inplace=True)
            data.columns = data.columns.str.lower()
        return data.drop_duplicates(subset="date")

    def get_monthly_data(self, symbol, ext=False, short=False):
        data = self.get_daily_data_long(symbol, ext, short)
        data.set_index("date", inplace=True)
        data = self.resample_data(data, "1M")
        return data.reset_index()

    def resample_data(self, data, freq, **kwargs):

        logic = {
            "close": "last",
            "open": "first",
            "volume": "sum",
            "low": "min",
            "high": "max",
        }
        if freq in ["1H"]:
            offset = "00:30:00"
        else:
            offset = None
        if type(data.index) == pd.Int64Index:
            data["date"] = pd.DatetimeIndex(data["date"])
            data.set_index("date", inplace=True)

        # print(data)
        return data.resample(freq, offset=offset, **kwargs).agg(logic).dropna()

    def yf_get_options(self, symbol, strike_count=18, num_friday=2):
        ticker = self.yf.Ticker(symbol)
        friday_exps = pd.Series(
            [next_friday(exp) for exp in pd.DatetimeIndex(ticker.options)]
        )
        friday_idx = friday_exps[friday_exps.ne(friday_exps.shift())].index
        options = {
            e: ticker.option_chain(date=e)
            for e in ticker.options[: friday_idx[num_friday]]
        }
        num_strike_lower = np.int(strike_count / 2)
        num_strike_upper = np.int(strike_count / 2) + strike_count % 2
        for exp, option in options.items():
            options[exp].puts["expirationDate"] = exp
            options[exp].puts["putCall"] = "PUT"
            otm_puts = (
                options[exp]
                .puts.groupby("inTheMoney")
                .get_group(False)
                .sort_values("strike", ascending=False)[:num_strike_lower]
            )
            itm_puts = (
                options[exp]
                .puts.groupby("inTheMoney")
                .get_group(True)
                .sort_values("strike", ascending=True)[:num_strike_upper]
            )

            options[exp].calls["expirationDate"] = exp
            options[exp].calls["putCall"] = "CALL"
            otm_calls = (
                options[exp]
                .calls.groupby("inTheMoney")
                .get_group(False)
                .sort_values("strike", ascending=True)[:num_strike_upper]
            )
            itm_calls = (
                options[exp]
                .calls.groupby("inTheMoney")
                .get_group(True)
                .sort_values("strike", ascending=False)[:num_strike_lower]
            )
            options[exp] = pd.concat(
                [otm_calls, itm_calls, otm_puts, itm_puts], axis=0, ignore_index=True
            )
        options = pd.concat(list(options.values()))
        options.rename(
            columns={
                "volume": "totalVolume",
                "strike": "strikePrice",
                "volatility": "impliedVolatility",
            },
            inplace=True,
        )
        options["mark"] = (
            options["bid"].astype(float) + options["ask"].astype(float)
        ) / 2
        now = datetime.datetime.now()
        options["daysToExpiration"] = (
            pd.DatetimeIndex(options["expirationDate"]) - now
        ).days
        return options

    def get_options(self, symbol, side=None, strike_count=18, num_friday=2):
        now = datetime.datetime.now()
        option_chain = self.get_option_chain(
            symbol,
            strike_count=strike_count,
            to_date=next_friday(now, num_friday=num_friday),
        )
        option_chain = option_chain.json()

        df = []
        for k1, v1 in option_chain["putExpDateMap"].items():
            for k2, v2 in v1.items():
                df.append(
                    pd.json_normalize(
                        option_chain, record_path=["putExpDateMap", k1, k2]
                    )
                )
        puts = pd.concat(df)
        puts["expirationDate"] = pd.DatetimeIndex(
            [
                datetime.datetime.fromtimestamp(f / 1e3).date()
                for f in puts["expirationDate"]
            ]
        )
        puts_subset = puts[
            [
                "putCall",
                "mark",
                "totalVolume",
                "volatility",
                "openInterest",
                "timeValue",
                "strikePrice",
                "expirationDate",
                "intrinsicValue",
                "daysToExpiration",
                "percentChange",
                "markPercentChange",
                "inTheMoney",
                "delta",
            ]
        ]
        puts_subset.set_index(["expirationDate", "strikePrice"], inplace=True)

        df = []
        for k1, v1 in option_chain["callExpDateMap"].items():
            for k2, v2 in v1.items():
                df.append(
                    pd.json_normalize(
                        option_chain, record_path=["callExpDateMap", k1, k2]
                    )
                )
        calls = pd.concat(df)
        calls["expirationDate"] = pd.DatetimeIndex(
            [
                datetime.datetime.fromtimestamp(f / 1e3).date()
                for f in calls["expirationDate"]
            ]
        )

        calls_subset = calls[
            [
                "putCall",
                "mark",
                "totalVolume",
                "volatility",
                "openInterest",
                "timeValue",
                "strikePrice",
                "expirationDate",
                "intrinsicValue",
                "daysToExpiration",
                "percentChange",
                "markPercentChange",
                "inTheMoney",
                "delta",
            ]
        ]
        calls_subset.set_index(["expirationDate", "strikePrice"], inplace=True)

        calls_subset["PutCallVolumeTotal"] = (
            calls_subset["totalVolume"] + puts_subset["totalVolume"]
        )
        puts_subset["PutCallVolumeTotal"] = (
            calls_subset["totalVolume"] + puts_subset["totalVolume"]
        )
        calls_subset["PutCallVolumeDiff"] = (
            calls_subset["totalVolume"] + puts_subset["totalVolume"]
        )
        puts_subset["PutCallVolumeDiff"] = (
            calls_subset["totalVolume"] - puts_subset["totalVolume"]
        )
        calls_subset["PutCallOIDiff"] = (
            calls_subset["openInterest"] - puts_subset["openInterest"]
        )
        puts_subset["PutCallOIDiff"] = (
            calls_subset["openInterest"] - puts_subset["openInterest"]
        )
        ops = pd.concat([puts_subset, calls_subset])
        ops["ATM"] = ops["inTheMoney"].groupby(level="expirationDate").diff().abs() == 1
        atm_straddle = (
            ops[ops.ATM.fillna(False)].groupby(level="expirationDate")["mark"].sum()
        )
        first_strangle = (
            ops[ops.ATM.shift().fillna(False)]
            .groupby(level="expirationDate")["mark"]
            .last()
            + ops[ops.ATM.shift(-1).fillna(False)]
            .groupby(level="expirationDate")["mark"]
            .first()
        )
        second_strangle = (
            ops[ops.ATM.shift(2).fillna(False)]
            .groupby(level="expirationDate")["mark"]
            .last()
            + ops[ops.ATM.shift(-2).fillna(False)]
            .groupby(level="expirationDate")["mark"]
            .first()
        )
        expected_move = (
            0.6 * atm_straddle + 0.3 * first_strangle + 0.1 * second_strangle
        )
        ops["expectedMove"] = ops.align(expected_move, axis=0)[1]

        if side == "puts":
            return puts_subset
        elif side == "calls":
            return calls_subset
        else:
            return ops

    def option_plots(
        self,
        symbol,
        side=None,
        strike_count=18,
        num_friday=2,
        return_option_OI=False,
        source="TD",
        query=None,
    ):
        if source == "TD":
            try:
                df = self.get_options(
                    symbol, side=side, strike_count=strike_count, num_friday=num_friday
                )
                df = df.reset_index().drop("index", axis=1, errors="ignore")
            except Exception as e:
                print(e)
                print("Fallback to Yahoo option source")
                df = self.yf_get_options(
                    symbol, strike_count=strike_count, num_friday=num_friday
                )
        else:
            df = self.yf_get_options(
                symbol, strike_count=strike_count, num_friday=num_friday
            )

        if query:
            df = df.query(query)

        df.columns = [
            str(c) for c in df.columns
        ]  # update columns to strings in case they are numbers

        # data = self.get_daily_data(symbol)
        # df["DiffToStrike"] = (df["strikePrice"] - data.iloc[-1].close) * df[
        #     "putCall"
        # ].eq("PUT").astype(int).replace(0, -1)
        # df["Premium"] = df["DiffToStrike"] * df["openInterest"]  # .abs()

        # maxpain = df.groupby(["strikePrice"])["Premium"].sum().idxmax()
        # print(df.groupby(["strikePrice"])["Premium"].sum())
        # print(maxpain)

        if not query:
            expdate = df["expirationDate"].iloc[0]
            maxpain_df = df.query(f"expirationDate == '{expdate.strftime('%Y-%m-%d')}'")
        else:
            maxpain_df = df.copy()
        # maxpain_df["premiums"] = np.sum(
        #     [
        #         maxpain_df["openInterest"]
        #         * (
        #             (maxpain_df["strikePrice"] - strike)
        #             * maxpain_df["putCall"].eq("CALL").astype(int).replace(0, -1)
        #         ).apply(lambda x: max(0, x))
        #         for strike in np.sort(np.unique(maxpain_df.strikePrice))
        #     ],
        #     axis=0,
        # )
        # maxpain = (
        #     maxpain_df.groupby(["expirationDate", "strikePrice"])["premiums"]
        #     .sum()
        #     .idxmin()[1]
        # )
        maxpain = max_pain(maxpain_df.reset_index())

        chart_data = pd.concat(
            [
                df["strikePrice"],
                df["openInterest"],
                df["putCall"],
                df["expirationDate"],
                df["expectedMove"],
            ],
            axis=1,
        )
        expected_move = chart_data["expectedMove"].min()
        chart_data = chart_data.query(
            """(`putCall` == 'CALL') or (`putCall` == 'PUT')"""
        )
        chart_data = chart_data.sort_values(
            ["expirationDate", "putCall", "strikePrice"]
        )
        chart_data = chart_data.rename(columns={"strikePrice": "x"})
        chart_data_sum = chart_data.groupby(
            ["expirationDate", "putCall", "x"], dropna=True
        )[["openInterest"]].sum()
        chart_data_sum.columns = ["openInterest|sum"]
        chart_data = chart_data_sum.reset_index()
        chart_data["expirationDate"] = chart_data["expirationDate"].astype(str)

        fig = px.bar(
            chart_data,
            x="x",
            y="openInterest|sum",
            color="putCall",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.D3,
            title=f"{symbol} - Max Pain = {maxpain} - EM = ±{expected_move:.1f}",
        )  # , animation_frame="expirationDate")

        OI_total = (
            df.groupby("putCall")[["openInterest", "percentChange"]]
            .agg({"openInterest": "sum", "percentChange": "mean"})
            .reset_index()
        )
        fig2 = px.bar(
            OI_total,
            x="putCall",
            y="openInterest",
            color="percentChange",
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.write_image("option_fig1.png", scale=3)
        fig2.write_image("option_fig2.png", scale=3)
        if return_option_OI:
            return fig, fig2, chart_data
        else:
            return fig, fig2


def gpt_support_resistance_by_vwap(data):

    # Compute the typical price
    typical_price = (data["high"] + data["low"] + data["close"]) / 3

    # Compute the volume-weighted typical price
    vwap = (data["volume"] * typical_price).cumsum() / data["volume"].cumsum()

    # Compute the price deviation from the VWAP
    price_deviation = typical_price - vwap

    # Define the minimum volume required to consider a support or resistance level
    min_volume = 0.5 * data["volume"].mean()

    # Find the local minima of the price deviation where the volume is above the minimum threshold
    support_levels = data.loc[
        (price_deviation.shift(1) > 0)
        & (price_deviation < 0)
        & (data["volume"] > min_volume)
    ]

    # Find the local maxima of the price deviation where the volume is above the minimum threshold
    resistance_levels = data.loc[
        (price_deviation.shift(1) < 0)
        & (price_deviation > 0)
        & (data["volume"] > min_volume)
    ]

    supports = []
    resistances = []
    for row, sup in support_levels.iterrows():
        # print(sup)
        supports.append({"Time Stamp": row, "Level": sup.low})
    for row, res in resistance_levels.iterrows():
        resistances.append({"Time Stamp": row, "Level": res.high})
    supports = sorted(supports, key=lambda d: d["Time Stamp"])
    resistances = sorted(resistances, key=lambda d: d["Time Stamp"])
    return supports, resistances


def gpt_support_resistance_by_volume_spread(data):

    # Identify swing highs and lows
    highs = data.high
    lows = data.low
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]

    # Calculate the volume spread
    volumes = data.volume
    spread = volumes * (2 * data.close - data.high - data.low) / (data.high - data.low)

    # Identify major support and resistance levels
    support_levels = []
    resistance_levels = []
    for level in swing_highs:
        if (spread[data.high == level] > spread.quantile(0.7)).any():
            resistance_levels.append(
                {"Time Stamp": data[data.high == level].index[0], "Level": level}
            )
    for level in swing_lows:
        if (spread[data.low == level] > spread.quantile(0.7)).any():
            support_levels.append(
                {"Time Stamp": data[data.low == level].index[0], "Level": level}
            )

    support_levels = sorted(support_levels, key=lambda d: d["Time Stamp"])
    resistance_levels = sorted(resistance_levels, key=lambda d: d["Time Stamp"])

    return support_levels, resistance_levels


def gpt_support_resistance_by_price_spread(data):

    # Identify swing highs and lows
    highs = data.high
    lows = data.low
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]

    # Calculate the price spread
    spread = (2 * data.close - data.high - data.low) / (data.high - data.low)

    # Identify major support and resistance levels
    support_levels = []
    resistance_levels = []
    for level in swing_highs:
        if (spread[data.high == level] > spread.quantile(0.7)).any():
            resistance_levels.append(
                {"Time Stamp": data[data.high == level].index[0], "Level": level}
            )
    for level in swing_lows:
        if (spread[data.low == level] > spread.quantile(0.7)).any():
            support_levels.append(
                {"Time Stamp": data[data.low == level].index[0], "Level": level}
            )

    support_levels = sorted(support_levels, key=lambda d: d["Time Stamp"])
    resistance_levels = sorted(resistance_levels, key=lambda d: d["Time Stamp"])

    return support_levels, resistance_levels


def gpt_support_resistance_by_pivots(data):
    data = data.copy()
    # Calculate pivot points, support and resistance levels
    data["Pivot"] = (data["high"] + data["low"] + data["close"]) / 3
    data["S1"] = (2 * data["Pivot"]) - data["high"]
    data["S2"] = data["Pivot"] - (data["high"] - data["low"])
    data["R1"] = (2 * data["Pivot"]) - data["low"]
    data["R2"] = data["Pivot"] + (data["high"] - data["low"])

    # Identify major support and resistance levels
    major_support_levels = []
    major_resistance_levels = []

    for i in range(1, len(data)):
        if (
            data["low"].iloc[i] > data["S1"].iloc[i - 1]
            and data["low"].iloc[i - 1] <= data["S1"].iloc[i - 1]
        ):
            major_support_levels.append(
                {"Time Stamp": data.index[i], "Level": data["S1"].iloc[i - 1]}
            )
        elif (
            data["low"].iloc[i] > data["S2"].iloc[i - 1]
            and data["low"].iloc[i - 1] <= data["S2"].iloc[i - 1]
        ):
            major_support_levels.append(
                {"Time Stamp": data.index[i], "Level": data["S2"].iloc[i - 1]}
            )
        elif (
            data["high"].iloc[i] < data["R1"].iloc[i - 1]
            and data["high"].iloc[i - 1] >= data["R1"].iloc[i - 1]
        ):
            major_resistance_levels.append(
                {"Time Stamp": data.index[i], "Level": data["R1"].iloc[i - 1]}
            )
        elif (
            data["high"].iloc[i] < data["R2"].iloc[i - 1]
            and data["high"].iloc[i - 1] >= data["R2"].iloc[i - 1]
        ):
            major_resistance_levels.append(
                {"Time Stamp": data.index[i], "Level": data["R2"].iloc[i - 1]}
            )
    major_support_levels = sorted(major_support_levels, key=lambda d: d["Time Stamp"])
    major_resistance_levels = sorted(
        major_resistance_levels, key=lambda d: d["Time Stamp"]
    )
    return major_support_levels, major_resistance_levels


def gpt_supply_demand_zones_by_volume_profile(df, price_step=1):

    df = df.reset_index()
    # Convert the 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Calculate the volume profile
    vp = pd.DataFrame(columns=["price", "volume"])
    for price in range(int(df["low"].min()), int(df["high"].max()) + 1, price_step):
        volume = df.loc[(df["low"] <= price) & (df["high"] > price), "volume"].sum()
        if volume > 0:
            vp = pd.concat(
                [vp, pd.DataFrame({"price": [price], "volume": [volume]})],
                ignore_index=True,
            )

    # Calculate the average volume
    avg_volume = vp["volume"].mean()

    # Identify the support and resistance levels
    supports = []
    resistances = []
    for i in range(1, len(vp) - 1):
        if (
            vp["volume"][i] > avg_volume
            and vp["volume"][i] > vp["volume"][i - 1]
            and vp["volume"][i] > vp["volume"][i + 1]
        ):
            supports.append(
                {
                    "Time Stamp": df.loc[df["low"] <= vp["price"][i], "date"].max(),
                    "Level": vp["price"][i],
                }
            )
        elif (
            vp["volume"][i] > avg_volume
            and vp["volume"][i] < vp["volume"][i - 1]
            and vp["volume"][i] < vp["volume"][i + 1]
        ):
            resistances.append(
                {
                    "Time Stamp": df.loc[df["high"] >= vp["price"][i], "date"].max(),
                    "Level": vp["price"][i],
                }
            )

    # Print the support and resistance levels
    # print({'Major Support Levels': supports, 'Major Resistance Levels': resistances})
    supports = sorted(supports, key=lambda d: d["Time Stamp"])
    resistances = sorted(resistances, key=lambda d: d["Time Stamp"])
    return supports, resistances


def combine_group_of_levels(combined_list, proximity=1):
    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(combined_list)

    # Sort the DataFrame by the Level column
    df = df.sort_values("Level")

    # Calculate the difference between adjacent levels
    diffs = df["Level"].diff()

    # Assign a group number to each row based on the difference between the current and previous level
    df["Group"] = (diffs > proximity).cumsum()

    combined = df

    group_count = combined.groupby("Group").count()["Level"]
    combined["Cnt"] = combined.Group.apply(lambda x: group_count[x])
    new_list = []
    for i, row in combined[combined.Cnt.gt(2)].groupby("Group").min().iterrows():
        new_list.append({"Time Stamp": row["Time Stamp"], "Level": row["Level"]})
    new_list = sorted(new_list, key=lambda d: d["Time Stamp"])

    return new_list


if __name__ == "__main__":
    pd.options.display.max_rows = 500
    
    # constants
    TOKEN_PATH = "tdtoken.json"
    
    dotenv.load_dotenv("/.env")
    ACC_NUMBER = os.getenv("ACC_NUMBER")
    API_KEY = os.getenv("API_KEY")
    REDIRECT_URI = "https://localhost"

    # Setting up the client
    import tda

    c = tda.auth.easy_client(
        token_path=TOKEN_PATH,  # follow this: https://developer.tdameritrade.com/content/simple-auth-local-apps
        # https://auth.tdameritrade.com/auth?response_type=code&redirect_uri=https%3A%2F%2Flocalhost&client_id=MTPZYEHVAXFY9Y3MPO1RF56RXG2Q7EP1%40AMER.OAUTHAP
        api_key=API_KEY,
        redirect_uri=REDIRECT_URI,
    )
    c = client(session=c.session, api_key=c.api_key)
    df = c.get_five_minute_data("SPY")
    # from TALibrary import dsma, vwap, ma_double_cloud_signal, linreg_signal

    # dsma(df.open, 67).plot()
    # vwap(df.open, df.volume, 67).plot()
    # gpt_support_resistance_by_volume_spread(df)
    from TALibrary import ma_double_cloud_signal
    ma_double_cloud_signal(
        df.set_index("date"),
        ma_length1=11,
        ma_length2=23,
        ma_length3=70,
        ma_length4=70,
        target_pct=0.69,
        price_for_ma="open",
        shift=True,
    )

    # df.set_index("date").low.lt(pt.linreg(df.set_index("date").low, 20))
    # pt.linreg(df.set_index("date").low, 20)
    # pt.linreg(df.low, 20,tsf=True).tail(78).plot()
    # df.low.tail(78).plot()

    # linreg_signal(df, 20,shift=True).tail(78)
