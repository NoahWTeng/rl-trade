import numpy as np
import pandas as pd
from tensortrade.data.cdd import CryptoDataDownload
import os
from datetime import timezone, datetime
import quantstats as qs
import torch
import pandas_ta as ta
import ta as ta1
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

np.seterr(
    divide="ignore",
    invalid="ignore",
)

pd.options.mode.use_inf_as_na = True

qs.extend_pandas()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def crypto_download_data(
    path: str = "dataset",
    exchange: str = "Bitfinex",
    coin: str = "BTC",
    fiat: str = "USD",
    timeframe: str = "1h",
    export_csv: bool = True,
) -> pd.DataFrame:

    fiat = fiat.upper()
    coin = coin.upper()
    timeframe = timeframe.lower()
    exchange = exchange.capitalize()

    cdd = CryptoDataDownload()

    df = cdd.fetch(exchange, fiat, coin, timeframe)

    if export_csv:
        _save_dataset(df, path, exchange, fiat, coin, timeframe)

    return df


def _save_dataset(
    df: pd.DataFrame,
    path: str = "dataset",
    exchange: str = "Bitfinex",
    fiat: str = "USD",
    coin: str = "BTC",
    timeframe: str = "1h",
) -> None:
    os.makedirs(
        path,
        exist_ok=True,
    )

    df = df[["date", "unix", "open", "high", "low", "close", "volume"]]

    # Convert the date column type from string to datetime for proper sorting.
    df["date"] = pd.to_datetime(df["date"])

    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by="date", ascending=True, inplace=True)

    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %I:%M %p")

    df.to_csv(f"{path}/{exchange}_{coin}_{fiat}_{timeframe}.csv", header=True)


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df["volume"] = np.int64(df["volume"])
    df["date"] = pd.to_datetime(df["date"])

    df.sort_values(by="date", ascending=True, inplace=True)
    df.drop(columns=["Unnamed: 0"], inplace=True)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d %I:%M %p")
    df["unix"] = df["date"].apply(
        lambda x: int(
            datetime.strptime(x, "%Y-%m-%d %I:%M %p")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    )

    return df


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    _data = pd.read_csv(file_name)
    return _data


def data_split(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df["unix"] > start) & (df["unix"] < end)]
    data = data.sort_values(["unix"], ignore_index=True)
    data.index = data.unix.factorize()[0]
    return data


def add_all_indicator(df: pd.DataFrame) -> pd.DataFrame:

    # Automatically-generated using pandas_ta
    data = df.copy()

    strategies = [
        "candles",
        "cycles",
        "momentum",
        "overlap",
        "performance",
        "statistics",
        "trend",
        "volatility",
        "volume",
    ]

    data.index = pd.DatetimeIndex(data.index)

    cores = os.cpu_count()
    data.ta.cores = cores

    # for strategy in strategies:
    #     data.ta.strategy(strategy, exclude=["kvo"])

    data = data.set_index("date")

    # Generate all default indicators from ta library
    # ta1.add_all_ta_features(data, "open", "high", "low", "close", "volume", fillna=True)

    # Naming convention across most technical indicator libraries
    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    # Custom indicators
    features = pd.DataFrame.from_dict(
        {
            "prev_open": data["Open"].shift(1),
            "prev_high": data["High"].shift(1),
            "prev_low": data["Low"].shift(1),
            "prev_close": data["Close"].shift(1),
            "prev_volume": data["Volume"].shift(1),
            # "vol_5": data["Close"].rolling(window=5).std().abs(),
            # "vol_10": data["Close"].rolling(window=10).std().abs(),
            # "vol_20": data["Close"].rolling(window=20).std().abs(),
            # "vol_30": data["Close"].rolling(window=30).std().abs(),
            # "vol_50": data["Close"].rolling(window=50).std().abs(),
            # "vol_60": data["Close"].rolling(window=60).std().abs(),
            # "vol_100": data["Close"].rolling(window=100).std().abs(),
            # "vol_200": data["Close"].rolling(window=200).std().abs(),
            # "ma_5": data["Close"].rolling(window=5).mean(),
            # "ma_10": data["Close"].rolling(window=10).mean(),
            # "ma_20": data["Close"].rolling(window=20).mean(),
            # "ma_30": data["Close"].rolling(window=30).mean(),
            # "ma_50": data["Close"].rolling(window=50).mean(),
            # "ma_60": data["Close"].rolling(window=60).mean(),
            # "ma_100": data["Close"].rolling(window=100).mean(),
            # "ma_200": data["Close"].rolling(window=200).mean(),
            # "lr_open": np.log(data["Open"]).diff().fillna(0),
            # "lr_high": np.log(data["High"]).diff().fillna(0),
            # "lr_low": np.log(data["Low"]).diff().fillna(0),
            # "lr_close": np.log(data["Close"]).diff().fillna(0),
            # "r_volume": data["Close"].diff().fillna(0),
        }
    )
    # Concatenate both manually and automatically generated features
    data = pd.concat([data, features], axis="columns").fillna(method="pad")

    # Remove potential column duplicates
    data = data.loc[:, ~data.columns.duplicated()]

    data = data.reset_index()

    # Generate all default quantstats features
    # df_quantstats = _generate_all_default_quantstats_features(data)

    # Concatenate both manually and automatically generated features
    # data = pd.concat([data, df_quantstats], axis="columns").fillna(method="pad")

    # Remove potential column duplicates
    # data = data.loc[:, ~data.columns.duplicated()]

    # A lot of indicators generate NaNs at the beginning of DataFrames, so remove them
    data = data.reset_index(drop=True)
    data = _fix_dataset_inconsistencies(data, fill_value=None)

    return data


def _generate_all_default_quantstats_features(data: pd.DataFrame) -> pd.DataFrame:
    excluded_indicators = [
        "compare",
        "greeks",
        "information_ratio",
        "omega",
        "r2",
        "r_squared",
        "rolling_greeks",
        "warn",
        "treynor_ratio",
        "compsum",
    ]

    indicators_list = [
        f for f in dir(qs.stats) if f[0] != "_" and f not in excluded_indicators
    ]

    df = data.copy()
    df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)

    for indicator_name in indicators_list:
        try:
            # print(indicator_name)
            indicator = qs.stats.__dict__[indicator_name](df["Close"])
            if isinstance(indicator, pd.Series):
                indicator = indicator.to_frame(name=indicator_name)
                df = pd.concat([df, indicator], axis="columns")
        except (pd.errors.InvalidIndexError, ValueError):
            pass

    df = df.reset_index()
    return df


def _fix_dataset_inconsistencies(
    dataframe: pd.DataFrame, fill_value: str = None
) -> pd.DataFrame:
    dataframe = dataframe.replace([-np.inf, np.inf], np.nan)

    # This is done to avoid filling middle holes with backfilling.
    if fill_value is None:
        dataframe.iloc[0, :] = dataframe.apply(
            lambda column: column.iloc[column.first_valid_index()], axis="index"
        )
    else:
        dataframe.iloc[0, :] = dataframe.iloc[0, :].fillna(fill_value)

    return dataframe.fillna(axis="index", method="pad").dropna(axis="columns")


def split_data(data: pd.DataFrame):
    X = data.copy()
    Y = X["Close"].pct_change()

    X_train_test, X_valid, y_train_test, y_valid = train_test_split(
        X, Y, train_size=0.8, test_size=0.20, shuffle=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, y_train_test, train_size=0.70, test_size=0.30, shuffle=False
    )

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def low_variance(data: pd.DataFrame) -> pd.DataFrame:
    """
    https://towardsdatascience.com/how-to-use-variance-thresholding-for-robust-feature-selection-a4503f2b5c3f
    """
    sel = VarianceThreshold(threshold=(0.86 * (1 - 0.86)))
    date = data[["date"]].copy()
    dataset = data.drop(columns=["date"])

    sel.fit(dataset)
    mask = sel.get_support(indices=True)

    dataset[dataset.columns[mask]]
    dataset = pd.concat([date, dataset], axis="columns")

    return dataset


def normalizing(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    column_names = df.columns.tolist()
    for column in column_names[1:]:
        # Logging and Differencing
        test = np.log(df[column]) - np.log(df[column].shift(1))
        if test[1:].isnull().any():
            df[column] = df[column] - df[column].shift(1)
        else:
            df[column] = np.log(df[column]) - np.log(df[column].shift(1))
        # Min Max Scaler implemented
        Min = df[column].min()
        Max = df[column].max()
        df[column] = (df[column] - Min) / (Max - Min)

    df = _fix_dataset_inconsistencies(df)
    df.drop(columns=["date", "unix"], inplace=True)
    return df
