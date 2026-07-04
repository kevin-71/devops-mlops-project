from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FEATURE_COLUMNS = ["year", "month", "co2", "Temp_Lag_1", "Temp_Lag_12"]


@dataclass(slots=True)
class TimeSeriesSplitBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    dates_test: pd.Series
    split_index: int


def split_time_series(frame: pd.DataFrame, train_ratio: float = 0.8) -> TimeSeriesSplitBundle:
    split_index = int(len(frame) * train_ratio)
    X = frame[FEATURE_COLUMNS]
    y = frame["Temperature_Anomaly"]
    return TimeSeriesSplitBundle(
        X_train=X.iloc[:split_index],
        X_test=X.iloc[split_index:],
        y_train=y.iloc[:split_index],
        y_test=y.iloc[split_index:],
        dates_test=frame["Date"].iloc[split_index:],
        split_index=split_index,
    )
