from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_co2_trend(climate_frame: pd.DataFrame) -> LinearRegression:
    features = pd.DataFrame(
        {
            "year": climate_frame["year"],
            "year_sq": climate_frame["year"] ** 2,
        }
    )
    target = climate_frame["co2"]
    model = LinearRegression()
    model.fit(features, target)
    return model


def build_future_frame(climate_frame: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    last_date = pd.Timestamp(climate_frame["Date"].max())
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS")
    future = pd.DataFrame({"Date": future_dates})
    future["year"] = future["Date"].dt.year
    future["month"] = future["Date"].dt.month

    trend_model = fit_co2_trend(climate_frame)
    co2_features = pd.DataFrame(
        {
            "year": future["year"],
            "year_sq": future["year"] ** 2,
        }
    )
    future["co2"] = trend_model.predict(co2_features)
    return future


def forecast_temperature(
    model,
    climate_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    noise_scale: float = 0.0,
    model_kind: str = "classic",
    scaler=None,
) -> pd.DataFrame:
    history = climate_frame["Temperature_Anomaly"].tolist()
    predictions: list[float] = []
    rng = np.random.default_rng(42)

    if len(history) < 12:
        raise ValueError("Not enough history to compute 12-month lags")

    for _, row in future_frame.iterrows():
        feature_row = pd.DataFrame(
            [
                {
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "co2": float(row["co2"]),
                    "Temp_Lag_1": float(history[-1]),
                    "Temp_Lag_12": float(history[-12]),
                }
            ]
        )

        if model_kind in {"ann", "cnn", "gcn"}:
            if scaler is None:
                raise ValueError("Scaler is required for ANN/CNN/GCN forecasting")
            scaled_features = scaler.transform(feature_row)
            if model_kind in {"cnn", "gcn"}:
                scaled_features = scaled_features.reshape(1, scaled_features.shape[1], 1)
            prediction = float(model.predict(scaled_features, verbose=0).flatten()[0])
        else:
            prediction = float(model.predict(feature_row)[0])

        if noise_scale > 0:
            prediction += float(rng.normal(0, noise_scale))
        predictions.append(prediction)
        history.append(prediction)

    result = future_frame.copy()
    result["Temperature_Forecast"] = predictions
    return result
