from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .settings import DATA_DIR, LEGACY_DATA_DIR

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

TEMP_FILENAME = "GLB.Ts+dSST.csv"
CO2_FILENAME = "co2_mm_mlo.csv"


def _resolve_data_file(filename: str) -> Path:
    for base_dir in (DATA_DIR, LEGACY_DATA_DIR):
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate {filename} in data/ or code/")


def load_temperature_raw() -> pd.DataFrame:
    path = _resolve_data_file(TEMP_FILENAME)
    frame = pd.read_csv(path)
    if "Year" not in frame.columns:
        frame = pd.read_csv(path, skiprows=1)
    return frame


def load_co2_raw() -> pd.DataFrame:
    path = _resolve_data_file(CO2_FILENAME)
    frame = pd.read_csv(path, comment="#")
    frame.columns = frame.columns.str.strip().str.lower()
    rename_map = {}
    if "monthly_average" in frame.columns:
        rename_map["monthly_average"] = "co2"
    elif "average" in frame.columns:
        rename_map["average"] = "co2"
    if rename_map:
        frame = frame.rename(columns=rename_map)
    return frame


def build_climate_frame() -> pd.DataFrame:
    temperature = load_temperature_raw()
    temperature = temperature[["Year"] + MONTHS]

    melted = temperature.melt(
        id_vars=["Year"],
        value_vars=MONTHS,
        var_name="Month_Name",
        value_name="Temperature_Anomaly",
    )
    melted["Month"] = melted["Month_Name"].map({month: index + 1 for index, month in enumerate(MONTHS)})
    melted["Temperature_Anomaly"] = pd.to_numeric(
        melted["Temperature_Anomaly"].replace("***", np.nan),
        errors="coerce",
    )
    melted = melted.rename(columns={"Year": "year", "Month": "month"})

    co2 = load_co2_raw()
    co2 = co2[["year", "month", "co2"]]

    climate = pd.merge(
        melted[["year", "month", "Temperature_Anomaly"]],
        co2,
        on=["year", "month"],
        how="inner",
    )
    climate = climate.dropna().sort_values(["year", "month"]).reset_index(drop=True)
    climate["Date"] = pd.to_datetime(climate[["year", "month"]].assign(day=1))
    return climate[["Date", "year", "month", "Temperature_Anomaly", "co2"]]


def build_ml_frame(climate_frame: pd.DataFrame) -> pd.DataFrame:
    frame = climate_frame.copy()
    frame["Temp_Lag_1"] = frame["Temperature_Anomaly"].shift(1)
    frame["Temp_Lag_12"] = frame["Temperature_Anomaly"].shift(12)
    return frame.dropna().reset_index(drop=True)


def dataset_summary(climate_frame: pd.DataFrame) -> dict:
    first_date = climate_frame["Date"].min()
    last_date = climate_frame["Date"].max()
    return {
        "rows": int(len(climate_frame)),
        "start_date": first_date.date().isoformat() if pd.notna(first_date) else None,
        "end_date": last_date.date().isoformat() if pd.notna(last_date) else None,
        "temperature_mean": float(climate_frame["Temperature_Anomaly"].mean()),
        "co2_mean": float(climate_frame["co2"].mean()),
    }
