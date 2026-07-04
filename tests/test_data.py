from backend.climate.data import build_climate_frame, build_ml_frame


def test_climate_frame_has_expected_columns() -> None:
    frame = build_climate_frame()
    assert not frame.empty
    assert {"Date", "year", "month", "Temperature_Anomaly", "co2"}.issubset(frame.columns)


def test_ml_frame_contains_lag_features() -> None:
    climate_frame = build_climate_frame()
    ml_frame = build_ml_frame(climate_frame)
    assert not ml_frame.empty
    assert {"Temp_Lag_1", "Temp_Lag_12"}.issubset(ml_frame.columns)
