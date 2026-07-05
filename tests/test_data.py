import pytest

from backend.climate import data as climate_data
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


def test_data_loader_does_not_fallback_to_legacy_directory(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    legacy_dir = tmp_path / "code"
    legacy_dir.mkdir()
    (legacy_dir / "co2_mm_mlo.csv").write_text("year,month,average\n2020,1,1.0\n", encoding="utf-8")

    monkeypatch.setattr(climate_data, "DATA_DIR", data_dir)
    if hasattr(climate_data, "LEGACY_DATA_DIR"):
        monkeypatch.setattr(climate_data, "LEGACY_DATA_DIR", legacy_dir)

    with pytest.raises(FileNotFoundError):
        climate_data._resolve_data_file("co2_mm_mlo.csv")
