from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np

from .data import build_climate_frame, build_ml_frame, dataset_summary
from .features import split_time_series
from .forecasting import build_future_frame, forecast_temperature
from .models import SimpleGCN, train_and_evaluate_models
from .settings import MODEL_DIR

ARTIFACT_PATH = MODEL_DIR / "climate_artifacts.joblib"
SUMMARY_PATH = MODEL_DIR / "climate_summary.json"
SERIALIZABLE_MODEL_NAMES = {
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
}
DL_MODEL_FILES = {
    "ANN": MODEL_DIR / "ann_model.keras",
    "CNN": MODEL_DIR / "cnn_model.keras",
    "GCN": MODEL_DIR / "gcn_model.keras",
}
DL_MODEL_KIND = {
    "ANN": "ann",
    "CNN": "cnn",
    "GCN": "gcn",
}
SCALER_FILE = MODEL_DIR / "dl_scaler.joblib"


class ClimateService:
    def __init__(self) -> None:
        self._state: dict[str, Any] | None = None

    def _serialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": state["summary"],
            "metrics": state["metrics"],
            "best_model_name": state["best_model_name"],
            "available_models": state["available_models"],
            "noise_scale": state["noise_scale"],
            "climate_frame": state["climate_frame"],
            "ml_frame": state["ml_frame"],
            "dates_test": state["dates_test"],
            "y_test": state["y_test"],
            "predictions": state["predictions"],
            "models": state["models"],
            "artifacts": state.get("artifacts", {}),
        }

    def _persist_state(self, state: dict[str, Any]) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        serializable_state = dict(state)
        serializable_state["models"] = {name: model for name, model in state["models"].items() if name in SERIALIZABLE_MODEL_NAMES}
        serializable_state["artifacts"] = state.get("artifacts", {})
        joblib.dump(serializable_state, ARTIFACT_PATH)

        if state.get("artifacts", {}).get("dl_scaler") is not None:
            joblib.dump(state["artifacts"]["dl_scaler"], SCALER_FILE)

        for model_name, model_path in DL_MODEL_FILES.items():
            if model_name in state["models"]:
                state["models"][model_name].save(model_path, overwrite=True)

        summary_payload = {
            "summary": state["summary"],
            "metrics": state["metrics"],
            "best_model_name": state["best_model_name"],
            "available_models": state["available_models"],
            "noise_scale": state["noise_scale"],
        }
        SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_state_from_disk(self) -> dict[str, Any] | None:
        if not ARTIFACT_PATH.exists():
            return None
        state = joblib.load(ARTIFACT_PATH)
        try:
            from tensorflow.keras.models import load_model

            for model_name, model_path in DL_MODEL_FILES.items():
                if model_path.exists():
                    state["models"][model_name] = load_model(
                        model_path,
                        compile=False,
                        custom_objects={"SimpleGCN": SimpleGCN},
                    )
            if SCALER_FILE.exists():
                state.setdefault("artifacts", {})["dl_scaler"] = joblib.load(SCALER_FILE)
        except Exception:
            pass
        return state

    def train(self, force: bool = False) -> dict[str, Any]:
        if not force:
            cached = self.load_or_train()
            if cached is not None:
                return self._public_state(cached)

        climate_frame = build_climate_frame()
        ml_frame = build_ml_frame(climate_frame)
        split = split_time_series(ml_frame)

        result = train_and_evaluate_models(split.X_train, split.y_train, split.X_test, split.y_test)
        best_predictions = np.asarray(result["predictions"][result["best_model_name"]])
        residuals = split.y_test.to_numpy() - best_predictions
        noise_scale = float(np.std(residuals)) if len(residuals) else 0.0

        state = {
            "summary": dataset_summary(climate_frame),
            "metrics": result["metrics"],
            "best_model_name": result["best_model_name"],
            "available_models": list(result["models"].keys()),
            "noise_scale": noise_scale,
            "climate_frame": climate_frame,
            "ml_frame": ml_frame,
            "dates_test": split.dates_test.tolist(),
            "y_test": split.y_test.tolist(),
            "predictions": result["predictions"],
            "models": result["models"],
            "artifacts": result.get("artifacts", {}),
        }
        self._state = self._serialize_state(state)
        self._persist_state(self._state)
        return self._public_state(self._state)

    def load_or_train(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        loaded = self._load_state_from_disk()
        if loaded is not None:
            self._state = loaded
            return loaded
        self.train(force=True)
        if self._state is None:
            raise RuntimeError("Training did not produce an in-memory state")
        return self._state

    def forecast(self, months_ahead: int = 60, model_name: str | None = None) -> dict[str, Any]:
        state = self.load_or_train()
        selected_model_name = model_name or state["best_model_name"]
        if selected_model_name not in state["models"]:
            raise ValueError(f"Unknown model '{selected_model_name}'")

        model = state["models"][selected_model_name]
        future_frame = build_future_frame(state["climate_frame"], months_ahead)
        forecast_frame = forecast_temperature(
            model,
            state["climate_frame"],
            future_frame,
            noise_scale=state["noise_scale"],
            model_kind=DL_MODEL_KIND.get(selected_model_name, "classic"),
            scaler=state.get("artifacts", {}).get("dl_scaler"),
        )

        historical = state["climate_frame"][["Date", "Temperature_Anomaly"]].copy()
        historical["Date"] = historical["Date"].dt.strftime("%Y-%m-%d")

        return {
            "model_name": selected_model_name,
            "months_ahead": months_ahead,
            "historical": historical.to_dict(orient="records"),
            "forecast": forecast_frame.assign(Date=forecast_frame["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        }

    def get_status(self) -> dict[str, Any]:
        state = self.load_or_train()
        return self._public_state(state)

    def _public_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": state["summary"],
            "metrics": state["metrics"],
            "best_model_name": state["best_model_name"],
            "available_models": state["available_models"],
            "noise_scale": state["noise_scale"],
            "dates_test": [date.strftime("%Y-%m-%d") for date in state["dates_test"]],
            "y_test": state["y_test"],
            "predictions": state["predictions"],
        }
