from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow

# MLflow retired: no MLflow imports
import numpy as np

from .data import build_climate_frame, build_ml_frame, dataset_summary
from .features import split_time_series
from .forecasting import build_future_frame, forecast_temperature

# NOTE: .models must be imported before mlflow. It imports tensorflow
# internally before mlflow for the same reason (see comment in models.py):
# importing mlflow first can break TensorFlow's native DLL loading on
# Windows. Since Python only executes a module's imports once, mlflow must
# not be imported anywhere above this line either.
from .models import SimpleGCN, train_and_evaluate_models
from .settings import (
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    DAGSHUB_USER_TOKEN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    MODEL_DIR,
)


def _configure_mlflow() -> bool:
    """Point MLflow at DagsHub if credentials are available.

    Two supported setups, tried in order:
    1. DAGSHUB_USER_TOKEN set -> use dagshub.init(), which configures both
       the tracking URI and auth from that single token.
    2. MLFLOW_TRACKING_URI + MLFLOW_TRACKING_USERNAME/PASSWORD set -> classic
       manual basic-auth setup.

    Returns True if tracking is active, False if neither is configured, in
    which case training proceeds exactly as before with no MLflow calls made.
    """
    if DAGSHUB_USER_TOKEN:
        os.environ.setdefault("DAGSHUB_USER_TOKEN", DAGSHUB_USER_TOKEN)
        import dagshub

        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return True

    if not MLFLOW_TRACKING_URI:
        return False
    if MLFLOW_TRACKING_USERNAME:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", MLFLOW_TRACKING_USERNAME)
    if MLFLOW_TRACKING_PASSWORD:
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", MLFLOW_TRACKING_PASSWORD)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return True


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
MODEL_REGISTRY_NAME = "Climate_Model"


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

        mlflow_active = _configure_mlflow()
        # A shared, human-readable identifier for this training pass. Every
        # nested run (Linear Regression, XGBoost, ANN, CNN, GCN, ...) gets
        # tagged with the same session_id, and the parent run's name embeds
        # it too. Without this, runs from different training sessions are
        # only distinguishable by "Created" timestamp, and the DagsHub/MLflow
        # table interleaves them by model name — several sessions' "ANN" runs
        # end up sitting next to each other with no easy way to tell which
        # session each one belongs to. Filter/sort by the "session_id" tag
        # column (or search `tags.session_id = '...'`) to see one session's
        # runs together.
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_context = mlflow.start_run(run_name=f"climate-training-{session_id}") if mlflow_active else contextlib.nullcontext()

        with run_context:
            if mlflow_active:
                mlflow.set_tag("session_id", session_id)

            climate_frame = build_climate_frame()
            ml_frame = build_ml_frame(climate_frame)
            split = split_time_series(ml_frame)

            if mlflow_active:
                mlflow.log_params(
                    {
                        "n_train": len(split.X_train),
                        "n_test": len(split.X_test),
                        "n_features": split.X_train.shape[1],
                    }
                )

            result = train_and_evaluate_models(
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
                session_id=session_id if mlflow_active else None,
            )
            best_predictions = np.asarray(result["predictions"][result["best_model_name"]])
            residuals = split.y_test.to_numpy() - best_predictions
            noise_scale = float(np.std(residuals)) if len(residuals) else 0.0

            if mlflow_active:
                mlflow.set_tag("best_model", result["best_model_name"])
                mlflow.log_metrics({f"best_{k}": v for k, v in result["metrics"][result["best_model_name"]].items()})

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

            if mlflow_active:
                mlflow.log_artifact(str(SUMMARY_PATH))

                # Register the selected best model in the MLflow Model Registry
                try:
                    best_name = state["best_model_name"]
                    best_model = state["models"][best_name]
                    # Choose appropriate logging method depending on model type
                    try:
                        # sklearn/xgboost classic models
                        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="best_model", registered_model_name=MODEL_REGISTRY_NAME)
                    except Exception:
                        try:
                            # tensorflow/keras models
                            import mlflow.tensorflow as _mlflow_tf

                            _mlflow_tf.log_model(best_model, artifact_path="best_model", registered_model_name=MODEL_REGISTRY_NAME)
                        except Exception:
                            # Best-effort registration; don't fail the training run if registry logging fails
                            pass
                except Exception:
                    pass

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

    def reset_state(self) -> None:
        """Clear the in-memory cached state so the service will retrain on next use."""
        self._state = None

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
