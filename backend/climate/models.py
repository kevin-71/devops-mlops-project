from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import tensorflow as tf
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, Layer
    from tensorflow.keras.regularizers import l2
except Exception:  # pragma: no cover - optional dependency
    tf = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    return candidates


def _build_ann(input_dim: int) -> Any:
    if tf is None:
        return None
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            Dense(1),
        ],
        name="ANN_Model",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _build_cnn(input_shape: tuple[int, int]) -> Any:
    if tf is None:
        return None
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=32, kernel_size=2, activation="relu", kernel_regularizer=l2(0.001)),
            Flatten(),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(1),
        ],
        name="CNN_Model",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


class SimpleGCN(Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.A = self.add_weight(shape=(input_shape[1], input_shape[1]), initializer="ones", trainable=True)

    def call(self, inputs):
        h = tf.matmul(inputs, self.W)
        output = tf.matmul(self.A, h)
        return tf.nn.relu(output)


def _build_gcn(input_shape: tuple[int, int]) -> Any:
    if tf is None:
        return None
    gnn_input = Input(shape=input_shape)
    x = SimpleGCN(16)(gnn_input)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    gnn_output = Dense(1)(x)
    model = Model(inputs=gnn_input, outputs=gnn_output, name="GCN_Model")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_evaluate_models(X_train, y_train, X_test, y_test) -> dict[str, Any]:
    models = build_model_candidates()
    trained_models: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    metrics: dict[str, dict[str, float]] = {}
    artifacts: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        trained_models[name] = model
        predictions[name] = prediction.tolist()
        metrics[name] = {
            "rmse": _rmse(y_test, prediction),
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }

    if tf is not None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        callbacks = [EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)]

        ann_model = _build_ann(X_train_scaled.shape[1])
        ann_model.fit(
            X_train_scaled,
            y_train,
            epochs=100,
            batch_size=16,
            verbose=0,
            validation_split=0.1,
            callbacks=callbacks,
        )
        ann_prediction = ann_model.predict(X_test_scaled, verbose=0).flatten()
        trained_models["ANN"] = ann_model
        predictions["ANN"] = ann_prediction.tolist()
        metrics["ANN"] = {
            "rmse": _rmse(y_test, ann_prediction),
            "mae": float(mean_absolute_error(y_test, ann_prediction)),
            "r2": float(r2_score(y_test, ann_prediction)),
        }

        cnn_train = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        cnn_test = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        cnn_model = _build_cnn((cnn_train.shape[1], 1))
        cnn_model.fit(
            cnn_train,
            y_train,
            epochs=100,
            batch_size=16,
            verbose=0,
            validation_split=0.1,
            callbacks=callbacks,
        )
        cnn_prediction = cnn_model.predict(cnn_test, verbose=0).flatten()
        trained_models["CNN"] = cnn_model
        predictions["CNN"] = cnn_prediction.tolist()
        metrics["CNN"] = {
            "rmse": _rmse(y_test, cnn_prediction),
            "mae": float(mean_absolute_error(y_test, cnn_prediction)),
            "r2": float(r2_score(y_test, cnn_prediction)),
        }

        gcn_model = _build_gcn((cnn_train.shape[1], 1))
        gcn_model.fit(
            cnn_train,
            y_train,
            epochs=100,
            batch_size=16,
            verbose=0,
            validation_split=0.1,
            callbacks=callbacks,
        )
        gcn_prediction = gcn_model.predict(cnn_test, verbose=0).flatten()
        trained_models["GCN"] = gcn_model
        predictions["GCN"] = gcn_prediction.tolist()
        metrics["GCN"] = {
            "rmse": _rmse(y_test, gcn_prediction),
            "mae": float(mean_absolute_error(y_test, gcn_prediction)),
            "r2": float(r2_score(y_test, gcn_prediction)),
        }
        artifacts["dl_scaler"] = scaler

    best_model_name = min(metrics, key=lambda candidate: metrics[candidate]["rmse"])
    return {
        "models": trained_models,
        "predictions": predictions,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "artifacts": artifacts,
    }
