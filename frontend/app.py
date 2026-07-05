from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="Climate ML MLOps", page_icon="🌍", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")


def _get_local_service():
    if BACKEND_URL:
        return None
    from backend.climate.service import ClimateService

    return ClimateService()


LOCAL_SERVICE = _get_local_service()

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(12, 44, 67, 0.92), rgba(8, 15, 28, 0.98)), 
                linear-gradient(180deg, #08111c 0%, #0b1726 100%);
            color: #edf6ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 1.2rem;
            background: rgba(255,255,255,0.04);
            box-shadow: 0 20px 60px rgba(0,0,0,0.28);
        }
        .metric-card {
            border-radius: 1rem;
            padding: 0.85rem 1rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _request(method: str, path: str, timeout: int = 120, **kwargs: Any) -> dict[str, Any]:
    if not BACKEND_URL:
        raise RuntimeError("Backend URL not configured")
    response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def load_status() -> dict[str, Any]:
    if BACKEND_URL:
        return _request("GET", "/status")
    if LOCAL_SERVICE is None:
        raise RuntimeError("Local backend service is unavailable")
    return LOCAL_SERVICE.get_status()


def load_train_result() -> dict[str, Any]:
    # NOTE: this used to be wrapped in `except Exception: pass`, which
    # silently swallowed timeouts, connection errors and backend 500s. That
    # made a failed retrain look identical to a successful one from the
    # user's point of view (no error shown, stale data reused). Any failure
    # here must now be visible to the caller so the "Entraîner" button can
    # report it instead of silently no-op'ing.
    # Training (especially the Keras models) can comfortably exceed the
    # default 120s timeout used for read-only calls, so this uses a much
    # longer timeout specifically for training.
    if BACKEND_URL:
        return _request("POST", "/train", timeout=900, json={"refresh": True})
    if LOCAL_SERVICE is None:
        raise RuntimeError("Local backend service is unavailable")
    return LOCAL_SERVICE.train(force=True)


def get_forecast(months_ahead: int, model_name: str | None) -> dict[str, Any]:
    if BACKEND_URL:
        params = {"months_ahead": months_ahead}
        if model_name:
            params["model_name"] = model_name
        return _request("GET", "/forecast", params=params)
    if LOCAL_SERVICE is None:
        raise RuntimeError("Local backend service is unavailable")
    return LOCAL_SERVICE.forecast(months_ahead=months_ahead, model_name=model_name)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div style='font-size:0.78rem;opacity:0.72'>{label}</div><div style='font-size:1.25rem;font-weight:700'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        "<div class='hero'><h1>Climate ML MLOps</h1><p>Streamlit dashboard for the temperature anomaly and CO2 regression pipeline.</p></div>",  # noqa: E501
        unsafe_allow_html=True,
    )

    try:
        status = load_status()
    except Exception as exc:
        st.error(f"Impossible de récupérer le statut du backend : {exc}")
        st.stop()

    with st.sidebar:
        st.header("Pilotage")
        st.caption("L'API backend est utilisée si `BACKEND_URL` est défini.")
        if st.button("Entraîner / recharger les modèles", use_container_width=True):
            load_status.clear()
            with st.spinner("Entraînement en cours (peut prendre plusieurs minutes avec ANN/CNN/GCN)..."):
                try:
                    status = load_train_result()
                except Exception as exc:
                    st.error(f"Échec de l'entraînement : {exc}")
                else:
                    st.success("Modèles entraînés.")

        selected_model = st.selectbox(
            "Modèle de prévision",
            options=status["available_models"],
            index=status["available_models"].index(status["best_model_name"]),
        )
        months_ahead = st.slider("Prévision en mois", min_value=12, max_value=300, value=300, step=12)

    summary = status["summary"]
    metrics = status["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Observations", f"{summary['rows']:,}")
    with col2:
        render_metric_card("Début", summary["start_date"] or "n/a")
    with col3:
        render_metric_card("Fin", summary["end_date"] or "n/a")
    with col4:
        render_metric_card("Meilleur modèle", status["best_model_name"])

    tab_data, tab_models, tab_forecast = st.tabs(["Données", "Modèles", "Prévision"])

    with tab_data:
        st.subheader("Aperçu du jeu de données")
        climate_frame = LOCAL_SERVICE.load_or_train()["climate_frame"] if not BACKEND_URL else None
        if climate_frame is not None:
            st.dataframe(climate_frame.head(12), use_container_width=True)
        else:
            st.info("Les aperçus détaillés sont disponibles en mode local.")
        st.json(summary)

        if climate_frame is None and not BACKEND_URL:
            climate_frame = LOCAL_SERVICE.load_or_train()["climate_frame"]

        if climate_frame is not None:
            chart_frame = climate_frame.copy()
            chart_frame["Date"] = pd.to_datetime(chart_frame["Date"])
            fig = px.line(
                chart_frame,
                x="Date",
                y=["Temperature_Anomaly", "co2"],
                title="Évolution historique",
                labels={"value": "Valeur", "variable": "Série"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_models:
        st.subheader("Métriques du test")
        metrics_table = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
        st.dataframe(metrics_table.sort_values("rmse"), use_container_width=True, hide_index=True)

        dates_test = pd.to_datetime(pd.Series(status["dates_test"]))
        y_test = pd.Series(status["y_test"], name="Actual")
        comparison = pd.DataFrame({"Date": dates_test, "Actual": y_test})
        for model_name, preds in status["predictions"].items():
            comparison[model_name] = preds
        plot_frame = comparison.melt(id_vars=["Date", "Actual"], var_name="Model", value_name="Prediction")
        fig = px.line(
            plot_frame,
            x="Date",
            y="Prediction",
            color="Model",
            title="Comparaison des prédictions sur le test",
        )
        fig.add_scatter(
            x=comparison["Date"],
            y=comparison["Actual"],
            name="Actual",
            line=dict(color="black", width=3),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_forecast:
        st.subheader("Projection future")
        if st.button("Générer la prévision", use_container_width=True):
            forecast_payload = get_forecast(months_ahead=months_ahead, model_name=selected_model)
            future = pd.DataFrame(forecast_payload["forecast"])
            history = pd.DataFrame(forecast_payload["historical"])
            history["Date"] = pd.to_datetime(history["Date"])
            future["Date"] = pd.to_datetime(future["Date"])

            chart_history = history.rename(columns={"Temperature_Anomaly": "Value"})
            chart_future = future.rename(columns={"Temperature_Forecast": "Value"})
            chart_history["Segment"] = "Historique"
            chart_future["Segment"] = "Prévision"
            forecast_chart = pd.concat(
                [
                    chart_history[["Date", "Value", "Segment"]],
                    chart_future[["Date", "Value", "Segment"]],
                ],
                ignore_index=True,
            )
            fig = px.line(
                forecast_chart,
                x="Date",
                y="Value",
                color="Segment",
                title=f"Prévision de température avec {selected_model}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(future.tail(12), use_container_width=True, hide_index=True)
        else:
            st.info("Clique sur le bouton pour calculer la projection future.")


if __name__ == "__main__":
    main()
