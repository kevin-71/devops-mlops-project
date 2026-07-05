import os
import sys
import time

import requests
from mlflow.tracking import MlflowClient

# Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "Climate_Model")
STAGING_URL = os.environ.get("STAGING_URL", "http://localhost:8000/forecast")
LATENCY_THRESHOLD_MS = float(os.environ.get("LATENCY_THRESHOLD_MS", 500))
RMSE_THRESHOLD = float(os.environ.get("RMSE_THRESHOLD", 0.5))
SUMMARY_PATH = os.environ.get("SUMMARY_PATH", "models/climate_summary.json")
PRODUCTION_MARKER = os.environ.get("PRODUCTION_MARKER", "models/production.json")

client = MlflowClient()


def get_latest_model_version():
    """Récupère la dernière version du modèle enregistrée."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as exc:
        # Could not reach MLflow server
        raise RuntimeError(f"MLflow unavailable: {exc}") from exc
    if not versions:
        raise LookupError("no_versions")
    latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    return latest_version


def run_quality_gates():
    print("🚀 Début des Quality Gates...")

    # Try MLflow first; if unavailable or no model versions, fallback to local summary
    use_mlflow = True
    try:
        latest_version = get_latest_model_version()
        run_id = latest_version.run_id
        version_num = latest_version.version
        print(f"📦 Modèle évalué via MLflow : Version {version_num} (Run ID: {run_id})")
        run = client.get_run(run_id)
        rmse = run.data.metrics.get("RMSE", run.data.metrics.get("rmse", None))
        if rmse is None:
            print("⚠️  RMSE absent du run MLflow; bascule vers résumé local.")
            use_mlflow = False
    except LookupError:
        print("⚠️  Aucun modèle trouvé dans le registre MLflow; bascule vers résumé local.")
        use_mlflow = False
    except Exception as exc:
        print(f"⚠️  Impossible d'accéder à MLflow: {exc}; bascule vers résumé local.")
        use_mlflow = False

    if not use_mlflow:
        # Fallback: read local summary file
        if not os.path.exists(SUMMARY_PATH):
            print(f"❌ Aucune source de métriques disponible (ni MLflow ni {SUMMARY_PATH}).")
            sys.exit(1)
        import json

        with open(SUMMARY_PATH, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        best = summary.get("best_model_name")
        metrics = summary.get("metrics", {})
        if best is None or best not in metrics:
            print("❌ Aucune métrique disponible pour le meilleur modèle dans le résumé local.")
            sys.exit(1)
        rmse = metrics[best].get("rmse")
        if rmse is None:
            print("❌ RMSE introuvable pour le modèle dans le résumé local.")
            sys.exit(1)
        print(f"📦 Modèle évalué via résumé local : {best}, RMSE = {rmse}")
        # version_num is unknown in local fallback
        version_num = None

    # Gate 2: Smoke Test & Latency
    print("⏳ Test de l'API de Staging...")
    time.sleep(5)
    start_time = time.time()
    try:
        response = requests.get(STAGING_URL, params={"months_ahead": 5}, timeout=10)
        latency = (time.time() - start_time) * 1000
    except Exception as e:
        print(f"❌ Échec Gate 2: API inaccessible - {e}")
        sys.exit(1)

    if response.status_code != 200:
        print(f"❌ Échec Gate 2: Code HTTP {response.status_code}")
        sys.exit(1)

    if latency > LATENCY_THRESHOLD_MS:
        print(f"❌ Échec Gate 2: Latence trop élevée ({latency:.2f}ms > {LATENCY_THRESHOLD_MS}ms)")
        sys.exit(1)

    print(f"✅ Gate 2 passée : Status 200, Latence = {latency:.2f}ms")

    # Promotion
    if use_mlflow and version_num is not None:
        print("🎉 Toutes les gates ont passé. Promotion en Production via MLflow...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_num,
            stage="Production",
            archive_existing_versions=True,
        )
        print("✅ Modèle promu avec succès dans MLflow !")
    else:
        # Local promotion: write a simple production marker file
        try:
            import json
            from datetime import datetime

            marker = {
                "best_model_name": summary.get("best_model_name") if not use_mlflow else None,
                "rmse": rmse,
                "promoted_at": datetime.utcnow().isoformat() + "Z",
            }
            os.makedirs(os.path.dirname(PRODUCTION_MARKER), exist_ok=True)
            with open(PRODUCTION_MARKER, "w", encoding="utf-8") as fh:
                json.dump(marker, fh, indent=2, ensure_ascii=False)
            print(f"✅ Modèle promu localement (marker: {PRODUCTION_MARKER})")
        except Exception as e:
            print(f"⚠️  Échec de la promotion locale: {e}")


if __name__ == "__main__":
    try:
        run_quality_gates()
    except SystemExit as e:
        # Gate script called sys.exit() (failure case)
        code = e.code if isinstance(e.code, int) else 1
        print("\n❌ QUALITY GATES FAILED — model stays in Staging; production unchanged.")
        sys.exit(code)
    except Exception as e:
        print("\n❌ QUALITY GATES ERROR — model stays in Staging; production unchanged.")
        print(f"Error: {e}")
        sys.exit(1)
    else:
        print("\n✅ QUALITY GATES PASSED — promotion completed (MLflow or local marker).")
