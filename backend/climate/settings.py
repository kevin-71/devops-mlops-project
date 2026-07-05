import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# NOTE: previously these ignored the DATA_DIR/MODEL_DIR environment
# variables entirely and always resolved relative to PROJECT_ROOT. That
# made the `MODEL_DIR=/app/models` / `DATA_DIR=/app/data` entries in
# docker-compose.yml silent no-ops: the container would still write to
# whatever PROJECT_ROOT/models resolves to inside the image, not to
# /app/models. Read the env vars first, and only fall back to the
# PROJECT_ROOT-relative path when they aren't set (e.g. running locally
# without Docker).
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow / DagsHub tracking configuration -------------------------------
# Preferred: set DAGSHUB_USER_TOKEN + DAGSHUB_REPO_OWNER + DAGSHUB_REPO_NAME.
# The dagshub library reads DAGSHUB_USER_TOKEN itself, so only one secret is
# needed. Get a token from DagsHub > avatar > Settings > Tokens.
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "kevin-71")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "climate-mlops")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "climate-ml")

# Fallback: manual basic-auth setup (still supported if someone prefers it,
# or if DAGSHUB_USER_TOKEN isn't set).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
