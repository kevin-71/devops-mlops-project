import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

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
