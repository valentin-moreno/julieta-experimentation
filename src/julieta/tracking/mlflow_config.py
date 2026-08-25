import os
from pathlib import Path

import mlflow
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "mlflow.yaml"
TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"


def configure_mlflow(experiment_id: str, experiment_name: str) -> str:
    """Point the MLflow client at the centralized Azure ML tracking server
    and set the active experiment.

    Requires the MLFLOW_TRACKING_URI environment variable (see configs/mlflow.yaml
    for how to obtain it once the Azure ML Workspace exists). Returns the full
    experiment name that was set, so callers can log it if useful.
    """
    tracking_uri = os.environ.get(TRACKING_URI_ENV_VAR)
    if not tracking_uri:
        raise RuntimeError(
            f"{TRACKING_URI_ENV_VAR} no está definida. Ver configs/mlflow.yaml "
            "para cómo obtener la URL del Azure ML Workspace."
        )

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    prefix = config["experiment_name_prefix"]

    mlflow.set_tracking_uri(tracking_uri)
    full_experiment_name = f"{prefix}/{experiment_id}-{experiment_name}"
    mlflow.set_experiment(full_experiment_name)
    return full_experiment_name
