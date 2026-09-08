import os
import subprocess
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "mlflow.yaml"
TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"

load_dotenv(REPO_ROOT / ".env")


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


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run(
    experiment_id: str,
    experiment_name: str,
    author: str,
    config: dict | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[str] | None = None,
    tags: dict[str, str] | None = None,
    run_name: str | None = None,
) -> str:
    """Log a complete MLflow run following this repo's convention (ver ADR 0009):

    - Todos los parámetros de `config` (ej. el contenido de un
      experiments/IDXX-*/configs/*.yaml), no solo algunos.
    - Todas las métricas en `metrics`, no solo la métrica primaria de
      metadata.yaml — comparar runs completos, no solo un número.
    - Tags automáticos: autor y commit de git, para poder ir del run al
      código exacto que lo generó.
    - `artifacts`: rutas de archivo a subir (plots, modelo entrenado, el
      config usado). **Nunca una muestra cruda de datos sensibles** — ver
      docs/architecture/data-governance.md, la misma regla que aplica a
      `data/raw` y al remoto de DVC aplica aquí.

    Debe llamarse después de que MLFLOW_TRACKING_URI esté configurada. Loguea
    todo dentro de un solo run y devuelve su run_id.

    `run_name` identifica este run dentro del experimento en la UI de MLflow/Azure
    ML — sin él, MLflow le asigna un nombre aleatorio (ej. "loyal-picture-m0jqptwt")
    que no dice nada sobre qué variante es. Usa el mismo nombre que el archivo de
    `configs/*.yaml` que generó este run (ej. "baseline").
    """
    configure_mlflow(experiment_id, experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "author": author,
                "git_commit": _current_git_commit(),
                **(tags or {}),
            }
        )
        if config:
            mlflow.log_params(config)
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
        for artifact_path in artifacts or []:
            mlflow.log_artifact(artifact_path)

        return run.info.run_id
