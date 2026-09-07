# ADR 0005: Tracking centralizado de experimentos con MLflow sobre Azure ML

- Estado: aceptada — conectado al Azure ML Workspace `ml-salva-dev` (resource group `ml-ops`)
- Fecha: 2026-08-25 (conectado: 2026-09-03)

## Contexto

El repo trackea experimentos hoy solo con `metadata.yaml` + `generate_summary.py`: es liviano y versionado en git, pero no registra corridas individuales (métricas por paso, artefactos, comparación entre runs) ni tiene un registro de modelos real. La empresa usa Azure, y Azure Machine Learning trae un tracking server compatible con MLflow ya administrado — no hace falta operar un servidor de MLflow propio (base de datos de backend, storage de artefactos, etc.).

## Decisión

- El tracking detallado de runs se hace con **MLflow**, apuntando al tracking server administrado del **Azure ML Workspace** `ml-salva-dev` (resource group `ml-ops`).
- `metadata.yaml` **no se reemplaza**: sigue siendo el resumen liviano y versionado en git que lee `generate_summary.py` (útil para ver el estado de todos los experimentos sin abrir MLflow). MLflow guarda el detalle de cada corrida. Son complementarios, no redundantes.
- La URL de tracking nunca se comitea (identifica recursos concretos de la suscripción de Azure): vive en la variable de entorno `MLFLOW_TRACKING_URI`, documentada en [configs/mlflow.yaml](../../configs/mlflow.yaml) (config global del repo — distinta de `experiments/IDXX-*/configs/`, que es configuración específica de cada experimento).
- [`julieta.tracking.mlflow_config.configure_mlflow(experiment_id, experiment_name)`](../../src/julieta/tracking/mlflow_config.py) centraliza cómo cualquier notebook o pipeline se conecta: valida que la variable de entorno exista, y agrupa los runs bajo `<prefijo>/<id>-<nombre>` en el workspace.
- Dependencias añadidas a `pyproject.toml`: `mlflow`, `azureml-mlflow` (el conector que traduce las llamadas de MLflow al backend de Azure ML).

## Consecuencias

- Si `MLFLOW_TRACKING_URI` no está definida, `configure_mlflow` falla con un mensaje explícito (no falla en silencio ni cae a un tracking local).
- Conexión real verificada el 2026-09-03: un run de prueba (`log_run`) quedó registrado en el workspace y visible en la pestaña Jobs de `ml.azure.com`.
- Cada científico necesita el rol **AzureML Data Scientist** sobre el workspace para poder loguear runs con su propia identidad (autenticación vía `az login`, no hay credenciales compartidas).
- Si en el futuro se decide auto-hospedar MLflow en vez de usar Azure ML, este ADR queda obsoleto — la única pieza a cambiar sería `MLFLOW_TRACKING_URI` y quitar `azureml-mlflow`, ya que el resto del código usa la API estándar de MLflow.
