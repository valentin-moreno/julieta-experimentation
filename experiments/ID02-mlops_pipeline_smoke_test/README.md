# ID02 - mlops_pipeline_smoke_test

## Resumen

Smoke test end-to-end del pipeline de MLOps del repo: datos sintéticos → DVC → entrenamiento con código real de `src/julieta` → tracking en MLflow/Azure ML → metadata/resumen del repo. No es un experimento de negocio — es la validación de que la plumbing completa funciona antes de que el equipo la use con datos reales.

## Motivación

El repo tenía las tres integraciones de Azure (DVC, MLflow, y CI/CD) documentadas pero nunca ejercitadas de punta a punta: cero datasets en DVC, cero runs en MLflow, cero experimentos reales corridos por el flujo completo (ver `experiments/ID01-*`, que quedó como contenido semilla sin terminar). Antes de pedirle a un científico que confíe en este flujo para un experimento real, valía la pena probarlo con algo desechable y sin ningún dato sensible de por medio.

## Hipótesis

El pipeline completo — `dvc add`/`dvc push` sobre el remote de Azure Blob, entrenamiento reusando `julieta.data.split_data`, `julieta.models.classifiers` y `julieta.models.metrics`, logging con `julieta.tracking.mlflow_config.log_run` hacia el Azure ML Workspace, y `julieta-validate`/`julieta-summary` sobre este `metadata.yaml` — corre sin intervención manual y deja un run real, visible en Azure ML Studio.

## Datos

- Dataset: sintético, generado con `sklearn.datasets.make_classification` (2000 filas, 12 features numéricas sin nombre de negocio, ~30% clase positiva). `dataset_version: synthetic_v1`.
- Fuente: ninguna — no viene de ningún sistema de Salva Health.
- Nivel de sensibilidad: **Nivel 0 (no sensible)** según [docs/architecture/data-governance.md](../../docs/architecture/data-governance.md) — no contiene ningún dato real ni de pacientes, así que no aplica ningún proceso de anonimización. El CSV se generó y se subió directo a `data/raw/` desde el notebook (sin pasar por `pipelines/data/download`, que sigue vacío) porque no hay ninguna fuente externa que ingerir — cuando exista un dataset real, ese sí debe entrar por ese pipeline.

## Metodología

Modelo: `julieta.models.classifiers.BalancedXGBClassifier` (XGBoost con sample weights balanceados automáticos — el dataset sintético es ~70/30, similar en proporción a un dataset clínico real desbalanceado). Split train/test 80/20 estratificado con `julieta.data.split_data.DataSplitter`. Métricas con `julieta.models.metrics.ClassificationMetrics`. Sin tuning de hiperparámetros — no es el objetivo de este smoke test.

## Métricas de éxito

- Métrica primaria: `roc_auc`.
- Objetivo (`target_value`): 0.85 — umbral arbitrario, solo para tener un target de referencia como cualquier experimento real.
- Valor obtenido: ver `metrics.final_value` en `metadata.yaml` y la sección Resultados abajo.

## Estructura

- `metadata.yaml` / `configs/baseline.yaml`: metadata del experimento e hiperparámetros usados.
- `notebooks/01_synthetic_smoke_test.ipynb`: notebook único (no hace falta separar EDA/features/modeling para un smoke test) que genera el dataset, lo sube a DVC, entrena, evalúa train/test, loguea el run a MLflow y genera el reporte.
- `results/`: `confusion_matrix_{train,test}.png`, `classification_report_{train,test}.png` y `run_summary.json` (run_id, run_name y las 16 métricas obtenidas) — no se versiona en git (ver `.gitignore`), es regenerable corriendo el notebook.
- `reports/report.html`: reporte autocontenido (metadata + métricas + las 4 imágenes embebidas en base64) generado automáticamente por `julieta.utils.generate_experiment_report.build_report` — ver [utils/README.md](../../src/julieta/utils/README.md). Tampoco se versiona; se regenera solo o a mano con `uv run julieta-report experiments/ID02-mlops_pipeline_smoke_test`.

## Cómo reproducir

1. `uv sync --all-groups` y `az login` (ver Setup en el [README](../../README.md) del repo).
2. Abrir y correr `notebooks/01_synthetic_smoke_test.ipynb` de principio a fin (o `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_synthetic_smoke_test.ipynb`).
3. El notebook deja el run logueado en el workspace de Azure ML `ml-salva-dev`, experimento `julieta/IDXX-mlops_pipeline_smoke_test` (el nombre real usa el ID ya asignado, no `IDXX`), y genera `reports/report.html`.
4. Para regenerar solo el reporte sin volver a entrenar (ej. si cambiaste metadata.yaml): `uv run julieta-report experiments/ID02-mlops_pipeline_smoke_test`.

## Resultados

Métricas completas, train vs. test (todas las de `ClassificationMetrics` más `accuracy` y `julieta_score`, la métrica propia del repo):

| Métrica | Train | Test |
|---|---|---|
| roc_auc (primaria, target 0.85) | 1.00 | **0.94** |
| accuracy | 0.98 | 0.87 |
| f1_macro | 0.98 | 0.85 |
| sensitivity | 1.00 | 0.85 |
| specificity | 0.98 | 0.88 |
| ppv | 0.95 | 0.75 |
| npv | 1.00 | 0.93 |
| julieta_score | 0.99 | 0.86 |

Train casi perfecto vs. test más bajo es overfitting esperable (200 árboles, profundidad 4, sin tuning, sobre un dataset sintético fácil de separar) — no invalida el smoke test, es justo la clase de señal que comparar train/test debe mostrar.

Run de MLflow: `run_id 6d95d186-77f5-436d-adf6-140d0ccbc6a2`, **run_name `baseline`** (antes salía con un nombre random tipo `loyal_picture_m0jqptwt` — se corrigió agregando el parámetro `run_name` a `julieta.tracking.mlflow_config.log_run`, ver [ADR 0009](../../docs/decisions/0009-mlflow-logging-convention.md)), status `FINISHED`, visible en el workspace de Azure ML `ml-salva-dev` bajo el experimento `julieta/ID02-mlops_pipeline_smoke_test` — con las 16 métricas de arriba (8 train + 8 test), los params de `configs/baseline.yaml`, y 4 artifacts: `confusion_matrix_{train,test}.png` y `classification_report_{train,test}.png`. Tags automáticos (`author`, `git_commit`) y manuales (`purpose`, `data_sensitivity`) presentes y correctos.

`data/raw/synthetic_smoke_test.csv.dvc` quedó creado por `dvc add`, y `dvc push` subió el contenido al remote `azure-storage` sin error.

## Conclusiones y siguientes pasos

El pipeline completo corre de punta a punta sin intervención manual: generar datos → `dvc add`/`dvc push` → entrenar con código real de `src/julieta` → evaluar train y test con todas las métricas disponibles → `log_run` a MLflow/Azure ML con un nombre de run legible → `julieta-validate`/`julieta-summary` reconocen el experimento. Esto confirma que la infraestructura (DVC + MLflow) está lista para que el equipo la use con un experimento real — ver [ID01](../ID01-prioritization_new_feature/README.md), que sigue pendiente de completarse.

Siguiente paso natural: que un experimento real del equipo (no este smoke test) pase por el mismo flujo. El CI/CD (Azure Pipelines) sigue sin conectar — ver sección CI/CD del [README](../../README.md) del repo — así que por ahora este flujo se valida corriendo el notebook localmente, no vía PR con checks automáticos.

## Limitaciones y riesgos conocidos

Es un smoke test con datos sintéticos — que esto funcione no valida nada sobre calidad de modelo ni sobre datos reales, solo que la infraestructura (DVC, MLflow, metadata) está lista para usarse. El pipeline de CI/CD (Azure Pipelines) sigue sin conectar — ver sección CI/CD del [README](../../README.md) del repo — así que este experimento se corrió y valida solo localmente, no vía PR.

## Referencias

- [ADR 0005 — MLflow tracking](../../docs/decisions/0005-mlflow-tracking.md)
- [ADR 0007 — DVC data versioning](../../docs/decisions/0007-dvc-data-versioning.md)
- [ADR 0009 — MLflow logging convention](../../docs/decisions/0009-mlflow-logging-convention.md)
