# ID02 - mlops_pipeline_smoke_test

## Resumen

Smoke test end-to-end del pipeline de MLOps del repo: datos sintéticos → DVC → entrenamiento con código real de `src/julieta` → tracking en MLflow/Azure ML → metadata/resumen del repo. No es un experimento de negocio — es la validación de que la plumbing completa funciona antes de que el equipo la use con datos reales.

## Motivación

El repo tenía las tres integraciones de Azure (DVC, MLflow, y CI/CD) documentadas pero nunca ejercitadas de punta a punta: cero datasets en DVC, cero runs en MLflow, cero experimentos reales corridos por el flujo completo (ver `experiments/ID01-*`, que quedó como contenido semilla sin terminar). Antes de pedirle a un científico que confíe en este flujo para un experimento real, valía la pena probarlo con algo desechable y sin ningún dato sensible de por medio.

## Hipótesis

El pipeline completo — `dvc add`/`dvc push` sobre el remote de Azure Blob, entrenamiento reusando `julieta.data.split_data`, `julieta.models.classifiers` y `julieta.models.metrics`, logging con `julieta.tracking.mlflow_config.log_run` hacia el Azure ML Workspace, y `julieta_validate`/`julieta_summary` sobre este `metadata.yaml` — corre sin intervención manual y deja un run real, visible en Azure ML Studio.

## Datos

- Dataset: sintético, generado con `sklearn.datasets.make_classification` (2000 filas, 12 features numéricas sin nombre de negocio, ~30% clase positiva). `dataset_version: dataset-synthetic_smoke_test-v1` — ese valor es un tag real de git (no solo texto descriptivo): `git checkout dataset-synthetic_smoke_test-v1 -- data/raw/synthetic_smoke_test.csv.dvc && dvc pull` recupera exactamente este archivo.
- Fuente: ninguna — no viene de ningún sistema de Salva Health.
- Nivel de sensibilidad: **Nivel 0 (no sensible)** según [docs/architecture/data_governance.md](../../docs/architecture/data_governance.md) — no contiene ningún dato real ni de pacientes, así que no aplica ningún proceso de anonimización. El CSV se generó y se subió directo a `data/raw/` desde el notebook (sin pasar por `pipelines/data/download`, que sigue vacío) porque no hay ninguna fuente externa que ingerir — cuando exista un dataset real, ese sí debe entrar por ese pipeline.

## Metodología

Split train/test 80/20 estratificado con `julieta.data.split_data.DataSplitter` — el mismo split para los 3 clasificadores, comparación justa. Métricas con `julieta.models.metrics.ClassificationMetrics`. Sin tuning de hiperparámetros — no es el objetivo de este smoke test.

**3 clasificadores comparados, cada uno como su propio run de MLflow** (mismo experimento, `run_name` = la clave del clasificador en `configs/baseline.yaml`) — no es solo un smoke test de DVC/MLflow, también valida el patrón de comparar varios modelos a la vez:

- `xgboost` → `julieta.models.classifiers.BalancedXGBClassifier` (XGBoost con sample weights balanceados automáticos — el dataset sintético es ~70/30, similar en proporción a un dataset clínico real desbalanceado).
- `logistic_regression` → `sklearn.linear_model.LogisticRegression`.
- `random_forest` → `sklearn.ensemble.RandomForestClassifier`.

## Métricas de éxito

- Métrica primaria: `roc_auc`.
- Objetivo (`target_value`): 0.85 — umbral arbitrario, solo para tener un target de referencia como cualquier experimento real.
- Valor obtenido: ver `metrics.final_value` en `metadata.yaml` y la sección Resultados abajo.

## Estructura

- `metadata.yaml` / `configs/baseline.yaml`: metadata del experimento e hiperparámetros usados.
- `notebooks/01_synthetic_smoke_test.ipynb`: notebook único (no hace falta separar EDA/features/modeling para un smoke test) que genera el dataset, lo sube a DVC, entrena y evalúa los 3 clasificadores, y loguea cada uno a MLflow.
- `results/`: por clasificador, `confusion_matrix_{train,test}_<clasificador>.png`, `classification_report_{train,test}_<clasificador>.png` — no se versiona en git (ver `.gitignore`), es regenerable corriendo el notebook. Estas imágenes viven solo aquí y como artifact de MLflow — nunca se comitean ni se duplican en ningún otro lado.

## Cómo reproducir

1. `uv sync --all-groups` y `az login` (ver Setup en el [README](../../README.md) del repo).
2. Abrir y correr `notebooks/01_synthetic_smoke_test.ipynb` de principio a fin (o `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_synthetic_smoke_test.ipynb`).
3. El notebook deja el run logueado en el workspace de Azure ML `ml-salva-dev`, experimento `julieta/IDXX-mlops_pipeline_smoke_test` (el nombre real usa el ID ya asignado, no `IDXX`), visible en Azure ML Studio.

## Resultados

Test roc_auc por clasificador (métrica primaria, target 0.85), cada uno su propio run de MLflow, mismo train/test:

| Clasificador | run_id | train_roc_auc | test_roc_auc | test_accuracy | test_julieta_score |
|---|---|---|---|---|---|
| **xgboost** (gana) | `d178a55d` | 1.00 | **0.94** | 0.87 | 0.86 |
| random_forest | `c90a94ff` | 0.97 | 0.92 | 0.82 | 0.71 |
| logistic_regression | `4dced268` | 0.80 | 0.80 | 0.81 | 0.71 |

Los 3 superan el target de 0.85 salvo logistic_regression (0.80) — comparación creíble, no forzada para que todos ganen. xgboost y random_forest muestran train >> test (overfitting esperable, sin tuning, dataset sintético fácil de separar); logistic_regression no, porque es un modelo lineal con menos capacidad de sobreajustar aquí. Métricas completas (16 por run: 8 train + 8 test, incluyendo `f1_macro`, `sensitivity`, `specificity`, `ppv`, `npv`, `julieta_score`) en cada run de MLflow/Azure ML.

Los 3 runs están `FINISHED` en el workspace de Azure ML `ml-salva-dev`, mismo experimento `julieta/ID02-mlops_pipeline_smoke_test`, cada uno con su `run_name` (`xgboost`/`logistic_regression`/`random_forest` — ver [ADR 0009](../../docs/decisions/0009_mlflow_logging_convention.md) sobre por qué nombrar el run), sus propios params (`configs/baseline.yaml`, sección `models.<clasificador>`) y sus 4 artifacts. Tags automáticos (`author`, `git_commit`) y manuales (`purpose`, `data_sensitivity`, `model_key`) presentes y correctos en los 3.

`data/raw/synthetic_smoke_test.csv.dvc` quedó creado por `dvc add`, y `dvc push` subió el contenido al remote `azure-storage` sin error.

**Bug real encontrado corriendo esto**: `julieta.tracking.mlflow_config` usaba `load_dotenv()`, que vuelca *todo* `.env` al proceso — un `dvc push` corrido como subprocess desde el mismo notebook heredaba `AZURE_STORAGE_CONNECTION_STRING` (de una cuenta de Azure distinta, para otro propósito) y DVC la priorizaba sobre la sesión de `az login`, apuntando a la cuenta equivocada y fallando con un error engañoso (`No such container: 'dvc-storage'` — el container sí existe, solo que no en esa otra cuenta). Corregido a `dotenv_values()` (lee `.env` a un dict local, sin tocar el resto del proceso), con test de regresión en `test_mlflow_config.py`.

## Conclusiones y siguientes pasos

El pipeline completo corre de punta a punta sin intervención manual: generar datos → `dvc add`/`dvc push` → entrenar con código real de `src/julieta` → evaluar train y test con todas las métricas disponibles → `log_run` a MLflow/Azure ML con un nombre de run legible → `julieta_validate`/`julieta_summary` reconocen el experimento. Esto confirma que la infraestructura (DVC + MLflow) está lista para que el equipo la use con un experimento real — ver [ID01](../ID01-prioritization_new_feature/README.md), que sigue pendiente de completarse.

Siguiente paso natural: que un experimento real del equipo (no este smoke test) pase por el mismo flujo. El CI/CD (Azure Pipelines) sigue sin conectar — ver sección CI/CD del [README](../../README.md) del repo — así que por ahora este flujo se valida corriendo el notebook localmente, no vía PR con checks automáticos.

## Limitaciones y riesgos conocidos

Es un smoke test con datos sintéticos — que esto funcione no valida nada sobre calidad de modelo ni sobre datos reales, solo que la infraestructura (DVC, MLflow, metadata) está lista para usarse. El pipeline de CI/CD (Azure Pipelines) sigue sin conectar — ver sección CI/CD del [README](../../README.md) del repo — así que este experimento se corrió y valida solo localmente, no vía PR.

## Referencias

- [ADR 0005 — MLflow tracking](../../docs/decisions/0005_mlflow_tracking.md)
- [ADR 0007 — DVC data versioning](../../docs/decisions/0007_dvc_data_versioning.md)
- [ADR 0009 — MLflow logging convention](../../docs/decisions/0009_mlflow_logging_convention.md)
