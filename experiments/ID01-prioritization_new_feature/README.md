# ID01 - prioritization_new_feature

## Resumen

Experimento en curso (`status: ongoing`): evaluar si agregar features temporales de actividad de usuario mejora el ROC-AUC del modelo de priorización de usuarios (XGBoost) sobre el baseline actual. Hay un `final_value: 0.83` registrado en `metadata.yaml`, pero el estado sigue en `ongoing` y `conclusion` dice "en progreso" — **confirmar si ese 0.83 es un resultado parcial o si el experimento ya debería marcarse `completed`.**

## Motivación

[completar: qué problema de negocio/producto motiva mejorar la priorización de usuarios, quién lo pidió, qué pasa si no se hace. `metadata.yaml` no tiene esta información — no se puede inferir con confianza, hay que llenarlo con contexto real.]

## Hipótesis

Agregar features temporales de actividad de usuario mejora el ROC-AUC del modelo de priorización (XGBoost) hasta alcanzar o superar 0.85 sobre el baseline actual (derivado de `objective` y `metrics.target_value` en `metadata.yaml` — confirmar que la afirmación es exacta).

## Datos

- Dataset: `v2.1_active_users` (`dataset_version` en `metadata.yaml`).
- Fuente y período que cubre: [completar].
- Nivel de sensibilidad y confirmación de que se siguió [docs/architecture/data_governance.md](../../docs/architecture/data_governance.md) — ¿el dataset ya llegó anonimizado? [completar — no hay registro de esto todavía].

## Metodología

Modelo: `xgboost_classifier`. Features consideradas: temporales de actividad de usuario (`tags: temporal_features`), sobre el set existente usado para priorización. [completar: diseño de validación — train/test split o cross-validation, y cómo se evita data leakage con las features temporales nuevas].

## Métricas de éxito

- Métrica primaria: `roc_auc`.
- Objetivo (`target_value`): 0.85.
- Valor actual (`final_value`): 0.83 — **ver nota en Resumen sobre la inconsistencia con `status: ongoing`.**

## Estructura

- `metadata.yaml`: información básica, estado y métricas del experimento. `type` clasifica qué clase de trabajo es (`experiment`/`data_report`/`analysis`) y `objective` resume en una frase qué se busca lograr — junto con `conclusion`, forma el par "qué buscaba → qué encontré" legible desde `julieta_summary` sin abrir este README. Ver [ADR 0008](../../docs/decisions/0008_experiment_type_and_objective.md).
- `configs/`: configuración usada por el experimento (hiperparámetros, paths, etc.). Nombra cada archivo por lo que prueba, no por quién lo hizo: `baseline.yaml`, `xgboost_more_depth.yaml`. Si dos colaboradores tunean variantes en paralelo, antepón tu nombre: `maria_xgboost_v2.yaml`.
- `notebooks/`: sin subcarpetas fijas, prefijo numérico para el orden de ejecución (ver [ADR 0001](../../docs/decisions/0001_flatten_experiment_notebooks.md)). **Está vacía todavía** — no hay notebooks commiteados pese a que el experimento ya está `ongoing`.
- `results/` / `reports/`: artefactos y reportes (no se versiona el contenido, solo la carpeta).

## Cómo reproducir

[completar: pasos concretos — comandos, en qué orden correr los notebooks, configuración necesaria. No se puede documentar todavía porque `notebooks/` está vacía.]

## Resultados

`final_value: 0.83` registrado en `metadata.yaml`, por debajo del `target_value` de 0.85. No hay gráficos ni tabla de resultados en `reports/` todavía. [completar cuando haya más avance, y aclarar si este 0.83 es parcial o definitivo — ver nota en Resumen.]

## Conclusiones y siguientes pasos

"En progreso. Extrayendo datos temporales para el baseline." (`conclusion` en `metadata.yaml`). [completar: próximos pasos concretos, y si el experimento sigue, se pausa, o se marca `completed`.]

## Limitaciones y riesgos conocidos

[completar.]

## Referencias

[completar: PRs, tickets o experimentos previos relacionados, si los hay.]
