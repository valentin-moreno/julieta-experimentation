# ID03 - healthy_model_groupsplit

## Resumen

Comparación de 7 clasificadores para priorizar pacientes por riesgo (BI-RADS binarizado: clase 2 = alta sospecha vs 0+1 = resto), sobre features de bioimpedancia Cole-Cole/Nyquist. Es el primer experimento **real** de este repo (ID02 fue un smoke test sintético) — portado desde `julieta-models/notebooks/modeling/VMV-healthy_model_experimentos_groupsplit.ipynb`.

## Motivación

Con la infraestructura ya probada de punta a punta con datos sintéticos (ID02), hacía falta probarla con un caso real de negocio: un experimento que ya existía en julieta-models, con resultados conocidos, para validar que el repo nuevo puede reproducir (y mejorar) ese flujo — features compartidas y reutilizables, DVC real sobre el dataset de features, y el modelo entrenado logueado en MLflow (no solo métricas).

## Qué cambió al portar este experimento (importante, léase antes de comparar contra el original)

El notebook original leía un CSV ya cacheado (`healthy_advanced_nyquist_dataset_qc.csv`, generado por separado, ~35-40 min de cómputo) y no existía en disco en ningún lado al momento de portar esto — ni en este repo ni en julieta-models. Para poder reproducirlo de verdad, se reconstruyó desde cero:

- **[`pipelines/data/build/build_healthy_advanced_nyquist_dataset.py`](../../pipelines/data/build/build_healthy_advanced_nyquist_dataset.py)** hace lo que antes hacía `pipelines/data/build_training_dataset.py` de julieta-models — reescrito leyendo el comportamiento real del original método por método (no reinventado a ojo), porque:
  - Ese script depende de `data/raw/disconnections.xlsx` (relabeling manual), que **no existe** en ningún lado al momento de portar esto. Se usa en su lugar el archivo de control de calidad por estudio (`disconnections_<estudio>.csv`, que sí se descarga hoy vía `pipelines/data/download/download_clinical_data.py`) — la fuente que el propio código viejo señalaba como la que de verdad importa.
  - Depende de `MakeDataset`, una clase de 2211 líneas de la que solo hacían falta 2 métodos (leer un `.h5` y aplanar un dict a DataFrame) — se leyó su código real y se reimplementó igual en [`src/julieta/data/loaders/impedance_h5.py`](../../src/julieta/data/loaders/impedance_h5.py), **incluyendo el deduplicado por `(patient_id, side)` con `keep="last"`** que el original aplica y que sí cambia el resultado: los `.h5` traen filas duplicadas por paciente (confirmado: 6 en SURA, 144 en CAFAM, 50 en CLINICA_DE_MAMA). Sin este fix, el dataset queda inflado y algunos pacientes pesan más que otros en el entrenamiento sin razón.
- **`src/julieta/features/compute_features.py`** (nuevo, fase 2 de [ADR 0010](../../docs/decisions/0010_migration_from_julieta_models.md)): solo `ComputeColeColeFeatures`, `ComputeAdvancedNyquistFeatures` y `ComputeBreastLevelStatistics` — las 3 piezas maduras que este experimento usa de verdad, de un archivo origen de 3076 líneas y 9 clases (el resto son variantes marcadas como experimentales o sin evidencia de uso real).
- **Las métricas usan `ClassificationMetrics(positive_classes=(2,))`** en vez de `triclass=True` — esa clase no existe con esa forma en este repo (ver [ADR 0010](../../docs/decisions/0010_migration_from_julieta_models.md), que ya portó y corrigió bugs reales de `models/metrics.py`). El efecto es el mismo que buscaba el notebook original: binarizar clase 2 (alta sospecha) como positivo.
- **El modelo ganador ya no se guarda como `.pkl` local** (el notebook original hacía `joblib.dump(...)`, sobrescribiéndolo en cada corrida, sin versionar). Ahora cada uno de los 7 modelos entrenados queda logueado en su propio run de MLflow (parámetro `model` nuevo en `log_run`, serializado con `joblib` y subido como artifact — no con `mlflow.sklearn.log_model`, que rompe contra el tracking server de Azure ML, ver conclusión abajo) — recuperable desde la UI, sin depender de un archivo local que se pisa solo.
- **El flujo de carga/features vive dentro del notebook, no en un script aparte** — igual que el original hace con `import build_training_dataset as btd` y llama sus funciones inline. `_load_diagnosis`, `_load_categoricals`, `_load_disconnected_ids`, `_build_signal_features` y `_filter_by_confirmed_laterality` son celdas del notebook (secciones "Load data"/"Load features"), no un import a un pipeline externo. El cálculo pesado de Cole-Cole/RS-CPE sí se cachea en disco por estudio (`pipelines/data/build/.signal_features_cache/`, no versionado) para no repetir ~15 min cada vez que se corre el notebook.

## Datos

- Fuente: SURA, CAFAM, CLINICA_DE_MAMA — categóricos, diagnóstico de mamografía y desconexiones vía la API en vivo (`pipelines/data/download/download_clinical_data.py`), señales de resistencia/reactancia vía los `.h5` cacheados en Azure Blob (`ai-julieta`).
- Nivel de sensibilidad: **Nivel 1/2** (diagnóstico real de pacientes, BI-RADS) — ver [docs/architecture/data_governance.md](../../docs/architecture/data_governance.md). La anonimización sigue **deferida** por decisión explícita anterior del equipo; `patient_id` real (uid de Mongo) vive en `data/raw`/`data/processed`, ambos fuera de git.
- `dataset_version: dataset-healthy_advanced_nyquist-v1` — tag real de git sobre el `.dvc` de `data/processed/healthy_advanced_nyquist_dataset.csv`.

## Metodología

1. Diagnóstico de mamografía por estudio, descartando BI RADS 0 (mamografía incompleta).
2. Categóricos del paciente (edad, copa, talla, peso, anticoncepción/terapia hormonal, menopausia) — con el fix de `height`/`weight`==0 tratado como NaN (dato faltante mal codificado: la mediana de ambas columnas en SURA es 0.0).
3. Exclusión de pacientes con medición no confiable, solo por señal fuerte (nunca por `gaps_detected` solo, demasiado ruidoso).
4. Features de señal: Cole-Cole/RS-CPE/geometría de Nyquist por nodo (47 features/nodo), agregadas por seno con mean/std/cv (~177 columnas), a partir de `resistance_<estudio>.h5` + `reactance_<estudio>.h5`.
5. Cruce por lateralidad confirmada: se queda con el seno que el diagnóstico marcó como afectado (o ambos, si el hallazgo es bilateral).
6. Split 75/25 por `patient_id` (`DataSplitter.split_by_group`) — ambos senos del mismo paciente siempre del mismo lado.
7. 7 clasificadores (SVC, LogisticRegression, RandomForest, XGBoost, XGBoost balanceado, GradientBoosting, KNN), cada uno con su propia búsqueda de hiperparámetros en Optuna (20 trials, semilla fija) optimizando directamente el índice J de Youden vía 5-fold CV.

## Métricas de éxito

- Métrica primaria: `youden_j` (sensibilidad + especificidad - 1) sobre el problema binarizado (clase 2 = positivo).
- Sin `target_value` fijo — el notebook original tampoco declaraba un umbral, es una comparación relativa entre los 7 clasificadores.

## Estructura

Hay **3 notebooks**, todos comparando los mismos 7 clasificadores sobre el mismo problema, con distinta procedencia de código:

- `notebooks/01_healthy_model_groupsplit.ipynb`: **reescritura propia** (resistencia/reactancia). Autocontenido, organizado en las mismas secciones que el original (Libraries, Load data, Load features, Data splitting, Feature selection, Training, Tabla final). Carga diagnóstico/categóricos/desconexiones y calcula (o lee del cache) las features de señal en celdas propias, sin depender de `build_training_dataset.py`.
- `notebooks/02_healthy_model_experimentos_magphase_groupsplit.ipynb`: **código real de julieta-models sin modificar** (magnitud/fase), salvo 2 cambios puntuales para que MLflow apunte a Azure (ver abajo).
- `notebooks/03_healthy_model_experimentos_groupsplit.ipynb`: igual que 02 pero **resistencia/reactancia** — el mismo feature_mode que el notebook histórico original.
- [`pipelines/data/build/build_healthy_advanced_nyquist_dataset.py`](../../pipelines/data/build/build_healthy_advanced_nyquist_dataset.py): la lógica de carga/features de la reescritura propia (01) como script de línea de comandos.
- [`pipelines/data/build_training_dataset.py`](../../pipelines/data/build_training_dataset.py): el script real de julieta-models, portado tal cual (usado por 02/03). Necesita `julieta.data.make_dataset.MakeDataset`, `julieta.data.label_mappings`, `julieta.utils.paths`/`logs` — también portados tal cual (ver ADR 0010).
- `results/`: matrices de confusión y reportes de clasificación (train/test) por clasificador — no versionado, regenerable.

**Los 2 cambios que sí se le hicieron a 02/03** (el resto del código es idéntico al original): agregar `mlflow.set_tracking_uri(...)` (el original no configuraba ninguno, dependía de un mlflow local) y reemplazar `mlflow.sklearn.log_model(...)` por `joblib.dump()` + `mlflow.log_artifact()` (ver bug real más abajo).

**Setup local necesario para correr 02/03** (no versionado en git, `data/raw/*` está en `.gitignore`): los nombres de archivo que espera `build_training_dataset.py` (`mammography_data_<ESTUDIO>.csv`, `categoricals_data_<ESTUDIO>.csv`, `disconnection_<estudio>.csv`) son distintos a los que descarga `download_clinical_data.py` (`mammography_<estudio>.csv`, etc.) — hace falta crear symlinks con el nombre viejo apuntando a los archivos reales, y un `data/raw/disconnections.xlsx` vacío (con columnas `uid, side, tipo, comentarios`) porque ese archivo de relabeling manual no existe en ningún lado.

## Cómo reproducir

1. `uv sync --all-groups`, `az login`.
2. Para 01: correr `notebooks/01_healthy_model_groupsplit.ipynb` de principio a fin. Para 02/03: crear primero los symlinks y el `disconnections.xlsx` vacío (ver arriba), luego correr el notebook.
3. La primera vez por feature_mode tarda ~15-20 min (fits de Cole-Cole/RS-CPE, sin cache); las siguientes corridas son casi instantáneas porque reusan el cache por estudio (`.signal_features_cache/`).

## Resultados

Los 3 notebooks corren de punta a punta contra datos reales y MLflow/Azure ML real:

| Notebook | Código | Feature mode | Filas | Ganador | youden_j |
|---|---|---|---|---|---|
| 01 | Reescritura propia | resistencia/reactancia | 1204 | SVC | 0.49 |
| 02 | Real (julieta-models) | magnitud/fase | 1269 | SVC | 0.47 |
| 03 | Real (julieta-models) | resistencia/reactancia | 1269 | LogisticRegression | 0.50 |
| _Histórico (21-ago-2026)_ | _Real, corrida original_ | _resistencia/reactancia_ | _1241_ | _SVC_ | _0.52_ |

Tabla completa de 03 (resistencia/reactancia, código real — la más comparable al histórico):

| Clasificador | roc_auc | sensibilidad | especificidad | youden_j |
|---|---|---|---|---|
| **LogisticRegression** (gana) | 0.81 | 0.76 | 0.74 | **0.50** |
| SVC | 0.79 | 0.79 | 0.66 | 0.45 |
| RandomForestClassifier | 0.80 | 0.74 | 0.71 | 0.45 |
| XGBClassifier_balanced | 0.81 | 0.54 | 0.82 | 0.36 |
| XGBClassifier | 0.81 | 0.46 | 0.89 | 0.35 |
| GradientBoostingClassifier | 0.80 | 0.46 | 0.89 | 0.35 |
| KNeighborsClassifier | 0.68 | 0.39 | 0.88 | 0.27 |

Todos los runs quedan `FINISHED` en el workspace de Azure ML `ml-salva-dev` (experimento `julieta/ID03-healthy_model_groupsplit`; los runs de 01 se loguearon antes de la asignación de ID real, bajo `julieta/IDXX-healthy_model_groupsplit`), cada uno con su `run_name`, params (mejores hiperparámetros de Optuna), 4 artifacts (matriz de confusión + reporte de clasificación de train/test) y el modelo entrenado (`.joblib`, vía `log_run(model=...)` en 01, o vía `joblib.dump`+`log_artifact` inline en 02/03).

## Comparación contra el notebook original (por qué no da exactamente igual)

El notebook original (corrida real del 2026-08-21, preservada en sus outputs) reportaba 1241 filas y **SVC como ganador** (youden_j 0.52). Investigando esa diferencia se encontró algo importante: `pipelines/data/build_training_dataset.py`, el script del que se porta toda la lógica de carga, **se creó el 2026-08-28 — una semana después** de esa corrida (confirmado con `git log` en julieta-models). El propio notebook original documenta en un markdown que su celda de carga fue reescrita después para usar ese script nuevo, sin volver a correr el notebook completo — es decir, el código que generó los números originales ya no existe en ningún lado, ni en este repo ni en julieta-models.

Con eso claro, se corrió `build_training_dataset.py` real (02/03) en vez de seguir con la reescritura propia (01), y **sí apareció un bug real**: los `.h5` de señal traen filas duplicadas por `(patient_id, side)` (6 en SURA, 144 en CAFAM, 50 en CLINICA_DE_MAMA) que el código original deduplica y la reescritura propia (01) no. Aun así, 02 y 03 dan **1269 filas, no 1241** — 28 pacientes más que la corrida original, consistente con que la base de datos en vivo siguió creciendo entre el 21 de agosto y hoy. Confirmado con 3 corridas limpias consecutivas de 02 que el resultado (1269 filas, SVC 0.47) es estable y reproducible — no es ruido.

## Conclusiones y siguientes pasos

**El pipeline completo corre de punta a punta con datos reales**, tanto con código propio (01) como con el código real de julieta-models sin modificar (02/03) — confirma que la infraestructura de este repo (no solo ID02, que era sintético) sostiene un experimento real de principio a fin, y que es posible correr un notebook heredado con cambios mínimos (solo la configuración de MLflow).

**El resultado no está forzado en ninguno de los 3**: en 01 y 02 gana SVC, no el de mejor `roc_auc`; en 03 gana LogisticRegression con sensibilidad y especificidad balanceadas (0.76/0.74) frente a modelos como XGBoost que sacrifican sensibilidad por especificidad. El "ganador" cambia según la representación de la señal (R/X vs magnitud/fase) y según si el código es el propio o el real, pero el rango de youden_j (0.44-0.50) es consistente entre las 3 corridas y cercano al histórico (0.52).

**Bugs reales encontrados y corregidos en el camino**:
- `log_run(model=...)` (01) y el `mlflow.sklearn.log_model()` original (02/03) rompen contra Azure ML: en `mlflow>=3` siempre intentan crear una entidad "Logged Model" vía `POST /api/2.0/mlflow/logged-models` — un endpoint que el tracking server de Azure ML no implementa (`404` real, confirmado con `artifact_path` y con `name`). Corregido a `joblib.dump()` + `mlflow.log_artifact()` en los 3 notebooks, con test de regresión en `test_mlflow_config.py`.
- Los `.h5` de señal traen pacientes duplicados (ver sección de arriba) — sin deduplicar, el dataset queda inflado y el resultado final cambia de ganador (confirmado: 01 sin este fix daba LogisticRegression 0.46 con 1218 filas).
- `ClassificationMetrics` de este repo no tenía `triclass` (solo `positive_classes`) — se agregó como parámetro de compatibilidad para que 02/03 corrieran sin tocar esa parte del código original.

Siguiente paso natural: que alguien del equipo revise si J de Youden es realmente la métrica correcta a optimizar para este caso de uso (priorización de pacientes), o si el costo real de un falso negativo (clase 2 no detectada) amerita ponderar la sensibilidad por encima de la especificidad en vez de pesarlas igual. También queda pendiente decidir si 01 (reescritura propia) sigue siendo útil una vez que 02/03 (código real) ya están funcionando, o si conviene retirarla.

## Limitaciones y riesgos conocidos

- La búsqueda interna de hiperparámetros (5-fold CV dentro de `train_val`) usa `StratifiedKFold` simple, **no agrupado por paciente** — un mismo paciente (ambos senos) podría caer en folds distintos dentro de esa CV interna. Es una limitación heredada del notebook original, no introducida al portarlo; el split externo train/test sí es por paciente.
- La exclusión por desconexión usa los CSV de control de calidad por estudio descargados hoy de la API en vivo, no el archivo `disconnections.xlsx` de relabeling manual del notebook original (no existe en ningún lado al momento de portar esto) — puede haber una diferencia pequeña en qué pacientes quedan excluidos frente a la corrida original.
- Anonimización deferida (ver `docs/architecture/data_governance.md`) — sigue pendiente antes de considerar este flujo listo para datos de producción sin restricciones de acceso.
- `dropna()` final descarta ~47% de las filas (en 01: 2273 → 1204). Investigado a fondo para 01: **1004 de 1069 filas descartadas (94%) son por una sola causa** — 588 de los 1165 pacientes de SURA (la mitad) nunca tienen `braCupSize`/`height`/`weight` registrados (los 3 a la vez), aunque sí tienen edad y antecedentes hormonales. Es una característica real de esos datos, no un bug: cualquier implementación correcta que use esas 3 columnas como feature tendría que descartar a los mismos pacientes. El resto son fits de Cole-Cole/RS-CPE que no convergieron en algún nodo. No se repitió este análisis para 02/03 (código real).

## Referencias

- [ADR 0009 — MLflow logging convention](../../docs/decisions/0009_mlflow_logging_convention.md)
- [ADR 0010 — Migración desde julieta-models](../../docs/decisions/0010_migration_from_julieta_models.md)
- `julieta-models/notebooks/modeling/VMV-healthy_model_experimentos_groupsplit.ipynb` (original, base de 01 y 03)
- `julieta-models/notebooks/modeling/VMV-healthy_model_experimentos_magphase_groupsplit.ipynb` (original, base de 02)
