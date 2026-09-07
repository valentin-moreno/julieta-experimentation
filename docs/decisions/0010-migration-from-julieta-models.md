# ADR 0010: Migración fase 1 de código reusable desde julieta-models

- Estado: aceptada
- Fecha: 2026-08-26

## Contexto

`julieta-models` (`github.com/Salva-Health/julieta-models`) es el repo viejo del equipo, con ~2 años de trabajo real sobre diagnóstico de cáncer de mama por impedancia bioeléctrica. Se exploró todo `src/julieta` (30 archivos), `pipelines/` (39 archivos), `docs/`, `notebooks/` (137 notebooks + un store local de MLflow no trackeado, 341MB) y las dependencias reales (`requirements.txt`, 250 paquetes) para decidir qué migrar a `julieta-experimentation` sin arrastrar deuda técnica, datos sensibles, ni infraestructura obsoleta.

Se confirmaron dos decisiones de alcance antes de tocar código:
- `pipelines/online_prediction/` (33 archivos, inferencia en producción) queda **fuera de esta migración** — consistente con [ADR 0006](0006-model-candidate-lifecycle.md): este repo es de experimentación, no de serving.
- Fase 1 **solo porta código genérico y limpio**, sin acople al esquema clínico viejo (nombres de campo en español, taxonomía BI-RADS hardcodeada). El resto queda de referencia para fases futuras, no se copia.

## Decisión

### Se portaron 7 archivos a `src/julieta/`

| Origen | Destino | Cambios al portar |
|---|---|---|
| `features/signal_complexity.py` | `features/signal_complexity.py` | Se quitaron ~120 líneas de código muerto comentado. |
| `data/split_data.py` | `data/split_data.py` | Ninguno — se preservó la atribución de licencia MIT. |
| `models/collapsing_classifier.py` | `models/collapsing_classifier.py` | Ninguno. |
| `models/classifiers.py` | `models/classifiers.py` | Se generalizó una referencia a un modelo específico del repo viejo en el docstring. |
| `models/metrics.py` | `models/metrics.py` | Ver bugs corregidos abajo. Se parametrizó `positive_classes` (antes hardcodeado a `class 2`) y el default de `class_names` en `plot_confusion_matrix`. Se limpiaron imports duplicados y un self-import roto. |
| `models/model_settings.py` | `models/model_settings.py` | Ver bug corregido abajo. Se quitaron 7 estimadores comentados (nunca activos) y sus imports sin uso. |
| `visualization/projections.py` | `visualization/projections.py` | Se quitaron ~140 líneas de una versión vieja de `visualize_projection` comentada, reemplazada por la activa. |

### Se excluyó `data/data_synthesis.py`

Dependía de `non-parametric-multivariate-data-generator` (PyPI). Se verificó antes de decidir: el paquete no tiene descripción real (`"A small example package"` — literalmente el texto de ejemplo del tutorial oficial de PyPI), sin autor listado, solo 3 releases. No es una dependencia confiable para código de librería. Este archivo queda fuera de fase 1; si se necesita más adelante, se reimplementa el mismo patrón (oversampling con cópulas) sobre una librería con procedencia verificable.

### Bugs reales encontrados y corregidos durante la migración (no solo copiados)

- `models/metrics.py`: `MultyClassificationMetrics.positive_likelihood_ratio()`/`negative_likelihood_ratio()` dividían listas de Python directo (`sens / (1 - spec)`, ambos listas) — esto explota con `TypeError` en cualquier caso multiclase real. Se corrigió convirtiendo a `np.array` y usando `np.where` para división segura. Hay un test que reproduce el caso multiclase para que esto no se rompa de nuevo en silencio.
- `models/model_settings.py`: `ModelConfig.get_estimator_config()` llamaba `self.estimators[self.estimator_name]()`, pero `self.estimators` ya guarda instancias, no clases — esto explota con `TypeError: '<Estimator>' object is not callable` en cualquier uso real. Corregido quitando el `()` de más.

Ambos bugs significan que, tal como estaba en el repo viejo, `ModelConfig.get_estimator_config()` y las métricas multiclase de likelihood ratio nunca funcionaron si alguien las hubiera llamado — el código muerto/no ejercitado escondía esto.

### Dependencias nuevas en `pyproject.toml`

`scipy`, `matplotlib`, `seaborn`, `sklearn-genetic-opt` (import `sklearn_genetic`), `umap-learn` (import `umap`). `lightgbm` **no** se agregó — estaba importado en el archivo viejo pero completamente sin uso (el único estimador que lo necesitaba estaba comentado); se puede agregar cuando alguien reactive ese estimador de verdad.

## Consecuencias

- `src/julieta/` deja de ser solo `__init__.py` + `utils/` (herramientas de gestión del repo) — ya tiene la primera librería de ML real, con tests que la ejercitan.
- Los 7 archivos portados quedan sujetos a las mismas reglas que todo lo demás en este repo: lint, formato, tests, y el hook de datos sensibles — ya se verificaron contra los tres.
- Las fases futuras (si se decide migrar algo de la lista de exclusión: `make_dataset.py`, `compute_features.py`, `clean_data.py`, etc.) van a requerir más trabajo de adaptación, no solo copiar — esos archivos están mucho más acoplados al esquema clínico viejo, como ya se documentó en la exploración previa a este ADR.
- `pipelines/` sigue sin código real — lo portado en esta fase es librería (`src/julieta/`), no pipelines de orquestación. Sigue pendiente escribir el primer pipeline real que use este código, cuando haya un experimento real que lo necesite.
