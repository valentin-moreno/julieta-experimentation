# ADR 0002: Separación entre `pipelines/` y `src/julieta/`

- Estado: aceptada
- Fecha: 2026-08-24

## Contexto

El repo tiene dos jerarquías de carpetas que a simple vista parecen solaparse: `pipelines/data/{download,build,validation}` + `pipelines/training`, y `src/julieta/data/{ingestion,loaders,validation}` + `src/julieta/{features,models}`. Sin explicar la relación, el nombre repetido `validation` en ambos lados en particular lee como duplicado.

## Decisión

- **`src/julieta/`** es la librería: funciones y clases reutilizables, puras en la medida de lo posible, testeables de forma aislada. No decide cuándo correr ni con qué configuración — solo implementa "cómo se hace algo".
- **`pipelines/`** es la orquestación: scripts ejecutables que leen configuración, encadenan llamadas a `src/julieta/`, y escriben resultados a `data/` o `models/`. Deciden "cuándo y con qué se hace".

Mapeo concreto entre cada pipeline y la librería que usa:

| Pipeline | Qué hace | Usa de `src/julieta` |
|---|---|---|
| `pipelines/data/download` | Trae datos crudos desde una fuente externa y los guarda en `data/raw/`. | `data/ingestion` (funciones que saben *cómo* traer datos de una fuente dada) |
| `pipelines/data/validation` | Corre checks de calidad sobre `data/raw` o `data/processed` como paso de un flujo, y falla/reporta si algo no cumple. | `data/validation` (funciones puras de chequeo: nulls, tipos, rangos — reciben datos, devuelven resultado) |
| `pipelines/data/build` | Toma `data/raw`, aplica transformaciones y produce `data/processed`. | `data/loaders` (leer datos de disco a memoria) + `features` (transformaciones de feature engineering) |
| `pipelines/training` | Entrena, evalúa y guarda un modelo candidato en `models/candidates/`. | `models` (wrappers de entrenamiento/evaluación) |

`src/julieta/utils` y `src/julieta/visualization` no tienen una pipeline dedicada: `utils` incluye herramientas de gestión del propio repo (`generate_summary.py`, `validate_metadata.py`), y `visualization` se usa tanto desde notebooks como desde cualquier pipeline que necesite generar gráficos/reportes.

Cada carpeta bajo `pipelines/` y `src/julieta/` tiene un `README.md` de una línea señalando esta relación, para que sea explícita sin depender de este ADR.

## Consecuencias

- Un mismo concepto (ej. "validación de datos") puede tener representación en ambos lados sin ser redundante: la definición del check vive en `src/julieta`, el "correrlo como parte de un flujo" vive en `pipelines`.
- Si en el futuro un pipeline no encaja en este mapeo (por ejemplo, necesita lógica que no es reutilizable en ningún otro lado), es una señal de que puede vivir directamente en `pipelines/` sin pasar por `src/julieta/` — no todo tiene que extraerse a librería antes de tener valor.
- Esta separación todavía no está probada con código real (todas las carpetas involucradas están vacías); si al implementar la primera pipeline el mapeo no cuadra, este ADR debe actualizarse o reemplazarse.
