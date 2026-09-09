# ADR 0009: Convención de logging en MLflow

- Estado: aceptada
- Fecha: 2026-08-26

## Contexto

[ADR 0005](0005_mlflow_tracking.md) definió que se usa MLflow sobre Azure ML para tracking, y dejó el mecanismo de conexión (`configure_mlflow`) listo. Pero conectar no es suficiente: si cada quien decide a mano qué loguear (un run con solo la métrica primaria, otro con todas; un run sin tags, otro con quién sabe qué), los runs no son comparables entre sí, y se pierde buena parte del valor de tener un tracking centralizado. El usuario pidió explícitamente que la convención sea completa — que nadie tenga que "pedir nada más" después de seguirla — cubriendo métricas, parámetros y artefactos, no solo el mecanismo de conexión.

## Decisión

Se agrega [`julieta.tracking.mlflow_config.log_run`](../../src/julieta/tracking/mlflow_config.py), que encapsula toda la convención en una sola llamada en vez de dejarla como una lista de reglas que cada quien debe recordar:

- **Todos los parámetros** del config usado (ej. el contenido de `experiments/IDXX-*/configs/*.yaml`) se loguean como `params` — no una selección arbitraria.
- **Todas las métricas calculadas**, no solo `metrics.primary_metric_name` de `metadata.yaml` — para poder comparar runs por cualquier métrica, no solo la principal.
- **Tags automáticos**: `author` (lo pasa quien llama, normalmente `metadata.yaml.author` o el colaborador que corrió ese run) y `git_commit` (detectado automáticamente vía `git rev-parse HEAD`) — así cualquiera puede ir de un run en MLflow al código exacto que lo generó, sin tener que preguntarle a nadie.
- **`artifacts`**: lista de rutas de archivo (plots, el config usado, el modelo entrenado) que se suben al run. **Aplica la misma regla de [docs/architecture/data_governance.md](../architecture/data_governance.md)**: nunca una muestra cruda de datos sensibles como artefacto — se agregó una sección nueva a ese documento extendiendo la regla explícitamente a MLflow.

### Hallazgos reales de la prueba local (no solo diseño en papel)

Se probó `log_run` de punta a punta contra un backend de MLflow local (sin esperar al recurso de Azure ML), y salieron dos cosas que no se sabían de antemano:

1. **El backend de archivos (`file:./mlruns`) está en modo mantenimiento** en la versión de MLflow que usa este repo — falla con una excepción explícita a menos que se fuerce con `MLFLOW_ALLOW_FILE_STORE=true`. Para pruebas locales, usar backend SQLite en su lugar: `MLFLOW_TRACKING_URI=sqlite:///ruta/a/mlflow.db`.
2. **El backend SQLite guarda los artefactos en `./mlruns` relativo al directorio donde se ejecuta el comando, no donde vive la base de datos.** Si se corre una prueba local sin fijarse en el directorio de trabajo, deja una carpeta `mlruns/` suelta donde sea que se haya corrido — pasó exactamente eso al probar esto, y tuvo que limpiarse a mano. Por eso `mlruns/`, `mlflow.db` y `*.db-journal` se agregaron a `.gitignore` como red de seguridad, y cualquier prueba local de este mecanismo debe correrse con `cwd` explícito en una carpeta de scratch, no en la raíz del repo.

Cuando exista el Azure ML Workspace, este problema no aplica — Azure ML gestiona su propio almacenamiento de artefactos, no depende del directorio local desde donde se corre nada.

## Consecuencias

- Nadie tiene que memorizar la convención para seguirla — llamar `log_run` con los parámetros correctos ya cumple todo. Si alguien loguea a mano con `mlflow.log_metric` directo en vez de usar `log_run`, se pierde la garantía de "todas las métricas, todos los parámetros, tags automáticos" — `log_run` debería ser el único punto de entrada normal para loguear un run completo.
- La detección automática del commit de git asume que el código corre dentro de un checkout de este repo con git disponible — si falla (no es un repo git, o `git` no está instalado), el tag `git_commit` queda como `"unknown"` en vez de romper el run completo.
- Falta decidir (no se resuelve en este ADR): si se necesita loguear métricas por paso/época (curvas de entrenamiento) en vez de un valor final, eso se hace llamando `mlflow.log_metric(nombre, valor, step=n)` directo dentro del loop de entrenamiento — `log_run` cubre el caso de un valor final por métrica, no series de tiempo.
