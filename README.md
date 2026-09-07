# julieta-experimentation

Repositorio de experimentación ML organizado con prácticas MLOps: cada experimento queda documentado y es reproducible, el código reutilizable vive en un paquete instalable, y los pipelines de datos/entrenamiento están separados de la exploración.

## Estructura

```
data/            # raw / interim / processed — contenido ignorado en git, solo se versiona la estructura
experiments/     # un directorio por experimento (ver más abajo)
pipelines/       # código productivo de datos y entrenamiento (data/build, data/download, data/validation, training)
models/          # candidates: modelos candidatos generados por experimentos/pipelines (no versionados en git)
src/julieta/     # paquete instalable: data, features, models, utils, visualization
tests/           # unit, integration
docs/
  architecture/    # diagramas, descripciones de arquitectura y gobernanza de datos
  decisions/       # ADRs — decisiones de diseño del repo, con contexto y consecuencias
  experimentation/ # guías de metodología de experimentación
```

## Quickstart

1. Instala `uv` si no lo tienes (una sola vez por máquina):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clona el repo e instala todo (`uv` descarga el Python correcto si no lo tienes, crea `.venv/` e instala las versiones exactas fijadas en `uv.lock`):
   ```bash
   git clone <url-del-repo>
   cd julieta-experimentation
   uv sync --all-groups
   ```
3. Activa los hooks de git (ruff, nbstripout, etc. corren automáticamente en cada commit local):
   ```bash
   uv run pre-commit install
   ```
4. Verifica que todo quedó bien:
   ```bash
   uv run pytest -q
   uv run julieta-validate
   ```

No hace falta activar el entorno manualmente: cualquier comando se corre con `uv run <comando>`. Si prefieres no escribir `uv run` cada vez, `source .venv/bin/activate`.

Comandos disponibles:

```bash
uv run julieta-validate     # valida experiments/*/metadata.yaml contra el schema
uv run julieta-summary      # genera experiments_summary.csv con todos los experimentos
uv run julieta-assign-ids   # asigna ID real a experimentos IDXX-* pendientes (lo corre el pipeline, no tú)
uv run pytest               # corre los tests
uv run ruff check .         # lint
```

## Datos sensibles

Este es un repositorio de una empresa de salud: **ningún dato identificable o de salud sin anonimizar puede tocar este repo** (ni `data/`, ni notebooks, ni resultados), sin importar que esas carpetas ya estén en `.gitignore`. Ver [docs/architecture/data-governance.md](docs/architecture/data-governance.md) antes de traer cualquier dataset nuevo — es borrador técnico pendiente de validación por legal/compliance, pero aplica ya como regla de trabajo.

## Cómo crear un experimento nuevo

1. Copia `experiments/_template` a `experiments/IDXX-nombre_del_experimento`, dejando el prefijo `IDXX` literal (no lo asignes a mano).
2. Llena `metadata.yaml` (autor, estado, dataset, métricas objetivo), dejando `id: "IDXX"`.
3. Corre `uv run julieta-validate` para confirmar que el metadata quedó bien formado.
4. Abre tu PR. Al mergear a `main`, el pipeline asigna el ID real y renombra la carpeta automáticamente — ver [ADR 0003](docs/decisions/0003-experiment-id-assignment.md).
5. Trabaja en `notebooks/` (ver convención de nombres en [experiments/_template/README.md](experiments/_template/README.md)).
6. Cuando el código de un experimento madura y deja de ser exploratorio, se traslada a `pipelines/` y `src/julieta/`, dejando el notebook solo como referencia del proceso original.

## Decisiones de arquitectura

Los cambios de diseño no triviales del repo se documentan como ADRs en [docs/decisions](docs/decisions). Empieza por [0001-flatten-experiment-notebooks.md](docs/decisions/0001-flatten-experiment-notebooks.md) para ver el formato.

## Tracking de experimentos (MLflow sobre Azure ML)

`julieta.tracking.mlflow_config.log_run(experiment_id, experiment_name, author, config, metrics, artifacts, tags)` loguea un run completo (todos los parámetros, todas las métricas, tags de autor/commit, artefactos) en una sola llamada, sin que cada quien tenga que recordar la convención — ver [ADR 0009](docs/decisions/0009-mlflow-logging-convention.md). Ya está conectado al **Azure ML Workspace** `ml-salva-dev` (resource group `ml-ops`) — ver [ADR 0005](docs/decisions/0005-mlflow-tracking.md).

**Setup (una sola vez por persona):**

1. Pide el rol **AzureML Data Scientist** sobre el workspace `ml-salva-dev` a quien administra el resource group `ml-ops`.
2. Autentícate con tu cuenta de Salva Health: `az login`.
3. Copia `.env.example` a `.env` (ya ignorado por git) y completa `MLFLOW_TRACKING_URI` con el valor real:
   ```bash
   az ml workspace show --name ml-salva-dev --resource-group ml-ops \
     --query mlflow_tracking_uri -o tsv
   ```
4. Listo — `log_run(...)` recoge la URL automáticamente desde `.env`, no hace falta exportarla a mano en cada sesión.

Si `.env`/`MLFLOW_TRACKING_URI` no está configurado, `configure_mlflow`/`log_run` fallan con un mensaje explícito en vez de fallar en silencio.

## Versionado de datos (DVC sobre Azure Blob Storage)

Los datasets se versionan con `dvc add`/`dvc push` en `pipelines/data/download` (después de anonimizar), en vez de quedar solo como texto libre en `dataset_version`. Ya está conectado al container `dvc-storage` del Storage Account `mlsalvadev2301648611` — ver [ADR 0007](docs/decisions/0007-dvc-data-versioning.md).

**Setup (una sola vez por persona):**

1. Pide el rol **Storage Blob Data Contributor** sobre `mlsalvadev2301648611` a quien administra el resource group `ml-ops`.
2. Autentícate con tu cuenta de Salva Health: `az login` (la autenticación de DVC usa la misma sesión, no hace falta ninguna clave ni connection string).
3. Listo — `dvc pull` trae los datos exactos de un experimento, `dvc push` sube uno nuevo.

## Roadmap

Este es el estado fundacional del repo (empaquetado, estructura, calidad de código, validación de metadata, gobernanza de datos). Tracking de experimentos con MLflow: **conectado**. Versionado de datos con DVC: **conectado**. Próximas fases: orquestación de pipelines en CI/CD, y monitoreo de modelos en producción.
