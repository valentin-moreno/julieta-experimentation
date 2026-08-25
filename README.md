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

El código ya está listo (`julieta.tracking.mlflow_config.configure_mlflow`, dependencias en `pyproject.toml`, config en [configs/mlflow.yaml](configs/mlflow.yaml)), pero depende de un **Azure ML Workspace** todavía sin aprovisionar — ver [ADR 0005](docs/decisions/0005-mlflow-tracking.md). Hasta que exista, `configure_mlflow` falla con un mensaje explícito en vez de fallar en silencio.

## Roadmap

Este es el estado fundacional del repo (empaquetado, estructura, calidad de código, validación de metadata, gobernanza de datos). Próximas fases: tracking de experimentos con MLflow (en progreso, bloqueado en el recurso de Azure), versionado de datos/modelos con DVC, orquestación de pipelines en CI/CD, y monitoreo de modelos en producción.
