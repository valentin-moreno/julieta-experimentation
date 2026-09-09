# Guía de colaboración: ramas, commits y Pull Requests

Esta es la referencia detallada de cómo se nombra una rama, cómo se escribe un commit, y cómo se abre/revisa un PR en este repositorio. [CONTRIBUTING.md](../../CONTRIBUTING.md) da el flujo general; este documento existe para que nadie tenga que adivinar los detalles ni inventar su propia convención sobre la marcha.

**Por qué existe un documento aparte para esto**: con varias personas trabajando en paralelo (ver [experiments/_template/README.md](../../experiments/_template/README.md) sobre cómo se reparte el trabajo dentro de un experimento), la fricción de colaborar no viene de la falta de reglas,  viene de que cada quien tenga una regla distinta en la cabeza. Esto se escribe una vez para que no haya que negociarlo cada vez.

## 1. Nombres de rama

Formato: `<tipo>/<identificador>_<slug_descriptivo>`, todo en minúsculas, palabras separadas por guion bajo (no guiones, no espacios, sin tildes) — snake_case en todo el repo, ver nota al final de esta sección.

| Tipo | Cuándo usarlo | Ejemplo |
|---|---|---|
| `exp/` | Trabajo dentro de un experimento (`experiments/IDXX-*` o `experiments/ID07-*`) | `exp/idxx_20260825_temporal_features` (experimento nuevo, ID pendiente) |
| `data/` | Traer o actualizar un dataset fuera del ciclo de un experimento puntual (ej. un `dvc add` de una fuente nueva) | `data/actualizar_dataset_pacientes_activos` |
| `feature/` | Funcionalidad nueva del repo (pipelines, `src/julieta`, tooling) | `feature/pipeline_training_baseline` |
| `fix/` | Corrección de un bug en código del repo (no en un experimento) | `fix/validate_metadata_fecha_nula` |
| `docs/` | Cambios que son solo documentación (READMEs, ADRs, esta guía) | `docs/adr_0009_serving_models` |
| `chore/` | Mantenimiento sin impacto funcional (deps, config, CI, formateo) | `chore/actualizar_ruff` |

**Por qué `exp/<ID>_...` y no `exp/<nombre_de_persona>_...`**: la rama identifica *qué* se está haciendo, no *quién* lo hace — eso ya lo dice el autor del commit y del PR. La única excepción es cuando **dos personas trabajan en el mismo experimento al mismo tiempo, cada una en su propio archivo** (la convención que ya define `experiments/_template/README.md` para notebooks/configs). Ahí sí se agrega el nombre al final para que las ramas no choquen: `exp/idxx_20260825_temporal_features_maria` y `exp/idxx_20260825_temporal_features_jose`. Cada quien abre su propio PR — no comparten rama, aunque trabajen en el mismo experimento (ver sección 3).

**Sobre el `IDXX` literal y la fecha**: la gran mayoría de las ramas `exp/` empiezan con `idxx` — nadie sabe su ID real hasta que el pipeline lo asigna al mergear (ver [ADR 0003](../decisions/0003_experiment_id_assignment.md)). El problema de usar solo `idxx`: si hay varios experimentos pendientes al tiempo, todas las ramas empiezan igual y una lista alfabética de ramas (o de PRs ordenados por título) no da ninguna pista de cuál es más vieja o más nueva — a diferencia de los commits, donde git ya guarda la fecha real y el orden nunca se pierde aunque el texto diga `idxx`. Por eso la rama lleva la fecha de creación justo después: `exp/idxx_20260825_temporal_features`. No hace falta coordinarse para esto — dos personas pueden crear su rama el mismo segundo sin chocar, porque el slug las distingue igual; la fecha solo da orden, no unicidad (esa la sigue dando el pipeline al asignar el ID real).

No pasa nada si el nombre de la rama queda "desactualizado" después de que el pipeline renombre la carpeta — la rama es temporal, la carpeta es lo que persiste. Solo cuando trabajas *sobre un experimento que ya está en `main` con su ID real* (una segunda ronda de PRs sobre el mismo experimento) la rama usa el número real en vez de `idxx`+fecha: `exp/id01_mas_profundidad`.

**Nota sobre la convención snake_case (todo el repo)**: ramas, commits, nombres de archivo de docs/ADRs y comandos CLI (`julieta_validate`, `julieta_summary`, etc.) usan guion bajo, no guion medio, para que la convención sea una sola en todo el repo (coincide con como ya se nombran módulos y carpetas de experimentos, ej. `mlops_pipeline_smoke_test`). La única excepción es `IDXX-nombre_del_experimento`: ese guion medio entre el ID y el nombre no es estilo, es un separador que el código realmente parsea (`assign_experiment_ids.py`, `validate_metadata.py`) — cambiarlo requeriría reescribir esa lógica, no es solo un rename.

## 2. Mensajes de commit

Formato de la primera línea: `<tipo>(<alcance>): <descripción en imperativo, minúscula, sin punto final>`

Tipos (los mismos que los prefijos de rama, más dos exclusivos de commits):

- `feat` — funcionalidad nueva de repo/tooling.
- `fix` — corrección de bug.
- `docs` — documentación.
- `chore` — mantenimiento, deps, config.
- `exp` — avance dentro de un experimento (EDA, feature engineering, un modelo nuevo, actualizar resultados).
- `data` — cambios de dataset (`dvc add`, actualizar `dataset_version`).
- `refactor` — reorganizar código sin cambiar comportamiento.
- `test` — agregar o corregir tests, sin cambiar comportamiento de producción.

El `<alcance>` es lo que se toca: un ID de experimento, una carpeta (`pipelines/training`), un módulo (`metadata_schema`).

**Ojo con el ID en el alcance — mismo problema que en el nombre de rama**: mientras el experimento no tenga ID real asignado (está en su primer PR, todavía no mergeado — ver [ADR 0003](../decisions/0003_experiment_id_assignment.md)), el alcance también es `idxx`, nunca un número inventado. Escribir `exp(id07): ...` en un commit de un experimento que todavía no existe en `main` es prometer un número que ni el pipeline ha decidido todavía. Recién cuando el experimento **ya está mergeado con su ID real**, el trabajo adicional sobre ese mismo experimento (en un PR posterior) sí usa el número real.

Ejemplos reales, con el vocabulario de este repo:

```
exp(idxx): agregar baseline con xgboost sobre features temporales   # experimento aún sin mergear, ID pendiente
exp(id01): agregar variante con más profundidad al modelo            # ID01 ya existe en main, esto es trabajo posterior
data(id01): actualizar dataset_version a v2.2 tras dvc add
fix(validate_metadata): aceptar status "paused" en el schema
docs(adr): registrar decisión de versionado de datos con DVC
chore(deps): actualizar ruff a 0.7
```

**Cómo buscar el historial de un experimento después de que cambió de `idxx` a su ID real**: no se busca por el texto del commit (los commits viejos van a decir `idxx` para siempre, son historia inmutable) — se busca por la ruta de la carpeta, con `--follow` para que git atraviese el rename que hizo el pipeline al asignar el ID:

```bash
git log --follow -- experiments/ID07-nombre_del_experimento/
```

Esto encuentra los commits de cuando la carpeta todavía se llamaba `IDXX-nombre_del_experimento`, porque git detecta el rename automáticamente (mismo contenido, solo cambió de nombre). Por esto el detalle del ID en el mensaje de commit importa menos de lo que parece  la trazabilidad real vive en la ruta del archivo, no en el texto del mensaje.

**Por qué esta estructura y no un mensaje libre**: con `julieta_summary` agregando experimentos en un CSV, y con varios ADRs ya conectando decisiones entre sí, tener el tipo y el alcance en el mensaje hace que `git log` sea buscable de la misma forma en que ya lo es el repo (por experimento, por tipo de cambio) — sin esto, el historial de git es la única parte del repo que queda sin la disciplina que le pusimos a todo lo demás.

**Reglas adicionales:**

- El *por qué*, no el *qué* — el diff ya dice qué cambió; el mensaje dice por qué era necesario. (Ver también la nota equivalente en [ADR 0000_template.md](../decisions/0000_template.md).)
- Un commit, un cambio lógico. No mezclar un `fix` de código con un `chore` de dependencias en el mismo commit.
- Si el commit toca `experiments/`, nunca se asigna el ID a mano en el mensaje ni en la carpeta — eso lo hace el pipeline (ADR 0003).
- Nunca commitear algo que `julieta_scan_sensitive` o `nbstripout` bloquearon saltándose el hook (`--no-verify` está prohibido salvo autorización explícita — ver [data_governance.md](../architecture/data_governance.md)).

## 3. Pull Requests

### Tamaño y cadencia

Ya está la regla base en [CONTRIBUTING.md](../../CONTRIBUTING.md#cadencia-de-prs-en-un-experimento-largo): PRs incrementales por hito, no uno gigante al final. Un "hito" en un experimento típicamente es: EDA terminado, baseline funcionando, cada variante relevante que se prueba, y el cierre (conclusión + decisión de qué sigue).

Si dos colaboradores trabajan en el mismo experimento en paralelo (cada uno en su propio notebook, cada uno en su propia rama `exp/idxx_<fecha>_<slug>_<nombre>`), **cada quien abre su propio PR**. No se comparte una rama "integradora" — como cada uno toca archivos distintos (la convención de un-notebook-por-persona ya lo garantiza), los PRs mergean sin pisarse.

### Título

Mismo formato que el commit principal del PR: `<tipo>(<alcance>): <qué cambia>`. Un título como "Update files" o "cambios" no dice nada — se rechaza en revisión, no por regla automática (nada en CI lo valida hoy), sino porque el propio revisor debe pedir que se corrija antes de aprobar.

### Descripción — checklist antes de abrir

- [ ] `uv run pre-commit run --all-files` pasó localmente.
- [ ] `uv run pytest -q` pasó localmente.
- [ ] Si el PR agrega o modifica un experimento: `metadata.yaml` tiene `type` y (si ya se sabe) `objective` llenos — ver [ADR 0008](../decisions/0008_experiment_type_and_objective.md).
- [ ] Si el PR trae datos nuevos: se siguió [data_governance.md](../architecture/data_governance.md) (anonimización antes de `data/raw` y antes de `dvc add`).
- [ ] Si el PR crea un experimento nuevo: la carpeta y el `id` en `metadata.yaml` siguen literalmente como `IDXX` — no se asignó a mano.
- [ ] Si el PR cambia cómo está organizado el repo (no solo agrega un experimento): hay un ADR nuevo en `docs/decisions/`.

### Revisión

Al menos una persona que no sea el autor revisa antes de mergear. Si el PR modifica un ADR o `docs/architecture/data_governance.md`, la revisión la hace preferiblemente quien tenga más contexto de esa decisión, no cualquiera — un cambio de gobernanza mal revisado es más caro de deshacer que uno de código.

### Qué bloquea el merge

El pipeline de Azure ([azure-pipelines.yml](../../azure-pipelines.yml), stage `Build`) corre lint, tests y `julieta_validate` en cada PR. Un PR no se mergea con el pipeline en rojo.

### Qué pasa automáticamente al mergear

Si el PR agregó un experimento nuevo (`experiments/IDXX-*`), el pipeline le asigna el ID real y renombra la carpeta después del merge, sin acción tuya — ver [ADR 0003](../decisions/0003_experiment_id_assignment.md). No hace falta (ni se debe) abrir un PR de seguimiento para "corregir" el nombre de la carpeta.

## Referencias

- [CONTRIBUTING.md](../../CONTRIBUTING.md) — flujo general, punto de entrada.
- [ADR 0001](../decisions/0001_flatten_experiment_notebooks.md) — convención de nombres de notebooks.
- [ADR 0003](../decisions/0003_experiment_id_assignment.md) — asignación de ID al mergear.
- [ADR 0008](../decisions/0008_experiment_type_and_objective.md) — campos `type`/`objective` en `metadata.yaml`.
- [docs/architecture/data_governance.md](../architecture/data_governance.md) — reglas sobre datos sensibles.
- [experiments/_template/README.md](../../experiments/_template/README.md) — cómo se reparte el trabajo dentro de un experimento.
