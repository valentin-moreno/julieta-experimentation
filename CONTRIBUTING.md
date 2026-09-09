# Cómo contribuir

## Flujo de trabajo

1. Crea una rama a partir de `main`, siguiendo la convención de nombres — ver [docs/experimentation/guia_de_colaboracion_git.md](docs/experimentation/guia_de_colaboracion_git.md#1-nombres-de-rama) para el detalle completo (tipos de rama, cómo se nombra cuando varias personas trabajan en el mismo experimento).
2. Escribe tus commits siguiendo el formato `<tipo>(<alcance>): <descripción>` — ver [la misma guía, sección 2](docs/experimentation/guia_de_colaboracion_git.md#2-mensajes-de-commit).
3. Antes de abrir el PR, corre localmente:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest -q
   ```
   (los hooks de pre-commit ya corren esto en cada commit local si activaste `uv run pre-commit install` — ver Quickstart en el [README](README.md)).
4. Abre el PR a `main` — checklist y qué revisar en [la guía, sección 3](docs/experimentation/guia_de_colaboracion_git.md#3-pull-requests). El pipeline de Azure ([azure-pipelines.yml](azure-pipelines.yml)) corre lint, tests y valida el metadata de experimentos automáticamente.
5. Si tu PR agrega un experimento nuevo (`experiments/IDXX-*`), **no le asignes un ID a mano** — el pipeline lo asigna automáticamente al mergear (ver [ADR 0003](docs/decisions/0003_experiment_id_assignment.md)).
6. Espera a que el pipeline pase en verde antes de mergear. El número de aprobaciones requeridas lo define la branch protection rule de `main`, no este documento.

## Cadencia de PRs en un experimento largo

No dejes todo el experimento en un solo PR gigante al final — así nadie del equipo ve progreso hasta que ya está "terminado" y es tarde para dar feedback útil. Prefiere PRs incrementales por hito: uno cuando el EDA está listo, otro cuando hay un baseline, otro por cada variante relevante que se prueba. Cada PR actualiza el `metadata.yaml` y el README del experimento al estado real en ese momento — no hace falta esperar a la conclusión final para que ambos digan algo útil. Detalle completo (incluyendo qué hacer si dos personas trabajan en el mismo experimento a la vez) en [la guía, sección 3](docs/experimentation/guia_de_colaboracion_git.md#3-pull-requests).

## Antes de traer datos nuevos

Lee [docs/architecture/data_governance.md](docs/architecture/data_governance.md) antes de agregar cualquier dataset — hay reglas concretas sobre qué tipo de dato puede y no puede tocar este repositorio, y quién anonimiza qué.

## Decisiones de diseño (ADRs)

Si tu cambio afecta cómo está organizado el repo (no solo agrega un experimento), documéntalo como ADR en `docs/decisions/`, siguiendo la plantilla en [0000_template.md](docs/decisions/0000_template.md). Revisa los ADRs existentes para ver el nivel de detalle esperado — cada uno explica el contexto, la decisión y las consecuencias, no solo "qué se hizo".

## Crear un experimento

Ver [experiments/_template/README.md](experiments/_template/README.md) para el paso a paso y la plantilla de contenido esperado.

## Dudas

Si algo de este flujo no aplica a tu caso o no queda claro, es señal de que falta un ADR o una actualización a este documento — no de que haya que improvisar en silencio y que el próximo se las arregle solo.
