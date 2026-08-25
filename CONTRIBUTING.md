# Cómo contribuir

## Flujo de trabajo

1. Crea una rama a partir de `main`. Convención sugerida: `exp/IDXX-nombre_del_experimento` para experimentos, `feature/lo-que-sea` o `fix/lo-que-sea` para cambios de código o infraestructura del repo.
2. Antes de abrir el PR, corre localmente:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest -q
   ```
   (los hooks de pre-commit ya corren esto en cada commit local si activaste `uv run pre-commit install` — ver Quickstart en el [README](README.md)).
3. Abre el PR a `main`. El pipeline de Azure ([azure-pipelines.yml](azure-pipelines.yml)) corre lint, tests y valida el metadata de experimentos automáticamente.
4. Si tu PR agrega un experimento nuevo (`experiments/IDXX-*`), **no le asignes un ID a mano** — el pipeline lo asigna automáticamente al mergear (ver [ADR 0003](docs/decisions/0003-experiment-id-assignment.md)).
5. Espera a que el pipeline pase en verde antes de mergear. El número de aprobaciones requeridas lo define la branch protection rule de `main`, no este documento.

## Antes de traer datos nuevos

Lee [docs/architecture/data-governance.md](docs/architecture/data-governance.md) antes de agregar cualquier dataset — hay reglas concretas sobre qué tipo de dato puede y no puede tocar este repositorio, y quién anonimiza qué.

## Decisiones de diseño (ADRs)

Si tu cambio afecta cómo está organizado el repo (no solo agrega un experimento), documéntalo como ADR en `docs/decisions/`, siguiendo la plantilla en [0000-template.md](docs/decisions/0000-template.md). Revisa los ADRs existentes para ver el nivel de detalle esperado — cada uno explica el contexto, la decisión y las consecuencias, no solo "qué se hizo".

## Crear un experimento

Ver [experiments/_template/README.md](experiments/_template/README.md) para el paso a paso y la plantilla de contenido esperado.

## Dudas

Si algo de este flujo no aplica a tu caso o no queda claro, es señal de que falta un ADR o una actualización a este documento — no de que haya que improvisar en silencio y que el próximo se las arregle solo.
