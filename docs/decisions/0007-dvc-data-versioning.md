# ADR 0007: Versionado de datasets con DVC sobre Azure Blob Storage

- Estado: aceptada — conectado al container `dvc-storage` en el Storage Account `mlsalvadev2301648611` (resource group `ml-ops`)
- Fecha: 2026-08-25 (conectado: 2026-09-03)

## Contexto

Cada experimento puede traer un dataset nuevo o una variante de uno existente. Hoy `dataset_version` en `metadata.yaml` es un string libre (`"v2.1_active_users"`) sin nada real detrás — dos personas pueden escribir el mismo texto queriendo decir cosas distintas, y no hay forma de reproducir exactamente qué datos usó un experimento pasado. Al mismo tiempo, `data/raw`, `data/interim`, `data/processed` están completamente fuera de git (`.gitignore`), así que git tampoco puede ayudar aquí — los datasets pesan demasiado para vivir en el historial de git de todas formas.

## Decisión

- Se versionan datasets con **DVC**, usando **Azure Blob Storage** como remoto (la empresa ya usa Azure Blob Storage, confirmado por el usuario) — vía el plugin `dvc[azure]`, ya agregado a `pyproject.toml`.
- `dvc init` ya corrió sobre el repo ([.dvc/config](../../.dvc/config)); se desactivó la telemetría anónima de DVC (`core.analytics = false`) por default conservador, dado el contexto de datos de salud.
- El `.gitignore` se ajustó para permitir los archivos puntero `*.dvc` dentro de `data/` (antes el patrón `data/raw/*` los hubiera ignorado también, dejando sin efecto el versionado).
- **Dónde corre `dvc add`/`dvc push`**: en `pipelines/data/download` ([ADR 0002](0002-pipelines-vs-src-separation.md)), el mismo punto que ya trae datos externos y ya anonimiza antes de escribir a `data/raw` ([docs/architecture/data-governance.md](../architecture/data-governance.md)). DVC no cambia esa responsabilidad, se inserta justo después.
- **`dataset_version` deja de ser texto libre**: pasa a ser un tag de git (ej. `data-id01-v2.1`) creado en el mismo commit donde se actualizó el `.dvc` correspondiente. `git checkout <tag> && dvc pull` reproduce el dataset exacto, byte por byte — no una intención escrita a mano.
- **Recurso real conectado**: el container `dvc-storage` dentro del Storage Account (`mlsalvadev2301648611`) que se creó automáticamente junto al Azure ML Workspace ya provisionado ([ADR 0005](0005-mlflow-tracking.md)) — se reutiliza ese storage account en vez de crear uno dedicado, indistinto para DVC. Configuración aplicada:
  ```bash
  dvc remote add -d azure-storage azure://dvc-storage/data
  dvc remote modify azure-storage account_name mlsalvadev2301648611
  ```
  Sin `account_key` ni connection string: la cuenta tiene el acceso por clave compartida deshabilitado ([ADR 0005](0005-mlflow-tracking.md)), así que la autenticación se resuelve vía Azure AD (`DefaultAzureCredential`), tomando la sesión activa de `az login` de cada persona — el mismo mecanismo que usa MLflow.

### Prueba real del mecanismo

Primero se probó el ciclo completo `dvc add` → `dvc push` → borrar todo localmente (simulando un clon nuevo) → `dvc pull` contra un remoto local de prueba (una carpeta en disco haciendo de "Azure Blob falso"), en una copia aislada del repo — antes de que el contenedor real existiera. Después, el 2026-09-03, se repitió exactamente el mismo ciclo contra el remoto real (`azure-storage` sobre `dvc-storage`): se subió un archivo de prueba, se verificó su llegada directo en Azure (`az storage blob list`), se borró localmente y se recuperó byte por byte con `dvc pull`. En todo momento el `.csv` real siguió invisible para git (solo el `.dvc` se trackea).

## Consecuencias

- `dvc add` + `dvc push`/`dvc pull` ya funcionan de verdad contra el remoto real — probado de punta a punta, no solo en un remoto local simulado.
- Los datasets pesados nunca tocan el historial de git, pero sí quedan versionados de forma real a través de los `.dvc` + tags, resolviendo el hueco que `dataset_version` tenía desde el inicio.
- La gobernanza de datos se extiende: el remoto de DVC es, igual que el historial de git, un lugar donde el dato queda casi permanentemente — la anonimización debe pasar antes de `dvc add`, nunca después (ver actualización en [docs/architecture/data-governance.md](../architecture/data-governance.md)).
- Si más adelante se decide usar otro backend de almacenamiento en vez de Azure Blob, solo cambia la configuración del remoto (`dvc remote`) — el resto del flujo (`dvc add` en `pipelines/data/download`, tags para `dataset_version`) no depende del backend específico.
