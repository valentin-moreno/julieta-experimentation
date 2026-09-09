# Gobernanza de datos sensibles

> **Estado: borrador técnico, pendiente de validación por legal/compliance.**
> Este documento define reglas de ingeniería para qué datos pueden tocar este repositorio y cómo. No sustituye una política de tratamiento de datos personales formal de Salva Health — si esa política no existe todavía, este documento puede servir de insumo inicial, pero el marco legal aplicable (Ley 1581 de 2012 y su Decreto Reglamentario 1377 de 2013 en Colombia que clasifican los datos de salud como **datos sensibles** bajo el Artículo 5 de la Ley 1581) debe ser confirmado y ampliado por quien tenga esa responsabilidad formal en la empresa, no por este repositorio.

## Por qué existe este documento

Este es un repositorio de experimentación de una empresa de salud. Los datasets que se van a tocar aquí muy probablemente contienen o derivan de datos de pacientes. A diferencia de bugs de código, un dato sensible que queda commiteado en git **no se puede simplemente borrar** — sigue en el historial, en cada clon del repo, potencialmente para siempre. Por eso esta regla se define antes de que exista el primer dataset real, no después de un incidente.

## Clasificación de datos

| Nivel | Qué incluye | Ejemplos |
|---|---|---|
| **0 — No sensible** | Datos agregados, sin capacidad de identificar a una persona. | Métricas de un experimento, distribuciones agregadas, código, configuración. |
| **1 — Identificable / cuasi-identificable** | Datos que, solos o combinados, permiten identificar a una persona. | Nombre, número de documento, teléfono, correo, dirección, fecha de nacimiento exacta. |
| **2 — Dato sensible de salud** | Cualquier dato de salud vinculado (directa o indirectamente) a una persona identificable. Es la categoría con mayor protección bajo la Ley 1581 (Art. 5). | Diagnósticos, historia clínica, resultados de laboratorio, tratamientos, cualquier variable clínica no agregada. |

## Regla central

**Ningún dato de Nivel 1 o Nivel 2 puede entrar a este repositorio en texto claro**, ni en `data/`, ni en notebooks, ni en `results/`, ni en `reports/`, ni en un output de celda, ni en un ejemplo hardcodeado en código. Esto aplica sin importar que `data/` ya esté en `.gitignore` — esa regla evita que los *archivos de datos* se suban, pero no evita que alguien pegue una fila de ejemplo con nombre y diagnóstico dentro de un notebook o un README.

Todo dato de Nivel 1 o 2 debe pasar por un proceso de **anonimización o seudonimización antes de tocar cualquier carpeta de este repositorio** (incluyendo `data/raw`, que ya está ignorado por git, pero igual puede tener datos reales sin anonimizar en el disco de alguien).

### Qué significa "anonimizar" en la práctica

- Remover o hashear identificadores directos (nombre, documento, teléfono, correo, dirección) antes de que el dato llegue a `data/raw`.
- Nunca usar el ID real de un paciente como llave dentro del repo — generar un ID sintético (hash con salt, o un mapeo que vive fuera del repo, gestionado por quien tenga acceso a la fuente original).
- Fechas exactas que sean cuasi-identificadores (ej. fecha de nacimiento) se generalizan (ej. a edad o rango etario) salvo que el experimento justifique explícitamente por qué necesita la fecha exacta, y esa justificación quede documentada en el `metadata.yaml` del experimento.
- Si un dataset no se puede anonimizar sin perder la señal que el experimento necesita, la decisión de usarlo igual no la toma quien corre el experimento — se escala a quien tenga la responsabilidad de tratamiento de datos en la empresa.

### Dónde vive esta responsabilidad en la arquitectura del repo

Según [ADR 0002](../decisions/0002_pipelines_vs_src_separation.md), `pipelines/data/download` es el único punto que trae datos desde una fuente externa. Es, por diseño, el lugar donde debe aplicarse la anonimización — **antes** de escribir cualquier cosa a `data/raw`. Ningún notebook ni pipeline aguas abajo (`build`, `validation`, `training`) debería necesitar tocar un identificador directo, porque para cuando el dato llega ahí ya debería estar anonimizado.

### El remoto de DVC es otro lugar donde el dato queda casi permanente

Desde [ADR 0007](../decisions/0007_dvc_data_versioning.md), los datasets se versionan con DVC sobre Azure Blob Storage. La misma regla aplica ahí: **la anonimización debe pasar antes de `dvc add`/`dvc push`, nunca después.** Un dato que llega al remoto de DVC sin anonimizar tiene el mismo problema que uno commiteado a git — queda en versiones anteriores del dataset, disponible para cualquiera con acceso al remoto, y no se puede simplemente "borrar" sin coordinar una purga.

### Los artefactos de MLflow son un tercer lugar con la misma regla

Desde [ADR 0009](../decisions/0009_mlflow_logging_convention.md), `log_run` sube archivos (`artifacts`) a MLflow — plots, el config usado, el modelo entrenado. **Nunca una muestra cruda de datos reales como artefacto**, ni siquiera "solo para revisar algo rápido": un CSV de ejemplo con filas reales subido como artefacto tiene exactamente el mismo problema que un dato sin anonimizar en `data/raw` o en el remoto de DVC — queda accesible para cualquiera con acceso al workspace, indefinidamente.

## Controles ya existentes en el repo (parciales, no suficientes por sí solos)

- `data/`, `models/candidates`, `experiments/**/results`, `experiments/**/reports` están en `.gitignore` — el contenido nunca se sube a git. Esto protege contra subir *archivos* de datos, pero no contra pegar un valor sensible directamente en un notebook o en código.
- `nbstripout` (vía pre-commit) limpia los outputs de los notebooks antes de cada commit — evita que una celda que imprimió una muestra de datos reales quede guardada en el `.ipynb` commiteado. Tampoco es suficiente solo: si alguien **hardcodea** un valor identificable en el código de una celda (no en el output), `nbstripout` no lo toca.
- **`julieta_scan_sensitive`** (vía pre-commit, hook `scan-sensitive-data`): busca en cada archivo `.py`/`.ipynb`/`.yaml`/`.yml`/`.csv`/`.json` que se va a commitear nombres de variable/columna que sugieren un dato de Nivel 1 o 2 sin anonimizar (`cedula`, `numero_documento`, `historia_clinica`, `nombre_paciente`, `diagnostico`, etc.) y direcciones de correo. **No es infalible** — es un detector de patrones y nombres, no entiende el contenido real de un dataset, así que no reemplaza la anonimización en el origen (`pipelines/data/download`), solo agrega una segunda capa. Un falso positivo legítimo (ej. el correo de una cuenta de servicio) se documenta en la misma línea con el marcador `# permitido-datos-sensibles: <razón>` — ver [scan_sensitive_data.py](../../src/julieta/utils/scan_sensitive_data.py).

## Qué falta (pendiente, no implementado todavía)

- Designar formalmente quién es el responsable del tratamiento de datos para este repositorio (rol que exige la Ley 1581), y a quién se escala una duda sobre si un dataset se puede anonimizar o no.
- Confirmar con legal/compliance si aplica algún otro marco además de la Ley 1581 (ej. normativa sectorial de salud, contratos con terceros que aporten datos).

## Qué hacer si algo sensible queda commiteado por accidente

1. No basta con borrar el archivo en un commit nuevo — sigue en el historial de git y en cada clon existente.
2. Notificar de inmediato a quien administre el repositorio y a quien lidere compliance/legal en la empresa — antes de purgar el historial, para evaluar si hubo una exposición que deba reportarse.
3. Purgar el historial de git de forma coordinada (herramientas como `git filter-repo`), y forzar a todo el equipo a re-clonar — esto reescribe el historial compartido, así que se hace una sola vez, coordinado, no por la persona que causó el incidente actuando sola.
