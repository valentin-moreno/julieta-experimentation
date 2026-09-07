# ADR 0008: Campos `type` y `objective` en `metadata.yaml`

- Estado: aceptada
- Fecha: 2026-08-25

## Contexto

Una colaboradora hizo notar que no todo lo que se necesita documentar encaja en `experiments/` tal como está diseñado: un reporte de calidad/caracterización de datos (sin modelo, sin hipótesis que probar) no tiene sentido forzado a través de la plantilla de README pensada para un experimento de ML. Se evaluaron tres opciones (ver discusión previa a este ADR): endurecer `domain`, agregar un campo `type` nuevo, o una convención de nombre de carpeta. Se descartó una carpeta nueva a propósito — el problema es de **clasificación**, no de **ubicación**.

También se identificó que `metadata.yaml` tenía `conclusion` (qué se encontró, al final) sin su contraparte: qué se buscaba lograr, desde el inicio. Sin eso, entender el propósito de un experimento requería siempre abrir el README completo — nunca alcanzaba con el metadata ni con el CSV de `julieta-summary`.

## Decisión

- **`type`** (obligatorio, enum: `experiment` | `data_report` | `analysis`) se agrega a `ExperimentMetadata` ([metadata_schema.py](../../src/julieta/utils/metadata_schema.py)), separado de `domain` a propósito: `domain` responde "de qué área es" (`nlp`, `feature_engineering`), `type` responde "qué clase de trabajo es". Mezclarlos en un solo campo hubiera obligado a valores como `domain: "nlp_data_report"`, ambiguo y sin escalar.
- Cuando `type` es `data_report` o `analysis`, las secciones Hipótesis/Metodología/Métricas de éxito/Resultados del README de la plantilla se interpretan distinto (marcadas con 🔁 en [experiments/_template/README.md](../../experiments/_template/README.md)) — sin necesidad de una plantilla ni carpeta separada.
- **`objective`** (opcional, string, default `""`) se agrega junto a `conclusion` en la sección "Objetivo y Resultados" de `metadata.yaml`. A diferencia de `conclusion` (se llena al final), `objective` se llena al crear el experimento. Juntos, dan el par "qué buscaba → qué encontré" legible desde `julieta-summary` sin abrir el README.
- `generate_summary.py` incluye ambos campos (`type`, `objective`) como columnas del CSV.
- `experiments/_template/metadata.yaml` y el experimento real existente (`ID01-prioritization_new_feature`) se actualizaron para incluir ambos campos.

## Consecuencias

- `type` es obligatorio (sin default) — cualquier `metadata.yaml` existente sin este campo falla `julieta-validate` hasta que se agregue. Es intencional: fuerza una clasificación explícita, igual que ya pasa con `status`.
- `objective` es opcional — no rompe metadata existente, pero se pierde parte del valor (el resumen "qué buscaba") si no se llena. No se hizo obligatorio para no bloquear experimentos que ya estaban en curso al momento de este cambio.
- Un reporte de datos ahora puede vivir en `experiments/IDXX-*` sin fricción, sin necesidad de una carpeta ni plantilla nueva — resuelve la observación original sin agregar estructura.
- Si en el futuro se necesita filtrar/reportar de forma distinta según `type` (ej. un `julieta-summary` que solo muestre `data_report`s), el campo ya está ahí para soportarlo sin otro cambio de schema.
