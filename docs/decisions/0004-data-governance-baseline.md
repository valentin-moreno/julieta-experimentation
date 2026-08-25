# ADR 0004: Línea base de gobernanza de datos sensibles

- Estado: propuesta — pendiente de validación por legal/compliance
- Fecha: 2026-08-25

## Contexto

Este repositorio pertenece a una empresa de salud (Salva Health) y va a alojar experimentos sobre datos que muy probablemente derivan de pacientes reales. Hasta ahora el repo tenía controles de infraestructura (`.gitignore` sobre `data/`, `nbstripout` en notebooks) pero ninguna regla explícita sobre qué tipo de dato puede tocar el repositorio ni cómo debe tratarse antes de llegar aquí. Ese vacío es más grave en este dominio que en un repo de ML genérico: un dato de salud identificable que queda commiteado no se puede simplemente borrar.

## Decisión

Se adopta [docs/architecture/data-governance.md](../architecture/data-governance.md) como línea base técnica:

- Clasificación de datos en 3 niveles (no sensible / identificable / dato sensible de salud).
- Regla central: nada de Nivel 1 o 2 entra al repositorio sin anonimizar antes, sin importar que la carpeta destino ya esté en `.gitignore`.
- La responsabilidad de anonimizar recae en `pipelines/data/download` (el único punto de contacto con fuentes externas, según [ADR 0002](0002-pipelines-vs-src-separation.md)) — para cuando un dato llega a `data/raw`, ya debe estar anonimizado.
- Se documenta explícitamente qué controles ya existen (`.gitignore`, `nbstripout`) y qué falta (detección automática de patrones sensibles, un responsable formal de tratamiento de datos designado).

Este ADR y el documento que referencia se marcan como **borrador técnico**, no como política legal validada — el marco regulatorio específico (Ley 1581 de 2012 y su decreto reglamentario en Colombia, y cualquier otro que aplique) debe confirmarlo quien tenga la responsabilidad formal de compliance en la empresa.

## Consecuencias

- A partir de ahora, cualquier PR que agregue un dataset o un notebook con datos reales puede evaluarse contra una regla escrita, no contra el criterio individual de quien lo sube.
- El repo todavía no tiene ningún control automático que haga cumplir esto — depende de disciplina humana hasta que se implemente una segunda capa (detección de patrones, revisión en PR).
- Si legal/compliance define un marco distinto o más estricto al descrito aquí, este ADR queda obsoleto y debe reemplazarse, no editarse en silencio — un cambio de esta naturaleza merece su propio ADR que explique qué cambió y por qué.
