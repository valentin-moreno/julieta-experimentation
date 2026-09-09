# ADR 0006: Ciclo de vida de `models/candidates` y promoción

- Estado: aceptada
- Fecha: 2026-08-25

## Contexto

`models/candidates` existe desde el esqueleto original, pero el nombre insinúa un ciclo de vida (candidato → algo más) que nunca se definió: no hay `models/production`, ni criterio de promoción, ni quién la aprueba. Al mismo tiempo, [ADR 0005](0005_mlflow_tracking.md) ya estableció que el tracking de experimentos usa el Model Registry de Azure ML/MLflow — que resuelve versionado y "stages" de modelos de forma nativa. Definir un `models/production/` local en este repo duplicaría, y probablemente desincronizaría, algo que Azure ML ya hace mejor.

## Decisión

- **`models/candidates/<experiment_id>/`** es *scratch* local: donde `pipelines/training` guarda el artefacto que produce cada corrida. Ya está en `.gitignore` — nunca se versiona en git, es desechable, se puede borrar y regenerar corriendo la pipeline de nuevo.
- **Este repositorio no tiene concepto de "modelo en producción".** Es un repositorio de experimentación: su responsabilidad termina en producir un candidato documentado, no en servirlo. Un modelo pasa a producción fuera de este repo (en el sistema de serving que corresponda), no moviendo carpetas aquí.
- **Cuando exista el Azure ML Workspace** ([ADR 0005](0005_mlflow_tracking.md)), `pipelines/training` registra el candidato directamente en el Model Registry (vía `mlflow.register_model` o el cliente de Azure ML) en vez de dejarlo solo en disco. "Promover" un modelo es una operación sobre ese registro (marcar una versión registrada como la vigente), no un `mv` entre carpetas de este repo.
- **Criterio para considerar un candidato listo para promoción** (mientras no exista el registro, y como criterio también después): el `metrics.final_value` de su `metadata.yaml` debe cumplir el `metrics.target_value` que se definió *antes* de correr el experimento, y el experimento debe tener su `status: completed` y la sección "Resultados"/"Conclusiones" de su README llena (ver la plantilla en [experiments/_template/README.md](../../experiments/_template/README.md)). Ningún experimento se promueve solo porque "corrió sin errores".
- La decisión de promover no es automática: la toma una persona, revisando lo anterior — no hay CI que promueva un modelo por sí solo.

## Consecuencias

- No se crea un `models/production/` local — evita mantener dos fuentes de verdad (el repo y el registry) que se pueden desincronizar.
- Hasta que exista el Azure ML Workspace, "promoción" no tiene una acción concreta que ejecutar en este repo — es una decisión humana documentada en el experimento, sin mecanismo. Eso es un estado intermedio aceptado, no un problema a resolver ahora.
- Si en el futuro este repo necesita orquestar despliegues (no solo experimentar), esta decisión debe revisarse — probablemente con un repo o pipeline aparte dedicado a serving, no agregando esa responsabilidad aquí.
