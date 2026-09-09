# ADR 0003: Asignación de ID de experimento al mergear, no al crear

- Estado: aceptada
- Fecha: 2026-08-24

## Contexto

Con varios científicos creando experimentos en paralelo, un esquema donde cada quien elige a mano el siguiente `IDNN` (`ID01`, `ID02`...) tiene una condición de carrera real: dos personas pueden elegir el mismo número sin saberlo y descubrirlo recién al mergear.

Se consideraron varias alternativas: IDs con timestamp o UUID (eliminan la colisión pero pierden el estilo legible `ID01`, `ID02`...), un chequeo en CI que solo detecta la colisión después de que ya ocurrió, o un CLI que reserva el ID contra el remoto de git antes de crear la carpeta. Se optó por una cuarta opción: no asignar el ID nunca en la máquina del científico, sino en el pipeline, en el único punto donde ya no hay concurrencia real.

## Decisión

- Un experimento nuevo se crea siempre con el prefijo literal `IDXX` (carpeta `experiments/IDXX-nombre_del_experimento`, y `id: "IDXX"` dentro de `metadata.yaml`). Nadie asigna un número a mano.
- Al mergear a `main` (no en la validación del PR), el stage `AssignExperimentIds` de [azure-pipelines.yml](../../azure-pipelines.yml) corre `julieta_assign_ids` ([assign_experiment_ids.py](../../src/julieta/utils/assign_experiment_ids.py)), que:
  1. Calcula el siguiente número libre a partir de los `ID<n>` ya asignados.
  2. Resuelve **todas** las carpetas `IDXX-*` pendientes en una sola pasada (no una por corrida), para que dos merges casi simultáneos no compitan por el mismo número.
  3. Renombra cada carpeta, reescribe el campo `id` en su `metadata.yaml`, y actualiza el encabezado `# IDXX - ...` de su `README.md` si existe (`metadata.yaml` sigue siendo la fuente de verdad — si el encabezado no matchea, no falla la asignación, solo no se toca).
  4. Comitea el resultado directamente a `main` con el mensaje `[skip ci]`.
- El trigger de CI usa `batch: true` para no correr dos pipelines en paralelo sobre `main`.
- **El código vive en GitHub; Azure Pipelines solo lo consume como fuente externa** (conexión de servicio de GitHub), no hay Azure Repos en este flujo. La rama `main` está protegida con una branch protection rule de GitHub (exigir PR), y el push automático del bot necesita una excepción a esa regla — en GitHub esto no es un permiso separado como en Azure Repos, sino una de estas dos formas:
  - El token/identidad que usa el pipeline para pushear pertenece a una cuenta con rol admin del repo, y la regla no tiene marcado "Do not allow bypassing the above settings" (los admins la saltan por defecto).
  - O, con GitHub Rulesets (el reemplazo moderno de branch protection), se agrega explícitamente esa identidad/app a la lista de "bypass" de la regla.

  Se evaluó la alternativa de que el bot abra un PR aparte en vez de pushear directo (más alineado con gobernanza estricta), pero se descartó por la fricción de requerir aprobación manual en cada asignación.
- `julieta_validate` exige que el `id` de cada experimento sea `"IDXX"` o siga el patrón `ID<numero>`, y que el prefijo de la carpeta coincida exactamente con ese `id` (ver [metadata_schema.py](../../src/julieta/utils/metadata_schema.py) y [validate_metadata.py](../../src/julieta/utils/validate_metadata.py)).

## Consecuencias

- Ningún científico puede provocar una colisión de ID, sin importar cuántos mergeen casi al mismo tiempo: el número siempre lo decide el pipeline, en un punto serializado.
- Se mantiene el estilo de ID legible (`ID01`, `ID02`...) sin pasar a timestamps o UUIDs.
- El pipeline necesita una excepción a la branch protection rule de `main` en GitHub — es una rendija deliberada y acotada a un solo bot, no una relajación general de la regla.
- Depender de que un admin de GitHub bypasea por defecto es más frágil que el permiso explícito y granular que existe en Azure Repos ("Bypass policies when pushing"); si la organización endurece esa configuración (marca "incluir administradores"), este ADR debe revisarse — la alternativa natural sigue siendo que el bot abra un PR de asignación en vez de pushear directo.
