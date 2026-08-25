# Cómo crear un experimento nuevo

1. Copia esta carpeta (`experiments/_template`) a `experiments/IDXX-nombre_del_experimento`. Deja el prefijo `IDXX` literal — **no le asignes un número a mano**, así evitamos que dos personas elijan el mismo ID a la vez.
2. Llena el resto de `metadata.yaml` (autor, estado, dataset, etc.) dejando `id: "IDXX"`.
3. Llena las secciones de más abajo — quedan como el README real de tu experimento.
4. Corre `uv run julieta-validate` para confirmar que el metadata quedó bien formado.
5. Abre tu PR normalmente. Al mergear a `main`, el pipeline asigna automáticamente el siguiente ID disponible (`ID07`, por ejemplo) y renombra la carpeta por ti — ver [ADR 0003](../../docs/decisions/0003-experiment-id-assignment.md).

Borra esta sección cuando copies el template — de aquí para abajo es lo que se llena.

---

# [ID] - Nombre del experimento

## Resumen

(2-3 líneas: qué se probó y cuál fue el resultado, en una frase. Se llena al final — es lo primero que lee alguien que entra a este experimento sin contexto.)

## Motivación

(Qué problema de negocio o producto motiva esto, quién lo pidió, qué pasa si no se hace. El "por qué" antes que el "cómo".)

## Hipótesis

(Qué se espera demostrar o refutar, como una afirmación verificable. Ej: "Agregar features temporales de actividad mejora el ROC-AUC del modelo de priorización en al menos 3 puntos sobre el baseline actual".)

## Datos

- Dataset y versión usada (debe coincidir con `dataset_version` en `metadata.yaml`).
- Fuente y período que cubre.
- Nivel de sensibilidad y confirmación de que se siguió [docs/architecture/data-governance.md](../../docs/architecture/data-governance.md) — ¿el dataset ya llegó anonimizado?

## Metodología

(Qué modelos/técnicas se prueban y por qué, qué features se consideran, cómo se diseñó la validación — train/test split, cross-validation — y cómo se evita data leakage.)

## Métricas de éxito

(Métrica primaria, baseline a superar, umbral que definiría éxito — debe ser consistente con la sección `metrics` de `metadata.yaml`.)

## Estructura

- `metadata.yaml`: información básica, estado y métricas del experimento (ver campos comentados en el archivo).
- `configs/`: configuración usada por el experimento (hiperparámetros, paths, etc.) — específica de este experimento, no confundir con `configs/` en la raíz del repo (esa es configuración global compartida).
- `notebooks/`: notebooks del experimento, sin subcarpetas fijas. Prefija con un número para indicar el orden de ejecución, por ejemplo:
  ```
  01_eda.ipynb
  02_feature_engineering.ipynb
  03_modeling_baseline.ipynb
  ```
  Ver [ADR 0001](../../docs/decisions/0001-flatten-experiment-notebooks.md) para el porqué de esta convención.
- `results/`: artefactos de resultados (no se versiona el contenido, solo la carpeta).
- `reports/`: reportes/resúmenes del experimento (no se versiona el contenido, solo la carpeta).

## Cómo reproducir

(Pasos concretos para correr esto de principio a fin: comandos, en qué orden correr los notebooks, configuración necesaria antes de empezar.)

## Resultados

(Tabla o resumen de las métricas obtenidas vs. las esperadas, gráficos clave o dónde encontrarlos en `reports/`, y el link al run de MLflow si ya existe — ver [ADR 0005](../../docs/decisions/0005-mlflow-tracking.md).)

## Conclusiones y siguientes pasos

(¿Se acepta o se rechaza la hipótesis? ¿Qué se aprendió? ¿Esto pasa a `pipelines/` y `src/julieta/` porque maduró, se descarta, o da pie a un experimento derivado? Debe ser consistente con `conclusion` en `metadata.yaml`.)

## Limitaciones y riesgos conocidos

(Sesgos conocidos, limitaciones del dataset, advertencias para quien quiera reusar esto.)

## Referencias

(ADRs relevantes, PRs, tickets, papers o experimentos previos relacionados.)
