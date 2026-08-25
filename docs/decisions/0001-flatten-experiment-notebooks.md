# ADR 0001: Notebooks de experimentos en una sola carpeta plana

- Estado: aceptada
- Fecha: 2026-08-24

## Contexto

La primera versión del template de experimentos (`experiments/_template`) forzaba subcarpetas fijas dentro de `notebooks/`: `eda/`, `data_preparation/`, `feature_engineering/`, `modeling/`. En la práctica no todos los experimentos siguen ese mismo flujo lineal (algunos no tienen EDA separado, otros iteran entre feature engineering y modeling varias veces), y la estructura rígida obligaba a decidir de antemano en qué carpeta iría cada notebook.

## Decisión

Se elimina la subdivisión fija y `notebooks/` queda como una única carpeta plana por experimento. La convención sugerida es prefijar los notebooks con un número que indique el orden de ejecución, por ejemplo:

```
notebooks/
  01_eda.ipynb
  02_feature_engineering.ipynb
  03_modeling_baseline.ipynb
  03b_modeling_alt_approach.ipynb
```

El número da orden de lectura/ejecución; el nombre después del número es libre. Esto se documenta también en `experiments/_template/README.md`.

## Consecuencias

- Cada experimento tiene libertad para organizar sus notebooks según su propio flujo, sin carpetas vacías impuestas por convención.
- Se pierde la garantía estructural de "todo EDA vive en la misma carpeta en todos los experimentos"; la disciplina de nombrado pasa a depender de la convención documentada, no de la estructura de carpetas.
- `generate_summary.py` y `validate_metadata.py` no se ven afectados: no dependen de la estructura interna de `notebooks/`.
