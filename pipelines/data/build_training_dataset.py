"""Script flexible para construir el dataset de entrenamiento con las columnas que quieras.

Sigue EXACTAMENTE los mismos pasos que
notebooks/modeling/MAVC-plantilla_modelo_desplegado.ipynb (diagnóstico ->
categóricos -> desconexiones -> features de medición -> filtros -> cruce por
lateralidad confirmada -> X/y), pero parametrizado: eliges qué señales, qué
estudios, qué columnas categóricas y qué filtros aplicar, en vez de editar
celdas de notebook a mano cada vez.

Reutiliza las mismas funciones "oficiales" de src/julieta que usa el notebook
(MakeDataset, label_mappings) -- no reimplementa el cálculo de nada, solo las
encadena de forma configurable.

CLI:
    python pipelines/data/build_training_dataset.py \\
        --signals impedance_phase resistance \\
        --studies SURA CAFAM CLINICA_DE_MAMA \\
        --select-nodes all

Como función (desde un notebook o script):
    from build_training_dataset import build_dataset
    X, y, df = build_dataset(signals=["impedance_phase"], studies=["SURA", "CAFAM"])
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from julieta.data.label_mappings import BREAST_DENSITY_LABEL_MAPPING, MASTER_LABEL_MAPPING
from julieta.data.make_dataset import MakeDataset
from julieta.features.compute_features import (
    ComputeAdvancedMagPhaseFeatures,
    ComputeAdvancedNyquistFeatures,
    ComputeBreastLevelStatistics,
)
from julieta.utils.logs import setup_logging
from julieta.utils.paths import data_processed_dir, data_raw_dir

logger = logging.getLogger(__name__)

# Estadísticos válidos para colapsar el eje de nodos en feature_mode="advanced_nyquist"
# (los mismos que expone ComputeBreastLevelStatistics.NODE_STATS).
VALID_NYQUIST_STATS = ("mean", "std", "cv", "median", "iqr", "min", "max")


def _load_diagnosis(studies):
    """Carga y limpia el diagnóstico de mamografía de cada estudio (igual que el notebook).

    Lee data/raw/mammography_data_<study>.csv por cada estudio, los junta, y
    descarta los BI RADS 0 (igual que hace la celda de limpieza del notebook).
    """
    frames = [
        MakeDataset.get_mammography_data(
            data_raw_dir(f"mammography_data_{study}.csv"),
            patient_id_column="uid",
            diagnosis_column="mammogramCategory",
        )
        for study in studies
    ]
    df_diagnosis = pd.concat(frames)
    id_col = "uid" if "uid" in df_diagnosis.columns else "patient_id"
    df_diagnosis.index = pd.Index(df_diagnosis[id_col].astype(str), name="patient_id")
    df_diagnosis = df_diagnosis.drop(columns=[id_col])
    df_diagnosis = df_diagnosis[df_diagnosis["mammogramCategory"] != "BI RADS 0"]
    return df_diagnosis


def _load_categoricals(studies, df_diagnosis=None):
    """Carga los categóricos de cada estudio y agrega las columnas derivadas del notebook.

    Además de los datos crudos del CSV (edad, peso, talla de copa, etc.), agrega:
    - breastDensity: tomada del diagnóstico y mapeada a número (solo si se pasa df_diagnosis).
    - age_categorical: 1 si el paciente tiene 50 años o más.
    - bmi_category: bins de índice de masa corporal.

    df_diagnosis es opcional: algunos estudios (ej. SANTAFE) no tienen
    mammography_data en absoluto, así que no hay de dónde sacar breastDensity
    -- para esos casos se pasa df_diagnosis=None y esa columna simplemente no
    se agrega, en vez de fallar.

    Estas columnas quedan disponibles para elegirlas con --categorical-columns
    igual que cualquier columna cruda del CSV.
    """
    frames = [
        MakeDataset.get_categoricals_data(
            data_raw_dir(f"categoricals_data_{study}.csv"),
            patient_id_column="uid",
        )
        for study in studies
    ]
    df_categoricals = pd.concat(frames)
    id_col = "uid" if "uid" in df_categoricals.columns else "patient_id"
    df_categoricals.index = pd.Index(df_categoricals[id_col].astype(str), name="patient_id")
    df_categoricals = df_categoricals.drop(columns=[id_col])
    # Algunos estudios (confirmado en SANTAFE: 18 de 1956 filas) tienen más de
    # una fila para el mismo patient_id (ej. visitas repetidas) -- sin esto,
    # cualquier reindex() posterior por patient_id revienta con
    # "cannot reindex on an axis with duplicate labels". Nos quedamos con la
    # última fila de cada paciente.
    n_duplicated = df_categoricals.index.duplicated(keep="last").sum()
    if n_duplicated > 0:
        logger.warning(
            "%s filas de categóricos duplicadas por patient_id (de %s estudios); se conserva la última.",
            n_duplicated,
            studies,
        )
        df_categoricals = df_categoricals[~df_categoricals.index.duplicated(keep="last")]

    if (
        "companyId" in df_categoricals.columns
        and "registeredByCompanyId" in df_categoricals.columns
    ):
        idx_na_companyid = df_categoricals[df_categoricals["companyId"].isna()].index
        df_categoricals.loc[idx_na_companyid, "companyId"] = df_categoricals.loc[
            idx_na_companyid, "registeredByCompanyId"
        ]

    df_categoricals["braCupSize"] = df_categoricals["braCupSize"].map(
        {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "dd": 5}
    )
    df_categoricals["hormonalContraception"] = df_categoricals["hormonalContraception"].astype(int)
    df_categoricals["hormonalTherapyTreatment"] = df_categoricals[
        "hormonalTherapyTreatment"
    ].astype(int)
    df_categoricals["menopause"] = df_categoricals["menopause"].astype(int)

    # "height"/"weight" == 0 es dato faltante mal codificado, no una estatura o
    # peso real (confirmado: en SURA la MEDIANA de ambas columnas es 0.0 --
    # más de la mitad de sus pacientes tienen esto). Sin este fix, el promedio
    # combinado de estudios queda arrastrado a valores fisiológicamente
    # imposibles (~103cm de estatura promedio).
    n_zero_height = (df_categoricals["height"] == 0).sum()
    n_zero_weight = (df_categoricals["weight"] == 0).sum()
    if n_zero_height > 0 or n_zero_weight > 0:
        logger.warning(
            "height==0 en %s filas y weight==0 en %s filas (de %s estudios); se tratan como NaN, no como valor real.",
            n_zero_height,
            n_zero_weight,
            studies,
        )
    df_categoricals.loc[df_categoricals["height"] == 0, "height"] = np.nan
    df_categoricals.loc[df_categoricals["weight"] == 0, "weight"] = np.nan
    df_categoricals["height"] = df_categoricals["height"] * 100

    if df_diagnosis is not None:
        df_categoricals["breastDensity"] = (
            df_diagnosis["breastDensity"]
            .reindex(df_categoricals.index)
            .map(BREAST_DENSITY_LABEL_MAPPING)
        )
    df_categoricals["age_categorical"] = (df_categoricals["age"] >= 50).astype(int)

    # "bmi" no existe en el CSV de todos los estudios (ej. SURA/CAFAM no lo
    # traen, solo CLINICA_DE_MAMA) -- si ninguno de los estudios pedidos lo
    # trae, se salta esta columna derivada en vez de reventar con KeyError.
    if "bmi" in df_categoricals.columns:
        bins = [0, 18.5, 25, 30, 35, 40, float("inf")]
        labels = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad I", "Obesidad II", "Obesidad III"]
        df_categoricals["bmi_category"] = pd.cut(
            df_categoricals["bmi"], bins=bins, labels=labels, right=False
        ).astype(str)
    else:
        logger.warning(
            "Ninguno de los estudios pedidos trae la columna 'bmi'; se omite 'bmi_category'."
        )

    return df_categoricals


# Mapeo estudio -> archivo de control de calidad propio de ese estudio.
# "cmtest" es el nombre real del archivo para CLINICA_DE_MAMA (confirmado por
# solape de patient_id: 403 de 404 pacientes coinciden).
_DISCONNECTION_FILENAME_BY_STUDY = {
    "SURA": "disconnection_sura.csv",
    "CAFAM": "disconnection_cafam.csv",
    "CLINICA_DE_MAMA": "disconnection_cmtest.csv",
    "SANTAFE": "disconnection_santafe.csv",
}


def _load_per_study_disconnected_ids(study):
    """Excluye pacientes según el archivo de control de calidad propio del estudio.

    Los 4 archivos NO comparten esquema:
    - SANTAFE y CAFAM tienen columnas "left"/"right" explícitas (1 = desconexión
      confirmada en ese lado).
    - CAFAM y CLINICA_DE_MAMA ("cmtest") tienen "valid" (False = medición no
      usable).
    - SURA solo tiene "gaps_detected" (~26% de las filas, demasiado amplio y
      ruidoso para usarlo solo -- casi ninguno terminó necesitando corrección
      real) y "needs_interpolation" (unas pocas filas, señal fuerte real).

    Por eso solo se excluye por señal FUERTE: left/right==1, valid==False, o
    needs_interpolation==True -- nunca por gaps_detected solo. Se usa la unión
    de las columnas que existan en el archivo de ese estudio en particular.
    """
    filename = _DISCONNECTION_FILENAME_BY_STUDY.get(study)
    if filename is None:
        return set()
    path = data_processed_dir(filename)
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    exclude_mask = pd.Series(False, index=df.index)
    if "left" in df.columns and "right" in df.columns:
        exclude_mask |= (df["left"] == 1) | (df["right"] == 1)
    if "valid" in df.columns:
        exclude_mask |= df["valid"] == False  # noqa: E712 (comparación explícita por claridad con NaN)
    if "needs_interpolation" in df.columns:
        exclude_mask |= df["needs_interpolation"] == True  # noqa: E712

    excluded_ids = set(df.loc[exclude_mask, "uid"].astype(str))
    if excluded_ids:
        logger.info(
            "%s: %s de %s pacientes excluidos por control de calidad propio del estudio (%s).",
            study,
            len(excluded_ids),
            len(df),
            filename,
        )
    return excluded_ids


def _load_disconnections(studies=None):
    """Carga la lista de pacientes con medición no confiable por desconexión de electrodo.

    Combina el reetiquetado manual (data/raw/disconnections.xlsx) con el CSV
    aparte del estudio Maicao (data/processed/disconnection_maicao.csv) --
    igual que hacía el notebook original -- MÁS los archivos de control de
    calidad propios de cada estudio pedido (data/processed/disconnection_<...>.csv),
    que antes no se usaban en absoluto (confirmado: SANTAFE solo tenía 0
    exclusiones automáticas antes de este fix, a pesar de tener 132 pacientes
    marcados con desconexión en su propio archivo).

    Devuelve solo los patient_id a excluir -- no se usa para entrenar nada, es
    una lista negra de mediciones dudosas.
    """
    all_disconnections_init = pd.read_excel(data_raw_dir("disconnections.xlsx"), index_col=[0, 1])
    all_disconnections_init = all_disconnections_init[
        (all_disconnections_init["tipo"] == "desconexion")
        | (
            (all_disconnections_init["tipo"] == "recuperable")
            & (all_disconnections_init["comentarios"] == "El gap es grande")
        )
    ]

    uid_unique = all_disconnections_init.index.get_level_values("uid").unique()
    all_disconnections = pd.DataFrame(
        np.zeros([len(uid_unique), 2]), index=uid_unique, columns=["left", "right"]
    )
    for uid in all_disconnections.index:
        if (uid, "left_breast") in all_disconnections_init.index:
            all_disconnections.loc[uid, "left"] = 1
        elif (uid, "right_breast") in all_disconnections_init.index:
            all_disconnections.loc[uid, "right"] = 1

    maicao_path = data_processed_dir("disconnection_maicao.csv")
    if maicao_path.exists():
        disconnections_maicao = pd.read_csv(maicao_path, index_col=0)
        disconnections_maicao = disconnections_maicao.query("left==1 or right==1")
        all_disconnections = pd.concat([all_disconnections, disconnections_maicao])

    disconnected_ids = set(all_disconnections.index.astype(str))

    for study in studies or []:
        disconnected_ids |= _load_per_study_disconnected_ids(study)

    return pd.Index(sorted(disconnected_ids))


def _load_signal(signal, studies, select_nodes, select_frequencies):
    """Trae UNA señal (ej. impedance_phase) de todos los estudios pedidos, ya aplanada.

    Por cada estudio: lee el .h5 correspondiente (get_measurements_data) y lo
    convierte de un cubo paciente x nodo x frecuencia a una tabla de una fila
    por seno (flatten_data, data_representation="breast").
    """
    frames = []
    for study in studies:
        data = MakeDataset.get_measurements_data(
            output_data=signal,
            select_nodes=select_nodes,
            select_frequencies=select_frequencies,
            study=study,
        )
        flat = MakeDataset.flatten_data(
            data,
            values_key="measurements",
            features_key="frequency_samples",
            channels_key="nodes",
            data_representation="breast",
        )
        flat["study"] = study
        frames.append(flat)
    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def _load_all_signals(signals, studies, select_nodes, select_frequencies):
    """Trae y combina VARIAS señales como columnas de un mismo dataset.

    Cada columna de features queda prefijada con el nombre de la señal (ej.
    "impedance_phase__14_37000") para que no se pisen entre señales distintas
    que compartan el mismo nombre de nodo-frecuencia. Esto aplica incluso si
    solo pides una señal, para que el nombre de columna sea consistente sin
    importar cuántas señales combines.
    """
    study_column = None
    feature_frames = []
    for signal in signals:
        flat = _load_signal(signal, studies, select_nodes, select_frequencies)
        if study_column is None:
            study_column = flat["study"]
        feature_cols = [c for c in flat.columns if c not in ("study", "birads_label")]
        feature_frames.append(flat[feature_cols].add_prefix(f"{signal}__"))
    # pd.concat con axis=1 alinea por índice (patient_id, side): si un paciente
    # no tiene una de las señales, esa parte queda en NaN y se elimina más
    # adelante con dropna(), igual que hace el notebook con una sola señal.
    combined = pd.concat(feature_frames, axis=1)
    combined["study"] = study_column
    return combined


def _load_complex_impedance(study, select_nodes, select_frequencies):
    """Construye la impedancia compleja Z = resistance + j*reactance.

    El .h5 "impedance_<study>.h5" (ya complejo) solo existe para SURA -- no
    para CAFAM ni CLINICA_DE_MAMA. "resistance" y "reactance" sí existen para
    las 4 clínicas y son la misma información en cartesianas (Z = R + jX), así
    que se reconstruye Z a partir de esas dos, de forma consistente para
    todos los estudios (no solo para los que les falta el .h5 de impedance).
    """
    resistance = MakeDataset.get_measurements_data(
        output_data="resistance",
        select_nodes=select_nodes,
        select_frequencies=select_frequencies,
        study=study,
    )
    reactance = MakeDataset.get_measurements_data(
        output_data="reactance",
        select_nodes=select_nodes,
        select_frequencies=select_frequencies,
        study=study,
    )

    # patient_id está indexado solo por birad y se asume compartido entre
    # laterality "left_breast"/"right_breast" (misma convención que usa
    # MakeDataset.flatten_to_breast_representation). resistance.h5 y
    # reactance.h5 pueden tener un número de pacientes ligeramente distinto
    # por birad (confirmado en CAFAM: 228 vs 231) -- se alinea por patient_id
    # real, no por posición, y se usa la intersección para ambos lados.
    common_patient_ids_by_birad = {}
    for birad, r_ids in resistance["patient_id"].items():
        x_ids = set(reactance["patient_id"].get(birad, []))
        common = [pid for pid in r_ids if pid in x_ids]
        if len(common) < len(r_ids):
            logger.warning(
                "Study=%s birad=%s: resistance tiene %s pacientes y reactance %s; "
                "se usan solo los %s pacientes presentes en ambos.",
                study,
                birad,
                len(r_ids),
                len(x_ids),
                len(common),
            )
        common_patient_ids_by_birad[birad] = common

    measurements = {}
    for laterality, by_birad_r in resistance["measurements"].items():
        measurements[laterality] = {}
        for birad, r_arr in by_birad_r.items():
            if (
                laterality not in reactance["measurements"]
                or birad not in reactance["measurements"][laterality]
            ):
                continue
            x_arr = reactance["measurements"][laterality][birad]
            common_ids = common_patient_ids_by_birad.get(birad, [])
            if not common_ids:
                continue

            r_ids = list(resistance["patient_id"][birad])
            x_ids = list(reactance["patient_id"][birad])
            if r_arr.shape[0] != len(r_ids) or x_arr.shape[0] != len(x_ids):
                logger.warning(
                    "Study=%s laterality=%s birad=%s: el tamaño del array no coincide "
                    "con su propia lista de patient_id (r=%s vs %s, x=%s vs %s); se omite este grupo.",
                    study,
                    laterality,
                    birad,
                    r_arr.shape[0],
                    len(r_ids),
                    x_arr.shape[0],
                    len(x_ids),
                )
                continue

            r_pos = {pid: i for i, pid in enumerate(r_ids)}
            x_pos = {pid: i for i, pid in enumerate(x_ids)}
            r_indices = [r_pos[pid] for pid in common_ids]
            x_indices = [x_pos[pid] for pid in common_ids]
            measurements[laterality][birad] = r_arr[r_indices] + 1j * x_arr[x_indices]

    return {
        "measurements": measurements,
        "frequency_samples": resistance["frequency_samples"],
        "patient_id": common_patient_ids_by_birad,
        "nodes": resistance["nodes"],
    }


def _load_advanced_nyquist_features(studies, select_nodes, select_frequencies, stats):
    """Reemplaza las columnas nodo-frecuencia crudas por features físicas agregadas.

    Por cada estudio:
    1. Construye la impedancia COMPLEJA (real+imaginaria) a partir de
       resistance+reactance (ver _load_complex_impedance) -- ComputeAdvancedNyquistFeatures
       necesita ambas partes, no solo magnitud o fase.
    2. ComputeAdvancedNyquistFeatures().compute_features(...) ajusta, por cada
       nodo, un modelo Cole-Cole y uno RS-CPE (con su AIC comparativo), calcula
       geometría del arco de Nyquist, y fase/magnitud a frecuencias clínicas
       fijas -- 47 features por nodo en vez de una columna por frecuencia.
    3. ComputeBreastLevelStatistics.aggregate(...) colapsa el eje de nodos a
       estadísticos (mean/std/cv/median/iqr/min/max) -- de "un nodo x una fila"
       a "un resumen por seno", eliminando el problema de columnas por nodo.
    4. flatten_data convierte esa estructura (ya 2D: fila x feature) a un
       DataFrame indexado por (patient_id, side), igual que el resto del script.

    `stats` filtra cuáles de los 7 estadísticos agregados quedarse (por
    defecto mean/std/cv, para no volver a inflar la dimensionalidad -- pasar
    los 7 da 47 * 7 = 329 columnas en vez de 47 * len(stats)).
    """
    extractor = ComputeAdvancedNyquistFeatures()
    study_column = None
    frames = []
    for study in studies:
        data = _load_complex_impedance(study, select_nodes, select_frequencies)
        node_features = extractor.compute_features(data)
        breast_features = ComputeBreastLevelStatistics.aggregate(node_features)
        flat = MakeDataset.flatten_data(
            breast_features,
            values_key="values",
            features_key="features",
            data_representation="breast",
        )
        # flatten_data nombra este nivel de índice "laterality" para datos ya
        # agregados (2D); el resto del script (filter_data_by_confirmed_laterality
        # incluido) espera que se llame "side", igual que en modo "raw".
        flat = flat.rename_axis(index={"laterality": "side"})
        flat = flat.drop(columns=[c for c in ("birads_label",) if c in flat.columns])
        flat["study"] = study
        if study_column is None:
            study_column = flat["study"]
        frames.append(flat)
    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]

    if stats is not None:
        keep_cols = [
            c for c in combined.columns if c == "study" or any(c.endswith(f"__{s}") for s in stats)
        ]
        combined = combined[keep_cols]

    return combined


def _load_advanced_magphase_features(studies, select_nodes, select_frequencies, stats):
    """Variante experimental de _load_advanced_nyquist_features(): magnitud/fase en vez de R/X.

    Mismos 4 pasos (impedancia compleja -> features por nodo -> agregación
    por seno -> aplanado), pero usando ComputeAdvancedMagPhaseFeatures en
    vez de ComputeAdvancedNyquistFeatures -- ver esa clase para el detalle
    de qué cambia (geometría sobre |Z|/fase, ajustes con residuo en
    log|Z|+fase). El bloque Bode (12 columnas) sale idéntico entre ambos
    modos, porque ya estaba basado en magnitud/fase desde el origen.
    """
    extractor = ComputeAdvancedMagPhaseFeatures()
    study_column = None
    frames = []
    for study in studies:
        data = _load_complex_impedance(study, select_nodes, select_frequencies)
        node_features = extractor.compute_features(data)
        breast_features = ComputeBreastLevelStatistics.aggregate(node_features)
        flat = MakeDataset.flatten_data(
            breast_features,
            values_key="values",
            features_key="features",
            data_representation="breast",
        )
        flat = flat.rename_axis(index={"laterality": "side"})
        flat = flat.drop(columns=[c for c in ("birads_label",) if c in flat.columns])
        flat["study"] = study
        if study_column is None:
            study_column = flat["study"]
        frames.append(flat)
    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]

    if stats is not None:
        keep_cols = [
            c for c in combined.columns if c == "study" or any(c.endswith(f"__{s}") for s in stats)
        ]
        combined = combined[keep_cols]

    return combined


def build_dataset(
    signals,
    studies,
    select_nodes="all",
    select_frequencies=None,
    categorical_columns=None,
    exclude_disconnections=True,
    require_categoricals_and_diagnosis=True,
    filter_by_confirmed_laterality=True,
    output_filename=None,
    persist=True,
    feature_mode="raw",
    nyquist_stats=("mean", "std", "cv"),
):
    """Construye el dataset de entrenamiento con los parámetros que elijas.

    Parámetros
    ----------
    signals : list[str]
        Señales a incluir como features, ej. ["impedance_phase", "resistance"].
        Valores válidos: los mismos que acepta MakeDataset.get_measurements_data
        (impedance_phase, impedance_magnitude, resistance, reactance, etc.).
    studies : list[str]
        Estudios/clínicas a incluir, ej. ["SURA", "CAFAM", "CLINICA_DE_MAMA"].
    select_nodes : str
        Qué nodos/electrodos incluir ("all", "singulars", "opposites", etc.
        -- ver MakeDataset.get_measurements_data para la lista completa).
    select_frequencies : list[str] | None
        Frecuencias a incluir, o None para incluirlas todas.
    categorical_columns : list[str] | None
        Qué columnas categóricas del paciente incluir como features (ej.
        ["age", "braCupSize"]). Si es None (default), NO se incluye ningún
        categórico en X -- igual que el notebook de referencia, que tampoco
        los mezcla con las features de medición. Columnas disponibles: las
        crudas del CSV de categóricos más breastDensity/age_categorical/
        bmi_category (derivadas). Si pides una columna con muchos NaN (el CSV
        crudo tiene varias, ej. bmi, breastCancerHistory, menopauseAge), el
        script te avisa con un warning antes de que el dropna() final borre
        filas por su culpa.
    exclude_disconnections : bool
        Si True (default), excluye pacientes con medición marcada como
        desconexión de electrodo (misma fuente que usa el notebook).
    require_categoricals_and_diagnosis : bool
        Si True (default), solo deja pacientes que tengan tanto datos
        categóricos como diagnóstico disponibles.
    filter_by_confirmed_laterality : bool
        Si True (default), solo deja el seno (izquierdo/derecho) que coincide
        con el lado donde el diagnóstico reportó el hallazgo.
    output_filename : str | None
        Nombre del CSV de salida. Si es None, se genera uno automático con las
        señales, estudios y un timestamp.
    persist : bool
        Si True (default), guarda el resultado como CSV en data/processed/
        junto con un manifiesto .json con los parámetros usados.
    feature_mode : "raw" | "advanced_nyquist" | "advanced_magphase"
        "raw" (default): columnas nodo-frecuencia crudas de `signals`, igual
        que el notebook de referencia.
        "advanced_nyquist": en vez de columnas crudas, ajusta Cole-Cole/RS-CPE
        por nodo y agrega por seno (ver `_load_advanced_nyquist_features`).
        Reduce ~2880 columnas a ~59*len(nyquist_stats) columnas con
        significado físico, analizando la curva (resistencia, reactancia).
        "advanced_magphase": mismo esquema de 59 features/nodo, pero
        analizando la curva (magnitud, fase) en vez de (resistencia,
        reactancia) -- ver `ComputeAdvancedMagPhaseFeatures` para el detalle
        de qué cambia. Experimento para comparar ambas bases contra el mismo
        set de clasificadores.
        Cuando se usa "advanced_nyquist" o "advanced_magphase", `signals` se
        ignora (siempre se calcula sobre la impedancia compleja completa).
    nyquist_stats : tuple[str] | None
        Solo aplica con feature_mode="advanced_nyquist". Qué estadísticos de
        agregación por nodo conservar, de VALID_NYQUIST_STATS. None = los 7
        (329 columnas). Default (mean, std, cv) = 141 columnas.

    Devuelve
    --------
    (X, y, df_completo) :
        X : pd.DataFrame -- solo las columnas numéricas de features (medición +
            categóricas numéricas), indexado por (patient_id, side).
        y : pd.Series -- la etiqueta mammogramCategory mapeada a clases (-1, 0, 1, 2).
        df_completo : pd.DataFrame -- X + y + columnas de contexto (study,
            mammogramCategory crudo, relevantFindingLaterality, breastDensity),
            que es exactamente lo que se guarda en el CSV.
    """
    logger.info(
        "Building dataset signals=%s studies=%s select_nodes=%s", signals, studies, select_nodes
    )

    # 1. Diagnóstico de mamografía por estudio.
    df_diagnosis = _load_diagnosis(studies)

    # 2. Categóricos del paciente, ya limpios y con las columnas derivadas.
    # OJO: el CSV crudo de categóricos tiene columnas casi vacías (ej.
    # breastImplantType, lastMenstruation están 100% NaN; bmi y
    # breastCancerHistory están ~76% NaN). Por eso, si no pides columnas
    # explícitas, NO se mete ningún categórico a X -- así se evita el problema
    # de que un dropna() global se coma todas las filas por una sola columna
    # casi vacía que nadie pidió a propósito.
    df_categoricals_full = _load_categoricals(studies, df_diagnosis)
    if categorical_columns is not None:
        missing_cat_cols = [c for c in categorical_columns if c not in df_categoricals_full.columns]
        if missing_cat_cols:
            raise ValueError(f"Categorical columns not available: {missing_cat_cols}")
        nan_ratios = df_categoricals_full[categorical_columns].isna().mean()
        for col, ratio in nan_ratios.items():
            if ratio > 0:
                logger.warning(
                    "Columna categórica '%s' tiene %.0f%% de valores NaN.", col, ratio * 100
                )
        df_categoricals = df_categoricals_full[categorical_columns]
    else:
        df_categoricals = df_categoricals_full[[]]

    # 3. Features de medición: una o varias señales crudas, o Cole-Cole/Nyquist
    # agregado por seno (ver docstring de feature_mode).
    if feature_mode == "raw":
        flat_data = _load_all_signals(signals, studies, select_nodes, select_frequencies)
    elif feature_mode == "advanced_nyquist":
        flat_data = _load_advanced_nyquist_features(
            studies, select_nodes, select_frequencies, nyquist_stats
        )
    elif feature_mode == "advanced_magphase":
        flat_data = _load_advanced_magphase_features(
            studies, select_nodes, select_frequencies, nyquist_stats
        )
    else:
        raise ValueError(
            f"Unknown feature_mode: {feature_mode!r}. Use 'raw', 'advanced_nyquist' or 'advanced_magphase'."
        )

    # 4. Filtros opcionales (todos activos por defecto, igual que el notebook).
    if exclude_disconnections:
        disconnected_ids = _load_disconnections(studies)
        flat_data = flat_data[
            ~flat_data.index.get_level_values("patient_id").isin(disconnected_ids)
        ]

    if require_categoricals_and_diagnosis:
        flat_data = flat_data[
            flat_data.index.get_level_values("patient_id").isin(df_categoricals.index)
            & flat_data.index.get_level_values("patient_id").isin(df_diagnosis.index)
        ]

    flat_data = flat_data.dropna()
    flat_data = flat_data.loc[:, (flat_data != 0).any(axis=0)]

    # 5. Cruce por lateralidad confirmada: solo se queda con el seno (izquierdo
    # o derecho) que coincide con el lado donde el diagnóstico reportó el
    # hallazgo (o ambos, si el hallazgo es bilateral). filter_data_by_confirmed_laterality
    # ya devuelve el resultado con el mismo índice (patient_id, side) de entrada.
    if filter_by_confirmed_laterality:
        df_diag_laterality = (
            df_diagnosis[["mammogramCategory", "relevantFindingLaterality", "breastDensity"]]
            .reset_index()
            .assign(patient_id=lambda d: d["patient_id"].astype(str))
        )
        flat_data = MakeDataset.filter_data_by_confirmed_laterality(
            flat_data,
            df_diagnosis=df_diag_laterality,
            laterality_column="relevantFindingLaterality",
        )

    # 6. Se separan las columnas de contexto/etiqueta de las columnas numéricas de features.
    context_cols = [
        c
        for c in ["study", "mammogramCategory", "relevantFindingLaterality", "breastDensity"]
        if c in flat_data.columns
    ]
    context_data = flat_data[context_cols].copy()
    feature_data = flat_data.drop(columns=context_cols)

    # 7. Se pegan las columnas categóricas del paciente (alineadas por patient_id).
    patient_ids = feature_data.index.get_level_values("patient_id")
    categoricals_aligned = df_categoricals.reindex(patient_ids)
    categoricals_aligned.index = feature_data.index

    X = pd.concat([feature_data, categoricals_aligned], axis=1)
    X = X.select_dtypes(include=[np.number]).copy()
    rows_before_dropna = X.shape[0]
    X = X.dropna()
    if X.shape[0] < rows_before_dropna:
        logger.warning(
            "dropna() eliminó %s de %s filas (quedaron %s) por NaN en alguna columna numérica seleccionada.",
            rows_before_dropna - X.shape[0],
            rows_before_dropna,
            X.shape[0],
        )

    # 8. La etiqueta se toma del diagnóstico (crudo del cruce por lateralidad si
    # se aplicó ese filtro, o directo de df_diagnosis si no) y se mapea a clase.
    if "mammogramCategory" in context_data.columns:
        y_raw = context_data["mammogramCategory"].reindex(X.index)
    else:
        y_raw = df_diagnosis.loc[X.index.get_level_values("patient_id"), "mammogramCategory"]
        y_raw.index = X.index
    y = y_raw.map(MASTER_LABEL_MAPPING)

    df_completo = X.copy()
    df_completo["mammogramCategory"] = y_raw
    df_completo["label"] = y
    for col in context_cols:
        if col != "mammogramCategory":
            df_completo[col] = context_data[col].reindex(X.index)

    logger.info(
        "Dataset final: %s filas, %s columnas de features. Distribución de etiqueta: %s",
        X.shape[0],
        X.shape[1],
        y.value_counts().sort_index().to_dict(),
    )

    if persist:
        # En modo advanced_nyquist "signals" puede venir None (se ignora para
        # calcular features); se usa el nombre del feature_mode para el archivo.
        signals_for_naming = signals if feature_mode == "raw" else [feature_mode]
        _persist_dataset(df_completo, signals_for_naming, studies, output_filename)

    return X, y, df_completo


def build_unlabeled_dataset(
    studies,
    feature_mode="advanced_nyquist",
    signals=None,
    select_nodes="all",
    select_frequencies=None,
    nyquist_stats=("mean", "std", "cv"),
    categorical_columns=None,
    exclude_disconnections=True,
    output_filename=None,
    persist=True,
):
    """Construye un dataset de FEATURES para estudios sin diagnóstico (ej. SANTAFE).

    A diferencia de build_dataset(), esta función NO requiere
    mammography_data para los estudios pedidos -- por eso no aplica el cruce
    por lateralidad confirmada (necesita el diagnóstico) ni arma una etiqueta
    `y`. Sirve para aprendizaje semisupervisado o para comparar la
    distribución de un estudio sin etiqueta contra el dataset etiquetado.

    Usa exactamente las mismas columnas/features que build_dataset() con el
    mismo feature_mode, para poder concatenar ambos datasets sin problema.

    Devuelve
    --------
    X_unlabeled : pd.DataFrame indexado por (patient_id, side), con las
        mismas columnas de features (+ categóricas si se piden) que
        produciría build_dataset() con la misma configuración.
    """
    logger.info(
        "Building UNLABELED dataset studies=%s feature_mode=%s (sin diagnóstico)",
        studies,
        feature_mode,
    )

    if feature_mode == "raw":
        flat_data = _load_all_signals(signals, studies, select_nodes, select_frequencies)
    elif feature_mode == "advanced_nyquist":
        flat_data = _load_advanced_nyquist_features(
            studies, select_nodes, select_frequencies, nyquist_stats
        )
    elif feature_mode == "advanced_magphase":
        flat_data = _load_advanced_magphase_features(
            studies, select_nodes, select_frequencies, nyquist_stats
        )
    else:
        raise ValueError(
            f"Unknown feature_mode: {feature_mode!r}. Use 'raw', 'advanced_nyquist' or 'advanced_magphase'."
        )

    if exclude_disconnections:
        disconnected_ids = _load_disconnections(studies)
        flat_data = flat_data[
            ~flat_data.index.get_level_values("patient_id").isin(disconnected_ids)
        ]

    flat_data = flat_data.dropna()
    flat_data = flat_data.loc[:, (flat_data != 0).any(axis=0)]

    study_col = flat_data["study"] if "study" in flat_data.columns else None
    feature_data = flat_data.drop(columns=[c for c in ("study",) if c in flat_data.columns])

    if categorical_columns is not None:
        df_categoricals_full = _load_categoricals(studies, df_diagnosis=None)
        missing_cat_cols = [c for c in categorical_columns if c not in df_categoricals_full.columns]
        if missing_cat_cols:
            raise ValueError(f"Categorical columns not available: {missing_cat_cols}")
        nan_ratios = df_categoricals_full[categorical_columns].isna().mean()
        for col, ratio in nan_ratios.items():
            if ratio > 0:
                logger.warning(
                    "Columna categórica '%s' tiene %.0f%% de valores NaN.", col, ratio * 100
                )
        df_categoricals = df_categoricals_full[categorical_columns]
        patient_ids = feature_data.index.get_level_values("patient_id")
        categoricals_aligned = df_categoricals.reindex(patient_ids)
        categoricals_aligned.index = feature_data.index
        feature_data = pd.concat([feature_data, categoricals_aligned], axis=1)

    X_unlabeled = feature_data.select_dtypes(include=[np.number]).copy()
    rows_before_dropna = X_unlabeled.shape[0]
    X_unlabeled = X_unlabeled.dropna()
    if X_unlabeled.shape[0] < rows_before_dropna:
        logger.warning(
            "dropna() eliminó %s de %s filas (quedaron %s) por NaN en alguna columna numérica seleccionada.",
            rows_before_dropna - X_unlabeled.shape[0],
            rows_before_dropna,
            X_unlabeled.shape[0],
        )

    logger.info(
        "Dataset SIN ETIQUETA final: %s filas, %s columnas.",
        X_unlabeled.shape[0],
        X_unlabeled.shape[1],
    )

    if persist:
        df_to_save = X_unlabeled.copy()
        if study_col is not None:
            df_to_save["study"] = study_col.reindex(X_unlabeled.index)
        signals_for_naming = signals if feature_mode == "raw" else [feature_mode]
        output_filename = output_filename or None
        _persist_dataset(df_to_save, signals_for_naming, studies, output_filename)

    return X_unlabeled


def apply_qc_filter(base_csv_filename, studies, output_filename=None):
    """Re-aplica el filtro de desconexiones por estudio a un CSV ya cacheado, CON manifiesto.

    Pensado para cuando ya existe un CSV caro de recalcular (advanced_nyquist
    tarda ~35-40 min por los fits Cole-Cole) y solo se necesita corregirlo con
    una versión más reciente de `_load_disconnections()` -- sin volver a
    calcular features desde cero.

    A diferencia de una corrección manual ad-hoc (editar el CSV a mano y
    guardarlo), esto SIEMPRE pasa por `_persist_dataset()`, así que el archivo
    corregido queda con su `.manifest.json` al lado. Antes de esta función,
    `healthy_advanced_nyquist_dataset_qc.csv` y las variantes de SANTAFE se
    corrigieron a mano y se quedaron sin manifiesto -- nadie podía saber
    después cuántos pacientes se habían excluido ni de qué CSV venían.
    """
    base_path = data_processed_dir(base_csv_filename)
    df = pd.read_csv(base_path, index_col=[0, 1])
    n_before = df.index.get_level_values("patient_id").nunique()

    disconnected_ids = _load_disconnections(studies)
    df_qc = df[~df.index.get_level_values("patient_id").isin(disconnected_ids)]
    n_after = df_qc.index.get_level_values("patient_id").nunique()
    logger.info(
        "apply_qc_filter: %s pacientes excluidos por desconexion (de %s, quedan %s).",
        n_before - n_after,
        n_before,
        n_after,
    )

    if output_filename is None:
        output_filename = base_csv_filename.replace(".csv", "_qc.csv")
    _persist_dataset(df_qc, ["advanced_nyquist"], studies, output_filename)
    return df_qc


def _persist_dataset(df, signals, studies, output_filename):
    """Guarda el dataset como CSV en data/processed/, con un manifiesto .json al lado.

    El manifiesto deja registro de qué señales/estudios se usaron y cuántas
    filas/columnas quedaron, para poder reconocer después con qué parámetros
    se generó cada CSV (igual que hace run_inference.py con cada corrida).
    """
    out_dir = Path(data_processed_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if output_filename is None:
        signals_part = "-".join(signals)
        studies_part = "-".join(studies)
        output_filename = f"dataset_{signals_part}_{studies_part}_{timestamp}.csv"

    csv_path = out_dir / output_filename
    df.to_csv(csv_path)

    manifest = {
        "signals": signals,
        "studies": studies,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "timestamp": timestamp,
        "csv_file": output_filename,
    }
    manifest_path = csv_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Dataset guardado en %s (manifiesto: %s)", csv_path, manifest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build a training dataset with configurable signals, studies, columns and filters."
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=None,
        help="e.g. impedance_phase resistance (ignorado si --feature-mode advanced_nyquist)",
    )
    parser.add_argument(
        "--studies", nargs="+", required=True, help="e.g. SURA CAFAM CLINICA_DE_MAMA"
    )
    parser.add_argument("--select-nodes", default="all")
    parser.add_argument("--select-frequencies", nargs="*", default=None)
    parser.add_argument("--categorical-columns", nargs="*", default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["raw", "advanced_nyquist", "advanced_magphase"],
        default="raw",
    )
    parser.add_argument(
        "--nyquist-stats",
        nargs="*",
        choices=list(VALID_NYQUIST_STATS),
        default=["mean", "std", "cv"],
    )
    parser.add_argument("--no-exclude-disconnections", action="store_true")
    parser.add_argument("--no-require-categoricals-diagnosis", action="store_true")
    parser.add_argument("--no-filter-laterality", action="store_true")
    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    if args.feature_mode == "raw" and not args.signals:
        parser.error("--signals is required when --feature-mode raw")

    setup_logging()
    X, y, _ = build_dataset(
        signals=args.signals,
        studies=args.studies,
        select_nodes=args.select_nodes,
        select_frequencies=args.select_frequencies,
        categorical_columns=args.categorical_columns,
        exclude_disconnections=not args.no_exclude_disconnections,
        require_categoricals_and_diagnosis=not args.no_require_categoricals_diagnosis,
        filter_by_confirmed_laterality=not args.no_filter_laterality,
        output_filename=args.output_filename,
        persist=not args.no_persist,
        feature_mode=args.feature_mode,
        nyquist_stats=tuple(args.nyquist_stats) if args.nyquist_stats else None,
    )
    print(f"X shape: {X.shape}")
    print(f"y distribution: {y.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
