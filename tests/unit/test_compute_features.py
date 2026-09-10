"""Smoke tests for the compute_features migration (fase 2, ver ADR 0010).

Datos 100% sintéticos, sin ninguna medición real -- solo confirma que las 3
funciones/clases portadas (ComputeColeColeFeatures, ComputeAdvancedNyquistFeatures,
ComputeBreastLevelStatistics) corren en este entorno y devuelven la forma
esperada, no que los valores sean clínicamente correctos.
"""

import numpy as np

from julieta.features.compute_features import (
    ComputeAdvancedNyquistFeatures,
    ComputeBreastLevelStatistics,
    ComputeColeColeFeatures,
)


def _synthetic_impedance_data_structure(n_samples=3, n_nodes=2, n_freqs=12):
    rng = np.random.default_rng(0)
    frequencies_hz = np.linspace(5_000, 100_000, n_freqs)

    def make_side():
        real = rng.uniform(200, 800, size=(n_samples, n_nodes, n_freqs))
        imag = -rng.uniform(20, 150, size=(n_samples, n_nodes, n_freqs))
        return real + 1j * imag

    return {
        "measurements": {
            "left_breast": {"synthetic_study": make_side()},
            "right_breast": {"synthetic_study": make_side()},
        },
        "frequency_samples": [str(int(f)) for f in frequencies_hz],
        "patient_id": {"synthetic_study": [f"p{i}" for i in range(n_samples)]},
        "nodes": [f"n{i}" for i in range(n_nodes)],
    }


def test_advanced_nyquist_features_run_on_synthetic_impedance():
    data = _synthetic_impedance_data_structure()
    extractor = ComputeAdvancedNyquistFeatures()

    result = extractor.compute_features(data)

    n_features = len(extractor.FEATURE_NAMES)
    assert result["features"] == list(extractor.FEATURE_NAMES)
    assert result["values"]["left_breast"]["synthetic_study"].shape == (3, 2, n_features)
    assert result["values"]["right_breast"]["synthetic_study"].shape == (3, 2, n_features)


def test_breast_level_statistics_collapses_node_axis():
    data = _synthetic_impedance_data_structure()
    node_features = ComputeAdvancedNyquistFeatures().compute_features(data)

    breast_features = ComputeBreastLevelStatistics.aggregate(node_features)

    n_features = len(node_features["features"])
    n_stats = len(ComputeBreastLevelStatistics.NODE_STATS)
    assert breast_features["values"]["left_breast"]["synthetic_study"].shape == (
        3,
        n_features * n_stats,
    )
    assert len(breast_features["features"]) == n_features * n_stats
    assert breast_features["features"][:n_stats] == [
        f"{node_features['features'][0]}__{stat}"
        for stat in ComputeBreastLevelStatistics.NODE_STATS
    ]


def test_cole_cole_compute_features_runs_on_synthetic_magnitude_phase():
    data = _synthetic_impedance_data_structure(n_samples=2, n_nodes=2, n_freqs=12)
    # ComputeColeColeFeatures ajusta sobre magnitud o fase, no impedancia compleja --
    # se deriva de la misma señal sintética de arriba.
    magnitude_structure = {
        "measurements": {
            laterality: {key: np.abs(array) for key, array in by_key.items()}
            for laterality, by_key in data["measurements"].items()
        },
        "frequency_samples": data["frequency_samples"],
        "patient_id": data["patient_id"],
        "nodes": data["nodes"],
    }

    result = ComputeColeColeFeatures().compute_features(
        magnitude_structure, measurements_type="impedance_magnitude", parallelize=False
    )

    assert len(result["features"]) == 8
    assert result["values"]["left_breast"]["synthetic_study"].shape == (2, 2, 8)
