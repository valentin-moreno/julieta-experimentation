"""Lee resistance_<study>.h5 + reactance_<study>.h5 de data/raw/ y arma la impedancia compleja.

Los .h5 que descarga pipelines/data/download/download_clinical_data.py traen
resistance y reactance por separado (partes real e imaginaria en cartesianas),
nunca la impedancia compleja directamente -- Z = R + jX se arma acá, no en la
fuente. Es lo único que ComputeAdvancedNyquistFeatures/ComputeColeColeFeatures
necesitan para ajustar sus modelos (ver features/compute_features.py).
"""

from pathlib import Path

import h5py


def _read_signal_h5(path: Path) -> dict:
    with h5py.File(path, "r") as h5file:
        study_key = next(iter(h5file["patient_id"].keys()))
        return {
            "study_key": study_key,
            "measurements": {
                laterality: h5file["measurements"][laterality][study_key][...]
                for laterality in h5file["measurements"]
            },
            "patient_id": [pid.decode() for pid in h5file["patient_id"][study_key][...]],
            "frequency_samples": [freq.decode() for freq in h5file["frequency_samples"][...]],
            "nodes": [node.decode() for node in h5file["nodes"][...]],
        }


def load_complex_impedance(study: str, data_dir: str | Path = "data/raw") -> dict:
    """Construye Z = resistance + j*reactance para un estudio, lista de nodos y frecuencias.

    Devuelve la forma que espera `ComputeAdvancedNyquistFeatures.compute_features`
    y `ComputeColeColeFeatures.compute_features`:
    `{"measurements": {laterality: {study_key: ndarray complejo}}, "frequency_samples": [...],
    "patient_id": {study_key: [...]}, "nodes": [...]}`.

    resistance.h5 y reactance.h5 pueden traer un número de pacientes ligeramente
    distinto para el mismo estudio -- se alinea por patient_id real (no por
    posición) y se usa la interseccion para ambas lateralidades.
    """
    data_dir = Path(data_dir)
    resistance = _read_signal_h5(data_dir / f"resistance_{study}.h5")
    reactance = _read_signal_h5(data_dir / f"reactance_{study}.h5")
    study_key = resistance["study_key"]

    r_ids, x_ids = resistance["patient_id"], reactance["patient_id"]
    common_ids = [pid for pid in r_ids if pid in set(x_ids)]
    r_pos = {pid: i for i, pid in enumerate(r_ids)}
    x_pos = {pid: i for i, pid in enumerate(x_ids)}
    r_indices = [r_pos[pid] for pid in common_ids]
    x_indices = [x_pos[pid] for pid in common_ids]

    measurements = {
        laterality: {
            study_key: (
                resistance["measurements"][laterality][r_indices]
                + 1j * reactance["measurements"][laterality][x_indices]
            )
        }
        for laterality in resistance["measurements"]
    }

    return {
        "measurements": measurements,
        "frequency_samples": resistance["frequency_samples"],
        "patient_id": {study_key: common_ids},
        "nodes": resistance["nodes"],
    }
