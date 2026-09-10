import h5py
import numpy as np

from julieta.data.loaders.impedance_h5 import load_complex_impedance


def _write_signal_h5(path, patient_ids, values_left, values_right, study_key="synthetic_study"):
    with h5py.File(path, "w") as h5file:
        measurements = h5file.create_group("measurements")
        left = measurements.create_group("left_breast")
        left.create_dataset(study_key, data=values_left)
        right = measurements.create_group("right_breast")
        right.create_dataset(study_key, data=values_right)

        patient_group = h5file.create_group("patient_id")
        patient_group.create_dataset(study_key, data=np.array(patient_ids, dtype="S24"))
        h5file.create_dataset(
            "frequency_samples", data=np.array(["5000", "10000", "50000"], dtype="S10")
        )
        h5file.create_dataset("nodes", data=np.array(["12", "13"], dtype="S2"))


def test_load_complex_impedance_combines_resistance_and_reactance(tmp_path):
    resistance_ids = ["p0", "p1", "p2"]
    reactance_ids = ["p1", "p2", "p3"]  # deliberadamente distinto: p0 y p3 no se solapan

    resistance_values = np.arange(3 * 2 * 3, dtype=float).reshape(3, 2, 3)
    reactance_values = -np.arange(3 * 2 * 3, dtype=float).reshape(3, 2, 3)

    _write_signal_h5(
        tmp_path / "resistance_SYNTH.h5", resistance_ids, resistance_values, resistance_values
    )
    _write_signal_h5(
        tmp_path / "reactance_SYNTH.h5", reactance_ids, reactance_values, reactance_values
    )

    result = load_complex_impedance("SYNTH", data_dir=tmp_path)

    # Solo p1 y p2 estan en ambos archivos -- esos son los unicos que deben sobrevivir.
    assert result["patient_id"]["synthetic_study"] == ["p1", "p2"]
    assert result["frequency_samples"] == ["5000", "10000", "50000"]
    assert result["nodes"] == ["12", "13"]

    left = result["measurements"]["left_breast"]["synthetic_study"]
    assert left.shape == (2, 2, 3)
    assert np.iscomplexobj(left)

    # p1 es la fila 1 en resistance y la fila 0 en reactance -- confirma que
    # se alinea por patient_id, no por posicion.
    expected_real = resistance_values[1]
    expected_imag = reactance_values[0]
    np.testing.assert_array_equal(left[0].real, expected_real)
    np.testing.assert_array_equal(left[0].imag, expected_imag)
