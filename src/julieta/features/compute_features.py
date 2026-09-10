"""Central de calculo de variables sobre las senales de bioimpedancia (los .h5 en data/raw).

Un solo lugar, compartido entre experimentos, con las mismas formulas -- ver
la nota de madurez en README.md de esta carpeta: solo entran funciones
maduras y reutilizables, no calculos de un solo uso.

Portado (fase 2 de ADR 0010, ver docs/decisions/0010_migration_from_julieta_models.md)
desde julieta-models/src/julieta/features/compute_features.py, que tiene 3076
lineas y mezcla codigo maduro con variantes explicitamente marcadas como
"experimentales" (ComputeWaveletFeatures, MahalanobisDepthFeature,
ComputeMarkovFeatures) o sin evidencia de uso real.

- ComputeColeColeFeatures: ajusta el modelo Cole-Cole (R0, Rinf, tau, alpha)
  por nodo -- modelo fisico estandar en bioimpedancia, no experimental.
- ComputeAdvancedNyquistFeatures: geometria del plot de Nyquist (lineal y
  log-log), features de Bode, y ajustes RS-CPE/Cole-Cole con comparacion por
  AIC -- 47 features por nodo. Usada por IDXX-healthy_model_groupsplit
  (feature_mode="advanced_nyquist").
- ComputeBreastLevelStatistics: colapsa el eje de nodos a estadisticos
  (mean/std/cv/median/iqr/min/max) y calcula asimetria izquierda/derecha.
- ComputeAdvancedMagPhaseFeatures: variante de ComputeAdvancedNyquistFeatures
  sobre (magnitud, fase) en vez de (resistencia, reactancia) -- su propio
  docstring original la marca como experimental, pero SI se usa de verdad
  (feature_mode="advanced_magphase" del mismo experimento de arriba), por eso
  se incluyo pese a no cumplir a rajatabla la barra de madurez de este README.
- compute_statistics: estadisticos genericos (mean/std/median/skew/kurtosis/
  range), usada por build_training_dataset.py.
"""

from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from scipy.stats import kurtosis, skew


def compute_statistics(
    data_structure: dict[
        str,
        dict[str, dict[str, np.ndarray]]
        | dict[str, np.ndarray]
        | list[int | str]
        | list[float | int],
    ],
    values_key: str,
    axis: int,
    use_raw_data: bool = False,
) -> dict[
    str,
    dict[str, dict[str, np.ndarray]] | dict[str, np.ndarray] | list[int | str] | list[float | int],
]:
    """Colapsa `data_structure[values_key]` a 6 estadisticos por (laterality, bi_rad): mean, std,
    median, skewness, kurtosis, range -- apilados a lo largo de `axis`. `patient_id`/`features` del
    input pasan sin cambios; `use_raw_data` decide si el resultado se etiqueta "measurements" o
    "values" (el resto de este modulo usa "values").
    """

    # Initialize statistics dictionary
    statistics = {
        laterality: {br: [] for br in item.keys()}
        for laterality, item in data_structure[values_key].items()
    }

    # Compute statistics for each laterality and BI-RADS category
    for laterality, data_dict in data_structure[values_key].items():
        for bi_rad, data in data_dict.items():
            statistics[laterality][bi_rad] = np.stack(
                [
                    np.mean(data, axis=axis),  # Mean
                    np.std(data, axis=axis),  # Standard deviation
                    np.median(data, axis=axis),  # Median
                    skew(data, axis=axis),  # Skewness
                    kurtosis(data, axis=axis),  # Kurtosis
                    data.max(axis=axis) - data.min(axis=axis),  # Range
                ],
                axis=axis,  # Stack features along a new axis
            )

    if use_raw_data:
        # Structure computed values into a final output dictionary
        statistics_structure = {
            "measurements": statistics,
            "patient_id": data_structure["patient_id"],
            "statistics": [
                "mean",
                "std",
                "median",
                "skewness",
                "kurtosis",
                "range",
            ],
            "frequency_samples": data_structure["frequency_samples"],
        }

    else:
        # Structure computed values into a final output dictionary
        statistics_structure = {
            "values": statistics,
            "patient_id": data_structure["patient_id"],
            "statistics": [
                "mean",
                "std",
                "median",
                "skewness",
                "kurtosis",
                "range",
            ],
            "features": data_structure["features"],
        }

    return statistics_structure


class ComputeColeColeFeatures:
    def __init__(self):
        pass

    def model(
        self,
        f: float | np.ndarray,
        R0: float,
        Rinf: float,
        tau: float,
        alpha: float,
    ) -> float | np.ndarray:
        """
        Computes the Cole-Cole model for complex impedance.

        Parameters
        ----------
        f : Union[float, np.ndarray]
            Frequency (in Hz). Can be a single float or an array of frequencies.
        R0 : float
            Resistance at zero frequency, indicating the system's low-frequency response.
        Rinf : float
            Resistance at infinite frequency, representing the system's high-frequency response.
        tau : float
            Characteristic time constant (in seconds), defining the relaxation behavior.
        alpha : float
            Dispersion constant between 0 and 1, describing the deviation from ideal behavior.

        Returns
        -------
        Union[float, np.ndarray]
            The complex impedance at the specified frequencies. The output will be:
            - A single complex number if `f` is a float.
            - A numpy array of complex numbers if `f` is an array.

        Notes
        -----
        - The real part of the impedance corresponds to the resistive behavior.
        - The imaginary part of the impedance corresponds to the reactive (capacitive or inductive)
        behavior.
        - This model is commonly used in electrical impedance spectroscopy to analyze biological
        tissues, materials, and electrochemical systems.
        """
        omega = 2 * np.pi * f  # Angular frequency
        Z = Rinf + (R0 - Rinf) / (1 + (1j * omega * tau) ** alpha)
        return Z

    def real_part(
        self,
        f: float | np.ndarray,
        R0: float,
        Rinf: float,
        tau: float,
        alpha: float,
    ) -> float | np.ndarray:
        """
        Computes the Cole-Cole model for the impedance real part.

        Parameters
        ----------
        f : Union[float, np.ndarray]
            Frequency (in Hz). Can be a single float or an array of frequencies.
        R0 : float
            Resistance at zero frequency, indicating the system's low-frequency response.
        Rinf : float
            Resistance at infinite frequency, representing the system's high-frequency response.
        tau : float
            Characteristic time constant (in seconds), defining the relaxation behavior.
        alpha : float
            Dispersion constant between 0 and 1, describing the deviation from ideal behavior.

        Returns
        -------
        Union[float, np.ndarray]
            The impedance real part at the specified frequencies. The output will be:
            - A single float if `f` is a float.
            - A numpy array of real numbers if `f` is an array.

        """
        omega = 2 * np.pi * f
        omega_tau_alpha = (omega * tau) ** alpha
        cos_term = np.cos(np.pi * alpha / 2)
        denominator = 1 + 2 * omega_tau_alpha * cos_term + omega_tau_alpha**2
        real_part = Rinf + (R0 - Rinf) * (1 + omega_tau_alpha * cos_term) / denominator
        return real_part

    def imag_part(
        self,
        f: float | np.ndarray,
        R0: float,
        Rinf: float,
        tau: float,
        alpha: float,
    ) -> float | np.ndarray:
        """
        Computes the Cole-Cole model for the impedance imaginary part.

        Parameters
        ----------
        f : Union[float, np.ndarray]
            Frequency (in Hz). Can be a single float or an array of frequencies.
        R0 : float
            Resistance at zero frequency, indicating the system's low-frequency response.
        Rinf : float
            Resistance at infinite frequency, representing the system's high-frequency response.
        tau : float
            Characteristic time constant (in seconds), defining the relaxation behavior.
        alpha : float
            Dispersion constant between 0 and 1, describing the deviation from ideal behavior.

        Returns
        -------
        Union[float, np.ndarray]
            The impedance imaginary part at the specified frequencies. The output will be:
            - A single float if `f` is a float.
            - A numpy array of real numbers if `f` is an array.

        """
        omega = 2 * np.pi * f
        omega_tau_alpha = (omega * tau) ** alpha
        sin_term = np.sin(np.pi * alpha / 2)
        denominator = 1 + 2 * omega_tau_alpha * np.cos(np.pi * alpha / 2) + omega_tau_alpha**2
        imag_part = (R0 - Rinf) * omega_tau_alpha * sin_term / denominator
        return imag_part

    def magnitude(
        self,
        f: float | np.ndarray,
        R0: float,
        Rinf: float,
        tau: float,
        alpha: float,
    ) -> float | np.ndarray:
        """
        Computes the Cole-Cole impedance magnitude.

        Parameters
        ----------
        f : Union[float, np.ndarray]
            Frequency (in Hz). Can be a single float or an array of frequencies.
        R0 : float
            Resistance at zero frequency, indicating the system's low-frequency response.
        Rinf : float
            Resistance at infinite frequency, representing the system's high-frequency response.
        tau : float
            Characteristic time constant (in seconds), defining the relaxation behavior.
        alpha : float
            Dispersion constant between 0 and 1, describing the deviation from ideal behavior.

        Returns
        -------
        Union[float, np.ndarray]
            The impedance magnitude at the specified frequencies. The output will be:
            - A single float if `f` is a float.
            - A numpy array of real numbers if `f` is an array.

        """
        # Compute real and imaginary parts
        real_part = self.real_part(f, R0, Rinf, tau, alpha)
        imag_part = self.imag_part(f, R0, Rinf, tau, alpha)

        # Compute the magnitude of impedance
        magnitude = np.sqrt(real_part**2 + imag_part**2)

        return magnitude

    def phase(
        self,
        f: float | np.ndarray,
        R0: float,
        Rinf: float,
        tau: float,
        alpha: float,
    ) -> float | np.ndarray:
        """
        Computes the Cole-Cole impedance phase.

        Parameters
        ----------
        f : Union[float, np.ndarray]
            Frequency (in Hz). Can be a single float or an array of frequencies.
        R0 : float
            Resistance at zero frequency, indicating the system's low-frequency response.
        Rinf : float
            Resistance at infinite frequency, representing the system's high-frequency response.
        tau : float
            Characteristic time constant (in seconds), defining the relaxation behavior.
        alpha : float
            Dispersion constant between 0 and 1, describing the deviation from ideal behavior.

        Returns
        -------
        Union[float, np.ndarray]
            The impedance phase at the specified frequencies. The output will be:
            - A single float if `f` is a float.
            - A numpy array of real numbers if `f` is an array.

        """
        # Compute real and imaginary parts
        real_part = self.real_part(f, R0, Rinf, tau, alpha)
        imag_part = self.imag_part(f, R0, Rinf, tau, alpha)

        # Compute the phase of impedance in degees
        phase = np.degrees(np.arctan2(imag_part, real_part))

        return phase

    def residuals(
        self,
        params: list[float],
        frequencies: np.ndarray,
        measured: np.ndarray,
        measurements_type: str,
    ) -> np.ndarray:
        """
        Computes the residuals between measured and modeled impedance.

        Parameters
        ----------
        params : List[float]
            Cole-Cole parameters `[R₀, R∞, τ, α]`.
        frequencies : np.ndarray
            Frequency values (in Hz).
        measured : np.ndarray
            Measured impedance magnitudes.
        measurements_type : Literal["impedance_magnitude", "impedance_phase"]
            Type of measurement to fit. Options are:
            - "impedance_magnitude": Fit based on impedance magnitude.
            - "impedance_phase": Fit based on impedance phase (in degrees).

        Returns
        -------
        np.ndarray
            Residuals for the given parameters and measured data.
        """
        R0, Rinf, tau, alpha = params
        if measurements_type == "impedance_magnitude":
            model = self.magnitude(frequencies, R0, Rinf, tau, alpha)
        elif measurements_type == "impedance_phase":
            model = np.unwrap(self.phase(frequencies, R0, Rinf, tau, alpha))
            measured = np.unwrap(measured)
        else:
            raise ValueError(f"Invalid measurements_type: {measurements_type}")

        return model - measured

    def compute_node_parameters(
        self,
        sample_idx: int,
        node_idx: int,
        signal: np.ndarray,
        frequencies: np.ndarray,
        measurements_type: str,
        initial_guess: list[float],
        bounds: tuple[list[float], list[float]],
    ) -> np.ndarray:
        """Computes the Cole-Cole parameters and derived metrics for a given node.

        This function performs least squares fitting of the Cole-Cole model to the impedance data
        for a specific node and computes derived metrics such as the low-to-high frequency
        resistance
        ratio and weighted time constant.

        Parameters
        ----------
        sample_idx : int
            Index of the sample (patient or dataset instance) to which the node belongs.
        node_idx : int
            Index of the node for which the parameters are being computed.
        signal : np.ndarray
            Measured signal for the node, typically the impedance magnitude or phase values
            across the provided frequencies.
        frequencies : np.ndarray
            Array of frequency values (in Hz) corresponding to the measurements in `signal`.
        measurements_type : str
            Type of measurement to fit the model to:
            - "impedance_magnitude": Fits the impedance magnitude.
            - "impedance_phase": Fits the impedance phase (in degrees).
        initial_guess : List[float]
            Initial guess for the Cole-Cole parameters `[R0, Rinf, tau, alpha]`.
        bounds : Tuple[List[float], List[float]]
            Bounds for the fitting parameters, defined as a tuple of two lists:
            - Lower bounds `[R0_min, Rinf_min, tau_min, alpha_min]`.
            - Upper bounds `[R0_max, Rinf_max, tau_max, alpha_max]`.

        Returns
        -------
        np.ndarray
            An array of computed parameters and derived metrics with the following structure:
            - `[R0, Rinf, tau, alpha, residual_sum, low2high_ratio, low2high_diff, tau_weighted]`,
            where:
                - R0: Resistance at zero frequency.
                - Rinf: Resistance at infinite frequency.
                - tau: Characteristic time constant.
                - alpha: Dispersion parameter.
                - residual_sum: Sum of squared residuals from the fitting process.
                - low2high_ratio: Ratio of low-to-high frequency resistances (R0/Rinf).
                - low2high_diff: Difference between low and high-frequency resistances (R0 - Rinf).
                - tau_weighted: Weighted time constant (tau * (R0 - Rinf)).
            If fitting fails, the function returns an array filled with NaN values.

        Notes
        -----
        - The function handles division by zero when computing `low2high_ratio` by returning NaN.
        - If the fitting process fails, the function prints an error message and fills the result
        with NaN values.
        - The derived metrics are computed to provide additional insights into the behavior of the
        system.

        Raises
        ------
        Exception
            If an unexpected error occurs during the fitting process, it is caught, and the failure
            is logged.
        """
        try:
            # Perform least squares fitting
            result = least_squares(
                self.residuals,
                x0=initial_guess,
                bounds=bounds,
                args=(frequencies, signal, measurements_type),
                # loss="soft_l1",
                # max_nfev=10000,
            )

            # Extract fitted parameters
            R0, Rinf, tau, alpha = result.x

            # Compute derived metrics
            low2high_ratio = R0 / Rinf if Rinf != 0 else np.nan
            low2high_diff = R0 - Rinf
            tau_weighted = tau * low2high_diff

            # Return the parameters and derived metrics
            return np.append(
                [R0, Rinf, tau, alpha],
                [
                    result.cost,  # Residual sum of squares
                    low2high_ratio,  # Low-to-high frequency resistance ratio
                    low2high_diff,  # Low-to-high frequency resistance difference
                    tau_weighted,  # Weighted time constant
                ],
            )
        except Exception as e:
            print(f"Fitting failed for sample {sample_idx}, node {node_idx}: {e}")
            return np.full(8, np.nan)  # 8 metrics to fill with NaNs

    def compute_parameters(
        self,
        data: np.ndarray,
        frequencies: np.ndarray,
        measurements_type: Literal["impedance_magnitude", "impedance_phase"],
        initial_guess: list[float] = [1500.0, 500.0, 0.2, 0.5],
        bounds: tuple[list[float], list[float]] = (
            [0.0, 0.0, 1.0e-9, 0.0],
            [np.inf, np.inf, np.inf, 1.0],
        ),
    ) -> np.ndarray:
        """
        Computes Cole-Cole parameters for all samples and nodes in the given dataset.

        This function sequentially computes the Cole-Cole parameters for a 3D dataset of impedance
        measurements using a specified measurement type (magnitude or phase). It iterates over
        all samples and nodes, fitting the Cole-Cole model parameters for each node individually.

        Parameters
        ----------
        data : np.ndarray
            A 3D array of impedance values with shape `(samples, nodes, frequencies)`, where:
            - `samples`: Number of samples (e.g., patients or datasets).
            - `nodes`: Number of measurement nodes for each sample.
            - `frequencies`: Impedance values corresponding to the provided frequencies.
        frequencies : np.ndarray
            A 1D array of frequency values (in Hz) corresponding to the last axis of `data`.
        measurements_type : Literal["impedance_magnitude", "impedance_phase"]
            Specifies the type of measurement to fit:
            - "impedance_magnitude": Fit based on impedance magnitude.
            - "impedance_phase": Fit based on impedance phase (in degrees).
        initial_guess : List[float], optional
            Initial guesses for the Cole-Cole parameters `[R₀, R∞, τ, α]`. Defaults to `[1500.0,
            500.0, 0.2, 0.5]`.
        bounds : Tuple[List[float], List[float]], optional
            Bounds for the Cole-Cole parameters, defined as a tuple of two lists:
            - Lower bounds `[R₀_min, R∞_min, τ_min, α_min]`.
            - Upper bounds `[R₀_max, R∞_max, τ_max, α_max]`.
            Defaults to `([0.0, 0.0, 1.0e-9, 0.0], [np.inf, np.inf, np.inf, 1.0])`.

        Returns
        -------
        np.ndarray
            A 3D array with shape `(samples, nodes, features)` where the last dimension contains:
            - `[R₀, R∞, τ, α, residual_sum, low2high_ratio, low2high_diff, tau_weighted]` for each
            sample and node.

        Raises
        ------
        ValueError
            If `data` is not a 3D array or if `frequencies` is not a 1D array.
        ValueError
            If the number of frequencies in `data` does not match the length of the `frequencies`
            array.

        Notes
        -----
        - Each node's parameters are computed sequentially, which may take longer for large
        datasets.
        - Derived metrics such as the low-to-high resistance ratio (`R₀ / R∞`), resistance
        difference
        (`R₀ - R∞`), and weighted time constant (`τ * (R₀ - R∞)`) are included in the output.
        - Fitting failures for any node will result in NaN values for that node's parameters and
        metrics.
        """
        # Validate inputs
        if data.ndim != 3 or frequencies.ndim != 1:
            raise ValueError("`data` must be 3D, and `frequencies` must be 1D.")
        if data.shape[-1] != len(frequencies):
            raise ValueError("Mismatch between `data` frequencies and `frequencies` array.")

        sample_data = []  # List to store parameters for all samples
        for sample_idx in range(data.shape[0]):  # Iterate over samples
            node_data = []  # List to store parameters for each node
            for node_idx in range(data.shape[1]):  # Iterate over nodes
                node_data.append(
                    self.compute_node_parameters(
                        sample_idx=sample_idx,
                        node_idx=node_idx,
                        signal=data[
                            sample_idx, node_idx, :
                        ],  # Extract the signal for the current node
                        frequencies=frequencies,
                        measurements_type=measurements_type,
                        initial_guess=initial_guess,
                        bounds=bounds,
                    )
                )

            # Stack parameters for all nodes in the current sample
            sample_data.append(np.stack(node_data))

        # Stack parameters for all samples
        return np.stack(sample_data, axis=0)

    def compute_parameters_parallel(
        self,
        data: np.ndarray,
        frequencies: np.ndarray,
        measurements_type: Literal["impedance_magnitude", "impedance_phase"],
        initial_guess: list[float] = [1500.0, 500.0, 0.2, 0.5],
        bounds: tuple[list[float], list[float]] = (
            [0.0, 0.0, 1.0e-9, 0.0],
            [np.inf, np.inf, np.inf, 1.0],
        ),
    ) -> np.ndarray:
        """Computes Cole-Cole parameters in parallel for all nodes and samples in the data.

        This function leverages parallel processing to speed up the fitting of Cole-Cole
        parameters for a 3D dataset of impedance measurements. Each node's parameters
        are computed independently, and the results are reshaped into the original
        `(samples, nodes, features)` format.

        Parameters
        ----------
        data : np.ndarray
            A 3D array of impedance values with shape `(samples, nodes, frequencies)`, where:
            - `samples`: Number of samples (e.g., patients or datasets).
            - `nodes`: Number of measurement nodes for each sample.
            - `frequencies`: Impedance values corresponding to the provided frequencies.
        frequencies : np.ndarray
            A 1D array of frequency values (in Hz) corresponding to the last axis of `data`.
        measurements_type : Literal["impedance_magnitude", "impedance_phase"]
            Specifies the type of measurement to fit:
            - "impedance_magnitude": Fit based on impedance magnitude.
            - "impedance_phase": Fit based on impedance phase (in degrees).
        initial_guess : List[float], optional
            Initial guesses for the Cole-Cole parameters `[R₀, R∞, τ, α]`. Defaults to
            `[1500.0, 500.0, 0.2, 0.5]`.
        bounds : Tuple[List[float], List[float]], optional
            Bounds for the Cole-Cole parameters, defined as a tuple of two lists:
            - Lower bounds `[R₀_min, R∞_min, τ_min, α_min]`.
            - Upper bounds `[R₀_max, R∞_max, τ_max, α_max]`.
            Defaults to `([0.0, 0.0, 1.0e-9, 0.0], [np.inf, np.inf, np.inf, 1.0])`.

        Returns
        -------
        np.ndarray
            A 3D array with shape `(samples, nodes, features)` where the last dimension contains:
            - `[R₀, R∞, τ, α, residual_sum, low2high_ratio, low2high_diff, tau_weighted]` for each
            sample and node.

        Raises
        ------
        ValueError
            If `data` is not a 3D array or if `frequencies` is not a 1D array.
        ValueError
            If the number of frequencies in `data` does not match the length of the `frequencies`
            array.

        Notes
        -----
        - This function uses `joblib.Parallel` to distribute computations across multiple CPU cores,
        significantly speeding up the parameter estimation process for large datasets.
        - Derived metrics such as the low-to-high resistance ratio (`R₀ / R∞`), resistance
        difference
        (`R₀ - R∞`), and weighted time constant (`τ * (R₀ - R∞)`) are included in the output.
        - Fitting failures for any node will result in NaN values for that node's parameters and
        metrics.
        """
        # Validate inputs
        if data.ndim != 3 or frequencies.ndim != 1:
            raise ValueError("`data` must be 3D, and `frequencies` must be 1D.")
        if data.shape[-1] != len(frequencies):
            raise ValueError("Mismatch between `data` frequencies and `frequencies` array.")

        results = Parallel(n_jobs=-1)(  # Use all available cores
            delayed(self.compute_node_parameters)(
                sample_idx,
                node_idx,
                data[sample_idx, node_idx, :],
                frequencies,
                measurements_type,
                initial_guess,
                bounds,
            )
            for sample_idx in range(data.shape[0])
            for node_idx in range(data.shape[1])
        )

        # Reshape the results back to the original `(samples, nodes, features)` shape
        return np.array(results).reshape(data.shape[0], data.shape[1], -1)

    def compute_features(
        self,
        data_structure: dict[
            str,
            dict[str, dict[str, np.ndarray]]
            | dict[str, np.ndarray]
            | list[int | str]
            | list[float | int],
        ],
        measurements_type: Literal["impedance_magnitude", "impedance_phase"],
        parallelize: bool = True,
    ) -> dict[
        str,
        dict[str, dict[str, np.ndarray]]
        | dict[str, np.ndarray]
        | list[int | str]
        | list[float | int],
    ]:
        """
        Computes features based on Cole-Cole parameters for the given impedance data structure.

        This function processes impedance measurements stored in a nested data structure, computes
        Cole-Cole parameters for each dataset (grouped by laterality and BI-RADS category), and
        organizes the computed features into a structured format.

        Parameters
        ----------
        data_structure : Dict
            A nested dictionary containing impedance data and metadata. Expected keys are:
            - "measurements": A dictionary where keys are laterality labels ("left", "right"),
            and values are dictionaries with BI-RADS categories as keys and 3D impedance
            data arrays (shape: `(samples, nodes, frequencies)`) as values.
            - "frequency_samples": A list or array of frequency values corresponding to the
            last dimension of the impedance data.
            - "patient_id": A dictionary mapping laterality labels to lists of patient IDs.
            - "nodes": A list of node labels for the impedance measurements.
        measurements_type : Literal["impedance_magnitude", "impedance_phase"]
            Specifies the type of measurement to fit:
            - "impedance_magnitude": Use the impedance magnitude for parameter computation.
            - "impedance_phase": Use the impedance phase for parameter computation.
        parallelize : bool, optional
            If True, computation is parallelized across nodes and samples for faster processing.
            Defaults to True.

        Returns
        -------
        Dict
            A dictionary containing the computed features and metadata. Keys include:
            - "features_values": Nested dictionary with the same structure as the input
            "measurements", but containing computed Cole-Cole parameters and derived metrics.
            - "patient_id": The original "patient_id" dictionary from the input data structure.
            - "nodes": The original list of node labels from the input data structure.
            - "features": A list of feature names corresponding to the computed parameters and
            metrics:
            - `R_0`: Resistance at zero frequency.
            - `R_inf`: Resistance at infinite frequency.
            - `tau`: Characteristic time constant.
            - `alpha`: Dispersion parameter.
            - `cost`: Residual sum of squares from the fit.
            - `low2high_ratio`: Ratio of low- to high-frequency resistances (`R_0 / R_inf`).
            - `low2high_diff`: Difference between low- and high-frequency resistances (`R_0 -
            R_inf`).
            - `tau_weighted`: Weighted time constant (`tau * (R_0 - R_inf)`).

        Raises
        ------
        ValueError
            If the input data structure is missing required keys.
        ValueError
            If the frequencies array is invalid or inconsistent with the impedance data.

        Notes
        -----
        - The function supports both sequential and parallel computation of Cole-Cole parameters.
        - Parallelization can significantly speed up processing for large datasets but may increase
        memory usage.
        """
        # Initialize features dictionary
        features = {
            laterality: {br: [] for br in item.keys()}
            for laterality, item in data_structure["measurements"].items()
        }

        # Scaled frequency vector for numerical stability
        frequencies = np.array(list(map(int, data_structure["frequency_samples"]))) / 1e3

        # Compute features for each laterality and BI-RADS category
        for laterality, data_dict in data_structure["measurements"].items():
            for bi_rad, data in data_dict.items():
                if parallelize:
                    features[laterality][bi_rad] = self.compute_parameters_parallel(
                        data,
                        frequencies=frequencies,
                        measurements_type=measurements_type,
                    )
                else:
                    features[laterality][bi_rad] = self.compute_parameters(
                        data,
                        frequencies=frequencies,
                        measurements_type=measurements_type,
                    )

        # Structure computed features into a final output dictionary
        features_structure = {
            "values": features,
            "patient_id": data_structure["patient_id"],
            "nodes": data_structure["nodes"],
            "features": [
                f"R_0_{measurements_type}",
                f"R_inf_{measurements_type}",
                f"tau_{measurements_type}",
                f"alpha_{measurements_type}",
                f"cost_{measurements_type}",
                f"low2high_ratio_{measurements_type}",
                f"low2high_diff_{measurements_type}",
                f"tau_weighted_{measurements_type}",
            ],
        }

        return features_structure


class ComputeAdvancedNyquistFeatures:
    """
    Advanced Nyquist/EIS feature extractor.

    Designed to run alongside ComputeNyquistPlotFeatures, returning a
    parallel feature array per (sample, node) with the same dict layout
    (values / patient_id / nodes / features). Does not modify or replace
    the original class.
    """

    #  Feature schema (ORDER MATTERS — defines output column order)
    FEATURE_NAMES: list[str] = [
        # Geometric in linear Nyquist (Re, -Im)
        "arc_length_lin",
        "chord_length_lin",
        "straightness_lin",
        "bulge_lin",
        "delta_re",
        "delta_im",
        "bbox_re_span",
        "bbox_im_span",
        "bbox_aspect",
        "centroid_x_raw",
        "centroid_y_raw",
        "centroid_x_bbox",
        "centroid_y_bbox",
        "low_slope_lin",
        "high_slope_lin",
        "slope_ratio_lin",
        "slope_diff_lin",
        "curvature_mean_lin",
        "curvature_max_lin",
        "curvature_std_lin",
        # Geometric in log-log Nyquist
        "arc_length_log",
        "chord_length_log",
        "straightness_log",
        "bulge_log",
        "low_slope_log",
        "high_slope_log",
        "slope_ratio_log",
        "slope_diff_log",
        "elbow_re_log",
        "elbow_im_log",
        "elbow_freq",
        "curvature_mean_log",
        "curvature_max_log",
        "curvature_std_log",
        # --- Bode-plane ---
        "Z_mag_at_5000",
        "Z_mag_at_10000",
        "Z_mag_at_50000",
        "Z_mag_at_100000",
        "phase_at_5000",
        "phase_at_10000",
        "phase_at_50000",
        "phase_at_100000",
        "phase_min",
        "phase_min_freq",
        "phase_mean",
        "logZ_logf_slope",
        # --- Model fits ---
        "rs_cpe_Rs",
        "rs_cpe_alpha",
        "rs_cpe_Y0",
        "rs_cpe_rmse",
        "rs_cpe_aic",
        "cole_R0",
        "cole_Rinf",
        "cole_tau",
        "cole_alpha",
        "cole_fc",
        "cole_rmse",
        "cole_aic",
        "aic_diff_cole_minus_rscpe",
    ]

    BODE_TARGET_FREQS: tuple[int, ...] = (5000, 10000, 50000, 100000)

    # Index window for low/high slope (kept compatible with the original
    # convention: indices 0..15 = 5..20 kHz, indices 35..95 = 40..100 kHz).
    LOW_SLOPE_RANGE: tuple[int, int] = (0, 15)
    HIGH_SLOPE_RANGE: tuple[int, int] = (35, -1)  # -1 => use the last index

    def __init__(self) -> None:
        self.n_features = len(self.FEATURE_NAMES)

    # Small helpers

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        if b is None or np.isnan(a) or np.isnan(b) or b == 0:
            return np.nan
        return a / b

    @staticmethod
    def _to_log(
        real: np.ndarray, imag_neg: np.ndarray, eps: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map (Re, -Im) to (log10 Re, log10(-Im)) with clipping."""
        return (
            np.log10(np.clip(real, eps, None)),
            np.log10(np.clip(imag_neg, eps, None)),
        )

    # Geometric features (work on any 2D parameterized curve)

    @staticmethod
    def arc_length(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(np.hypot(np.diff(x), np.diff(y))))

    @staticmethod
    def chord_length(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.hypot(x[-1] - x[0], y[-1] - y[0]))

    @classmethod
    def straightness(cls, x: np.ndarray, y: np.ndarray) -> float:
        L = cls.arc_length(x, y)
        if L == 0:
            return np.nan
        return cls.chord_length(x, y) / L

    @staticmethod
    def bulge(x: np.ndarray, y: np.ndarray) -> float:
        """
        Max perpendicular distance from curve points to the chord that joins
        its endpoints, normalised by the chord length. Zero for a straight
        line; grows with curvature/sag of the curve.
        """
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        chord = np.hypot(x1 - x0, y1 - y0)
        if chord == 0:
            return np.nan
        # Signed perpendicular distances times chord; take absolute / chord.
        d = np.abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) / chord
        return float(np.max(d) / chord)

    @staticmethod
    def discrete_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Discrete curvature at every interior point using the circumradius
        formula kappa = 4 * Area / (a * b * c) where a, b, c are the side
        lengths of the triangle formed by three consecutive points.
        Returns array of length len(x) - 2.
        """
        x0, x1, x2 = x[:-2], x[1:-1], x[2:]
        y0, y1, y2 = y[:-2], y[1:-1], y[2:]
        cross = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        a = np.hypot(x1 - x0, y1 - y0)
        b = np.hypot(x2 - x1, y2 - y1)
        c = np.hypot(x2 - x0, y2 - y0)
        denom = a * b * c
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.where(denom > 0, 2.0 * np.abs(cross) / denom, np.nan)
        return kappa

    @staticmethod
    def slope_polyfit(x: np.ndarray, y: np.ndarray, idx_lo: int, idx_hi: int) -> float:
        """First-order least-squares slope over x[idx_lo:idx_hi+1]."""
        if idx_hi <= idx_lo:
            return np.nan
        try:
            slope, _ = np.polyfit(x[idx_lo : idx_hi + 1], y[idx_lo : idx_hi + 1], 1)
            return float(slope)
        except Exception:
            return np.nan

    # Bode-plane features

    @classmethod
    def bode_block(
        cls, frequencies: np.ndarray, real: np.ndarray, imag_neg: np.ndarray
    ) -> dict[str, float]:
        """
        |Z| and phase features. Phase in degrees, with EIS convention
        phi = arctan(Im(Z)/Re(Z)) = -arctan(imag_neg/real) ∈ [-90, 0].
        """
        Z_mag = np.hypot(real, imag_neg)
        phase_deg = -np.degrees(np.arctan2(imag_neg, real))

        out: dict[str, float] = {}
        for f_target in cls.BODE_TARGET_FREQS:
            out[f"Z_mag_at_{f_target}"] = float(np.interp(f_target, frequencies, Z_mag))
            out[f"phase_at_{f_target}"] = float(np.interp(f_target, frequencies, phase_deg))

        # Phase extremes: "max capacitive" = most negative phase.
        idx_pm = int(np.argmin(phase_deg))
        out["phase_min"] = float(phase_deg[idx_pm])
        out["phase_min_freq"] = float(frequencies[idx_pm])
        out["phase_mean"] = float(np.mean(phase_deg))

        # log|Z| vs log f slope — for a pure CPE this is approximately -alpha.
        log_f = np.log10(frequencies)
        log_mag = np.log10(np.clip(Z_mag, 1e-12, None))
        try:
            slope, _ = np.polyfit(log_f, log_mag, 1)
            out["logZ_logf_slope"] = float(slope)
        except Exception:
            out["logZ_logf_slope"] = np.nan
        return out

    # Model fitting: R_s + CPE and Cole-Cole

    @staticmethod
    def _aic(ssr: float, n_obs: int, k: int) -> float:
        """AIC for least-squares with Gaussian residuals."""
        if ssr <= 0 or n_obs <= 0:
            return np.nan
        return float(n_obs * np.log(ssr / n_obs) + 2 * k)

    @staticmethod
    def fit_rs_cpe(
        frequencies: np.ndarray, real: np.ndarray, imag_neg: np.ndarray
    ) -> dict[str, float]:
        """
        Fit Z(w) = R_s + 1 / (Y0 * (jw)^alpha).

        Residual function uses both Re and Im components (concatenated),
        which fits the complex impedance directly rather than only one axis.
        """
        omega = 2.0 * np.pi * frequencies
        Z_data = real + 1j * (-imag_neg)  # back to native EIS convention

        def residuals(params):
            R_s, alpha, log_Y0 = params
            Y0 = np.exp(log_Y0)
            Z_model = R_s + 1.0 / (Y0 * (1j * omega) ** alpha)
            r = Z_model - Z_data
            return np.concatenate([r.real, r.imag])

        R_s_init = float(np.min(real))
        alpha_init = 0.7
        i_mid = len(omega) // 2
        Z_dev = Z_data[i_mid] - R_s_init
        denom = max(np.abs(Z_dev) * omega[i_mid] ** alpha_init, 1e-20)
        log_Y0_init = float(np.log(max(1.0 / denom, 1e-20)))

        try:
            res = least_squares(
                residuals,
                x0=[R_s_init, alpha_init, log_Y0_init],
                bounds=([0.0, 0.01, -60.0], [np.inf, 1.0, 60.0]),
                max_nfev=2000,
            )
            R_s, alpha, log_Y0 = res.x
            Y0 = float(np.exp(log_Y0))
            Z_model = R_s + 1.0 / (Y0 * (1j * omega) ** alpha)
            r = Z_model - Z_data
            ssr = float(np.sum(np.abs(r) ** 2))
            rmse = float(np.sqrt(ssr / len(omega)))
            aic = ComputeAdvancedNyquistFeatures._aic(ssr, 2 * len(omega), 3)
            return {
                "rs_cpe_Rs": float(R_s),
                "rs_cpe_alpha": float(alpha),
                "rs_cpe_Y0": Y0,
                "rs_cpe_rmse": rmse,
                "rs_cpe_aic": aic,
            }
        except Exception:
            return {
                k: np.nan
                for k in ("rs_cpe_Rs", "rs_cpe_alpha", "rs_cpe_Y0", "rs_cpe_rmse", "rs_cpe_aic")
            }

    @staticmethod
    def fit_cole_cole(
        frequencies: np.ndarray, real: np.ndarray, imag_neg: np.ndarray
    ) -> dict[str, float]:
        """
        Fit depressed Cole-Cole: Z(w) = R_inf + (R_0 - R_inf) / (1 + (j w tau)^alpha).

        Note: with frequency range 5 - 100 kHz and curves that do not show
        the -Im(Z) peak, R_0 and tau may be only weakly identifiable. The
        RMSE and AIC fields tell you whether the fit is trustworthy; use
        the ΔAIC against R_s+CPE for relative model evidence.
        """
        omega = 2.0 * np.pi * frequencies
        Z_data = real + 1j * (-imag_neg)

        def residuals(params):
            R_inf, dR, log_tau, alpha = params
            tau = np.exp(log_tau)
            Z_model = R_inf + dR / (1.0 + (1j * omega * tau) ** alpha)
            r = Z_model - Z_data
            return np.concatenate([r.real, r.imag])

        R_inf_init = float(np.min(real))
        dR_init = float(np.max(real) - np.min(real)) * 3.0  # arc not closed
        # Characteristic freq likely below window: tau slightly above 1/(2π f_min).
        tau_init = 1.0 / (2.0 * np.pi * float(frequencies[0]))
        log_tau_init = float(np.log(tau_init))
        alpha_init = 0.7

        try:
            res = least_squares(
                residuals,
                x0=[R_inf_init, dR_init, log_tau_init, alpha_init],
                bounds=([0.0, 0.0, -30.0, 0.01], [np.inf, np.inf, 30.0, 1.0]),
                max_nfev=4000,
            )
            R_inf, dR, log_tau, alpha = res.x
            tau = float(np.exp(log_tau))
            R_0 = float(R_inf + dR)
            f_c = float(1.0 / (2.0 * np.pi * tau)) if tau > 0 else np.nan
            Z_model = R_inf + dR / (1.0 + (1j * omega * tau) ** alpha)
            r = Z_model - Z_data
            ssr = float(np.sum(np.abs(r) ** 2))
            rmse = float(np.sqrt(ssr / len(omega)))
            aic = ComputeAdvancedNyquistFeatures._aic(ssr, 2 * len(omega), 4)
            return {
                "cole_R0": R_0,
                "cole_Rinf": float(R_inf),
                "cole_tau": tau,
                "cole_alpha": float(alpha),
                "cole_fc": f_c,
                "cole_rmse": rmse,
                "cole_aic": aic,
            }
        except Exception:
            return {
                k: np.nan
                for k in (
                    "cole_R0",
                    "cole_Rinf",
                    "cole_tau",
                    "cole_alpha",
                    "cole_fc",
                    "cole_rmse",
                    "cole_aic",
                )
            }

    # Node-level driver

    def _empty_feature_dict(self) -> dict[str, float]:
        return {name: np.nan for name in self.FEATURE_NAMES}

    def compute_node_advanced(
        self,
        sample_idx: int,
        node_idx: int,
        real: np.ndarray,
        imag_neg: np.ndarray,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        """
        Compute every advanced feature for a single (sample, node).

        Parameters
        ----------
        real      : Re(Z), shape (F,)
        imag_neg  : -Im(Z), shape (F,), positive Nyquist convention
        frequencies : Hz, shape (F,), monotonic increasing

        Returns
        -------
        ndarray of shape (n_features,) in FEATURE_NAMES order.
        """
        feats = self._empty_feature_dict()

        # Linear-scale geometric
        try:
            feats["arc_length_lin"] = self.arc_length(real, imag_neg)
            feats["chord_length_lin"] = self.chord_length(real, imag_neg)
            feats["straightness_lin"] = self.straightness(real, imag_neg)
            feats["bulge_lin"] = self.bulge(real, imag_neg)

            feats["delta_re"] = float(real[0] - real[-1])
            feats["delta_im"] = float(imag_neg[0] - imag_neg[-1])

            re_min, re_max = float(np.min(real)), float(np.max(real))
            im_min, im_max = float(np.min(imag_neg)), float(np.max(imag_neg))
            re_range = re_max - re_min
            im_range = im_max - im_min
            feats["bbox_re_span"] = re_range
            feats["bbox_im_span"] = im_range
            feats["bbox_aspect"] = self._safe_div(im_range, re_range)

            cx = float(np.mean(real))
            cy = float(np.mean(imag_neg))
            feats["centroid_x_raw"] = cx
            feats["centroid_y_raw"] = cy
            feats["centroid_x_bbox"] = self._safe_div(cx - re_min, re_range)
            feats["centroid_y_bbox"] = self._safe_div(cy - im_min, im_range)

            idx_hi = self.HIGH_SLOPE_RANGE[1]
            if idx_hi == -1:
                idx_hi = len(real) - 1
            low_s = self.slope_polyfit(real, imag_neg, *self.LOW_SLOPE_RANGE)
            high_s = self.slope_polyfit(real, imag_neg, self.HIGH_SLOPE_RANGE[0], idx_hi)
            feats["low_slope_lin"] = low_s
            feats["high_slope_lin"] = high_s
            feats["slope_ratio_lin"] = self._safe_div(high_s, low_s)
            feats["slope_diff_lin"] = (
                low_s - high_s if not (np.isnan(low_s) or np.isnan(high_s)) else np.nan
            )

            kappa = self.discrete_curvature(real, imag_neg)
            if np.any(~np.isnan(kappa)):
                feats["curvature_mean_lin"] = float(np.nanmean(kappa))
                feats["curvature_max_lin"] = float(np.nanmax(kappa))
                feats["curvature_std_lin"] = float(np.nanstd(kappa))
        except Exception as e:
            print(f"[lin-geom] sample={sample_idx} node={node_idx}: {e}")

        # --- Log-log geometric ---
        try:
            r_log, i_log = self._to_log(real, imag_neg)
            feats["arc_length_log"] = self.arc_length(r_log, i_log)
            feats["chord_length_log"] = self.chord_length(r_log, i_log)
            feats["straightness_log"] = self.straightness(r_log, i_log)
            feats["bulge_log"] = self.bulge(r_log, i_log)

            idx_hi = self.HIGH_SLOPE_RANGE[1]
            if idx_hi == -1:
                idx_hi = len(r_log) - 1
            low_s_log = self.slope_polyfit(r_log, i_log, *self.LOW_SLOPE_RANGE)
            high_s_log = self.slope_polyfit(r_log, i_log, self.HIGH_SLOPE_RANGE[0], idx_hi)
            feats["low_slope_log"] = low_s_log
            feats["high_slope_log"] = high_s_log
            feats["slope_ratio_log"] = self._safe_div(high_s_log, low_s_log)
            feats["slope_diff_log"] = (
                low_s_log - high_s_log
                if not (np.isnan(low_s_log) or np.isnan(high_s_log))
                else np.nan
            )

            kappa_log = self.discrete_curvature(r_log, i_log)
            if np.any(~np.isnan(kappa_log)):
                feats["curvature_mean_log"] = float(np.nanmean(kappa_log))
                feats["curvature_max_log"] = float(np.nanmax(kappa_log))
                feats["curvature_std_log"] = float(np.nanstd(kappa_log))
                # Elbow = point of max curvature in log-log
                idx_elbow = int(np.nanargmax(kappa_log)) + 1  # interior offset
                feats["elbow_re_log"] = float(r_log[idx_elbow])
                feats["elbow_im_log"] = float(i_log[idx_elbow])
                feats["elbow_freq"] = float(frequencies[idx_elbow])
        except Exception as e:
            print(f"[log-geom] sample={sample_idx} node={node_idx}: {e}")

        #  Bode
        try:
            feats.update(self.bode_block(frequencies, real, imag_neg))
        except Exception as e:
            print(f"[bode] sample={sample_idx} node={node_idx}: {e}")

        # R_s + CPE
        try:
            feats.update(self.fit_rs_cpe(frequencies, real, imag_neg))
        except Exception as e:
            print(f"[rs+cpe] sample={sample_idx} node={node_idx}: {e}")

        # Cole-Cole
        try:
            feats.update(self.fit_cole_cole(frequencies, real, imag_neg))
        except Exception as e:
            print(f"[cole-cole] sample={sample_idx} node={node_idx}: {e}")

        #  Model comparison
        cole_aic = feats.get("cole_aic", np.nan)
        rs_aic = feats.get("rs_cpe_aic", np.nan)
        if not (np.isnan(cole_aic) or np.isnan(rs_aic)):
            feats["aic_diff_cole_minus_rscpe"] = cole_aic - rs_aic

        return np.array([feats[name] for name in self.FEATURE_NAMES], dtype=float)

    # =====================================================================
    # Methods mirroring the original API
    # =====================================================================
    def compute_metrics(self, data: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute advanced features for every (sample, node) in a 3D complex
        array.

        Parameters
        ----------
        data : (S, N, F) complex
        frequencies : (F,) monotonic increasing in Hz

        Returns
        -------
        (S, N, n_features) float array
        """
        if data.ndim != 3 or frequencies.ndim != 1:
            raise ValueError("`data` must be 3D and `frequencies` 1D.")
        if data.shape[-1] != len(frequencies):
            raise ValueError("Mismatch between `data` frequency axis and `frequencies`.")

        S, N, _ = data.shape
        out = np.full((S, N, self.n_features), np.nan, dtype=float)

        for s in range(S):
            for n in range(N):
                real = data[s, n, :].real
                imag = data[s, n, :].imag
                imag_neg = -imag  # Nyquist y-axis convention
                out[s, n, :] = self.compute_node_advanced(s, n, real, imag_neg, frequencies)
        return out

    def compute_features(
        self,
        data_structure: dict[str, dict | list],
    ) -> dict[str, dict | list]:
        """
        Compute the advanced feature set across every laterality and BI-RADS
        category. Mirrors ComputeNyquistPlotFeatures.compute_features so the
        output plugs into the same downstream pipeline.

        NOTE on BI-RADS: BI-RADS labels are used here only as a grouping key
        from the input data_structure. No feature is computed using the
        label itself, so there is no risk of label leakage from feature
        extraction.
        """
        features = {
            laterality: {br: [] for br in item.keys()}
            for laterality, item in data_structure["measurements"].items()
        }

        frequencies = np.array(list(map(int, data_structure["frequency_samples"])))

        for laterality, data_dict in data_structure["measurements"].items():
            for bi_rad, data in data_dict.items():
                features[laterality][bi_rad] = self.compute_metrics(data, frequencies=frequencies)

        return {
            "values": features,
            "patient_id": data_structure["patient_id"],
            "nodes": data_structure["nodes"],
            "features": list(self.FEATURE_NAMES),
        }


class ComputeBreastLevelStatistics:
    """
    Aggregate node-level features (output of ComputeAdvancedNyquistFeatures
    or ComputeNyquistPlotFeatures) into breast-level summary statistics and
    patient-level left/right asymmetry.

    Operates purely on the feature dict; never reads BI-RADS labels for any
    computation — they are only used as grouping keys.
    """

    NODE_STATS: tuple[str, ...] = ("mean", "std", "cv", "median", "iqr", "min", "max")

    @staticmethod
    def _stats_1d(arr_1d: np.ndarray) -> dict[str, float]:
        a = arr_1d[~np.isnan(arr_1d)]
        if len(a) == 0:
            return {k: np.nan for k in ComputeBreastLevelStatistics.NODE_STATS}
        mean = float(np.mean(a))
        std = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        cv = std / abs(mean) if abs(mean) > 1e-12 else np.nan
        q25, q75 = np.percentile(a, [25, 75])
        return {
            "mean": mean,
            "std": std,
            "cv": cv,
            "median": float(np.median(a)),
            "iqr": float(q75 - q25),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    @classmethod
    def aggregate(
        cls,
        node_features: dict[str, dict | list],
    ) -> dict[str, dict | list]:
        """
        For each (laterality, bi_rad) collapse the node axis with the
        statistics in NODE_STATS.

        Input:  values[lat][br] of shape (S, N, F)
        Output: values[lat][br] of shape (S, F * len(NODE_STATS))
                features = [f"{feature}__{stat}" for feature in features
                                                  for stat   in NODE_STATS]
        """
        feature_names = list(node_features["features"])
        stat_names = cls.NODE_STATS
        flat_names = [f"{f}__{s}" for f in feature_names for s in stat_names]

        out_values: dict[str, dict[str, np.ndarray]] = {}
        for laterality, by_birad in node_features["values"].items():
            out_values[laterality] = {}
            for birad, arr in by_birad.items():
                S, N, F = arr.shape
                out = np.full((S, F * len(stat_names)), np.nan, dtype=float)
                for s in range(S):
                    for f in range(F):
                        sd = cls._stats_1d(arr[s, :, f])
                        for k, sname in enumerate(stat_names):
                            out[s, f * len(stat_names) + k] = sd[sname]
                out_values[laterality][birad] = out

        return {
            "values": out_values,
            "patient_id": node_features["patient_id"],
            "features": flat_names,
        }

    @classmethod
    def asymmetry(
        cls,
        breast_features: dict[str, dict | list],
        eps: float = 1e-12,
    ) -> dict[str, dict | list]:
        """
        Per-patient L vs R asymmetry on the aggregate breast-level features.

        Asymmetry index:
            A = (L - R) / (|L| + |R| + eps)   in [-1, 1]

        Requires that 'left' and 'right' share the same patient ordering and
        BI-RADS groups.

        Returns:
            { "values":     {bi_rad: ndarray (S, F)},
              "patient_id": ...,
              "features":   [f"{name}_asym" for name in input features] }
        """
        if (
            "left_breast" not in breast_features["values"]
            or "right_breast" not in breast_features["values"]
        ):
            raise ValueError(
                "Both 'left_breast' and 'right_breast' laterality required for asymmetry."
            )

        feat_names = breast_features["features"]
        asym_names = [f"{f}_asym" for f in feat_names]

        L_by_br = breast_features["values"]["left_breast"]
        R_by_br = breast_features["values"]["right_breast"]

        out_values: dict[str, np.ndarray] = {}
        for birad in L_by_br.keys():
            if birad not in R_by_br:
                continue
            L = L_by_br[birad]
            R = R_by_br[birad]
            if L.shape != R.shape:
                raise ValueError(f"Shape mismatch for {birad}: L={L.shape} R={R.shape}")
            asym = (L - R) / (np.abs(L) + np.abs(R) + eps)
            out_values[birad] = asym

        return {
            "values": out_values,
            "patient_id": breast_features["patient_id"],
            "features": asym_names,
        }


class ComputeAdvancedMagPhaseFeatures:
    """
    Variante experimental de ComputeAdvancedNyquistFeatures: analiza la curva
    (magnitud, fase) en vez de (resistencia, reactancia).

    Reutiliza EXACTAMENTE las mismas fórmulas geométricas (arco, cuerda,
    rectitud, abultamiento, pendientes, curvatura) de
    ComputeAdvancedNyquistFeatures -- son funciones genéricas sobre (x, y),
    así que no hace falta reescribirlas, solo alimentarlas con |Z| y fase en
    vez de con R y -X.

    Diferencias deliberadas frente al original (documentadas para que quede
    claro qué se cambió y por qué):
    - Grupo "log": solo se transforma la MAGNITUD a log10, la fase se deja
      lineal. La fase no es una cantidad multiplicativa que abarque décadas
      como sí lo hace la impedancia, y cruza por valores cercanos a cero en
      los extremos del arco -- tomarle log dispararía a -infinito.
    - Bloque Bode: queda IDÉNTICO al original (mismos 12 valores). Ya estaba
      basado en magnitud/fase desde el principio, así que no hay nada que
      cambiar de base acá -- el cambio real está en los grupos geométrico y
      paramétrico.
    - Ajustes RS-CPE y Cole-Cole: mismo modelo eléctrico que el original,
      pero el error que se minimiza es sobre log|Z| y fase (en vez de sobre
      parte real e imaginaria) -- una ponderación distinta y real en
      espectroscopía de impedancia ("modulus weighting"), no una
      aproximación inventada. La fase se normaliza dividiendo entre 90 para
      que pese de forma comparable a log|Z| en la suma de errores.
    """

    FEATURE_NAMES: list[str] = [
        # --- Geometric in linear mag-phase ---
        "arc_length_lin",
        "chord_length_lin",
        "straightness_lin",
        "bulge_lin",
        "delta_mag",
        "delta_phase",
        "bbox_mag_span",
        "bbox_phase_span",
        "bbox_aspect",
        "centroid_x_raw",
        "centroid_y_raw",
        "centroid_x_bbox",
        "centroid_y_bbox",
        "low_slope_lin",
        "high_slope_lin",
        "slope_ratio_lin",
        "slope_diff_lin",
        "curvature_mean_lin",
        "curvature_max_lin",
        "curvature_std_lin",
        # --- Geometric in log-magnitude / linear-phase ---
        "arc_length_log",
        "chord_length_log",
        "straightness_log",
        "bulge_log",
        "low_slope_log",
        "high_slope_log",
        "slope_ratio_log",
        "slope_diff_log",
        "elbow_mag_log",
        "elbow_phase_log",
        "elbow_freq",
        "curvature_mean_log",
        "curvature_max_log",
        "curvature_std_log",
        # --- Bode-plane (idéntico a ComputeAdvancedNyquistFeatures) ---
        "Z_mag_at_5000",
        "Z_mag_at_10000",
        "Z_mag_at_50000",
        "Z_mag_at_100000",
        "phase_at_5000",
        "phase_at_10000",
        "phase_at_50000",
        "phase_at_100000",
        "phase_min",
        "phase_min_freq",
        "phase_mean",
        "logZ_logf_slope",
        # --- Model fits (residuo en log-magnitud + fase, no en Re/Im) ---
        "rs_cpe_Rs",
        "rs_cpe_alpha",
        "rs_cpe_Y0",
        "rs_cpe_rmse",
        "rs_cpe_aic",
        "cole_R0",
        "cole_Rinf",
        "cole_tau",
        "cole_alpha",
        "cole_fc",
        "cole_rmse",
        "cole_aic",
        "aic_diff_cole_minus_rscpe",
    ]

    BODE_TARGET_FREQS = ComputeAdvancedNyquistFeatures.BODE_TARGET_FREQS
    LOW_SLOPE_RANGE = ComputeAdvancedNyquistFeatures.LOW_SLOPE_RANGE
    HIGH_SLOPE_RANGE = ComputeAdvancedNyquistFeatures.HIGH_SLOPE_RANGE

    def __init__(self) -> None:
        self.n_features = len(self.FEATURE_NAMES)

    def _empty_feature_dict(self) -> dict[str, float]:
        return {name: np.nan for name in self.FEATURE_NAMES}

    @staticmethod
    def fit_rs_cpe_magphase(
        frequencies: np.ndarray, mag: np.ndarray, phase_deg: np.ndarray, eps: float = 1e-12
    ) -> dict[str, float]:
        """Igual modelo que fit_rs_cpe, pero el residuo se mide en log|Z| + fase."""
        omega = 2.0 * np.pi * frequencies
        log_mag_data = np.log10(np.clip(mag, eps, None))

        def residuals(params):
            R_s, alpha, log_Y0 = params
            Y0 = np.exp(log_Y0)
            Z_model = R_s + 1.0 / (Y0 * (1j * omega) ** alpha)
            mag_model = np.abs(Z_model)
            phase_model = np.degrees(np.angle(Z_model))
            r_mag = np.log10(np.clip(mag_model, eps, None)) - log_mag_data
            r_phase = (phase_model - phase_deg) / 90.0
            return np.concatenate([r_mag, r_phase])

        R_s_init = float(np.min(mag))
        alpha_init = 0.7
        i_mid = len(omega) // 2
        denom = max(mag[i_mid] * omega[i_mid] ** alpha_init, 1e-20)
        log_Y0_init = float(np.log(max(1.0 / denom, 1e-20)))

        try:
            res = least_squares(
                residuals,
                x0=[R_s_init, alpha_init, log_Y0_init],
                bounds=([0.0, 0.01, -60.0], [np.inf, 1.0, 60.0]),
                max_nfev=2000,
            )
            R_s, alpha, log_Y0 = res.x
            Y0 = float(np.exp(log_Y0))
            r = residuals(res.x)
            ssr = float(np.sum(r**2))
            rmse = float(np.sqrt(ssr / len(omega)))
            aic = ComputeAdvancedNyquistFeatures._aic(ssr, 2 * len(omega), 3)
            return {
                "rs_cpe_Rs": float(R_s),
                "rs_cpe_alpha": float(alpha),
                "rs_cpe_Y0": Y0,
                "rs_cpe_rmse": rmse,
                "rs_cpe_aic": aic,
            }
        except Exception:
            return {
                k: np.nan
                for k in ("rs_cpe_Rs", "rs_cpe_alpha", "rs_cpe_Y0", "rs_cpe_rmse", "rs_cpe_aic")
            }

    @staticmethod
    def fit_cole_cole_magphase(
        frequencies: np.ndarray, mag: np.ndarray, phase_deg: np.ndarray, eps: float = 1e-12
    ) -> dict[str, float]:
        """Igual modelo que fit_cole_cole, pero el residuo se mide en log|Z| + fase."""
        omega = 2.0 * np.pi * frequencies
        log_mag_data = np.log10(np.clip(mag, eps, None))

        def residuals(params):
            R_inf, dR, log_tau, alpha = params
            tau = np.exp(log_tau)
            Z_model = R_inf + dR / (1.0 + (1j * omega * tau) ** alpha)
            mag_model = np.abs(Z_model)
            phase_model = np.degrees(np.angle(Z_model))
            r_mag = np.log10(np.clip(mag_model, eps, None)) - log_mag_data
            r_phase = (phase_model - phase_deg) / 90.0
            return np.concatenate([r_mag, r_phase])

        R_inf_init = float(np.min(mag))
        dR_init = float(np.max(mag) - np.min(mag)) * 3.0
        tau_init = 1.0 / (2.0 * np.pi * float(frequencies[0]))
        log_tau_init = float(np.log(tau_init))
        alpha_init = 0.7

        try:
            res = least_squares(
                residuals,
                x0=[R_inf_init, dR_init, log_tau_init, alpha_init],
                bounds=([0.0, 0.0, -30.0, 0.01], [np.inf, np.inf, 30.0, 1.0]),
                max_nfev=4000,
            )
            R_inf, dR, log_tau, alpha = res.x
            tau = float(np.exp(log_tau))
            R_0 = float(R_inf + dR)
            f_c = float(1.0 / (2.0 * np.pi * tau)) if tau > 0 else np.nan
            r = residuals(res.x)
            ssr = float(np.sum(r**2))
            rmse = float(np.sqrt(ssr / len(omega)))
            aic = ComputeAdvancedNyquistFeatures._aic(ssr, 2 * len(omega), 4)
            return {
                "cole_R0": R_0,
                "cole_Rinf": float(R_inf),
                "cole_tau": tau,
                "cole_alpha": float(alpha),
                "cole_fc": f_c,
                "cole_rmse": rmse,
                "cole_aic": aic,
            }
        except Exception:
            return {
                k: np.nan
                for k in (
                    "cole_R0",
                    "cole_Rinf",
                    "cole_tau",
                    "cole_alpha",
                    "cole_fc",
                    "cole_rmse",
                    "cole_aic",
                )
            }

    def compute_node_advanced(
        self,
        sample_idx: int,
        node_idx: int,
        real: np.ndarray,
        imag_neg: np.ndarray,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        feats = self._empty_feature_dict()
        F = ComputeAdvancedNyquistFeatures  # reutiliza la geometría genérica

        mag = np.hypot(real, imag_neg)
        phase_deg = -np.degrees(np.arctan2(imag_neg, real))

        # Geometría lineal en (magnitud, fase)
        try:
            feats["arc_length_lin"] = F.arc_length(mag, phase_deg)
            feats["chord_length_lin"] = F.chord_length(mag, phase_deg)
            feats["straightness_lin"] = F.straightness(mag, phase_deg)
            feats["bulge_lin"] = F.bulge(mag, phase_deg)

            feats["delta_mag"] = float(mag[0] - mag[-1])
            feats["delta_phase"] = float(phase_deg[0] - phase_deg[-1])

            mag_min, mag_max = float(np.min(mag)), float(np.max(mag))
            ph_min, ph_max = float(np.min(phase_deg)), float(np.max(phase_deg))
            mag_range = mag_max - mag_min
            ph_range = ph_max - ph_min
            feats["bbox_mag_span"] = mag_range
            feats["bbox_phase_span"] = ph_range
            feats["bbox_aspect"] = F._safe_div(ph_range, mag_range)

            cx = float(np.mean(mag))
            cy = float(np.mean(phase_deg))
            feats["centroid_x_raw"] = cx
            feats["centroid_y_raw"] = cy
            feats["centroid_x_bbox"] = F._safe_div(cx - mag_min, mag_range)
            feats["centroid_y_bbox"] = F._safe_div(cy - ph_min, ph_range)

            idx_hi = self.HIGH_SLOPE_RANGE[1]
            if idx_hi == -1:
                idx_hi = len(mag) - 1
            low_s = F.slope_polyfit(mag, phase_deg, *self.LOW_SLOPE_RANGE)
            high_s = F.slope_polyfit(mag, phase_deg, self.HIGH_SLOPE_RANGE[0], idx_hi)
            feats["low_slope_lin"] = low_s
            feats["high_slope_lin"] = high_s
            feats["slope_ratio_lin"] = F._safe_div(high_s, low_s)
            feats["slope_diff_lin"] = (
                low_s - high_s if not (np.isnan(low_s) or np.isnan(high_s)) else np.nan
            )

            kappa = F.discrete_curvature(mag, phase_deg)
            if np.any(~np.isnan(kappa)):
                feats["curvature_mean_lin"] = float(np.nanmean(kappa))
                feats["curvature_max_lin"] = float(np.nanmax(kappa))
                feats["curvature_std_lin"] = float(np.nanstd(kappa))
        except Exception as e:
            print(f"[magphase-lin-geom] sample={sample_idx} node={node_idx}: {e}")

        # --- Geometría en (log10 magnitud, fase lineal) ---
        try:
            mag_log = np.log10(np.clip(mag, 1e-12, None))
            feats["arc_length_log"] = F.arc_length(mag_log, phase_deg)
            feats["chord_length_log"] = F.chord_length(mag_log, phase_deg)
            feats["straightness_log"] = F.straightness(mag_log, phase_deg)
            feats["bulge_log"] = F.bulge(mag_log, phase_deg)

            idx_hi = self.HIGH_SLOPE_RANGE[1]
            if idx_hi == -1:
                idx_hi = len(mag_log) - 1
            low_s_log = F.slope_polyfit(mag_log, phase_deg, *self.LOW_SLOPE_RANGE)
            high_s_log = F.slope_polyfit(mag_log, phase_deg, self.HIGH_SLOPE_RANGE[0], idx_hi)
            feats["low_slope_log"] = low_s_log
            feats["high_slope_log"] = high_s_log
            feats["slope_ratio_log"] = F._safe_div(high_s_log, low_s_log)
            feats["slope_diff_log"] = (
                low_s_log - high_s_log
                if not (np.isnan(low_s_log) or np.isnan(high_s_log))
                else np.nan
            )

            kappa_log = F.discrete_curvature(mag_log, phase_deg)
            if np.any(~np.isnan(kappa_log)):
                feats["curvature_mean_log"] = float(np.nanmean(kappa_log))
                feats["curvature_max_log"] = float(np.nanmax(kappa_log))
                feats["curvature_std_log"] = float(np.nanstd(kappa_log))
                idx_elbow = int(np.nanargmax(kappa_log)) + 1
                feats["elbow_mag_log"] = float(mag_log[idx_elbow])
                feats["elbow_phase_log"] = float(phase_deg[idx_elbow])
                feats["elbow_freq"] = float(frequencies[idx_elbow])
        except Exception as e:
            print(f"[magphase-log-geom] sample={sample_idx} node={node_idx}: {e}")

        # --- Bode (idéntico al original: ya está en magnitud/fase) ---
        try:
            feats.update(F.bode_block(frequencies, real, imag_neg))
        except Exception as e:
            print(f"[magphase-bode] sample={sample_idx} node={node_idx}: {e}")

        # --- R_s + CPE (residuo en log|Z| + fase) ---
        try:
            feats.update(self.fit_rs_cpe_magphase(frequencies, mag, phase_deg))
        except Exception as e:
            print(f"[magphase-rs+cpe] sample={sample_idx} node={node_idx}: {e}")

        # --- Cole-Cole (residuo en log|Z| + fase) ---
        try:
            feats.update(self.fit_cole_cole_magphase(frequencies, mag, phase_deg))
        except Exception as e:
            print(f"[magphase-cole-cole] sample={sample_idx} node={node_idx}: {e}")

        cole_aic = feats.get("cole_aic", np.nan)
        rs_aic = feats.get("rs_cpe_aic", np.nan)
        if not (np.isnan(cole_aic) or np.isnan(rs_aic)):
            feats["aic_diff_cole_minus_rscpe"] = cole_aic - rs_aic

        return np.array([feats[name] for name in self.FEATURE_NAMES], dtype=float)

    def compute_metrics(self, data: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        if data.ndim != 3 or frequencies.ndim != 1:
            raise ValueError("`data` must be 3D and `frequencies` 1D.")
        if data.shape[-1] != len(frequencies):
            raise ValueError("Mismatch between `data` frequency axis and `frequencies`.")

        S, N, _ = data.shape
        out = np.full((S, N, self.n_features), np.nan, dtype=float)

        for s in range(S):
            for n in range(N):
                real = data[s, n, :].real
                imag = data[s, n, :].imag
                imag_neg = -imag
                out[s, n, :] = self.compute_node_advanced(s, n, real, imag_neg, frequencies)
        return out

    def compute_features(
        self,
        data_structure: dict[str, dict | list],
    ) -> dict[str, dict | list]:
        features = {
            laterality: {br: [] for br in item.keys()}
            for laterality, item in data_structure["measurements"].items()
        }

        frequencies = np.array(list(map(int, data_structure["frequency_samples"])))

        for laterality, data_dict in data_structure["measurements"].items():
            for bi_rad, data in data_dict.items():
                features[laterality][bi_rad] = self.compute_metrics(data, frequencies=frequencies)

        return {
            "values": features,
            "patient_id": data_structure["patient_id"],
            "nodes": data_structure["nodes"],
            "features": list(self.FEATURE_NAMES),
        }
