import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, iqr


def estimate_bins(data: np.ndarray, method: str = "fd") -> int:
    """
    Estimates the optimal number of bins for histogram computation.

    Parameters
    ----------
    data : np.ndarray
        The input signal, can be 1D or multi-dimensional.
    method : str, optional
        Method to estimate the number of bins:
        - "fd" (Freedman-Diaconis, default)
        - "sturges"
        - "sqrt"

    Returns
    -------
    int
        Estimated number of bins.
    """
    if method == "fd":
        signal = data.flatten()
        N = len(signal)
        bin_width = 2 * iqr(signal) / (N ** (1 / 3))
        bins = max(10, int(np.ceil((np.max(signal) - np.min(signal)) / bin_width)))  # Min 10 bins
    elif method == "sturges":
        N = data.shape[-1]
        bins = int(np.ceil(np.log2(N) + 1))
    elif method == "sqrt":
        N = data.shape[-1]
        bins = int(np.ceil(np.sqrt(N)))
    else:
        raise ValueError("Invalid method. Choose 'fd', 'sturges', or 'sqrt'.")

    return min(bins, 50)  # Limit max bins to avoid over-segmentation


class SignalComplexity:
    @staticmethod
    def compute_log_energy_entropy(
        data: np.ndarray,
        axis: int = -1,
    ) -> float | np.ndarray:
        """
        Computes the Log Energy Entropy (LEE) of a signal or a collection of signals.

        Log Energy Entropy measures the energy distribution of a signal using a
        logarithmic transformation. It is commonly used in signal processing and
        feature extraction for time-series analysis.

        Parameters
        ----------
        data : np.ndarray
            The input signal, can be 1D or multi-dimensional.
        axis : int, optional
            The axis along which the energy is computed. Default is -1.

        Returns
        -------
        Union[float, np.ndarray]
            The Log Energy Entropy value of the signal. If `data` is multi-dimensional, an array
            of entropy values is returned along the specified axis.

        Notes
        -----
        - A higher value indicates a more complex signal with higher energy concentration.
        - A lower value suggests a more uniform or less energetic signal.
        - A small constant (1e-10) is added to prevent `log(0)` errors.
        """
        energy = np.square(data)  # Compute signal energy
        log_energy_entropy = np.log(np.sum(energy, axis=axis) + 1e-10)  # Compute log entropy

        return log_energy_entropy

    @staticmethod
    def compute_shannon_entropy(
        data: np.ndarray,
        axis: int = -1,
    ) -> float | np.ndarray:
        """
        Computes the Shannon entropy of a signal or a collection of signals using
        histogram-based probability estimation.

        Parameters
        ----------
        data : np.ndarray
            Input signal, can be 1D or multi-dimensional.
        axis : int, optional
            Axis along which the entropy is computed (default is 0).

        Returns
        -------
        Union[float, np.ndarray]
            The computed Shannon entropy. If `data` is multi-dimensional, returns an array
            of entropy values computed along the specified axis.
        """
        # Ensure signal is a NumPy array
        data = np.asarray(data)

        if data.ndim == 1:
            # Estimate optimal number of bins
            num_bins = estimate_bins(data, method="fd")
            hist, _ = np.histogram(data, bins=num_bins, density=True)
            prob_distribution = hist / np.sum(hist)  # Normalize to get probabilities
        else:
            # Compute histogram bins based on the entire dataset for consistency
            num_bins = estimate_bins(data, method="sqrt")

            # Compute histograms along the specified axis with fixed bins
            def hist_func(x):
                hist, _ = np.histogram(x, bins=num_bins, density=True)
                return hist / np.sum(hist)  # Normalize probabilities

            prob_distribution = np.apply_along_axis(hist_func, axis=axis, arr=data)

        # Remove zero probabilities to avoid log(0) issues
        prob_distribution = np.where(prob_distribution > 0, prob_distribution, 1e-10)

        # Compute Shannon entropy (log base 2 for bits)
        return entropy(prob_distribution, base=2, axis=axis)

    @staticmethod
    def compute_spectral_entropy(
        data: np.ndarray, fs: int = 1, axis: int = -1
    ) -> float | np.ndarray:
        """
        Computes the spectral entropy of a signal or a collection of signals based on its
        Power Spectral Density (PSD).

        Spectral entropy quantifies the distribution of power across different frequencies,
        providing a measure of signal complexity or disorder.

        Parameters
        ----------
        data : np.ndarray
            Input signal, can be 1D or multi-dimensional.
        fs : int, optional
            Sampling frequency of the signal. Default is 1 (normalized frequency).
        axis : int, optional
            Axis along which the spectral entropy is computed. Default is -1 (last axis).

        Returns
        -------
        Union[float, np.ndarray]
            Spectral entropy of the signal. If `data` is multi-dimensional, an array
            of entropy values is returned along the specified axis.

        Notes
        -----
        - Uses Welch's method to estimate the Power Spectral Density (PSD).
        - Normalizes the PSD to create a probability distribution before computing entropy.
        """
        # Compute Power Spectral Density (PSD) using Welch's method
        nperseg = min(256, max(16, data.shape[axis] // 4))  # Ensure at least 16 samples per segment
        _, psd = welch(data, fs=fs, nperseg=nperseg, axis=axis)

        # Normalize PSD to obtain a probability distribution
        psd_sum = np.sum(psd, axis=axis, keepdims=True)  # Ensure broadcasting
        psd_norm = psd / np.clip(psd_sum, 1e-10, None)  # Avoid division by zero

        # Compute Shannon entropy of the PSD
        spectral_entropy = entropy(psd_norm, base=2, axis=axis)

        return spectral_entropy

    @staticmethod
    def compute_approximate_entropy(
        data: np.ndarray,
        m: int = 2,
        r: float = 0.2,
        axis: int = -1,
    ) -> float | np.ndarray:
        """
        Computes the Approximate Entropy (ApEn) for a signal or a collection of signals.

        Parameters
        ----------
        data : np.ndarray
            - If 1D: A single time series signal `(n_timepoints,)`.
            - If 3D: A collection of signals `(n_samples, n_nodes, n_timepoints)`.
        m : int, optional
            Embedding dimension (default is 2).
        r : float, optional
            Tolerance threshold, set as a fraction (default is 0.2) of the
            signal's standard deviation.
        axis : int, optional
            Axis along which to compute Approximate Entropy (default is -1).

        Returns
        -------
        Union[float, np.ndarray]
            - If input is 1D: Returns a single float.
            - If input is 3D: Returns an array of shape `(n_samples, n_nodes)`.

        Notes
        -----
        - Lower ApEn values indicate more regularity (predictability).
        - Higher ApEn values indicate greater complexity (less predictability).
        """
        # Get length of time series along the specified axis
        N = data.shape[axis]

        # Compute standard deviation per signal and scale r
        r_scaled = r * np.std(data, axis=axis, keepdims=True)

        def phi(m: int) -> np.ndarray:
            """Computes the phi function for embedding dimension m."""
            # Create overlapping subseries
            patterns = np.stack(
                [data[..., i : i + m] for i in range(N - m + 1)], axis=data.ndim - 1
            )

            # Compute max absolute distance (pairwise comparison)
            distances = np.abs(patterns[..., None, :, :] - patterns[..., :, None, :])
            max_distances = np.max(distances, axis=axis)  # Take max difference for each pattern

            # Compute similarity count
            similarity = np.mean(max_distances <= r_scaled[..., None], axis=axis)

            return np.mean(np.log(similarity + 1e-10), axis=axis)  # Avoid log(0)

        # Compute Approximate Entropy)
        ap_en = phi(m) - phi(m + 1)

        return ap_en


if __name__ == "__main__":
    # Example usage
    signal = np.sin(np.linspace(0, 10, 500)) + np.random.normal(0, 0.1, 500)  # Example signal

    # Log energy entropy computation
    log_energy_en = SignalComplexity.compute_log_energy_entropy(signal)
    print(f"Log Energy Entropy: {log_energy_en:.4f}")

    # Shannon entropy computation
    shannon_en = SignalComplexity.compute_shannon_entropy(signal)
    print(f"Shannon Entropy: {shannon_en:.4f}")

    # Spectral entropy computation
    spectral_en = SignalComplexity.compute_spectral_entropy(signal, fs=1)
    print(f"Spectral Entropy: {spectral_en:.4f}")

    # Approximate entropy computation
    ap_en = SignalComplexity.compute_approximate_entropy(signal, m=2, r=0.2)
    print(f"Approximate Entropy: {ap_en:.4f}")
