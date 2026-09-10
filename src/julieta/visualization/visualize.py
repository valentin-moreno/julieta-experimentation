import ast
import os
from collections.abc import Callable, Iterable
from itertools import chain
from typing import Literal

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix


class VisualizeData:
    @staticmethod
    def plot_categories_distribution(
        data: pd.Series | dict[str, int],
        title: str = "Category Distribution",
    ) -> None:
        """
        Plots the distribution of categories as a bar chart, with labels showing the height of each bar.

        Parameters
        ----------
        data : Union[pd.Series, Dict[str, int]]
            A dictionary where keys represent category names and values represent their corresponding counts.
        title : str, optional
            Title of the plot, by default "Category Distribution"
        """
        plt.figure(figsize=(10, 5))

        # Convert the dictionary to a pandas Series for plotting
        if isinstance(data, dict):
            data = pd.Series(data)

        ax = data.plot(kind="bar")
        # Add labels on top of each bar
        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 1),  # Offset for the text
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        # Add labels and title
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion(
        y_true: list[int],
        y_pred: list[int],
        title: str,
        labels: list[str] = None,
        labels_names: dict[str, str] = None,
    ) -> None:
        """
        Plot a normalized confusion matrix with annotated raw counts.

        Parameters
        ----------
        y_true : List[int]
            Ground-truth class labels.

        y_pred : List[int]
            Predicted class labels.

        title : str
            Title to display above the confusion matrix.

        labels : List[str], default=None
            Class names to use for axis tick labels. If None, numeric labels
            inferred from the confusion matrix indices will be used.

        Returns
        -------
        None
            This function displays the plot and does not return a value.
        """

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        # Custom colormap
        pink_purple_cmap = LinearSegmentedColormap.from_list("pink_purple", ["white", "#7a0177"])

        # Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap=pink_purple_cmap,
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            square=True,
        )
        plt.title(title, fontsize=14, weight="bold", pad=15)
        if labels_names is not None:
            plt.xlabel(labels_names["x"], fontsize=12)
            plt.ylabel(labels_names["y"], fontsize=12)
        else:
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("True", fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_magnitude_plotly(
        mag_r: np.ndarray,
        mag_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        data_type: Literal["impedance", "admitance"],
        decibels: bool = False,
        log_scale: bool = False,
    ):
        """
        Plots the magnitude data for left and right breast measurements across frequency samples.

        Parameters
        ----------
        mag_r : np.ndarray
            Right-side magnitude data, with shape `(frequencies, nodes)`.
        mag_l : np.ndarray
            Left-side magnitude data, with shape `(frequencies, nodes)`.
        frequency_samples : List[float]
            A list of frequency values (Hz) corresponding to the rows of `mag_l` and `mag_r`.
        nodes : List[int]
            A list of node indices corresponding to the columns of `mag_l` and `mag_r`.
        title : str
            Main title for the plot, typically identifying the patient or measurement session.
        data_type : Literal["impedance", "admitance"]
            Determines the y-axis label (|Z| for impedance, |Y| for admittance).
        decibels : bool, optional
            Whether to plot the magnitude data in decibels, default is False.
        log_scale : bool, optional
            Whether to plot the frequency axis in log scale, default is False.
        """
        # Convert to decibels if needed
        if decibels:
            mag_r = 20 * np.log10(mag_r)
            mag_l = 20 * np.log10(mag_l)
            y_label = "|Z| (Ω dB)" if data_type == "impedance" else "|Y| (S dB)"
        else:
            y_label = "|Z| (Ω)" if data_type == "impedance" else "|Y| (S)"

        # Create subplot: Right (1,1), Left (1,2)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Right Breast Measurements", "Left Breast Measurements"),
            shared_yaxes=True,
        )

        # Plot right breast
        for i in range(mag_r.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=frequency_samples,
                    y=mag_r[:, i],
                    mode="lines+markers",
                    name=f"Node {nodes[i]}",
                    hovertemplate=(
                        "<b>Node:</b> %{customdata[0]}<br>"
                        "<b>Frequency:</b> %{x} Hz<br>"
                        "<b>Magnitude:</b> %{y}<br>"
                    ),
                    customdata=np.column_stack([[nodes[i]] * len(frequency_samples)]),
                ),
                row=1,
                col=1,
            )

        # Plot left breast
        for i in range(mag_l.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=frequency_samples,
                    y=mag_l[:, i],
                    mode="lines+markers",
                    name=f"Node {nodes[i]}",
                    hovertemplate=(
                        "<b>Node:</b> %{customdata[0]}<br>"
                        "<b>Frequency:</b> %{x} Hz<br>"
                        "<b>Magnitude:</b> %{y}<br>"
                    ),
                    customdata=np.column_stack([[nodes[i]] * len(frequency_samples)]),
                ),
                row=1,
                col=2,
            )

        # Log scale if requested
        if log_scale:
            fig.update_xaxes(type="log", row=1, col=1)
            fig.update_xaxes(type="log", row=1, col=2)

        # Labels
        fig.update_yaxes(title_text=y_label, row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)

        fig.update_layout(
            height=500,
            width=1100,
            title_text=title,
            hovermode="closest",
        )

        fig.show()

    @staticmethod
    def plot_magnitude_plotly_single(
        mag: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        highlight_points: list[str] | None = None,
        show_fig=True,
        return_fig=False,
    ):
        # ---------------------------------
        # Parse highlighted points
        # ---------------------------------
        highlight_dict = {}
        if highlight_points:
            for item in highlight_points:
                node, freq = item.split("_")
                node = int(node)
                freq = int(freq)
                highlight_dict.setdefault(node, []).append(freq)

        fig = make_subplots(
            rows=1,
            cols=1,
            shared_yaxes=True,
        )

        for node in nodes:
            fig.add_trace(
                go.Scatter(
                    x=frequency_samples,
                    y=mag.loc[:, node],
                    mode="lines+markers",
                    name=f"Node {node}",
                    legendgroup=f"node_{node}",
                    customdata=np.column_stack([[node] * len(frequency_samples)]),
                    hovertemplate=(
                        "<b>Node:</b> %{customdata[0]}<br>"
                        "<b>Frequency:</b> %{x} Hz<br>"
                        "<b>Magnitude:</b> %{y}<br>"
                    ),
                ),
                row=1,
                col=1,
            )

            # highlight
            if node in highlight_dict:
                for freq in highlight_dict[node]:
                    fig.add_trace(
                        go.Scatter(
                            x=[freq],
                            y=[mag.loc[freq, node]],
                            mode="markers",
                            legendgroup=f"node_{node}",
                            marker=dict(
                                size=5, symbol="circle-open-dot", color="black", line=dict(width=1)
                            ),
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)

        fig.update_layout(
            height=500,
            width=1100,
            title_text=title,
            hovermode="closest",
        )
        if show_fig:
            fig.show()
        if return_fig:
            return fig

    @staticmethod
    def plot_magnitude_static(
        mag_r: np.ndarray,
        mag_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        data_type: Literal["impedance", "admitance"],
        decibels: bool = False,
        log_scale: bool = False,
    ):
        """

        Plot static magnitude curves for left and right breast measurements using matplotlib.

        This function generates a dual-panel static plot that displays the magnitude responses
        for both the right and left breast across the same frequency samples.

        Parameters
        ----------
        mag_r : np.ndarray
            2D array containing the magnitude values for the right breast.
            Shape must be `(n_frequencies, n_nodes)`, where:
            - Each row corresponds to a frequency.
            - Each column corresponds to a node.

        mag_l : np.ndarray
            2D array containing the magnitude values for the left breast.
            Must match the shape and structure of `mag_r`.

        frequency_samples : list of float
            List of frequency values (in Hz) that correspond to the rows of `mag_r` and `mag_l`.

        nodes : list of int
            List of node identifiers associated with each column of the magnitude arrays.

        title : str
            Global title for the figure. Typically includes patient ID and side information.

        data_type : {"impedance", "admitance"}
            Determines the y-axis label:
            - `"impedance"` → |Z| (Ω)
            - `"admitance"` → |Y| (S)

        decibels : bool, optional (default=False)
            If True, magnitude data is converted to decibels using:
            `20 * log10(magnitude)`.

        log_scale : bool, optional (default=False)
            If True, the frequency axis for both subplots is displayed using a logarithmic scale.

        """

        # Convert to dB if needed
        if decibels:
            mag_r = 20 * np.log10(mag_r)
            mag_l = 20 * np.log10(mag_l)
            y_label = "|Z| (Ω dB)" if data_type == "impedance" else "|Y| (S dB)"
        else:
            y_label = "|Z| (Ω)" if data_type == "impedance" else "|Y| (S)"

        # Create figure
        # Choose pastel colors (using seaborn)
        plt.style.use("seaborn-v0_8-whitegrid")
        pink_purple_blue = LinearSegmentedColormap.from_list(
            "pink_purple_blue",
            [
                "#262644",
                "#bd00ad",
                "#e57db0",
            ],
        )

        colors = [
            sns.desaturate(pink_purple_blue(x), 0.6) for x in np.linspace(0.15, 0.85, len(nodes))
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # --- RIGHT breast ---
        ax = axes[0]
        for i in range(mag_r.shape[1]):
            ax.plot(
                frequency_samples,
                mag_r[:, i],
                marker="o",
                color=colors[i],
                markersize=2.5,
                linewidth=1.5,
            )
        if log_scale:
            ax.set_xscale("log")
        ax.set_title("Right Breast Measurements")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        # --- LEFT breast ---
        ax = axes[1]
        for i in range(mag_l.shape[1]):
            ax.plot(
                frequency_samples,
                mag_l[:, i],
                marker="o",
                color=colors[i],
                markersize=2.5,
                linewidth=1.5,
                label=f"Node {nodes[i]}",
            )
        if log_scale:
            ax.set_xscale("log")
        ax.set_title("Left Breast Measurements")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, alpha=0.3)

        # Layout + Title
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()

    @staticmethod
    def plot_phase(
        phase_r: np.ndarray,
        phase_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        data_type: Literal["impedance", "admitance"],
        log_scale: bool = False,
    ) -> None:
        """
        Plots the phase data for right and left breast measurements across frequency samples.

        Parameters
        ----------
        phase_r : np.ndarray
            Right-side phase data, with shape `(frequencies, nodes)`, in degrees.
        phase_l : np.ndarray
            Left-side phase data, with shape `(frequencies, nodes)`, in degrees.
        frequency_samples : List[float]
            A list of frequency values (Hz) corresponding to the rows of `phase_r` and `phase_l`.
        nodes : List[int]
            A list of node indices corresponding to the columns of `phase_r` and `phase_l`.
        title : str
            Main title for the plot, typically identifying the patient or measurement session.
        data_type : Literal["impedance", "admitance"]
            Determines the y-axis label (`arg(Z)` for impedance, `arg(Y)` for admittance).
        log_scale : bool, optional
            Whether to plot the frequency axis in log scale, by default False.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        ylabel = r"$\arg(Z)$ (degrees)" if data_type == "impedance" else r"$\arg(Y)$ (degrees)"

        # # Define colormap for line colors
        # cmap = plt.get_cmap("jet")
        # num_nodes = len(nodes)
        # colors = [cmap(i / num_nodes) for i in range(num_nodes)]

        # Define plot function based on log scale
        plot_func = ax[0].semilogx if log_scale else ax[0].plot

        # Plot Right Breast Data in ax[0]
        for i in range(phase_r.shape[1]):  # Loop through nodes (columns)
            plot_func(
                frequency_samples,
                phase_r[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
                # color=colors[i],
            )
            # ax[0].text(frequency_samples[-1], phase_r[-1, i], nodes[i], fontsize=8)
        ax[0].set(title="Right Breast Measurements", xlabel="Frequency (Hz)", ylabel=ylabel)
        ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax[0].legend(title="Nodes", ncols=3)

        # Plot Left Breast Data in ax[1]
        plot_func = ax[1].semilogx if log_scale else ax[1].plot
        for i in range(phase_l.shape[1]):  # Loop through nodes (columns)
            plot_func(
                frequency_samples,
                phase_l[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
                # color=colors[i],
            )
            # ax[1].text(frequency_samples[-1], phase_l[-1, i], nodes[i], fontsize=8)
        ax[1].set(title="Left Breast Measurements", xlabel="Frequency (Hz)")
        ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax[1].legend(title="Nodes", ncols=3)

        # Set figure title and layout
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_real_part(
        real_r: np.ndarray,
        real_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        data_type: Literal["resistance", "conductance"],
        log_scale: bool = False,
    ) -> None:
        """
        Plots the real part of impedance or admittance data (resistance or conductance)
        for right and left breast measurements across frequency samples.

        Parameters
        ----------
        real_r : np.ndarray
            Right-side real part data (resistance/conductance), shape `(frequencies, nodes)`.
        real_l : np.ndarray
            Left-side real part data (resistance/conductance), shape `(frequencies, nodes)`.
        frequency_samples : List[float]
            A list of frequency values (Hz) corresponding to the rows of `real_r` and `real_l`.
        nodes : List[int]
            A list of node indices corresponding to the columns of `real_r` and `real_l`.
        title : str
            Main title for the plot, typically identifying the patient or measurement session.
        data_type : Literal["resistance", "conductance"]
            Determines the y-axis label (Resistance in Ohms, Conductance in Siemens).
        log_scale : bool, optional
            Whether to plot the frequency axis in log scale, by default False.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        ylabel = "Resistance (Ω)" if data_type == "resistance" else "Conductance (S)"

        # # Define colormap for line colors
        # cmap = plt.get_cmap("jet")
        # num_nodes = len(nodes)
        # colors = [cmap(i / num_nodes) for i in range(num_nodes)]

        # Choose log or linear scale
        plot_func = ax[0].semilogx if log_scale else ax[0].plot

        # Plot Right Breast Data in ax[0]
        for i in range(real_r.shape[1]):  # Loop through nodes (columns)
            plot_func(
                frequency_samples,
                real_r[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
                # color=colors[i],
            )
            # ax[0].text(frequency_samples[-1], real_r[-1, i], nodes[i], fontsize=8)
        ax[0].set(title="Right Breast Measurements", xlabel="Frequency (Hz)", ylabel=ylabel)
        ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax[0].legend(title="Nodes", ncols=3)

        # Plot Left Breast Data in ax[1]
        plot_func = ax[1].semilogx if log_scale else ax[1].plot
        for i in range(real_l.shape[1]):  # Loop through nodes (columns)
            plot_func(
                frequency_samples,
                real_l[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
                # color=colors[i],
            )
            # ax[1].text(frequency_samples[-1], real_l[-1, i], nodes[i], fontsize=8)
        ax[1].set(title="Left Breast Measurements", xlabel="Frequency (Hz)")
        ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax[1].legend(title="Nodes", ncols=3)

        # Set figure title and layout
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_imag_part(
        imag_r: np.ndarray,
        imag_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        data_type: Literal["reactance", "susceptance"],
        log_scale: bool = False,
    ) -> None:
        """
        Plots the imaginary part of impedance or admittance data (reactance or susceptance)
        for right and left breast measurements across frequency samples.

        Parameters
        ----------
        imag_r : np.ndarray
            Right-side imaginary part data (reactance/susceptance), shape `(frequencies, nodes)`.
        imag_l : np.ndarray
            Left-side imaginary part data (reactance/susceptance), shape `(frequencies, nodes)`.
        frequency_samples : List[float]
            A list of frequency values (Hz) corresponding to the rows of `imag_r` and `imag_l`.
        nodes : List[int]
            A list of node indices corresponding to the columns of `imag_r` and `imag_l`.
        title : str
            Main title for the plot, typically identifying the patient or measurement session.
        data_type : Literal["reactance", "susceptance"]
            Determines the y-axis label (Reactance in Ohms, Susceptance in Siemens).
        log_scale : bool, optional
            Whether to plot the frequency axis in log scale, by default False.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        ylabel = "Reactance (Ω)" if data_type == "reactance" else "Susceptance (S)"

        # # Define colormap for line colors
        # cmap = plt.get_cmap("jet")
        # num_nodes = len(nodes)
        # colors = [cmap(i / num_nodes) for i in range(num_nodes)]

        # Choose log or linear scale
        plot_func = ax[0].semilogx if log_scale else ax[0].plot

        # Plot Right Breast Data in ax[0]
        for i in range(imag_r.shape[1]):  # Loop through nodes (columns)
            plot_func(
                frequency_samples,
                imag_r[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
                # color=colors[i],
            )
            # ax[0].text(frequency_samples[-1], imag_r[-1, i], nodes[i], fontsize=8)
        ax[0].set(title="Right Breast Measurements", xlabel="Frequency (Hz)", ylabel=ylabel)
        ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax[0].legend(title="Nodes", ncols=3)

        # Plot Left Breast Data in ax[1]
        plot_func = ax[1].semilogx if log_scale else ax[1].plot
        for i in range(imag_l.shape[1]):  # Loop through nodes (columns)
            plot_func(
                frequency_samples,
                imag_l[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
                # color=colors[i],
            )
            # ax[1].text(frequency_samples[-1], imag_l[-1, i], nodes[i], fontsize=8)
        ax[1].set(title="Left Breast Measurements", xlabel="Frequency (Hz)")
        ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)
        # ax[1].legend(title="Nodes", ncols=3)

        # Set figure title and layout
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def bode_plot(
        mag_r: np.ndarray,
        mag_l: np.ndarray,
        phase_r: np.ndarray,
        phase_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
        data_type: Literal["impedance", "admitance"],
    ) -> None:
        """
        Pseudo Bode plot for the given magnitude and phase measurements.

        Plots the magnitude and phase data for right and left breast measurements
        across frequency samples, displayed in a 2x2 grid of subplots. The top row
        shows the magnitude on a log-log scale, and the bottom row shows the phase
        on a semi-log scale.

        Parameters
        ----------
        mag_r : np.ndarray
            Right-side magnitude data, with shape `(frequencies, nodes)`.
        mag_l : np.ndarray
            Left-side magnitude data, with shape `(frequencies, nodes)`.
        phase_r : np.ndarray
            Right-side phase data in degrees, with shape `(frequencies, nodes)`.
        phase_l : np.ndarray
            Left-side phase data in degrees, with shape `(frequencies, nodes)`.
        frequency_samples : List[float]
            A list of frequency values (Hz) corresponding to the rows of
            `mag_r`, `mag_l`, `phase_r`, and `phase_l`.
        nodes : List[int]
            A list of node indices corresponding to the columns of
            `mag_r`, `mag_l`, `phase_r`, and `phase_l`.
        title : str
            Main title for the plot, typically identifying the patient or measurement session.
        """
        # Create subplots for right and left measurements
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        if data_type == "impedance":
            ylabel_mag = r"$|Z|$ (Ω)"
            ylabel_phase = r"$\arg(Z)$ (degrees)"
        else:
            ylabel_mag = r"$|Y|$ (S)"
            ylabel_phase = r"$\arg(Y)$ (degrees)"

        # Plot right-side magnitude
        ax[0, 0].semilogx(frequency_samples, 20 * np.log10(mag_r), marker=".", linewidth=0.5)
        ax[0, 0].set_title("Right breast measurements")
        ax[0, 0].set_ylabel(ylabel_mag)
        ax[0, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[0, 0].legend(nodes, title="Nodes", ncols=3)

        # Plot left-side magnitude
        ax[0, 1].semilogx(frequency_samples, 20 * np.log10(mag_l), marker=".", linewidth=0.5)
        ax[0, 1].set_title("Left breast measurements")
        ax[0, 1].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[0, 1].legend(nodes, title="Nodes", ncols=3)
        ax[0, 1].sharey(ax[0, 0])

        # Plot right-side phase
        ax[1, 0].semilogx(frequency_samples, phase_r, marker=".", linewidth=0.5)
        ax[1, 0].set_xlabel("Frequency (Hz)")
        ax[1, 0].set_ylabel(ylabel_phase)
        ax[1, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[1, 0].legend(nodes, title="Nodes", ncols=3)

        # Plot left-side phase
        ax[1, 1].semilogx(frequency_samples, phase_l, marker=".", linewidth=0.5)
        ax[1, 1].set_xlabel("Frequency (Hz)")
        ax[1, 1].grid(True, which="both", linestyle="--", linewidth=0.5)
        ax[1, 1].legend(nodes, title="Nodes", ncols=3)
        ax[1, 1].sharey(ax[1, 0])

        # Add a common title and adjust layout
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def nyquist_plot(
        real_r: np.ndarray,
        real_l: np.ndarray,
        imag_r: np.ndarray,
        imag_l: np.ndarray,
        nodes: list[int],
        title: str,
        log_scale: bool = False,
    ) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

        colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(nodes)))

        # Guardar líneas para interacción
        right_lines = []
        left_lines = []

        # =========================
        # RIGHT BREAST
        # =========================
        for i, node in enumerate(nodes):
            (line_r,) = ax[0].plot(
                real_r[i, :],
                -imag_r[i, :],
                marker="o",
                markersize=3,
                linestyle="-",
                linewidth=1.5,
                label=f"{node}",
                color=colors[i],
            )

            right_lines.append(line_r)

        ax[0].set_title("Right breast measurement")
        ax[0].set_xlabel(r"$\mathrm{Real}(Z)$")
        ax[0].set_ylabel(r"$-\mathrm{Imag}(Z)$")

        if log_scale:
            ax[0].set_xscale("log", base=10)
            ax[0].set_yscale("log", base=10)

        ax[0].grid(True, which="both", linestyle="--", linewidth=0.5)

        # =========================
        # LEFT BREAST
        # =========================
        for i, node in enumerate(nodes):
            (line_l,) = ax[1].plot(
                real_l[i, :],
                -imag_l[i, :],
                marker="o",
                markersize=3,
                linestyle="-",
                linewidth=1.5,
                label=f"{node}",
                color=colors[i],
            )

            left_lines.append(line_l)

        ax[1].set_title("Left breast measurement")
        ax[1].set_xlabel(r"$\mathrm{Real}(Z)$")

        if log_scale:
            ax[1].set_xscale("log", base=10)
            ax[1].set_yscale("log", base=10)

        ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)

        # =========================
        # INTERACTIVE LEGEND
        # =========================
        legend = ax[0].legend(
            title="Nodes", ncols=5, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1)
        )

        legend_lines = legend.get_lines()

        # Relacionar cada item de leyenda con ambas curvas
        line_map = {}

        for legline, r_line, l_line in zip(legend_lines, right_lines, left_lines):
            legline.set_picker(True)
            legline.set_pickradius(10)

            line_map[legline] = (r_line, l_line)

        # Evento click
        def on_pick(event):
            legline = event.artist

            r_line, l_line = line_map[legline]

            visible = not r_line.get_visible()

            r_line.set_visible(visible)
            l_line.set_visible(visible)

            # transparencia en leyenda
            legline.set_alpha(1.0 if visible else 0.2)

            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)

        fig.suptitle(title)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_signals_difference(
        data_r: np.ndarray,
        data_l: np.ndarray,
        frequency_samples: list[float],
        nodes: list[int],
        title: str,
    ) -> None:
        """Generate differnce plot for the given signals.

        Parameters
        ----------
        signal_r : np.ndarray
            Right-side signal data, with shape `(frequencies, nodes)`.
        signal_l : np.ndarray
            Left-side signal data, with shape `(frequencies, nodes)`.
        frequency_samples : List[float]
            Frequency values corresponding to the rows of `signal_l` and `signal_r`.
        nodes : List[int]
            Node indices corresponding to the columns of `signal_l` and `signal_r`.
        title : str
            Title of the plot.
        """
        # Compute absolute diference
        node_difference = data_r - data_l

        plt.figure(figsize=(12, 4), constrained_layout=True)

        # Plot right Breast Data
        for i in range(data_r.shape[1]):  # Loop through nodes (columns)
            plt.plot(
                frequency_samples,
                node_difference[:, i],
                marker=".",
                linewidth=0.5,
                label=f"Node {nodes[i]}",
            )
            plt.text(frequency_samples[-1], node_difference[-1, i], nodes[i], fontsize=8)

        # plt.legend(nodes, title="Nodes", ncols=3)
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Difference (R-L)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    @staticmethod
    def plot_kde_distribution(
        df_merged_mammography: pd.DataFrame, column: str, ax: plt.Axes
    ) -> None:
        """
        Plot a Kernel Density Estimate (KDE) for the specified column in a DataFrame on a given Axes.

        Parameters:
        -----------
        df_merged_mammography : pd.DataFrame
            The input DataFrame containing the data to be plotted.
        column : str
            The column name in the DataFrame to be used for the KDE plot.
        ax : plt.Axes
            The Axes object where the KDE plot will be drawn.
        """
        # Extract key statistics
        col_min = df_merged_mammography[column].min()
        col_max = df_merged_mammography[column].max()
        col_mean = df_merged_mammography[column].mean()
        col_median = df_merged_mammography[column].median()

        # Create the KDE plot
        sns.kdeplot(
            data=df_merged_mammography[column].dropna(),
            color="plum",
            linewidth=2,
            fill=True,
            ax=ax,
            warn_singular=False,
        )

        # Add vertical lines for key statistics
        ax.axvline(col_min, color="plum", linestyle="-", label=f"Min: {col_min:.1f}")
        ax.axvline(col_max, color="plum", linestyle="-", label=f"Max: {col_max:.1f}")
        ax.axvline(col_mean, color="plum", linestyle="--", label=f"Mean: {col_mean:.1f}")
        ax.axvline(col_median, color="orchid", linestyle="--", label=f"Median: {col_median:.1f}")

        # Titles and labels
        ax.set_title(f"{column} \ndistribution", fontsize=12, fontweight="bold")
        ax.set_xlabel(column, fontsize=5)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    @staticmethod
    def plot_nulls(df: pd.DataFrame, title: str = "Null Values in the DataFrame") -> None:
        """
        Plot a missing-value matrix visualization for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame whose null values will be visualized.
        title : str, optional
            Title of the plot. Defaults to "Null Values in the DataFrame".

        Returns
        -------
        None
            Displays a missing-value matrix plot.
        """
        plt.figure(figsize=(12, 6))

        # Plot missingness matrix
        msno.matrix(df, color=(0.5, 0, 0.5))

        # Axis labels
        plt.xticks(ticks=range(df.shape[1]), labels=df.columns, rotation=90, fontsize=10)
        plt.title(title, fontsize=14)
        plt.xlabel("Columns", fontsize=12)
        plt.ylabel("Records", fontsize=12)

        plt.show()

    @staticmethod
    def plot_categorical_distribution(
        df_merged_mammography: pd.DataFrame,
        column: str,
        ax: plt.Axes,
        flat_list: bool = False,
        cols_to_flat: list | None = None,
    ) -> None:
        """
        Plot the frequency distribution (counts + percentages) of a categorical variable
        on a specified matplotlib Axes. Supports optional flattening of list-like strings.

        Parameters
        ----------
        df_merged_mammography : pd.DataFrame
            The DataFrame containing the column to be plotted.
        column : str
            Name of the column to visualize.
        ax : plt.Axes
            Matplotlib axis on which the plot will be rendered.
        flat_list : bool, optional
            Whether to flatten list-like string values (e.g., "['a', 'b']"). Default is False.
        cols_to_flat : list, optional
            List of column names for which flattening should be applied.

        Returns
        -------
        None
            Draws a bar plot with counts and percentages.
        """

        # --- Data preprocessing ---
        if flat_list and cols_to_flat and column in cols_to_flat:
            # Parse strings into Python lists
            parsed_lists = (
                df_merged_mammography[column]
                .dropna()
                .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            )

            # Flatten all lists into a single list
            flattened = list(chain.from_iterable(parsed_lists))
            value_counts = pd.Series(flattened).value_counts()

        else:
            # Standard categorical column
            value_counts = df_merged_mammography[column].dropna().value_counts()

        # Compute percentages
        percentages = (value_counts / value_counts.sum()) * 100

        # Sort for consistent visual appearance
        value_counts = value_counts.sort_index()
        percentages = percentages.sort_index()

        # --- Custom color palette ---
        plum_palette = sns.light_palette("plum", n_colors=len(value_counts), input="rgb")

        # --- Bar plot ---
        sns.barplot(
            x=value_counts.index,
            y=value_counts.values,
            hue=[str(val) for val in value_counts.index],
            palette=plum_palette,
            ax=ax,
            legend=False,
        )

        # --- Adjust Y limits to avoid overlap with title ---
        y_max = value_counts.max()
        ax.set_ylim(0, y_max * 1.25)  # Add 25% headroom

        # --- Annotate bars with count + percentage ---
        num_categories = len(value_counts)

        for i, (count, pct) in enumerate(zip(value_counts.values, percentages.values)):
            y_offset = y_max * 0.05  # dynamic offset

            ax.text(
                i,
                count + y_offset,
                f"{count} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,  # keep labels horizontal
                color="black",
            )

        # --- Aesthetics ---
        ax.set_title(f"{column}\nDistribution", fontsize=12, fontweight="bold")
        ax.set_xlabel(column, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis="x", rotation=90)
        ax.grid(alpha=0.3, axis="y")
        sns.despine(ax=ax)

    @staticmethod
    def plot_all_distributions(
        df_merged_mammography: pd.DataFrame,
        numerical_data: list[str],
        categorical_data: list[str],
        flat_list: bool = False,
        cols_to_flat: list[str] | None = None,
    ) -> None:
        """
        Plot all numerical and categorical distributions in a set of subplots.
        Numerical variables are plotted using KDE; categorical variables using bar plots.
        Supports flattening of list-like categorical columns.

        Parameters
        ----------
        df_merged_mammography : pd.DataFrame
            DataFrame containing all variables to be plotted.
        numerical_data : list of str
            List of column names corresponding to numerical variables.
        categorical_data : list of str
            List of column names corresponding to categorical variables.
        flat_list : bool, optional
            Whether to flatten list-like categorical columns. Default is False.
        cols_to_flat : list of str, optional
            Columns for which flattening should be applied.

        Returns
        -------
        None
            Displays a grid of distribution plots.
        """

        num_plots = len(numerical_data) + len(categorical_data)
        num_cols = 3
        num_rows = -(-num_plots // num_cols)  # Ceiling division

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True
        )
        axes = axes.flatten()

        # --- Plot numerical variables ---
        for i, var in enumerate(numerical_data):
            if df_merged_mammography[var].dropna().empty:
                continue
            else:
                VisualizeData.plot_kde_distribution(df_merged_mammography, var, axes[i])

        # --- Plot categorical variables ---
        for i, var in enumerate(categorical_data, start=len(numerical_data)):
            if df_merged_mammography[var].dropna().empty:
                continue
            else:
                VisualizeData.plot_categorical_distribution(
                    df_merged_mammography,
                    var,
                    axes[i],
                    flat_list=flat_list,
                    cols_to_flat=cols_to_flat,
                )

        # Remove unused subplot areas
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.show()

    @staticmethod
    def plot_single_categorical_distribution(df: pd.DataFrame, column: str) -> None:
        """
        Plot the frequency distribution of a categorical column,
        displaying both absolute counts and percentages above each bar.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame that contains the categorical column.

        column : str
            The name of the categorical column to visualize. The function
            computes value counts (excluding NaN values) and plots their
            distribution as a bar chart.

        Returns
        -------
        None
            The function produces a bar plot using matplotlib/seaborn and
            displays it directly. No value is returned.
        """

        value_counts = df[column].dropna().value_counts()
        percentages = (value_counts / value_counts.sum()) * 100

        colors = sns.light_palette("plum", n_colors=len(value_counts), reverse=True)

        plt.figure(figsize=(5, 5))
        ax = sns.barplot(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            hue=value_counts.index.astype(str),  # required for seaborn >= 0.14
            palette=colors,
            legend=False,  # hide redundant legend
        )

        # Add annotations
        for i, (count, pct) in enumerate(zip(value_counts.values, percentages.values)):
            ax.text(
                i,
                count + (max(value_counts.values) * 0.02),
                f"{count} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        ax.set_title(f"{column} Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel(column, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        sns.despine()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sample_magnitude_curves(
        items: Iterable[tuple[str, str]],
        title_prefix: str,
        build_measurements_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
        plot_magnitude_fn: Callable[..., None],
        study_name: str,
        data_path_builder: Callable[[str], str],
    ) -> None:
        """
        Plot impedance magnitude curves for a collection of items, where each item
        contains a test ID and the corresponding side label.

        Parameters
        ----------
        items : Iterable[Tuple[str, str]]
            Iterable containing elements of the form (test_id, side). Each element
            will be processed to load impedance magnitude data and generate
            corresponding plots.

        title_prefix : str
            Text used as the prefix of the title in each generated plot. The final
            title will append the test ID and side information.

        build_measurements_fn : Callable
            Function responsible for loading measurement data.
            Expected signature:
            `fn(path: str, output_data: str, delimiter: str, header: int)
                -> Tuple[pd.DataFrame, pd.DataFrame]`
            It must return magnitude_left_df, magnitude_right_df.

        plot_magnitude_fn : Callable
            Function used to generate magnitude plots.
            Expected parameters include:
            mag_r, mag_l, title, frequency_samples, nodes, data_type

        study_name : str
            Name of the study used to construct the file path for each test ID.

        data_path_builder : Callable[[str], str]
            Function that receives a relative file path and returns the full
            resolved path to the measurement file.

        Returns
        -------
        None
            The function generates plots directly and does not return any value.
        """

        for test_id, side in items:
            # Build full path to measurement file
            file_path = data_path_builder(os.path.join(study_name, f"{test_id}.txt"))

            # Load structured left and right breast data
            mag_l_df, mag_r_df = build_measurements_fn(
                path=file_path,
                output_data="impedance_magnitude",
                delimiter=";",
                header=1,
            )

            # Extract metadata and raw arrays
            frequency_samples = mag_l_df.index.tolist()
            nodes = mag_l_df.columns.tolist()
            mag_l = mag_l_df.values
            mag_r = mag_r_df.values

            # Plot
            plot_magnitude_fn(
                mag_r=mag_r,
                mag_l=mag_l,
                title=f"{title_prefix}\nTest ID: {test_id}  |  Side: {side}",
                frequency_samples=frequency_samples,
                nodes=nodes,
                data_type="impedance",
            )


# import plotly.express as px

# def plot_magnitude(
#         mag_r: np.ndarray,
#         mag_l: np.ndarray,
#         frequency_samples: list,
#         nodes: list,
#         title: str,
#         data_type: str,
#         decibels: bool = False,
#         log_scale: bool = False,
#         output_file: str = "plot.html"
#     ) -> None:
#     """
#     Creates an interactive scatter plot of magnitude data for left and right breast
#     measurements across frequency samples and saves it as an HTML file.

#     Parameters
#     ----------
#     mag_r : np.ndarray
#         Right-side magnitude data, with shape `(frequencies, nodes)`.
#     mag_l : np.ndarray
#         Left-side magnitude data, with shape `(frequencies, nodes)`.
#     frequency_samples : list
#         A list of frequency values (Hz) corresponding to the rows of `mag_l` and `mag_r`.
#     nodes : list
#         A list of node indices corresponding to the columns of `mag_l` and `mag_r`.
#     title : str
#         Main title for the plot.
#     data_type : str
#         Either "impedance" or "admittance", determines the y-axis label.
#     decibels : bool, optional
#         If True, converts magnitude data to decibels (default is False).
#     log_scale : bool, optional
#         If True, uses a logarithmic frequency axis (default is False).
#     output_file : str, optional
#         The name of the output HTML file (default is "plot.html").
#     """

#     # Convert to decibels if required
#     if decibels:
#         mag_r = 20 * np.log10(mag_r)
#         mag_l = 20 * np.log10(mag_l)
#         ylabel = "|Z| (Ω dB)" if data_type == "impedance" else "|Y| (S dB)"
#     else:
#         ylabel = "|Z| (Ω)" if data_type == "impedance" else "|Y| (S)"

#     # Convert data to a long-format Pandas DataFrame
#     df = pd.DataFrame({
#         "Frequency (Hz)": np.tile(frequency_samples, len(nodes) * 2),
#         "Magnitude": np.concatenate([mag_r.flatten(), mag_l.flatten()]),
#         "Node": np.tile(nodes, len(frequency_samples) * 2),
#         "Side": ["Right"] * (mag_r.size) + ["Left"] * (mag_l.size),
#     })

#     # Create scatter plot with Plotly Express
#     fig = px.scatter(
#         df,
#         x="Frequency (Hz)",
#         y="Magnitude",
#         color="Node",
#         symbol="Side",
#         facet_col="Side",
#         labels={"Magnitude": ylabel},
#         title=title,
#         hover_data=["Node", "Frequency (Hz)"],
#     )

#     # Set log scale if required
#     if log_scale:
#         fig.update_xaxes(type="log")

#     # Adjust layout for better readability
#     fig.update_layout(
#         legend_title="Node Index",
#         template="plotly_white"
#     )

#     # Save the plot as an interactive HTML file
#     fig.write_html(output_file)
#     print(f"Interactive plot saved as: {output_file}")
