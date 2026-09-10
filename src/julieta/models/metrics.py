from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score


class BinaryClassificationMetrics:
    """
    A utility class for computing multy classification metrics based on a confusion matrix.

    Attributes
    ----------
    y_true : Optional[list or np.ndarray]
        Ground truth multy labels (0,1,...).
    y_pred : Optional[list or np.ndarray]
        Predicted multy labels (0,1,...).
    cm : Optional[np.ndarray]
        Confusion matrix computed from `y_true` and `y_pred`.

    Methods
    -------
    accuracy():
        Computes the accuracy of the model.
    specificity():
        Computes the specificity (true negative rate).
    sensitivity():
        Computes the sensitivity (recall or true positive rate).
    negative_predictive_value(weighted=False):
        Computes the negative predictive value (VPN).
    positive_predictive_value(weighted=False):
        Computes the positive predictive value (VPP).
    julieta_score():
        Computes a custom metric as the average of specificity, sensitivity, and VPN.
    """

    def __init__(
        self,
        y_true: list | np.ndarray | None = None,
        y_pred: list | np.ndarray | None = None,
    ):
        """
        Initializes the BinaryClassificationMetrics class with true and predicted labels.

        Parameters
        ----------
        y_true : list or np.ndarray, optional
            Ground truth multy labels (0,1,...).
        y_pred : list or np.ndarray, optional
            Predicted multy labels (0,1,...).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.cm = None
        self.cr = None
        if y_true is not None and y_pred is not None:
            self.cm = confusion_matrix(y_true, y_pred)
            self.cr = classification_report(y_true, y_pred)

    def _update_confusion_matrix(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        labels: list[str | int],
    ) -> None:
        """
        Updates the confusion matrix based on new true and predicted labels.

        Parameters
        ----------
        y_true : list or np.ndarray
            Ground truth multy labels (0,1,...).
        y_pred : list or np.ndarray
            Predicted multy labels (0,1,...).
        labels : list with str or int
            List of labels to index the confusion matrix.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.cm = confusion_matrix(y_true, y_pred, labels=labels)

    def _update_classification_report(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        labels: list[str | int],
    ) -> None:
        """
        Updates the classification report based on new true and predicted labels.

        Parameters
        ----------
        y_true : list or np.ndarray
            Ground truth multy labels (0,1,...).
        y_pred : list or np.ndarray
            Predicted multy labels (0,1,...).
        labels : list with str or int
            List of labels to index the confusion matrix.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.cr = classification_report(y_true, y_pred, labels=labels, output_dict=True)

    def accuracy(self) -> float:
        """Computes the accuracy of the model.

        Returns
        -------
        float
            Accuracy value.
        """
        tn, fp, fn, tp = self.cm.ravel()
        return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

    def specificity(self) -> float:
        """Computes the specificity (true negative rate).

        Returns
        -------
        float
            Specificity value.
        """
        tn, fp, _, _ = self.cm.ravel()
        return tn / (tn + fp) if (tn + fp) != 0 else 0

    def sensitivity(self) -> float:
        """Computes the sensitivity (recall, true positive rate or detection rate).

        Returns
        -------
        float
            Sensitivity value.
        """
        _, _, fn, tp = self.cm.ravel()
        return tp / (tp + fn) if (tp + fn) != 0 else 0

    def negative_predictive_value(self, weighted: bool = False) -> float:
        """Computes the negative predictive value (NPV).

        Parameters
        ----------
        weighted : bool, optional
            Whether to use weighted computation (default is False).

        Returns
        -------
        float
            Negative predictive value.
        """
        tn, fp, fn, tp = self.cm.ravel()
        if weighted:
            w0 = 1 / (tn + fp) if (tn + fp) != 0 else 0
            w1 = 1 / (fn + tp) if (fn + tp) != 0 else 0
            denom = w0 * tn + w1 * fn
            return (w0 * tn) / denom if denom != 0 else 0
        return tn / (tn + fn) if (tn + fn) != 0 else 0

    def positive_predictive_value(self, weighted: bool = False) -> float:
        """Computes the positive predictive value (PPV).

        Parameters
        ----------
        weighted : bool, optional
            Whether to use weighted computation (default is False).

        Returns
        -------
        float
            Positive predictive value.
        """
        tn, fp, fn, tp = self.cm.ravel()
        if weighted:
            w0 = 1 / (tn + fp) if (tn + fp) != 0 else 0
            w1 = 1 / (fn + tp) if (fn + tp) != 0 else 0
            denom = w1 * tp + w0 * fp
            return (w1 * tp) / denom if denom != 0 else 0
        return tp / (tp + fp) if (tp + fp) != 0 else 0

    def positive_likelihood_ratio(self) -> float:
        """
        Computes the Positive Likelihood Ratio (LR+).

        LR+ = sensitivity / (1 - specificity)

        Returns
        -------
        float
            The Positive Likelihood Ratio (LR+). If specificity is 1 (perfectly specific),
            the denominator becomes zero and the function returns 0.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return sens / (1 - spec) if (1 - spec) != 0 else 0

    def negative_likelihood_ratio(self) -> float:
        """
        Computes the Negative Likelihood Ratio (LR-).

        LR- = (1 - sensitivity) / specificity

        Returns
        -------
        float
            The Negative Likelihood Ratio (LR-). If specificity is 0, returns 0.
        """
        sens = self.sensitivity()
        spec = self.specificity()
        return (1 - sens) / spec if spec != 0 else 0

    def julieta_score(self) -> float:
        """
        Computes a custom metric as the average of specificity, sensitivity, and VPN.

        Returns
        -------
        float
            Custom metric value.
        """
        spec = self.specificity()
        sens = self.sensitivity()
        vpn = self.negative_predictive_value(weighted=True)
        return (spec + sens + vpn) / 3

    def update(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        labels: list[str | int],
    ) -> None:
        """Updates the true and predicted labels for metric computation.

        Parameters
        ----------
        y_true : list or np.ndarray
            Ground truth binary labels (0,1).
        y_pred : list or np.ndarray
            Predicted binary labels (0,1).
        labels : list with str or int
            List of labels to index the confusion matrix.
        """
        self._update_confusion_matrix(y_true, y_pred, labels)
        self._update_classification_report(y_true, y_pred, labels)


class MultyClassificationMetrics:
    """
    A utility class for computing multiclass classification metrics based on a
    confusion matrix (one-vs-rest per class).

    Attributes
    ----------
    y_true : Optional[list or np.ndarray]
        Ground truth multy labels (0,1,...).
    y_pred : Optional[list or np.ndarray]
        Predicted multy labels (0,1,...).
    cm : Optional[np.ndarray]
        Confusion matrix computed from `y_true` and `y_pred`.
    num_classes : int
        Number of classes the confusion matrix covers.
    """

    def __init__(
        self,
        y_true: list | np.ndarray | None = None,
        y_pred: list | np.ndarray | None = None,
        num_classes: int = 3,
    ):
        """
        Initializes the MultyClassificationMetrics class with true and predicted labels.

        Parameters
        ----------
        y_true : list or np.ndarray, optional
            Ground truth multy labels (0,1,...).
        y_pred : list or np.ndarray, optional
            Predicted multy labels (0,1,...).
        num_classes : int, optional
            Number of classes to compute one-vs-rest metrics for (default 3).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.cm = None
        self.cr = None
        self.num_classes = num_classes
        if y_true is not None and y_pred is not None:
            self.cm = confusion_matrix(y_true, y_pred)
            self.cr = classification_report(y_true, y_pred)

    def _update_confusion_matrix(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        labels: list[str | int],
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.cm = confusion_matrix(y_true, y_pred, labels=labels)

    def _update_classification_report(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        labels: list[str | int],
    ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    def accuracy(self) -> float:
        cm = self.cm
        tp_tn_sum = 0
        total_predictions = np.sum(cm)

        for i in range(self.num_classes):
            tp = cm[i, i]  # Verdaderos positivos para la clase i
            tn = (
                np.sum(np.delete(cm, i, axis=0), axis=0).sum()
                + np.sum(np.delete(cm, i, axis=1), axis=1).sum()
            )  # Verdaderos negativos
            tp_tn_sum += tp + tn

        return tp_tn_sum / total_predictions if total_predictions != 0 else 0

    def specificity(self) -> list:
        """Computes the specificity (true negative rate) per class."""
        cm = self.cm

        specificities = []
        for i in range(self.num_classes):
            tn = (
                np.sum(np.delete(cm, i, axis=0), axis=0).sum()
                + np.sum(np.delete(cm, i, axis=1), axis=1).sum()
            )
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0
            specificities.append(specificity_i)

        return specificities

    def sensitivity(self) -> list:
        """Computes the sensitivity (recall, true positive rate) per class."""
        cm = self.cm
        sensitivities = []

        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity_i = tp / (tp + fn) if (tp + fn) != 0 else 0
            sensitivities.append(sensitivity_i)

        return sensitivities

    def negative_predictive_value(self, weighted: bool = False) -> list:
        """Computes the negative predictive value (NPV) per class."""
        cm = self.cm
        vpns = []
        if weighted:
            for i in range(self.num_classes):
                tn = (
                    np.sum(np.delete(cm, i, axis=0), axis=0).sum()
                    + np.sum(np.delete(cm, i, axis=1), axis=1).sum()
                )
                fp = np.sum(cm[:, i]) - cm[i, i]
                fn = np.sum(cm[i, :]) - cm[i, i]
                tp = cm[i, i]
                w0 = 1 / (tn + fp) if (tn + fp) != 0 else 0
                w1 = 1 / (fn + tp) if (fn + tp) != 0 else 0
                denom = w0 * tn + w1 * fn
                vpn_i = (w0 * tn) / denom if denom != 0 else 0
                vpns.append(vpn_i)
        else:
            for i in range(self.num_classes):
                tn = (
                    np.sum(np.delete(cm, i, axis=0), axis=0).sum()
                    + np.sum(np.delete(cm, i, axis=1), axis=1).sum()
                )
                fn = np.sum(cm[i, :]) - cm[i, i]
                denom = tn + fn
                vpn_i = tn / denom if denom != 0 else 0
                vpns.append(vpn_i)
        return vpns

    def positive_predictive_value(self, weighted: bool = False) -> list:
        """Computes the positive predictive value (PPV) per class."""
        cm = self.cm
        vpps = []
        if weighted:
            for i in range(self.num_classes):
                tn = (
                    np.sum(np.delete(cm, i, axis=0), axis=0).sum()
                    + np.sum(np.delete(cm, i, axis=1), axis=1).sum()
                )
                fp = np.sum(cm[:, i]) - cm[i, i]
                fn = np.sum(cm[i, :]) - cm[i, i]
                tp = cm[i, i]
                w0 = 1 / (tn + fp) if (tn + fp) != 0 else 0
                w1 = 1 / (fn + tp) if (fn + tp) != 0 else 0
                denom = w1 * tp + w0 * fp
                vpp_i = (w1 * tp) / denom if denom != 0 else 0
                vpps.append(vpp_i)
        else:
            for i in range(self.num_classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                denom = tp + fp
                vpp_i = tp / denom if denom != 0 else 0
                vpps.append(vpp_i)
        return vpps

    def positive_likelihood_ratio(self):
        sens = np.array(self.sensitivity())
        spec = np.array(self.specificity())
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((1 - spec) != 0, sens / (1 - spec), 0)

    def negative_likelihood_ratio(self):
        sens = np.array(self.sensitivity())
        spec = np.array(self.specificity())
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(spec != 0, (1 - sens) / spec, 0)

    def julieta_score(self) -> float:
        """
        Computes a custom metric as the average of the worst-case specificity,
        sensitivity, and NPV across all classes.
        """
        spec = np.min(self.specificity())
        sens = np.min(self.sensitivity())
        vpn = np.min(self.negative_predictive_value(weighted=True))
        return (spec + sens + vpn) / 3

    def update(
        self,
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        labels: list[str | int],
    ) -> None:
        self._update_confusion_matrix(y_true, y_pred, labels)
        self._update_classification_report(y_true, y_pred, labels)


def julieta_score(y_true, y_pred) -> float:
    """Binary julieta_score as a free function (no state), for quick one-off use."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    w0 = 1 / (tn + fp) if (tn + fp) != 0 else 0
    w1 = 1 / (fn + tp) if (fn + tp) != 0 else 0
    denom = w0 * tn + w1 * fn
    vpn = (w0 * tn) / denom if denom != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

    return (specificity + sensitivity + vpn) / 3


def julieta_score_triclass(y_true, y_pred) -> float:
    """Multiclass julieta_score as a free function (no state), for quick one-off use."""
    cm = confusion_matrix(y_true, y_pred)

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.diag(cm) / np.sum(cm, axis=1)
        specificity = np.diag(cm) / np.sum(cm, axis=0)
        vpn = np.diag(cm) / (np.diag(cm) + np.sum(cm, axis=1) - np.diag(cm))

        sensitivity = np.nan_to_num(sensitivity)
        specificity = np.nan_to_num(specificity)
        vpn = np.nan_to_num(vpn)

    return (np.min(sensitivity) + np.min(specificity) + np.min(vpn)) / 3


# Code adapted from:
# https://github.com/malejav02/breast-cancer-ml-research/blob/main/src/evaluation/metrics.py
#
# Original author: Maria Alejandra Velez Clavijo
# License: MIT
#
# Modified by: Maria Alejandra Velez Clavijo (2026)


class ClassificationMetrics:
    """
    Computes classification metrics and provides plotting utilities
    for confusion matrix and classification report.

    Attributes:
        y_true (np.ndarray or list): True labels.
        y_pred (np.ndarray or list): Predicted labels.
        y_proba (Optional[np.ndarray or list]): Predicted probabilities for class 1.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        positive_classes: tuple | None = None,
        triclass: bool = False,
    ):
        """
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Optional predicted probabilities for positive class (for ROC-AUC).
            positive_classes: If given, maps y_true/y_pred to binary (1 if the
                original label is in `positive_classes`, else 0) before computing
                metrics. Leave as None to use y_true/y_pred as-is (already binary).
            triclass: Compatibilidad con notebooks portados de julieta-models --
                equivalente a `positive_classes=(2,)`. Preferir `positive_classes`
                en código nuevo (mismo efecto, explícito en vez de un flag opaco).
        """
        if triclass:
            positive_classes = (2,)
        if positive_classes is not None:
            self.y_true = self.map_to_binary(y_true, positive_classes)
            self.y_pred = self.map_to_binary(y_pred, positive_classes)
        else:
            self.y_true = np.array(y_true)
            self.y_pred = np.array(y_pred)
        self.y_proba = np.array(y_proba) if y_proba is not None else None

        self.cm = confusion_matrix(self.y_true, self.y_pred)
        self.tn, self.fp, self.fn, self.tp = self.cm.ravel()

    @staticmethod
    def map_to_binary(y, positive_classes: tuple):
        return [1 if val in positive_classes else 0 for val in y]

    def roc_auc(self) -> float | None:
        """Compute ROC-AUC score, rounded to 2 decimals.

        Returns None if probabilities are not provided.
        """
        if self.y_proba is None:
            return None
        return round(roc_auc_score(self.y_true, self.y_proba), 2)

    def f1_macro(self) -> float:
        """Compute F1 macro score, rounded to 2 decimals."""
        return round(f1_score(self.y_true, self.y_pred, average="macro"), 2)

    def sensitivity(self) -> float:
        """Compute sensitivity (recall for positive class), rounded to 2 decimals."""
        return round(self.tp / (self.tp + self.fn), 2)

    def specificity(self) -> float:
        """Compute specificity (recall for negative class), rounded to 2 decimals."""
        return round(self.tn / (self.tn + self.fp), 2)

    def positive_predictive_value(self) -> float:
        """Compute Positive Predictive Value (PPV / VPP), rounded to 2 decimals."""
        if (self.tp + self.fp) == 0:
            return 0.0
        return round(self.tp / (self.tp + self.fp), 2)

    def negative_predictive_value(self) -> float:
        """Compute Negative Predictive Value (NPV / VPN), rounded to 2 decimals."""
        if (self.tn + self.fn) == 0:
            return 0.0
        return round(self.tn / (self.tn + self.fn), 2)

    def get_metrics(self) -> dict[str, float | None]:
        """
        Return all metrics as a dictionary with values rounded to 2 decimals.

        Returns:
            Dictionary with keys: 'roc_auc', 'f1_macro', 'sensitivity', 'specificity'.
        """
        return {
            "roc_auc": self.roc_auc(),
            "f1_macro": self.f1_macro(),
            "sensitivity": self.sensitivity(),
            "specificity": self.specificity(),
            "ppv": self.positive_predictive_value(),
            "npv": self.negative_predictive_value(),
        }

    def plot_confusion_matrix(
        self, class_names: list | None = None, return_fig: bool = True
    ) -> plt.Figure | None:
        """
        Plot the confusion matrix.

        Args:
            class_names: Optional list of class names (default ["Negative", "Positive"]).
            return_fig: If True, returns matplotlib Figure object; else shows plot.

        Returns:
            Matplotlib Figure if return_fig=True, else None.
        """
        if class_names is None:
            class_names = ["Negative", "Positive"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            self.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        if return_fig:
            plt.close(fig)
            return fig
        else:
            plt.show()
            return None

    def plot_classification_report(
        self, return_fig: bool = True, print_df: bool = False
    ) -> plt.Figure | None:
        """
        Plot the classification report as a table.

        Args:
            print_df: If True, prints the dataframe with metrics.
            return_fig: If True, returns matplotlib Figure object; else shows plot.

        Returns:
            Matplotlib Figure if return_fig=True, else None.
        """
        report: dict[str, Any] = classification_report(
            self.y_true, self.y_pred, output_dict=True, zero_division=0
        )
        df: pd.DataFrame = pd.DataFrame(report).transpose()
        df = df.round(2)
        if print_df:
            print(df)
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        ax.set_title("Classification Report")
        fig.tight_layout()

        if return_fig:
            plt.close(fig)
            return fig
        else:
            plt.show()
            return None
