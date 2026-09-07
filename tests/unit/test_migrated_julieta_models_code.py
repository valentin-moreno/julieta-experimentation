"""Smoke tests for code ported from julieta-models in the fase 1 migration.

Not exhaustive test coverage of every method — just enough per module to
confirm it imports and runs correctly in this repo's environment/dependency
set. See docs/decisions/0010-migration-from-julieta-models.md.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from julieta.data.split_data import DataSplitter
from julieta.features.signal_complexity import SignalComplexity, estimate_bins
from julieta.models.classifiers import BalancedXGBClassifier
from julieta.models.collapsing_classifier import ClassCollapsingClassifier
from julieta.models.metrics import (
    BinaryClassificationMetrics,
    ClassificationMetrics,
    MultyClassificationMetrics,
    julieta_score,
    julieta_score_triclass,
)
from julieta.models.model_settings import ModelConfig
from julieta.visualization.projections import VisualizeProjections


def test_signal_complexity_metrics_run_on_a_synthetic_signal():
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 10, 200)) + rng.normal(0, 0.1, 200)

    assert isinstance(SignalComplexity.compute_log_energy_entropy(signal), float)
    assert isinstance(SignalComplexity.compute_shannon_entropy(signal), float)
    assert isinstance(SignalComplexity.compute_spectral_entropy(signal), float)
    assert isinstance(SignalComplexity.compute_approximate_entropy(signal), float)
    assert estimate_bins(signal) > 0


def test_data_splitter_split_by_group_keeps_groups_together():
    index = pd.MultiIndex.from_tuples(
        [(f"p{i}", side) for i in range(20) for side in ("left", "right")],
        names=["patient_id", "side"],
    )
    X = pd.DataFrame({"feature": range(len(index))}, index=index)
    y = pd.Series([i % 2 for i in range(20) for _ in range(2)], index=index)

    X_train, X_test, y_train, y_test = DataSplitter().split_by_group(X, y, test_size=0.3)

    train_patients = set(X_train.index.get_level_values("patient_id"))
    test_patients = set(X_test.index.get_level_values("patient_id"))
    assert train_patients.isdisjoint(test_patients)


def test_class_collapsing_classifier_predicts_binary_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = rng.integers(0, 3, size=60)

    from sklearn.linear_model import LogisticRegression

    clf = ClassCollapsingClassifier(LogisticRegression(), negative_classes=(0, 1))
    clf.fit(X, y)

    assert set(clf.predict(X)) <= {0, 1}
    assert clf.predict_proba(X).shape == (60, 2)


def test_balanced_xgb_classifier_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.array([0] * 35 + [1] * 5)

    clf = BalancedXGBClassifier(n_estimators=5)
    clf.fit(X, y)

    assert set(clf.predict(X)) <= {0, 1}


def test_binary_classification_metrics_and_free_functions():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    metrics = BinaryClassificationMetrics(y_true, y_pred)
    assert 0 <= metrics.julieta_score() <= 1
    assert 0 <= julieta_score(y_true, y_pred) <= 1


def test_multy_classification_metrics_likelihood_ratios_do_not_crash():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2, 0, 1, 2]

    metrics = MultyClassificationMetrics(y_true, y_pred, num_classes=3)
    # Antes de portar, esto explotaba con TypeError (dividía listas de Python
    # directo) — ver ADR 0010.
    plr = metrics.positive_likelihood_ratio()
    nlr = metrics.negative_likelihood_ratio()

    assert len(plr) == 3
    assert len(nlr) == 3
    assert 0 <= julieta_score_triclass(y_true, y_pred) <= 1


def test_classification_metrics_with_parameterized_positive_classes():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 2]

    metrics = ClassificationMetrics(y_true, y_pred, positive_classes=(2,))
    result = metrics.get_metrics()

    assert result["sensitivity"] is not None
    assert result["specificity"] is not None


def test_visualize_projections_pca_returns_expected_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"f{i}" for i in range(5)])

    result = VisualizeProjections().pca_projection(X, transformer=StandardScaler(), n_components=2)

    assert list(result.columns) == ["Component_1", "Component_2"]
    assert len(result) == 30


def test_model_config_get_estimator_config_does_not_crash():
    # Antes de portar, esto explotaba con TypeError porque el código llamaba
    # una instancia ya creada como si fuera una clase — ver ADR 0010.
    config = ModelConfig("XGBClassifier").get_estimator_config()

    assert "estimator" in config
    assert "search_space" in config
    assert "n_estimators" in config["search_space"]
