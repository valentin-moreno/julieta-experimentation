import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class ClassCollapsingClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps a multi-class classifier and collapses its predictions to binary
    at inference time. Trains normally on all original classes.

    Parameters
    ----------
    base_estimator : sklearn-compatible classifier
        The underlying multi-class model.
    negative_classes : tuple
        Original classes to map to 0. All remaining classes map to 1.
        Default: (0, 1) → collapsed 0; class 2 → collapsed 1.
    """

    def __init__(self, base_estimator, negative_classes=(0, 1)):
        self.base_estimator = base_estimator
        self.negative_classes = negative_classes

    def fit(self, X, y):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        self.classes_ = np.array([0, 1])  # collapsed binary classes
        return self

    def _collapse_labels(self, y):
        return np.where(np.isin(y, self.negative_classes), 0, 1)

    def predict(self, X):
        return self._collapse_labels(self.estimator_.predict(X))

    def predict_proba(self, X):
        proba = self.estimator_.predict_proba(X)
        orig_classes = self.estimator_.classes_
        # P(collapsed=1) = sum of probabilities of all non-negative original classes
        neg_mask = np.isin(orig_classes, self.negative_classes)
        pos_proba = proba[:, ~neg_mask].sum(axis=1)
        return np.column_stack([1 - pos_proba, pos_proba])

    # ------------------------------------------------------------------ #
    # Proper get_params / set_params so GASearchCV puede tunar el
    # base_estimator transparentemente con el prefijo base_estimator__param
    # ------------------------------------------------------------------ #
    def get_params(self, deep=True):
        params = {
            "base_estimator": self.base_estimator,
            "negative_classes": self.negative_classes,
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            for k, v in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{k}"] = v
        return params

    def set_params(self, **params):
        base_params = {}
        for k, v in params.items():
            if k in {"base_estimator", "negative_classes"}:
                setattr(self, k, v)
            elif k.startswith("base_estimator__"):
                base_params[k[len("base_estimator__") :]] = v
        if base_params:
            self.base_estimator.set_params(**base_params)
        return self
