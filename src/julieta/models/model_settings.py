# Code adapted from:
# https://github.com/malejav02/breast-cancer-ml-research/blob/main/src/models/model_settings.py
#
# Original author: Maria Alejandra Velez Clavijo
# License: MIT
#
# Modified by: Maria Alejandra Velez Clavijo (2026)


from typing import Any

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn_genetic.space import Categorical, Continuous, Integer


class ModelConfig:
    def __init__(self, estimator_name: str = "SVC") -> None:
        """
        Initialize the model configuration object.

        This class provides a unified interface to access machine learning
        estimators and their corresponding hyperparameter search spaces used
        for optimization procedures.

        Parameters
        ----------
        estimator_name : str, default="SVC"
            Name of the estimator whose configuration will be used. The name must
            correspond to one of the supported estimators defined in the internal
            registry.

            Supported estimators are:

            - "LogisticRegression"
            - "SVC"
            - "XGBClassifier"

            Otros estimadores de sklearn/xgboost/lightgbm se pueden agregar acá
            cuando haya una necesidad real de usarlos, siguiendo el mismo patrón
            (registrar la instancia en `self.estimators` y su espacio de búsqueda
            en `get_hyperparameter_search_space`).

        Raises
        ------
        ValueError
            If the provided estimator name is not supported.

        """

        self.estimators = {
            "LogisticRegression": LogisticRegression(),
            "SVC": SVC(max_iter=1000),
            "XGBClassifier": xgb.XGBClassifier(),
        }
        self.estimator_name = estimator_name
        if self.estimator_name not in self.estimators.keys():
            raise ValueError(
                f"Unsupported estimator name: {estimator_name}. "
                f"Choose from: {list(self.estimators.keys())}"
            )

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """
        Return the hyperparameter search space for a given classifier.

        This function defines hyperparameter search spaces for multiple classifiers.
        These spaces are intended to be used with genetic algorithm optimization methods.

        Returns
        -------
        Dict[str, Any]
            A dictionary where the keys correspond to hyperparameter names and the
            values define the search space.

            This dictionary can be directly used in genetic hyperparameter
            optimization procedures.

        """

        hyperparameter_spaces = {
            "LogisticRegression": {
                "tol": Continuous(1e-9, 1e-1),
                "fit_intercept": Categorical(choices=[True, False]),
                "intercept_scaling": Continuous(0, 1),
                "class_weight": Categorical(choices=["balanced"]),
                "C": Continuous(0.00001, 1),
                "solver": Categorical(
                    choices=[
                        "lbfgs",
                        "liblinear",
                        "saga",
                        "newton-cg",
                        "newton-cholesky",
                        "sag",
                    ]
                ),
            },
            "SVC": {
                "C": Continuous(0.0001, 100),
                "gamma": Categorical(["scale", "auto"]),
                "kernel": Categorical(["linear"]),
                "tol": Continuous(1e-5, 1e-3),
                "class_weight": Categorical(choices=["balanced"]),
                "probability": Categorical([True]),
            },
            "XGBClassifier": {
                "n_estimators": Integer(50, 500),
                "learning_rate": Continuous(0.0001, 0.5),
                "max_depth": Integer(2, 10),
                "subsample": Continuous(0, 1),
                "gamma": Continuous(0, 1),
            },
        }

        grid = hyperparameter_spaces[self.estimator_name]

        return grid

    def get_estimator_config(self) -> dict[str, Any]:
        """
        Retrieve the configuration associated with the selected estimator.

        This method returns a dictionary containing the instantiated estimator
        object and its corresponding hyperparameter search space.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the estimator configuration with the following keys:

            - "estimator" : BaseEstimator
                Instantiated machine learning estimator compatible with the
                scikit-learn API.

            - "search_space" : Dict[str, Any]
                Hyperparameter search space defining the parameters to optimize.
                The values correspond to search space objects such as
                ``Integer``, ``Continuous``, or ``Categorical``.
        """

        estimator_config_dict = {
            "estimator": self.estimators[self.estimator_name],
            "search_space": self.get_hyperparameter_search_space(),
        }

        return estimator_config_dict
