# Code adapted from:
# https://github.com/malejav02/breast-cancer-ml-research/blob/main/src/data/split_data.py
#
# Original author: Maria Alejandra Velez Clavijo
# License: MIT
#
# Modified by: Maria Alejandra Velez Clavijo (2026)


import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Data splitter for train/test or train/val/test.
    """

    def __init__(self, random_state: int = 1124):
        self.random_state = 1124

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float | None = None,
        stratify: bool = True,
    ) -> tuple:
        """
        Split data into train/test or train/val/test.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        test_size : float
            Proportion of data for test set
        val_size : float, optional
            Proportion of training data to use as validation set. If None, only train/test split
        stratify : bool
            Whether to stratify by target

        Returns
        -------
        If val_size is None: X_train, X_test, y_train, y_test
        Else: X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )

        if val_size is not None:
            stratify_train = y_train if stratify else None
            val_relative = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_relative,
                random_state=self.random_state,
                stratify=stratify_train,
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_test, y_train, y_test

    def split_by_group(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_level: str = "patient_id",
        test_size: float = 0.2,
        val_size: float | None = None,
        stratify: bool = True,
    ) -> tuple:
        """
        Split data into train/test (or train/val/test) keeping every row that
        shares the same `group_level` value (e.g. all breasts/rows of the same
        patient_id) on the same side of the split.

        Use this instead of `split()` whenever a single subject can contribute
        more than one row (ej. left/right breast por paciente): partir por fila
        en vez de por grupo deja pasar filas del mismo paciente a train Y test
        a la vez -- fuga de datos, porque comparten columnas categoricas
        (edad, peso, etc.) y probablemente rasgos fisiologicos correlacionados
        entre ambos senos.

        Assumes every row within a group shares the same label (true for este
        dataset: un paciente con ambos senos siempre tiene el mismo `label` en
        las dos filas -- si no fuera asi, se usa la primera etiqueta de cada
        grupo para la estratificacion).

        Parameters
        ----------
        X : pd.DataFrame
            Features, indexed with `group_level` as one of the index levels
            (ej. MultiIndex (patient_id, side)).
        y : pd.Series
            Target, aligned to X.index.
        group_level : str
            Name of the index level to group by before splitting.
        test_size, val_size, stratify :
            Same meaning as in `split()`.

        Returns
        -------
        Same shape as `split()`: (X_train, X_test, y_train, y_test) or, with
        val_size, (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        groups = X.index.get_level_values(group_level)
        group_labels = y.groupby(groups).first()

        stratify_param = group_labels if stratify else None
        train_groups, test_groups = train_test_split(
            group_labels.index,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )

        def _select(group_ids):
            mask = groups.isin(group_ids)
            return X.loc[mask], y.loc[mask]

        X_train, y_train = _select(train_groups)
        X_test, y_test = _select(test_groups)

        if val_size is not None:
            train_group_labels = group_labels.loc[train_groups]
            stratify_train = train_group_labels if stratify else None
            val_relative = val_size / (1 - test_size)
            train_groups_final, val_groups = train_test_split(
                train_group_labels.index,
                test_size=val_relative,
                random_state=self.random_state,
                stratify=stratify_train,
            )
            X_train, y_train = _select(train_groups_final)
            X_val, y_val = _select(val_groups)
            return X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_test, y_train, y_test
