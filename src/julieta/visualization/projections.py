from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS
from sklearn.pipeline import Pipeline


class VisualizeProjections:
    """
    A class to perform projections and visualize them in 2D and 3D within the same figure.
    """

    def __init__(self):
        pass

    def pca_projection(
        self,
        X: pd.DataFrame | np.ndarray,
        transformer: TransformerMixin,
        n_components: int = 2,
        return_projector: bool = False,
    ) -> pd.DataFrame:
        """
        Performs PCA projection on the input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input feature matrix.
        n_components : int, optional
            Number of principal components to retain, by default 2.
        transformer : TransformerMixin
            A preprocessing transformer (e.g., StandardScaler, MinMaxScaler) to apply before PCA.
        return_projector : bool, optional
            If True, returns the fitted PCA object along with the transformed DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed dataset with principal components.
        """
        # Define PCA pipeline with preprocessing transformer
        pipeline = Pipeline(
            [
                ("transformer", transformer),
                ("PCA", PCA(n_components=n_components)),
            ]
        )

        # Apply PCA transformation
        X_reduced = pipeline.fit_transform(X)
        print(f"Explained variance ratio: {pipeline.named_steps['PCA'].explained_variance_ratio_}")
        if return_projector:
            return pd.DataFrame(
                data=X_reduced,
                columns=[f"Component_{idx + 1}" for idx in range(n_components)],
                index=X.index if isinstance(X, pd.DataFrame) else None,
            ), pipeline
        else:
            # Construct output DataFrame
            return pd.DataFrame(
                data=X_reduced,
                columns=[f"Component_{idx + 1}" for idx in range(n_components)],
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )

    def kpca_projection(
        self,
        X: pd.DataFrame | np.ndarray,
        transformer: TransformerMixin,
        n_components: int = 2,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Performs Kernel PCA (KPCA) projection on the input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input feature matrix.
        transformer : TransformerMixin
            A preprocessing transformer (e.g., StandardScaler, MinMaxScaler) to apply before KPCA.
        n_components : int, optional
            Number of dimensions for KPCA projection, by default 2.
        kwargs : dict
            Additional keyword arguments for KPCA.

        Returns
        -------
        pd.DataFrame
            Transformed dataset with KPCA components.
        """
        # Define KPCA pipeline with preprocessing transformer
        pipeline = Pipeline(
            [
                ("transformer", transformer),
                ("KPCA", KernelPCA(n_components=n_components, **kwargs)),
            ]
        )

        # Apply KPCA transformation
        X_reduced = pipeline.fit_transform(X)

        # Construct output DataFrame
        return pd.DataFrame(
            data=X_reduced,
            columns=[f"Component_{idx + 1}" for idx in range(n_components)],
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

    def umap_projection(
        self,
        X: pd.DataFrame | np.ndarray,
        transformer: TransformerMixin,
        n_components: int = 2,
        labels: pd.Series | None = None,
        return_projector: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Performs UMAP projection on the input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input feature matrix.
        transformer : TransformerMixin
            A preprocessing transformer (e.g., StandardScaler, MinMaxScaler) to apply before UMAP.
        n_components : int, optional
            Number of dimensions for UMAP projection, by default 2.
        labels : Optional[pd.Series], optional
            Supervised labels for UMAP. If provided, they are converted to numerical form if needed.
        kwargs : dict
            Additional keyword arguments for UMAP.
        return_projector : bool, optional
            If True, returns the fitted UMAP object along with the transformed DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed dataset with UMAP components.
        """
        if labels is not None:
            labels = labels.astype("category").cat.codes  # Ensure numerical labels for UMAP

        pipeline = Pipeline(
            [
                ("transformer", transformer),
                ("UMAP", umap.UMAP(n_components=n_components, **kwargs)),
            ]
        )
        X_reduced = pipeline.fit_transform(X)
        if return_projector:
            return pd.DataFrame(
                data=X_reduced,
                columns=[f"Component_{idx + 1}" for idx in range(n_components)],
                index=X.index if isinstance(X, pd.DataFrame) else None,
            ), pipeline
        return pd.DataFrame(
            data=X_reduced,
            columns=[f"Component_{idx + 1}" for idx in range(n_components)],
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

    def mds_projection(
        self,
        X: pd.DataFrame | np.ndarray,
        transformer: TransformerMixin,
        n_components: int = 2,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Performs MDS projection on the input data.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input feature matrix.
        transformer : TransformerMixin
            A preprocessing transformer (e.g., StandardScaler, MinMaxScaler) to apply before MDS.
        n_components : int, optional
            Number of dimensions for MDS projection, by default 2.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        pd.DataFrame
            Transformed dataset with MDS components.
        """
        # Define MDS pipeline with preprocessing transformer
        pipeline = Pipeline(
            [
                ("transformer", transformer),
                ("MDS", MDS(n_components=n_components, **kwargs)),
            ]
        )

        # Apply PCA transformation
        X_reduced = pipeline.fit_transform(X)

        # Construct output DataFrame
        return pd.DataFrame(
            data=X_reduced,
            columns=[f"Component_{idx + 1}" for idx in range(n_components)],
            index=X.index if isinstance(X, pd.DataFrame) else None,
        )

    def visualize_projection(
        self,
        X: pd.DataFrame,
        transformer: TransformerMixin,
        method: Literal["PCA", "KPCA", "UMAP", "MDS"],
        y: pd.Series | None = None,
        supervised_umap: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Performs projection and visualizes both 2D and 3D scatter plots.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        transformer : TransformerMixin
            Preprocessing transformer.
        method : Literal["PCA", "KPCA", "UMAP", "MDS"]
            Projection method.
        y : Optional[pd.Series]
            Labels used for coloring.
        supervised_umap : bool
            Whether to use supervised UMAP.
        kwargs : dict
            Additional arguments for projection method.

        Returns
        -------
        pd.DataFrame
            2D projected dataframe.
        """

        # =========================================================
        # Ensure alignment between X and y
        # =========================================================
        X = X.copy()

        if y is not None:
            y = y.loc[X.index]

            # Remove rows with NaNs in X or y
            mask = ~(X.isna().any(axis=1) | y.isna())

            X = X.loc[mask]
            y = y.loc[mask]

            print(f"Remaining samples after NaN filtering: {len(X)}")
            print(y.value_counts(dropna=False))

        # =========================================================
        # Compute projections
        # =========================================================
        if method == "PCA":
            X_reduced_2D = self.pca_projection(
                X,
                n_components=2,
                transformer=transformer,
            )

            X_reduced_3D = self.pca_projection(
                X,
                n_components=3,
                transformer=transformer,
            )

        elif method == "UMAP":
            label_data = y if supervised_umap else None

            X_reduced_2D = self.umap_projection(
                X,
                n_components=2,
                transformer=transformer,
                labels=label_data,
                **kwargs,
            )

            X_reduced_3D = self.umap_projection(
                X,
                n_components=3,
                transformer=transformer,
                labels=label_data,
                **kwargs,
            )

        elif method == "MDS":
            X_reduced_2D = self.mds_projection(
                X,
                n_components=2,
                transformer=transformer,
                **kwargs,
            )

            X_reduced_3D = self.mds_projection(
                X,
                n_components=3,
                transformer=transformer,
                **kwargs,
            )

        elif method == "KPCA":
            X_reduced_2D = self.kpca_projection(
                X,
                n_components=2,
                transformer=transformer,
                **kwargs,
            )

            X_reduced_3D = self.kpca_projection(
                X,
                n_components=3,
                transformer=transformer,
                **kwargs,
            )

        else:
            raise ValueError(f"Unsupported method: {method}")

        # =========================================================
        # Ensure indices are preserved
        # =========================================================
        X_reduced_2D.index = X.index
        X_reduced_3D.index = X.index

        # =========================================================
        # Prepare plotting data
        # =========================================================
        df_2D = X_reduced_2D.copy()
        df_3D = X_reduced_3D.copy()

        point_colors = None
        unique_labels = None
        color_map = None

        if y is not None:
            # Attach labels
            df_2D = pd.concat([df_2D, y], axis=1)
            df_3D = pd.concat([df_3D, y], axis=1)

            # Convert to categorical
            y_coded = y.astype("category")
            unique_labels = list(y_coded.cat.categories)

            # Explicit colors (MUCH safer than cmap)
            cmap = plt.get_cmap("tab20")

            color_map = {
                label: cmap(i / max(len(unique_labels) - 1, 1))
                for i, label in enumerate(unique_labels)
            }

            point_colors = y.map(color_map)

            print("\nClass-color mapping:")
            for k, v in color_map.items():
                print(f"{k} -> {v}")

        # =========================================================
        # Plotting
        # =========================================================
        fig = plt.figure(figsize=(12, 6))

        # ---------------------------------------------------------
        # 2D Plot
        # ---------------------------------------------------------
        ax1 = fig.add_subplot(121)

        ax1.scatter(
            df_2D["Component_1"],
            df_2D["Component_2"],
            c=point_colors if y is not None else None,
            alpha=0.7,
            edgecolors="k",
        )

        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        if "title" in kwargs:
            ax1.set_title(f"{method} 2D Projection\n {kwargs['title']}")
        else:
            ax1.set_title(f"{method} 2D Projection")

        # Legend
        if y is not None:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map[label],
                    markeredgecolor="k",
                    markersize=8,
                    linestyle="",
                )
                for label in unique_labels
            ]

            ax1.legend(
                handles,
                unique_labels,
                title=y.name,
                loc="best",
            )

        # ---------------------------------------------------------
        # 3D Plot
        # ---------------------------------------------------------
        ax2 = fig.add_subplot(122, projection="3d")

        ax2.scatter(
            df_3D["Component_1"],
            df_3D["Component_2"],
            df_3D["Component_3"],
            c=point_colors if y is not None else None,
            alpha=0.7,
            edgecolors="k",
        )

        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.set_zlabel("Component 3")
        if "title" in kwargs:
            ax2.set_title(f"{method} 3D Projection\n {kwargs['title']}")
        else:
            ax2.set_title(f"{method} 3D Projection")

        plt.tight_layout()
        plt.show()

        return df_2D
