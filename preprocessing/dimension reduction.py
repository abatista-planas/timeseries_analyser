from typing import Literal, Optional

import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    method: Literal["PCA", "UMAP", "t-SNE"] = "PCA",
    n_components: int = 2,
    pca_variance: Optional[float] = None,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Reduces feature dimensionality, always keeping time and target columns.

    Args:
        df (pd.DataFrame): DataFrame with time and target columns.
        time_column (str): Name of the time column to keep.
        target_column (str): Name of the target column to keep.
        method (str): One of "PCA", "UMAP", "t-SNE".
        n_components (int): Number of output dimensions (ignored if pca_variance is set).
        pca_variance (float, optional): For PCA, set n_components to cover this fraction of variance.
        random_state (int): For reproducibility.
        **kwargs: Additional args for the reducer.

    Returns:
        pd.DataFrame: DataFrame with time, target, and reduced feature columns.
    """
    # Extract feature columns
    feature_cols = [
        col for col in df.columns if col not in [time_column, target_column]
    ]
    X = df[feature_cols].to_numpy()
    df_out = df[[time_column, target_column]].copy()

    if method == "PCA":

        if pca_variance:
            reducer = PCA(
                n_components=pca_variance, random_state=random_state, **kwargs
            )
        else:
            reducer = PCA(
                n_components=n_components, random_state=random_state, **kwargs
            )
        X_reduced = reducer.fit_transform(X)
        comp_names = [f"PCA_{i+1}" for i in range(X_reduced.shape[1])]

    elif method == "UMAP":

        reducer = umap.UMAP(
            n_components=n_components, random_state=random_state, **kwargs
        )
        X_reduced = reducer.fit_transform(X)
        comp_names = [f"UMAP_{i+1}" for i in range(n_components)]

    elif method == "t-SNE":

        reducer = TSNE(n_components=n_components, random_state=random_state, **kwargs)
        X_reduced = reducer.fit_transform(X)
        comp_names = [f"TSNE_{i+1}" for i in range(n_components)]

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    # Add reduced columns to the output DataFrame
    df_reduced = pd.DataFrame(X_reduced, columns=comp_names, index=df.index)
    df_out = pd.concat([df_out, df_reduced], axis=1)
    return df_out
