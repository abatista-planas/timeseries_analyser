import os
from typing import List

from preprocessing.auto_encoder import (
    encode_with_autoencoder,
    fit_autoencoder,
    parallel_search_optimal_bottleneck,
)
from preprocessing.dimension_reduction import reduce_dimensions
from preprocessing.feature_engineering import add_features
from preprocessing.handle_nan import treat_nan_dataframe
from preprocessing.high_correlation import drop_highly_correlated_columns
from preprocessing.load_timeseries import load_timeseries_file
from preprocessing.low_variance import drop_low_variance_columns
from preprocessing.normalization import normalize_features


def processing_data(
    file_path: str,
    time_column: str,
    target_column: str,
    dropped_columns: List[str] | None = None,
    algorithm: str = "PCA",
    n_components: int = 10,
    pca_variance: float = 1.0,
    rolling_windows=[10],
    stats=["mean", "std"],
    mean_centric: bool = False,
):
    """
    Processes a time series data file into a pandas DataFrame.

    Args:
        file_path (str): Path to the data file.
        time_column (str): Name of the time/datetime column.
        target_column (str): Name of the target/label column.
        dropped_columns (list): Columns to drop from the DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if not file_path or not time_column or not target_column:
        raise ValueError(
            "file_path, time_column, and target_column are required arguments."
        )

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = load_timeseries_file(file_path, time_column, target_column, dropped_columns)

    df = treat_nan_dataframe(df, time_column, k_neighbors=10)

    df = drop_low_variance_columns(df, time_column, target_column, threshold=1e-8)

    df = drop_highly_correlated_columns(df, time_column, target_column, threshold=0.99)

    df = normalize_features(df, time_column, target_column, strategy="StandardScaler")

    if algorithm == "PCA":
        if pca_variance is not None:
            print(
                f"Reducing dimensions with PCA to cover {pca_variance*100:.1f}% variance"
            )
            df = reduce_dimensions(
                df, time_column, target_column, method="PCA", pca_variance=pca_variance
            )
        else:
            df = reduce_dimensions(
                df, time_column, target_column, method="PCA", n_components=n_components
            )
    elif algorithm == "UMAP":
        df = reduce_dimensions(
            df, time_column, target_column, method="UMAP", n_components=n_components
        )

    elif algorithm == "Autoencoder":

        # 1. Select only feature columns
        feature_cols = [c for c in df.columns if c not in [time_column, target_column]]
        X_np = df[feature_cols].values

        # 2. Parallel search for best bottleneck size using 90/10 validation average
        print("Searching for optimal autoencoder bottleneck size (with validation)...")
        losses, best_dim = parallel_search_optimal_bottleneck(
            X_np, bottleneck_sizes=[2, 4, 8, 16, 32, 36, 40, 44, 48], epochs=30
        )
        print(f"Best bottleneck size found: {best_dim}")

        # 3. Fit final autoencoder on ALL data (no split)
        autoencoder, recon_loss = fit_autoencoder(
            X_np, bottleneck_dim=best_dim, device="cuda:0", epochs=30
        )
        print(f"Final autoencoder fit, reconstruction loss: {recon_loss:.5f}")

        # 4. Encode features
        X_encoded = encode_with_autoencoder(autoencoder, X_np)
        encoded_df = df[[time_column, target_column]].copy()
        for i in range(X_encoded.shape[1]):
            encoded_df[f"enc_{i+1}"] = X_encoded[:, i]

        df = encoded_df
        print(f"Encoded DataFrame shape: {df.shape}")
        print("df columns:", df.head())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    df = normalize_features(df, time_column, target_column, strategy="StandardScaler")

    df = add_features(
        df,
        time_column,
        target_column,
        rolling_windows=rolling_windows,
        stats=stats,
    )

    if mean_centric:
        df[target_column] = (
            df[target_column] - df[f"{target_column}_roll{rolling_windows[0]}_mean"]
        )

    df = treat_nan_dataframe(df, time_column, k_neighbors=10)

    df = drop_low_variance_columns(df, time_column, target_column, threshold=1e-8)

    df = drop_highly_correlated_columns(df, time_column, target_column, threshold=0.98)

    print(f"Processed DataFrame shape: {df.shape}")
    return df
