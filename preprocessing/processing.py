import os
from typing import List

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

    df = drop_highly_correlated_columns(df, time_column, target_column, threshold=0.98)

    df = normalize_features(df, time_column, target_column, strategy="StandardScaler")

    df = reduce_dimensions(
        df, time_column, target_column, method="PCA", pca_variance=0.975
    )

    df = add_features(
        df,
        time_column,
        target_column,
        rolling_windows=[10],
        stats=["mean", "std", "min", "max"],
    )

    df = treat_nan_dataframe(df, time_column, k_neighbors=10)

    df = drop_low_variance_columns(df, time_column, target_column, threshold=1e-8)

    df = drop_highly_correlated_columns(df, time_column, target_column, threshold=0.98)

    print(f"Processed DataFrame shape: {df.shape}")
    return df
