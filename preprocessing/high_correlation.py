import numpy as np
import pandas as pd


def drop_highly_correlated_columns(
    df: pd.DataFrame, time_column: str, target_column: str, threshold: float = 0.95
) -> pd.DataFrame:
    """
    Drops columns with absolute correlation higher than the given threshold,
    always keeping the time_column and target_column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_column (str): Column always to keep.
        target_column (str): Column always to keep.
        threshold (float): Absolute correlation threshold above which to drop a column.

    Returns:
        pd.DataFrame: DataFrame with highly correlated columns dropped.
    """
    # Always keep time and target columns
    cols_to_exclude = [time_column, target_column]

    # Consider only numeric columns except time and target
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in cols_to_exclude
    ]
    to_drop = set()
    if len(feature_cols) < 2:
        return df

    # Compute correlation matrix for feature columns
    corr_matrix = df[feature_cols].corr().abs()

    # Only examine upper triangle (avoid duplicate checks)
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            col1 = feature_cols[i]
            col2 = feature_cols[j]
            corr_value = corr_matrix.loc[col1, col2]
            if corr_value > threshold:
                # Drop the second column (col2) by default
                to_drop.add(col2)

    if to_drop:
        print(f"Number of Dropped highly correlated columns: {len(list(to_drop))}")
    keep_cols = cols_to_exclude + [col for col in feature_cols if col not in to_drop]
    return df[keep_cols]
