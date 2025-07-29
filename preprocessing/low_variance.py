import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def drop_low_variance_columns(
    df: pd.DataFrame, time_column: str, target_column: str, threshold: float = 1e-8
) -> pd.DataFrame:
    """
    Uses sklearn's VarianceThreshold to drop low-variance columns,
    always preserving the time column and target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_column (str): The column to always keep.
        target_column (str): The target column to always keep.
        threshold (float): Minimum variance for a feature to be kept.

    Returns:
        pd.DataFrame: DataFrame with low-variance columns dropped.
    """
    # Columns always to keep
    cols_to_exclude = [time_column, target_column]

    # Select numeric columns not in excluded
    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in cols_to_exclude
    ]
    if not feature_cols:
        print("No numeric feature columns to check for variance.")
        return df

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[feature_cols])
    high_var_cols = [
        col for col, keep in zip(feature_cols, selector.get_support()) if keep
    ]

    columns_to_keep = cols_to_exclude + high_var_cols
    dropped = [col for col in feature_cols if col not in high_var_cols]
    if dropped:
        print(f"Dropping low-variance columns: {dropped}")

    return df[columns_to_keep]
