from typing import Tuple

import pandas as pd


def split_features_target(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    test_split: float = 0.2,
    strategy: str = "temporal",
    shuffle_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split DataFrame into train/test and return X_train, y_train, X_test, y_test.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_col (str): Name of the time column (will be dropped from features).
        target_col (str): Name of the target column.
        test_split (float): Fraction of data to use as test set.
        strategy (str): "temporal" or "random".
        shuffle_seed (int): Seed for reproducibility (random only).

    Returns:
        X_train, y_train, X_test, y_test
    """
    n = len(df)
    n_test = int(n * test_split)

    if strategy == "temporal":
        df_sorted = df.sort_values(time_col)
        test = df_sorted.iloc[-n_test:]
        train = df_sorted.iloc[:-n_test]
    elif strategy == "random":
        df_shuffled = df.sample(frac=1, random_state=shuffle_seed)
        test = df_shuffled.iloc[:n_test]
        train = df_shuffled.iloc[n_test:]
    else:
        raise ValueError("strategy must be either 'temporal' or 'random'")

    # Prepare features and targets, drop time column
    X_train = train.drop([time_col, target_col], axis=1).reset_index(drop=True)
    y_train = train[target_col].reset_index(drop=True)
    X_test = test.drop([time_col, target_col], axis=1).reset_index(drop=True)
    y_test = test[target_col].reset_index(drop=True)

    return X_train, y_train, X_test, y_test
