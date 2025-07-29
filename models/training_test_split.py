import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    test_split: float = 0.2,
    strategy: str = "temporal",
    shuffle_seed: int = 42,
):
    """
    Split DataFrame into train/test and return X_train, y_train, X_test, y_test, train_time, test_time.
    Uses sklearn's train_test_split for random strategy.

    Returns:
        X_train, y_train, X_test, y_test, train_time, test_time
    """
    if strategy == "temporal":
        n = len(df)
        n_test = int(n * test_split)
        df_sorted = df.sort_values(time_col)
        train = df_sorted.iloc[:-n_test]
        test = df_sorted.iloc[-n_test:]

        X_train = train.drop([time_col, target_col], axis=1).reset_index(drop=True)
        y_train = train[target_col].reset_index(drop=True)
        train_time = train[time_col].reset_index(drop=True)

        X_test = test.drop([time_col, target_col], axis=1).reset_index(drop=True)
        y_test = test[target_col].reset_index(drop=True)
        test_time = test[time_col].reset_index(drop=True)

    elif strategy == "random":
        features = df.drop([time_col, target_col], axis=1)
        target = df[target_col]
        timevals = df[time_col]

        X_train, X_test, y_train, y_test, train_time, test_time = train_test_split(
            features, target, timevals, test_size=test_split, random_state=shuffle_seed
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        train_time = train_time.reset_index(drop=True)
        test_time = test_time.reset_index(drop=True)
    else:
        raise ValueError("strategy must be either 'temporal' or 'random'")

    return X_train, y_train, X_test, y_test, train_time, test_time
