import numpy as np
import pandas as pd


def add_features(
    df,
    time_column,
    target_column,
    rolling_windows=[3, 6, 12],
    stats=["mean", "std", "min", "max"],
):
    """
    Adds cyclical time features and rolling window statistics (for target only) to DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with datetime, target, and features.
        time_column (str): Name of datetime column.
        target_column (str): Name of target column.
        rolling_windows (list): List of window sizes (in rows) for rolling stats.
        stats (list): Which statistics to compute for rolling windows.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    # Cyclical features
    df["minute"] = df[time_column].dt.minute
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["hour"] = df[time_column].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"] = df[time_column].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Rolling window features (target only)
    for w in rolling_windows:
        roll = df[target_column].rolling(window=w, min_periods=1)
        if "mean" in stats:
            df[f"{target_column}_roll{w}_mean"] = roll.mean()
        if "std" in stats:
            df[f"{target_column}_roll{w}_std"] = roll.std()
        if "min" in stats:
            df[f"{target_column}_roll{w}_min"] = roll.min()
        if "max" in stats:
            df[f"{target_column}_roll{w}_max"] = roll.max()

    return df
