import pandas as pd


def add_features(
    df,
    time_column,
    target_column,
    rolling_windows=[3, 6, 12, 20],
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
    feature_cols = [
        col for col in df.columns if col not in [time_column, target_column]
    ]

    print("Feature columns:", feature_cols)
    # Cyclical features
    # df["minute"] = df[time_column].dt.minute
    # df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    # df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    # df["hour"] = df[time_column].dt.hour
    # df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    # df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # df["dow"] = df[time_column].dt.dayofweek
    # df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    # df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Rolling window features (target only)
    for w in rolling_windows:
        roll = df[target_column].rolling(w, min_periods=1)
        if "mean" in stats:
            df[f"{target_column}_roll{w}_mean"] = roll.mean()
        if "std" in stats:
            df[f"{target_column}_roll{w}_std"] = roll.std()
        if "skew" in stats:
            df[f"{target_column}_roll{w}_skew"] = roll.skew()
        if "kurt" in stats:
            df[f"{target_column}_roll{w}_kurt"] = roll.kurt()
        if "max" in stats:
            df[f"{target_column}_roll{w}_max"] = roll.max()
        if "min" in stats:
            df[f"{target_column}_roll{w}_min"] = roll.min()
        if "median" in stats:
            df[f"{target_column}_roll{w}_median"] = roll.median()

    # Time difference in seconds (as float)
    dt = df[time_column].astype("int64").diff() / 1e9  # nanoseconds to seconds
    mycols = [
        target_column,
    ]
    for col in mycols:
        dy = df[col].diff()
        df[f"{col}_derivative"] = dy / dt
    print("Derivative shape:", df[f"{target_column}_derivative"].shape)
    # Second derivative
    d1 = df[f"{target_column}_derivative"]
    d1t = dt
    dd1 = d1.diff()
    ddtt = d1t  # (dt for the derivative series is still the time between points)
    df[f"{target_column}_second_derivative"] = dd1 / ddtt

    return df
