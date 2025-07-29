import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def normalize_features(
    df: pd.DataFrame, time_col: str, target_col: str, strategy: str = "StandardScaler"
) -> pd.DataFrame:
    """
    Normalize feature columns in the DataFrame using the selected strategy.
    The time and target columns are not modified.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_col (str): Name of the time column.
        target_col (str): Name of the target column.
        strategy (str): Normalization method ('MinMaxScaler', 'MinMax_ext', 'StandardScaler', 'RobustScaler').

    Returns:
        pd.DataFrame: DataFrame with normalized features, original time/target columns.
    """
    # Identify feature columns to scale (exclude time and target)
    feature_cols = [c for c in df.columns if c not in [time_col, target_col]]
    features = df[feature_cols].values

    # Choose scaler
    if strategy == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif strategy == "MinMax_ext":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif strategy == "StandardScaler":
        scaler = StandardScaler()
    elif strategy == "RobustScaler":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    # Fit and transform features
    features_scaled = scaler.fit_transform(features)

    # Rebuild DataFrame
    df_scaled = pd.DataFrame(features_scaled, columns=feature_cols, index=df.index)
    df_scaled[time_col] = df[time_col]
    df_scaled[target_col] = df[target_col]

    # Optional: to preserve column order as in original df
    cols = [time_col, target_col] + feature_cols
    return df_scaled[cols]


# Example usage:
# df_norm = normalize_df(df, time_col='timestamp', target_col='y', strategy='MinMax_ext')
