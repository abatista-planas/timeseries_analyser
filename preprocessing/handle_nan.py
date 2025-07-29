import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def handle_nan_column(
    series: pd.Series,
    predictors: pd.DataFrame = None,
    strategy: str = "remove_row",
    model_threshold: float = 0.985,
    k_neighbors: int = 10,
    nan_threshold: float = 0.33,  # 33%
):
    """
    Handle NaN in a Series using the specified strategy.

    Args:
        series (pd.Series): Target column to process.
        predictors (pd.DataFrame, optional): Feature columns (must align with series).
        strategy (str): 'remove_row', 'mean', 'model', 'neighbors_mean'.
        model_threshold (float): Minimum R² for model imputation.
        k_neighbors (int): Window for neighbors_mean.
        nan_threshold (float): If NaN percent > threshold, always use mean.

    Returns:
        Tuple[pd.Series, list]: (filled_series, drop_indexes)
    """
    s = series.copy()
    nan_mask = s.isna()
    drop_idxs = []
    nan_ratio = nan_mask.mean()

    if nan_ratio > nan_threshold:
        # Force mean strategy
        print(
            f"Column has {nan_ratio:.1%} NaNs, filling all NaNs with mean (overrides chosen strategy)."
        )
        s = s.fillna(s.mean())
        return s, []

    if not nan_mask.any():
        return s, []

    if strategy == "remove_row":
        drop_idxs = s.index[nan_mask].tolist()
        return s, drop_idxs

    elif strategy == "mean":
        s = s.fillna(s.mean())
        return s, []

    elif strategy == "model":
        # Use predictors if provided, else use index as the only feature
        if predictors is not None and predictors.shape[1] > 0:
            X = predictors
        else:
            X = pd.DataFrame({"index": s.index.values}, index=s.index)

        nonan_mask = ~nan_mask
        X_nonan = X.loc[nonan_mask]
        y_nonan = s.loc[nonan_mask]
        X_nan = X.loc[nan_mask]

        if len(y_nonan) < 10 or X_nonan.shape[1] == 0:
            drop_idxs = s.index[nan_mask].tolist()
            return s, drop_idxs

        # If datetime, convert to int
        y_for_fit = y_nonan
        if np.issubdtype(y_for_fit.dtype, np.datetime64):
            y_for_fit = y_for_fit.astype("int64")

        X_train, X_test, y_train, y_test = train_test_split(
            X_nonan, y_for_fit, test_size=0.25, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        if r2 < model_threshold:
            drop_idxs = s.index[nan_mask].tolist()
            return s, drop_idxs
        else:
            nan_y_pred = model.predict(X_nan)
            if np.issubdtype(s.dtype, np.datetime64):
                nan_y_pred = pd.to_datetime(nan_y_pred.astype("int64"))
            s.loc[nan_mask] = nan_y_pred
            return s, []

    elif strategy == "neighbors_mean":
        s_filled = s.copy()
        for idx in s[nan_mask].index:
            i = s.index.get_loc(idx)
            neighbors = []
            for offset in range(1, k_neighbors // 2 + 1):
                if i - offset >= 0 and not pd.isna(s.iloc[i - offset]):
                    neighbors.append(s.iloc[i - offset])
                if i + offset < len(s) and not pd.isna(s.iloc[i + offset]):
                    neighbors.append(s.iloc[i + offset])
            if neighbors:
                s_filled.at[idx] = np.mean(neighbors)
        return s_filled, []

    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")
