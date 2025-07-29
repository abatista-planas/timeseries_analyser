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
):
    """
    Handle NaN values in a data column (Series) with various strategies.

    Args:
        series (pd.Series): Data column to process.
        predictors (pd.DataFrame, optional): Other features for model-based imputation.
        strategy (str): 'remove_row', 'mean', 'model', or 'neighbors_mean'.
        model_threshold (float): Minimum R^2 for model-based fill.
        k_neighbors (int): Used in 'neighbors_mean' strategy.

    Returns:
        Tuple: (processed_series, drop_index_list)
    """
    s = series.copy()
    nan_mask = s.isna()
    drop_idxs = []

    if not nan_mask.any():
        return s, []

    if strategy == "remove_row":
        drop_idxs = s.index[nan_mask].tolist()
        print(f"Strategy: remove_row -> Suggest removing {len(drop_idxs)} rows.")
        return s, drop_idxs

    elif strategy == "mean":
        col_mean = s.mean()
        s = s.fillna(col_mean)
        print(f"Strategy: mean -> Filled {nan_mask.sum()} NaNs with mean {col_mean}.")
        return s, []

    elif strategy == "model":
        if predictors is None:
            raise ValueError("Predictors required for model strategy.")

        nonan_mask = ~nan_mask
        not_nan_X = predictors.loc[nonan_mask]
        not_nan_y = s.loc[nonan_mask]
        nan_X = predictors.loc[nan_mask]

        if len(not_nan_X) < 10 or not_nan_X.shape[1] == 0:
            print("Not enough data or features for modeling. Suggest row removal.")
            drop_idxs = s.index[nan_mask].tolist()
            return s, drop_idxs

        y_for_fit = not_nan_y
        # Convert to int for datetime
        if np.issubdtype(y_for_fit.dtype, np.datetime64):
            y_for_fit = y_for_fit.astype("int64")
        X_train, X_test, y_train, y_test = train_test_split(
            not_nan_X, y_for_fit, test_size=0.25, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Strategy: model -> Random Forest R² on 25% test: {r2:.4f}")

        if r2 < model_threshold:
            print(f"Model R² below threshold {model_threshold}. Suggest row removal.")
            drop_idxs = s.index[nan_mask].tolist()
            return s, drop_idxs
        else:
            nan_y_pred = model.predict(nan_X)
            # If datetime, convert back
            if np.issubdtype(s.dtype, np.datetime64):
                nan_y_pred = pd.to_datetime(nan_y_pred.astype("int64"))
            s.loc[nan_mask] = nan_y_pred
            print(f"Filled {nan_mask.sum()} NaNs with model prediction.")
            return s, []

    elif strategy == "neighbors_mean":
        s_filled = s.copy()
        for idx in s[nan_mask].index:
            i = s.index.get_loc(idx)
            # Get k neighbors before and after
            neighbors = []
            for offset in range(1, k_neighbors // 2 + 1):
                if i - offset >= 0 and not pd.isna(s.iloc[i - offset]):
                    neighbors.append(s.iloc[i - offset])
                if i + offset < len(s) and not pd.isna(s.iloc[i + offset]):
                    neighbors.append(s.iloc[i + offset])
            if neighbors:
                s_filled.at[idx] = np.mean(neighbors)
            else:
                print(f"No neighbors found for idx {idx}, leaving NaN.")
        print(
            f"Strategy: neighbors_mean -> Filled NaNs with mean of {k_neighbors} nearest neighbors."
        )
        return s_filled, []

    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")
