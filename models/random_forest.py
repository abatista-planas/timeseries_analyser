import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_evaluate_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    verbose=True,
):
    """
    Train and evaluate a Random Forest regressor.
    Returns the fitted model and evaluation metrics for both train and test sets.
    """
    # Train model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf.fit(X_train, y_train)

    # Predict
    y_pred_test = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    if verbose:
        print(f"Random Forest R²:   {r2_train:.4f} / {r2_test:.4f}  (train/test)")
        print(f"Random Forest RMSE: {rmse_train:.4f} / {rmse_test:.4f}  (train/test)")

    metrics = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }

    return rf, metrics
