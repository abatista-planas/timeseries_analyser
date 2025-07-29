import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_evaluate_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=True,
):
    """
    Train and evaluate an XGBoost regressor.
    Returns the fitted model and evaluation metrics for both train and test sets.
    """
    # Train model
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,  # Suppress XGBoost's own output
    )
    xgb.fit(X_train, y_train)

    # Predict
    y_pred_train = xgb.predict(X_train)
    y_pred_test = xgb.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    if verbose:
        print(f"XGBoost R²:   {r2_train:.4f} / {r2_test:.4f}  (train/test)")
        print(f"XGBoost RMSE: {rmse_train:.4f} / {rmse_test:.4f}  (train/test)")

    metrics = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }

    return xgb, metrics
