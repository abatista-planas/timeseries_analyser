import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def train_evaluate_stacking(
    models,  # list of unfitted base models
    model_names,  # names for the base models
    X_train,
    y_train,
    X_test,
    y_test,
    meta_model=None,
    verbose=True,
):
    """
    Train and evaluate a stacking ensemble regressor.
    Returns the fitted stacking model and evaluation metrics (train/test).
    """
    if meta_model is None:
        meta_model = Ridge()

    # Build stacking regressor (base models and names as (str, estimator) pairs)
    estimators = list(zip(model_names, models))
    stacking = StackingRegressor(
        estimators=estimators, final_estimator=meta_model, n_jobs=-1, passthrough=False
    )
    print("Fitting stacking model...")
    stacking.fit(X_train, y_train)
    print("END Fitting stacking model...")

    # Predict
    y_pred_train = stacking.predict(X_train)
    y_pred_test = stacking.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    if verbose:
        print(f"Stacking R²:   {r2_train:.4f} / {r2_test:.4f}  (train/test)")
        print(f"Stacking RMSE: {rmse_train:.4f} / {rmse_test:.4f}  (train/test)")

    metrics = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }

    return stacking, metrics
