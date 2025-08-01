import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_evaluate_arimax(
    y_train, X_train, y_test, X_test, order=(1, 0, 0), verbose=True  # AR, I, MA
):
    """
    Fit an ARIMAX model (ARIMA with exogenous variables).
    Returns fitted model and predictions on train/test, with metrics.
    """
    # Fit the model
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    # Predict
    y_pred_train = fit.predict(start=0, end=len(y_train) - 1, exog=X_train)
    y_pred_test = fit.predict(
        start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test
    )

    # Metrics

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    if verbose:
        print(f"ARIMAX R²:   {r2_train:.4f} / {r2_test:.4f}  (train/test)")
        print(f"ARIMAX RMSE: {rmse_train:.4f} / {rmse_test:.4f}  (train/test)")

    metrics = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }

    return fit, y_pred_train, y_pred_test, metrics
