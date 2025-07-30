import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error


def regression_metrics_sidebyside(
    models, model_names, X_train, y_train, X_test, y_test, digits=4
):
    """
    Returns a DataFrame with one row per model, and columns with train/test stats in the same cell.
    """
    rows = []
    for model, name in zip(models, model_names):
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        def fmt(a, b):  # Format as 'train / test'
            return f"{round(a, digits)} / {round(b, digits)}"

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)

        mape_train = (
            np.mean(
                np.abs(
                    (y_train - y_train_pred) / np.where(y_train != 0, y_train, np.nan)
                )
            )
            * 100
        )
        mape_test = (
            np.mean(
                np.abs((y_test - y_test_pred) / np.where(y_test != 0, y_test, np.nan))
            )
            * 100
        )
        maxerr_train = np.max(np.abs(y_train - y_train_pred))
        maxerr_test = np.max(np.abs(y_test - y_test_pred))

        row = {
            "Model": name,
            "R2": fmt(r2_train, r2_test),
            "RMSE": fmt(rmse_train, rmse_test),
            "MAPE %": fmt(mape_train, mape_test),
            "Max Error": fmt(maxerr_train, maxerr_test),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # df = df.set_index("Model")
    return df


def show_metrics_table_as_fig(
    models,
    model_names,
    X_train,
    y_train,
    X_test,
    y_test,
    title="",
    fontsize=12,
    colWidths=None,
    ax=None,
    digits=4,
):
    """
    Calls regression_metrics_sidebyside and displays the table in a matplotlib figure or axis.
    """
    df = regression_metrics_sidebyside(
        models, model_names, X_train, y_train, X_test, y_test, digits=digits
    )
    if ax is None:
        _, ax = plt.subplots(
            figsize=(min(2 + 2 * len(df.columns), 12), 1 + 0.5 * len(df))
        )

    ax.axis("off")  # Hide axes

    table = ax.table(
        cellText=df.values,
        rowLabels=None,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colWidths=colWidths if colWidths else [1.0 / len(df.columns)] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.1, 1.2)  # Adjust cell size

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=10)
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax
