import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_residuals_kde(
    models,
    model_names,
    X_train,
    y_train,
    X_test,
    y_test,
    plot_type="both",  # "train", "test", or "both"
    figsize=(8, 4),
    ax=None,  # Optional axis to plot on
):
    assert len(models) == len(model_names), "Each model must have a corresponding name."
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # Use tab10 for distinct colors (cycle if many models)
    color_map = plt.get_cmap("tab10")
    n_colors = color_map.N

    for i, (model, name) in enumerate(zip(models, model_names)):
        color_train = color_map((2 * i) % n_colors)
        color_test = color_map((2 * i + 1) % n_colors)
        if plot_type in ["train", "both"]:
            residuals_train = np.array(y_train) - model.predict(X_train)
            kde_train = gaussian_kde(residuals_train)
            x_grid = np.linspace(np.min(residuals_train), np.max(residuals_train), 500)
            ax.plot(
                x_grid,
                kde_train(x_grid),
                color=color_train,
                lw=2,
                alpha=0.85,
                label=f"{name} (Train)",
            )
            ax.fill_between(x_grid, kde_train(x_grid), color=color_train, alpha=0.25)
        if plot_type in ["test", "both"]:
            residuals_test = np.array(y_test) - model.predict(X_test)
            kde_test = gaussian_kde(residuals_test)
            x_grid = np.linspace(np.min(residuals_test), np.max(residuals_test), 500)
            ax.plot(
                x_grid,
                kde_test(x_grid),
                color=color_test,
                lw=2,
                alpha=0.6,
                label=f"{name} (Test)",
            )
            ax.fill_between(x_grid, kde_test(x_grid), color=color_test, alpha=0.15)
    ax.axvline(0, color="k", linewidth=1, label="Zero Residual")
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Density")
    ax.set_title("KDE of Residuals")
    ax.legend()
