import matplotlib.pyplot as plt
import numpy as np


def plot_residuals_vs_time(
    models,
    model_names,
    X_train,
    y_train,
    train_time,
    X_test,
    y_test,
    test_time,
    plot_type="both",  # "train", "test", or "both"
    figsize=(12, 4),
    ax=None,  # Optional axis to plot on
):
    assert len(models) == len(model_names), "Each model must have a corresponding name."
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # Use tab10 which gives 10 distinct colors, repeat if more models
    color_map = plt.get_cmap("tab10")
    n_colors = color_map.N

    for i, (model, name) in enumerate(zip(models, model_names)):
        color_train = color_map(i % n_colors)
        color_test = color_map((i + 1) % n_colors)
        if plot_type in ["train", "both"]:
            residuals_train = np.array(y_train) - model.predict(X_train)
            ax.scatter(
                train_time,
                residuals_train,
                color=color_train,
                alpha=0.8,
                s=5,
                label=f"{name} (Train)",
                marker="o",
            )
        if plot_type in ["test", "both"]:
            residuals_test = np.array(y_test) - model.predict(X_test)
            ax.scatter(
                test_time,
                residuals_test,
                color=color_test,
                alpha=0.8,
                s=5,
                label=f"{name} (Test)",
                marker=".",
            )
    ax.axhline(0, color="k", linestyle="--", linewidth=1, label="Zero Residual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (y_true - y_pred)")
    ax.set_title("Residuals vs Time")
    ax.legend()
