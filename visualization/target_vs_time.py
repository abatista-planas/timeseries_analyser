import matplotlib.pyplot as plt


def plot_target_vs_time(
    X_train,
    y_train,
    train_time,
    X_test,
    y_test,
    test_time,
    models,
    model_names,
    plot_type="both",  # "train", "test", or "both"
    figsize=(12, 4),
    ax=None,  # Optional axis to plot on
):
    """
    Line plot of model predictions vs. time for train and/or test sets.
    """
    assert len(models) == len(model_names), "Number of models and names must match."
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # Use tab10 for distinct prediction colors
    color_map = plt.get_cmap("tab10")
    n_colors = color_map.N

    # Plot actual values as lines
    if plot_type in ["train", "both"]:
        ax.plot(
            train_time,
            y_train,
            color="gray",
            alpha=0.7,
            label="Actual (Train)",
            linewidth=1,
        )
    if plot_type in ["test", "both"]:
        ax.plot(
            test_time,
            y_test,
            color="black",
            alpha=0.7,
            label="Actual (Test)",
            linewidth=1,
        )

    # Plot model predictions as lines, with separate color for train and test
    for i, (model, name) in enumerate(zip(models, model_names)):
        color_train = color_map((2 * i) % n_colors)
        color_test = color_map((2 * i + 1) % n_colors)
        if plot_type in ["train", "both"]:
            y_train_pred = model.predict(X_train)
            ax.plot(
                train_time,
                y_train_pred,
                color=color_train,
                alpha=0.85,
                label=f"{name} (Train Pred.)",
                linewidth=1,
            )
        if plot_type in ["test", "both"]:
            y_test_pred = model.predict(X_test)
            ax.plot(
                test_time,
                y_test_pred,
                color=color_test,
                alpha=0.7,
                label=f"{name} (Test Pred.)",
                linewidth=1,
                linestyle="--",
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.set_title("Predictions vs Time (Line Plot)")
    ax.legend()
