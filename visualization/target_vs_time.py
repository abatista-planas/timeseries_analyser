import matplotlib.colors as mcolors
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
):
    """
    Scatter plot of model predictions vs. time for train and/or test sets.
    """
    assert len(models) == len(model_names), "Number of models and names must match."

    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(models) > len(colors):
        colors += list(mcolors.CSS4_COLORS.values())

    plt.figure(figsize=figsize)

    # Plot actual values
    if plot_type in ["train", "both"]:
        plt.scatter(
            train_time,
            y_train,
            color="gray",
            alpha=0.5,
            label="Actual (Train)",
            marker="o",
        )
    if plot_type in ["test", "both"]:
        plt.scatter(
            test_time,
            y_test,
            color="black",
            alpha=0.5,
            label="Actual (Test)",
            marker="o",
        )

    # Plot model predictions
    for i, (model, name) in enumerate(zip(models, model_names)):
        color = colors[i]
        if plot_type in ["train", "both"]:
            y_train_pred = model.predict(X_train)
            plt.scatter(
                train_time,
                y_train_pred,
                color=color,
                alpha=0.85,
                label=f"{name} (Train)",
                marker=".",
            )
        if plot_type in ["test", "both"]:
            y_test_pred = model.predict(X_test)
            plt.scatter(
                test_time,
                y_test_pred,
                color=color,
                alpha=0.4,
                label=f"{name} (Test)",
                marker=".",
            )

    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.title("Predictions vs Time (Scatter)")
    plt.legend()
    plt.tight_layout()
    plt.show()
