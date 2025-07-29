import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_vs_actual(
    models,
    model_names,
    X_train,
    y_train,
    X_test,
    y_test,
    plot_type="test",  # 'test', 'train', or 'both'
    figsize=(8, 6),
):
    """
    Plots prediction vs actual for multiple models on train and/or test data.

    Args:
        models (list): Trained model objects supporting .predict().
        model_names (list): Names for each model (for legend).
        X_train, y_train, X_test, y_test: Data splits.
        plot_type (str): 'train', 'test', or 'both'.
        figsize (tuple): Figure size.
    """
    assert len(models) == len(model_names), "Each model must have a corresponding name."

    plt.figure(figsize=figsize)
    colors = list(mcolors.TABLEAU_COLORS.values())
    n_models = len(models)
    if n_models > len(colors):
        # Use more if needed
        colors += list(mcolors.CSS4_COLORS.values())

    for i, (model, name) in enumerate(zip(models, model_names)):
        color = colors[i]
        # Train predictions
        if plot_type in ["train", "both"]:
            y_train_pred = model.predict(X_train)
            plt.scatter(
                y_train,
                y_train_pred,
                color=color,
                alpha=0.85,
                marker="o",
                label=f"{name} (train)",
            )
        # Test predictions
        if plot_type in ["test", "both"]:
            y_test_pred = model.predict(X_test)
            # Lighter color for test
            lighter_color = mcolors.to_rgba(color, alpha=0.45)
            plt.scatter(
                y_test,
                y_test_pred,
                color=lighter_color,
                marker="o",
                label=f"{name} (test)",
            )

    # 1:1 reference line
    all_y = np.concatenate([y_train, y_test])
    y_min, y_max = np.min(all_y), np.max(all_y)
    plt.plot(
        [y_min, y_max], [y_min, y_max], "k--", linewidth=1, label="Perfect Prediction"
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.show()
