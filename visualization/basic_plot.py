import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from visualization.kde_residuals import plot_residuals_kde
from visualization.pred_vs_actual import plot_prediction_vs_actual
from visualization.residual_vs_time import plot_residuals_vs_time
from visualization.table_stats import show_metrics_table_as_fig
from visualization.target_vs_time import plot_target_vs_time


def plot_full_summary_grid(
    models,
    model_names,
    X_train,
    y_train,
    train_time,
    X_test,
    y_test,
    test_time,
    figsize=(16, 12),
    target_name="",
):
    """
    Plots a 3x3 grid summary with:
        Row 1: Prediction vs Actual (train), (test), KDE residuals (train & test)
        Row 2: Target vs Time (spanning cols 0,1) + metrics table (col 2, spanning rows 1&2)
        Row 3: Residuals vs Time (spanning cols 0,1)
    """

    # Import your plot functions as needed

    fig = plt.figure(figsize=figsize)
    fig.canvas.manager.set_window_title(
        f"Regression Diagnostics (Target: {target_name})"
    )
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # Top left: Prediction vs Actual (train)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_prediction_vs_actual(
        models,
        model_names,
        X_train,
        y_train,
        X_test,
        y_test,
        plot_type="train",
        ax=ax1,
    )

    # Top center: Prediction vs Actual (test)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_prediction_vs_actual(
        models,
        model_names,
        X_train,
        y_train,
        X_test,
        y_test,
        plot_type="test",
        ax=ax2,
    )

    # Top right: KDE of residuals (train and test)
    ax_kde = fig.add_subplot(gs[0, 2])
    plot_residuals_kde(
        models,
        model_names,
        X_train,
        y_train,
        X_test,
        y_test,
        plot_type="both",
        ax=ax_kde,
    )

    # Middle row, spanning columns 0,1: Target vs Time
    ax3 = fig.add_subplot(gs[1, :2])
    plot_target_vs_time(
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        models,
        model_names,
        plot_type="both",
        ax=ax3,
    )
    ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax3.set_xlabel("")

    # Bottom row, spanning columns 0,1: Residuals vs Time
    ax4 = fig.add_subplot(gs[2, :2])
    plot_residuals_vs_time(
        models,
        model_names,
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        plot_type="both",
        ax=ax4,
    )
    ax4.set_title("")

    # Metrics table: span col 2, rows 1 and 2
    ax_table = fig.add_subplot(gs[1:, 2])

    show_metrics_table_as_fig(
        models, model_names, X_train, y_train, X_test, y_test, ax=ax_table, digits=3
    )

    plt.tight_layout()
    plt.show()
