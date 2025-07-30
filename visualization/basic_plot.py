import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from visualization.kde_residuals import plot_residuals_kde
from visualization.pred_vs_actual import plot_prediction_vs_actual
from visualization.residual_vs_time import plot_residuals_vs_time
from visualization.target_vs_time import plot_target_vs_time


def plot_full_summary_grid(
    rf_model_temp,
    X_train,
    y_train,
    train_time,
    X_test,
    y_test,
    test_time,
    model_name="RF",
    figsize=(16, 12),
):
    """
    Plots a 3x3 grid summary with:
        Row 1: Prediction vs Actual (train), (test), KDE residuals (train & test)
        Row 2: Target vs Time (spanning all columns)
        Row 3: Residuals vs Time (spanning all columns)
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # Top left: Prediction vs Actual (train)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_prediction_vs_actual(
        [rf_model_temp],
        [model_name],
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
        [rf_model_temp],
        [model_name],
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
        [rf_model_temp],
        [model_name],
        X_train,
        y_train,
        X_test,
        y_test,
        plot_type="both",
        ax=ax_kde,
    )

    # Middle row, spanning all columns: Target vs Time
    ax3 = fig.add_subplot(gs[1, :])
    plot_target_vs_time(
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        [rf_model_temp],
        [model_name],
        plot_type="both",
        ax=ax3,
    )
    ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax3.set_xlabel("")

    # Bottom row, spanning all columns: Residuals vs Time
    ax4 = fig.add_subplot(gs[2, :])
    plot_residuals_vs_time(
        [rf_model_temp],
        [model_name],
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

    plt.tight_layout()
    plt.subplots_adjust()  # Adjust vertical space between rows
    plt.show()
