import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from models.random_forest import train_evaluate_random_forest
from models.training_test_split import split_features_target
from preprocessing.processing import processing_data
from visualization.pred_vs_actual import plot_prediction_vs_actual
from visualization.target_vs_time import plot_target_vs_time


def main():
    data_path = "/home/albplanas/Desktop/Programming/IoTC/Data/15Days/15_Day_RD1.csv"
    time_column = "Time"
    target_column = "Q2910"
    dropped_columns = ["Q2934", "Q2933"]

    df = processing_data(
        data_path,
        time_column,
        target_column,
        dropped_columns,
        algorithm="PCA",
        n_components=10,
        rolling_windows=[5, 10, 20],
        stats=["mean", "std", "skew", "kurt", "max", "min", "median"],
    )

    X_train, y_train, X_test, y_test, train_time, test_time = split_features_target(
        df,
        time_column,
        target_column,
        test_split=0.2,
        strategy="temporal",
    )

    rf_model_temp, _ = train_evaluate_random_forest(
        X_train, y_train, X_test, y_test, n_estimators=500, max_depth=100, n_jobs=-1
    )
    print("Test size:", len(X_test))
    print("Train size:", len(X_train))

    # xgb_model, _ = train_evaluate_xgboost(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     n_estimators=500,
    #     max_depth=100,
    #     learning_rate=0.05,
    # )

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Top left: Prediction vs Actual (train)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_prediction_vs_actual(
        [rf_model_temp],
        ["Random Forest (Temporal)"],
        X_train,
        y_train,
        X_test,
        y_test,
        plot_type="train",
        ax=ax1,  # <<---- axis passed here
    )

    # Top right: Prediction vs Actual (test)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_prediction_vs_actual(
        [rf_model_temp],
        ["RF"],
        X_train,
        y_train,
        X_test,
        y_test,
        plot_type="test",
        ax=ax2,  # <<---- axis passed here
    )

    # Bottom row, spanning both columns: Target vs Time
    ax3 = fig.add_subplot(gs[1, :])
    plot_target_vs_time(
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        [rf_model_temp],
        ["RF"],
        plot_type="both",
        ax=ax3,  # <<---- axis passed here
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
