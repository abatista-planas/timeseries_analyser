import matplotlib.pyplot as plt
import seaborn as sns

from models.correlation import model_prediction_correlation
from models.random_forest import train_evaluate_random_forest
from models.training_test_split import split_features_target
from models.xgboost import train_evaluate_xgboost
from preprocessing.processing import processing_data
from visualization.basic_plot import plot_full_summary_grid


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

    xgb_model, _ = train_evaluate_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=500,
        max_depth=100,
        learning_rate=0.05,
    )

    plot_full_summary_grid(
        [rf_model_temp],
        ["Random Forest"],
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        target_name=target_column,
    )
    plot_full_summary_grid(
        [xgb_model],
        ["XGBoost"],
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        target_name=target_column,
    )

    # --- Correlation between models ---
    corr_matrix_test = model_prediction_correlation(
        [rf_model_temp, xgb_model], ["Random Forest", "XGBoost"], X_test
    )
    print("Correlation between model predictions (test set):")
    print(corr_matrix_test)
    # Optional: plot as heatmap

    plt.figure(figsize=(4, 3))
    sns.heatmap(corr_matrix_test, annot=True, cmap="coolwarm")
    plt.title("Model Prediction Correlation (Test Set)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
