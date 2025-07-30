from models.random_forest import train_evaluate_random_forest
from models.training_test_split import split_features_target
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

    plot_full_summary_grid(
        rf_model_temp,
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        model_name="RF",  # or any other string
        figsize=(16, 12),
        hspace=0.12,
    )


if __name__ == "__main__":
    main()
