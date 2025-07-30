from models.random_forest import train_evaluate_random_forest
from models.training_test_split import split_features_target
from models.xgboost import train_evaluate_xgboost
from preprocessing.processing import processing_data


def main():
    data_path = "/home/albplanas/Desktop/Programming/IoTC/Data/15Days/15_Day_RD1.csv"
    time_column = "Time"
    target_column = "Q2934"
    dropped_columns = ["Q2910", "Q2933"]

    df = processing_data(
        data_path,
        time_column,
        target_column,
        dropped_columns,
        algorithm="UMAP",
        n_components=1,
        rolling_windows=[5, 10, 20],
        stats=["mean", "std", "skew", "kurt", "max", "min", "median"],
    )

    X_train, y_train, X_test, y_test, train_time, test_time = split_features_target(
        df,
        time_column,
        target_column,
        test_split=0.4,
        strategy="temporal",
    )

    rf_model_temp, _ = train_evaluate_random_forest(
        X_train, y_train, X_test, y_test, n_estimators=500, max_depth=100, n_jobs=-1
    )
    print("Test size:", len(X_test))
    xgb_model, _ = train_evaluate_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=500,
        max_depth=100,
        learning_rate=0.2,
    )

    # plot_prediction_vs_actual(
    #     [ rf_model_temp],
    #     ["Random Forest (Temporal)"],
    #     X_train, y_train,
    #     X_test, y_test,
    #     plot_type='test',  # 'test', 'train', or 'both'
    #     figsize=(8, 6)
    # )

    # plot_target_vs_time(X_train, y_train, train_time,
    #                     X_test, y_test, test_time,
    #                     [rf_model_temp],
    #                     ['RF_temp', ],
    #                     plot_type='test',)


if __name__ == "__main__":
    main()
