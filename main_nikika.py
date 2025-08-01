import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from models.correlation import model_prediction_correlation
from models.embedding import train_evaluate_stacking
from models.random_forest import train_evaluate_random_forest
from models.training_test_split import split_features_target
from models.xgboost import train_evaluate_xgboost
from preprocessing.processing import processing_data
from visualization.basic_plot import plot_full_summary_grid


def main():
    data_path = "/home/albplanas/Downloads/NaKika_MCL_merged.xlsx"
    time_column = "Datetime"
    target_column = "Primary process value (ILSX_140:Value)"
    dropped_columns = ["Date", "Time", "ModelTime"]

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

    # Unfitted base models
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=100, n_jobs=-1, random_state=42
    )
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100, max_depth=10, learning_rate=0.05, random_state=42
    )

    models = [rf, xgb_reg]
    model_names = ["Random Forest", "XGBoost"]

    stacking_model, stacking_metrics = train_evaluate_stacking(
        models,
        model_names,
        X_train,
        y_train,
        X_test,
        y_test,
        meta_model=None,  # Default is Ridge, you can pass your own
        verbose=True,
    )

    # 6. Optionally, use all models for comparison/plotting:
    # (You may want to fit and evaluate each separately if desired.)
    plot_full_summary_grid(
        [stacking_model],  # includes base models and stacking model
        ["Stacking"],
        X_train,
        y_train,
        train_time,
        X_test,
        y_test,
        test_time,
        target_name=target_column,
    )


if __name__ == "__main__":
    main()
