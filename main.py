from models.training_test_split import split_features_target
from preprocessing.processing import processing_data


def main():
    data_path = "/home/albplanas/Desktop/Programming/IoTC/Data/15Days/15_Day_RD1.csv"
    time_column = "Time"
    target_column = "Q2910"
    dropped_columns = ["Q2933", "Q2934"]

    df = processing_data(data_path, time_column, target_column, dropped_columns)

    X_train, y_train, X_test, y_test = split_features_target(
        df,
        time_column,
        target_column,
        test_split=0.2,
        strategy="temporal",
    )


if __name__ == "__main__":
    main()
