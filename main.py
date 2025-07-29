from preprocessing.processing import processing_data


def main():
    data_path = "/home/albplanas/Desktop/Programming/IoTC/Data/15Days/15_Day_RD1.csv"
    time_column = "Time"
    target_column = "Q2910"
    dropped_columns = ["Q2933", "Q2934"]

    df = processing_data(data_path, time_column, target_column, dropped_columns)
    print("df size:", df.shape)


if __name__ == "__main__":
    main()
