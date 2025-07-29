import os
from typing import List

from preprocessing.load_timeseries import load_timeseries_file


def processing_data(
    file_path: str,
    time_column: str,
    target_column: str,
    dropped_columns: List[str] | None = None,
):
    """
    Processes a time series data file into a pandas DataFrame.

    Args:
        file_path (str): Path to the data file.
        time_column (str): Name of the time/datetime column.
        target_column (str): Name of the target/label column.
        dropped_columns (list): Columns to drop from the DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if not file_path or not time_column or not target_column:
        raise ValueError(
            "file_path, time_column, and target_column are required arguments."
        )

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = load_timeseries_file(file_path, time_column, target_column, dropped_columns)

    return df
