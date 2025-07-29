import os
from typing import List

import pandas as pd

from preprocessing.handle_nan import treat_nan_dataframe


def load_timeseries_file(
    file_path: str,
    time_column: str,
    target_column: str,
    dropped_columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Loads a file into a pandas DataFrame based on file extension,
    checks required columns, and drops specified columns.

    Args:
        file_path (str): Path to the data file.
        time_column (str): Name of the time/datetime column (must exist in the file).
        target_column (str): Name of the target/label column (must exist in the file).
        dropped_columns (list): Columns to drop from the DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame.

    Raises:
        ValueError: If required arguments are missing or columns not found.
        FileNotFoundError: If file_path does not exist.
        NotImplementedError: If the file extension is not supported.
    """
    if not file_path or not time_column or not target_column:
        raise ValueError(
            "file_path, time_column, and target_column are required arguments."
        )

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Identify extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Choose reader based on extension
    if ext in [".csv"]:
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext in [".parquet"]:
        df = pd.read_parquet(file_path)
    elif ext in [".feather"]:
        df = pd.read_feather(file_path)
    else:
        raise NotImplementedError(f"File extension {ext} not supported.")

    # Check required columns
    for col in [time_column, target_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the loaded data.")

    # Drop columns if specified
    if dropped_columns:
        df = df.drop(columns=[c for c in dropped_columns if c in df.columns])

    df = df.sort_values(time_column).reset_index(drop=True)

    df = treat_nan_dataframe(df, time_column, k_neighbors=10)

    return df
