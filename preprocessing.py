import pandas as pd
from typing import List, Optional, Tuple

class DataTimeSeries:
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        target: str,
        features: Optional[List[str]] = None,
    ):
        """
        Basic Time Series Data Preprocessing Class

        Args:
            df (pd.DataFrame): Input dataframe.
            time_col (str): Name of the column containing timestamps.
            target (str): Name of the target column.
            features (List[str], optional): Feature columns. If None, uses all except time/target.
        """
        self.df = df.copy()
        self.time_col = time_col
        self.target = target
        self.features = features or [
            col for col in df.columns if col not in [time_col, target]
        ]
        self._prepare()

    def _prepare(self):
        # Ensure datetime format and sorting
        pass

    def basic_clean(
        self, 
    ):
        """
        Basic missing value handling: forward fill or drop rows.

        Args:
            fill_method (str): Method for filling NA ('ffill', 'bfill', None).
            dropna (bool): Drop rows with NA if True.
        """
        pass

    def get_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns features X and target y."""
        X = self.df[self.features]
        y = self.df[self.target]
        return X, y

    def train_test_split(
        self,     
    ): 
        """
        Chronological split (not randomized!).

        Args:
            test_size (int/float): Number or fraction of rows for test.
        Returns:
            train_df, test_df (pd.DataFrame)
        """
        pass

    def summary(self):
        print("Time range:", self.df.index.min(), "to", self.df.index.max())
        print("Total samples:", len(self.df))
        print("Features:", self.features)
        print("Target:", self.target)

# --- Example usage ---

# df = pd.read_csv("your_timeseries.csv")
# ts = DataTimeSeries(df, time_col="Sample Time", target="Lab Value")
# ts.basic_clean()
# ts.summary()
# train_df, test_df = ts.train_test_split(0.2)