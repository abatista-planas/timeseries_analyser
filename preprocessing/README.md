# timeseries_analyser

Timeseries Analyser

## Preprocessing

The `load_timeseries_file` function loads a time series dataset from a CSV or Excel
file, drops optional columns and sorts the resulting DataFrame by the provided
``time_column``. This ensures that downstream analysis always works with data in
chronological order.
