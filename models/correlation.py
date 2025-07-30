import pandas as pd


def model_prediction_correlation(models, model_names, X, corr_method="pearson"):
    """
    Computes the correlation matrix between model predictions on the same dataset.

    Parameters:
    - models: list of trained models (with .predict method)
    - model_names: list of model names (same order as models)
    - X: input features to predict on
    - corr_method: 'pearson', 'spearman', or 'kendall'

    Returns:
    - pandas DataFrame correlation matrix
    """
    preds = {}
    for model, name in zip(models, model_names):
        preds[name] = model.predict(X)
    df_preds = pd.DataFrame(preds)
    corr_matrix = df_preds.corr(method=corr_method)
    return corr_matrix
