import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def safe_numeric(series: pd.Series):
    """Convert a series to numeric, coerce errors to NaN."""
    return pd.to_numeric(series, errors='coerce')

def zscore_outliers(df, col, threshold=3):
    """
    Detect outliers using Z-score method.
    Returns list of row indices considered outliers.
    """
    try:
        data = safe_numeric(df[col]).dropna()
        z_scores = (data - data.mean()) / data.std()
        return z_scores[abs(z_scores) > threshold].index.tolist()
    except Exception as e:
        print(f"Outlier detection error for {col} (Z-score): {e}")
        return []

def iqr_outliers(df, col, factor=1.5):
    """
    Detect outliers using IQR method.
    Returns list of row indices considered outliers.
    """
    try:
        data = safe_numeric(df[col]).dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - factor * IQR) | (data > Q3 + factor * IQR)]
        return outliers.index.tolist()
    except Exception as e:
        print(f"Outlier detection error for {col} (IQR): {e}")
        return []

def isolation_forest_outliers(df, cols=None, contamination=0.01):
    """
    Detect outliers using Isolation Forest for multiple numeric columns.
    Returns dict: {col_name: list_of_outlier_indices}
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    outliers = {}
    for col in cols:
        try:
            data = safe_numeric(df[col]).dropna().values.reshape(-1, 1)
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(data)
            outlier_idx = np.where(preds == -1)[0].tolist()
            if outlier_idx:
                outliers[col] = outlier_idx
        except Exception as e:
            print(f"Outlier detection error for {col} (Isolation Forest): {e}")
    return outliers
