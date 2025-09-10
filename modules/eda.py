# modules/eda.py
import pandas as pd
import numpy as np

def load_file(uploaded):
    """Read csv/xlsx/json from Streamlit uploader file-like object."""
    name = uploaded.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded)
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded)
    if name.endswith('.json'):
        return pd.read_json(uploaded)
    raise ValueError("Unsupported file type")

def get_dataset_info(df):
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "num_cols": df.select_dtypes(include=np.number).columns.tolist(),
        "cat_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "date_cols": df.select_dtypes(include=['datetime', 'datetime64[ns]', 'datetime64']).columns.tolist()
    }
    return info

def missing_value_report(df):
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": miss,
        "missing_pct": miss_pct
    }).sort_values("missing_pct", ascending=False)
    return report

def descriptive_stats(df):
    desc_num = df.describe(include=[np.number]).T
    desc_cat = df.describe(include=['object', 'category']).T
    return desc_num, desc_cat

def basic_cleaning(df, strategy_num='median', strategy_cat='mode', drop_thresh=0.9):
    """Drop columns with > drop_thresh missing % (default keep columns with >=10% non-null),
       fill numeric with median/mean, categorical with mode."""
    # drop columns with very high missing (if threshold is fraction to keep, e.g., 0.1 min non-null -> drop_thresh is keep fraction)
    min_non_null = int(len(df) * (1 - drop_thresh))  # if drop_thresh=0.9 -> min_non_null = 0.1*len
    # The previous line yields tiny threshold; to be safe we'll implement as: drop col if missing_pct > drop_thresh
    df = df.loc[:, df.isnull().mean() <= drop_thresh].copy()
    # drop duplicates
    df = df.drop_duplicates()
    # numeric impute
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        if df[c].isnull().any():
            if strategy_num == 'mean':
                df[c].fillna(df[c].mean(), inplace=True)
            else:
                df[c].fillna(df[c].median(), inplace=True)
    # categorical impute
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "", inplace=True)
    return df