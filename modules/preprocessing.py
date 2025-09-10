import pandas as pd

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple preprocessing: drop duplicates, reset index, handle NaNs.
    Expand later as needed.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill missing numeric values with median
    for col in df.select_dtypes(include='number'):
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical values with mode
    for col in df.select_dtypes(include='object'):
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df