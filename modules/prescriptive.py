def generate_recommendations(df, num_cols, cat_cols):
    recs = []
    if df.isna().sum().sum() > 0:
        recs.append("Consider filling or imputing missing values.")
    if len(num_cols) > 0:
        recs.append("Analyze top numeric KPIs for trends and anomalies.")
    if len(cat_cols) > 0:
        recs.append("Check category distributions and rare categories.")
    return recs
