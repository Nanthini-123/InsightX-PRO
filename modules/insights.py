# modules/insights.py
import numpy as np
import pandas as pd

def generate_basic_insights(df, max_pairs=5):
    insights = []
    # size
    insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    # missing values
    miss_total = int(df.isnull().sum().sum())
    insights.append(f"Total missing values in dataset: {miss_total}.")
    # numeric overview
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        sample = []
        for c in num_cols[:5]:
            mean = df[c].mean()
            std = df[c].std()
            sample.append(f"{c} (mean={mean:.2f}, std={std:.2f})")
        insights.append("Numeric columns (sample): " + "; ".join(sample))
    # top correlations
    corr = df.select_dtypes(include='number').corr().abs()
    if not corr.empty:
        tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = tri.stack().sort_values(ascending=False).head(max_pairs)
        for (a,b),val in pairs.items():
            insights.append(f"High correlation: {a} & {b} = {val:.2f}")
    # skewness hints
    for c in num_cols[:5]:
        s = df[c].dropna()
        if s.shape[0] > 0:
            skew = s.skew()
            if skew > 1:
                insights.append(f"{c} is highly right-skewed (long tail on the right).")
            elif skew < -1:
                insights.append(f"{c} is highly left-skewed (long tail on the left).")
    return insights

# Simple question-answer (rule-based) for the chatbot
def answer_query(df, query):
    q = query.lower()
    if "top" in q and "by" in q:
        # example: "top 5 products by revenue"
        try:
            parts = q.split("by")
            left = parts[0].strip()
            right = parts[1].strip()
            n = 5
            if "top" in left:
                try:
                    n = int([w for w in left.split() if w.isdigit()][0])
                except:
                    n = 5
            col = right.split()[0]
            if col in df.columns:
                res = df.groupby(col).size().sort_values(ascending=False).head(n)
                return res.to_string()
        except Exception:
            pass
    if "mean" in q or "average" in q:
        for c in df.select_dtypes(include='number').columns:
            if c.lower() in q:
                return f"Average {c} = {df[c].mean():.2f}"
    return "Sorry, I can't answer that yet. Try questions like 'Top 5 by <column>' or 'Average <column>'."