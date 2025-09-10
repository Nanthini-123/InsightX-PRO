# modules/business_insights.py
import pandas as pd
import numpy as np

def compute_kpis(df, revenue_col=None, customer_id_col=None, date_col=None):
    """
    Returns KPI dict for header cards:
    - total_rows, total_revenue, avg_order_value, unique_customers, missing_pct
    """
    kpis = {}
    kpis['total_rows'] = len(df)
    kpis['missing_pct'] = round(df.isnull().mean().mean() * 100, 2)
    if revenue_col and revenue_col in df.columns:
        total_revenue = df[revenue_col].dropna().sum()
        kpis['total_revenue'] = float(total_revenue)
        if customer_id_col and customer_id_col in df.columns:
            # Customer Lifetime Value (simple avg revenue per customer)
            clv = df.groupby(customer_id_col)[revenue_col].sum().mean()
            kpis['clv'] = float(clv)
        else:
            kpis['clv'] = None
    else:
        kpis['total_revenue'] = None
        kpis['clv'] = None
    if 'date' in (date_col or '').lower() and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            monthly = df.set_index(date_col).resample('M').sum(numeric_only=True)
            kpis['last_month_change_pct'] = None
            if monthly.shape[0] >= 2:
                last = monthly.iloc[-1].select_dtypes(include=np.number).sum()
                prev = monthly.iloc[-2].select_dtypes(include=np.number).sum()
                diff = (last - prev).sum()
                kpis['last_month_change_pct'] = float((diff / prev.sum())*100) if prev.sum() != 0 else None
        except Exception:
            pass
    return kpis

def stakeholder_recommendations(df, kpis, revenue_col=None):
    recs = []
    # Example rules
    if kpis.get('missing_pct',0) > 20:
        recs.append("High missing data (>20%). Recommend improving data collection or imputing before decisions.")
    if kpis.get('total_revenue'):
        if kpis['total_revenue'] < 10000:
            recs.append("Revenue low. Consider short-term promotions or optimizing ad spend.")
        else:
            recs.append("Revenue looks healthy; consider scaling marketing in top regions.")
    if kpis.get('clv'):
        recs.append(f"Average customer value (CLV): {kpis['clv']:.2f}. Consider retention offers for high-value customers.")
    return recs