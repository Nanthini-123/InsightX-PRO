# app.py - Unified InsightX PRO (Modular Dashboard + Ultimate Scaffold)
# Single app with sidebar switch. Keep modular imports; replace function names if your modules differ.

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import os
from datetime import datetime
from pathlib import Path



# --- Modular imports (your existing modules) ---
from modules.eda import load_file, get_dataset_info, missing_value_report, descriptive_stats, basic_cleaning
from modules.preprocessing import preprocess_dataset
from modules.outlier import zscore_outliers, iqr_outliers, isolation_forest_outliers
from modules.visualizer import plotly_hist, plotly_box, plotly_scatter, plotly_bar, plotly_pie, save_matplotlib_heatmap
from modules.business_insights import compute_kpis, stakeholder_recommendations
from modules.chatbot import DataChatbot
from modules.gemini_chat import ask_gemini
from modules.report import PDFReport
from modules.auth import init_users_db, create_user, check_user


# --- New/advanced modules (you said you already have these) ---
from modules.forecasting import prophet_forecast, lstm_forecast
from modules.interpretability import run_shap_explain
from modules.auth import init_users_db, create_user, check_user
from modules.layout_manager import save_layout, load_layouts_for_user

# --- Initialize customizable colors for visualizations ---
if 'viz_colors' not in st.session_state:
    st.session_state['viz_colors'] = {
        'histogram': '#636EFA',  # blue
        'bar': '#EF553B',        # red
        'scatter': '#00CC96',    # green
        'pie': '#AB63FA',        # purple
        'box': '#FFA15A',        # orange
        'line': '#19D3F3',       # cyan
    }

# --- Optional libs used by modules / fallback checks ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# --- App dirs ---
APP_DIR = Path.cwd() / "insightx_data"
APP_DIR.mkdir(exist_ok=True)
LAYOUTS_DIR = APP_DIR / "layouts"
LAYOUTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR = Path.cwd() / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Ensure auth DB is initialized (module function)
try:
    init_users_db()
except TypeError:
    # some auth implementations may expect a path arg — try both ways silently
    try:
        init_users_db(str(APP_DIR / "users.db"))
    except Exception:
        pass
except Exception:
    pass

# --- Streamlit config & CSS ---
st.set_page_config(page_title="InsightX PRO: Autonomous AI-Driven Data Intelligence and Visual Storytelling Through a No Code AI System", layout="wide", page_icon="🔍")
st.markdown("""
    <style>
    body {background-color:#f5f5f5;}
    .stMetricValue {font-size: 1.6rem; font-weight:700;}
    .kpi-card {background:#fff;padding:12px;border-radius:8px;box-shadow:0 4px 10px rgba(0,0,0,0.06);margin-bottom:8px}
    </style>
""", unsafe_allow_html=True)

# Sidebar: app mode switch
with st.sidebar:
    if (ASSETS_DIR / "logo.png").exists():
        st.image(str(ASSETS_DIR / "logo.png"), use_container_width=True)
    st.title("InsightX PRO")
    app_mode = st.radio("Mode", ["📊 Modular Dashboard", "🚀 Ultimate Scaffold", "📈 Analytics Pipeline"])
    st.markdown("---")
    st.markdown("Support: drop datasets into the upload widget on the chosen mode.")
    with st.sidebar.expander("🎨 Visualization Colors"):
      for viz_type in st.session_state['viz_colors']:
        st.session_state['viz_colors'][viz_type] = st.color_picker(
            f"Choose {viz_type} color",
            st.session_state['viz_colors'][viz_type],
            key=f"color_{viz_type}"
        )
# -----------------------
# Helper functions used below
# -----------------------
def safe_load_dataset(file_obj):
    """Load using your modules. Returns preprocessed dataframe or raises."""
    df = load_file(file_obj)
    df = preprocess_dataset(df)
    return df

def safe_download_dataframe(df, filename="data.xlsx"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    buf.seek(0)
    st.download_button("Download dataset", buf, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===========================================
# MODE 1: MODULAR DASHBOARD
# ===========================================
if app_mode == "📊 Modular Dashboard":
    st.title("InsightX PRO — Modular Dashboard")
    st.markdown("Upload or select your dataset to explore. This view uses your existing modular pipeline.")

    uploaded_file = st.file_uploader("Upload CSV / XLSX / JSON (max 200MB)", type=["csv", "xlsx", "xls", "json"], key="mod_upload")
    if uploaded_file:
        try:
            df = safe_load_dataset(uploaded_file)
            st.session_state['mod_df'] = df
            st.success("Dataset loaded and preprocessed.")
        except Exception as e:
            st.exception(f"Failed to load dataset: {e}")
            st.stop()
    elif 'mod_df' not in st.session_state:
        st.info("Upload a dataset to begin. (Or switch to Ultimate Scaffold for advanced demo.)")
        st.stop()

    df = st.session_state['mod_df']
    st.write(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Detect column types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64[ns]','datetime64']).columns.tolist()

    # Top KPIs
    st.subheader("Executive KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{df.shape[0]:,}")
    missing_pct = df.isna().mean().mean() * 100
    k2.metric("Missing %", f"{missing_pct:.2f}%")
    k3.metric("Numeric cols", len(num_cols))
    k4.metric("Categorical cols", len(cat_cols))

    # Quick dataset info & descriptive stats
    with st.expander("Dataset info & brief"):
        try:
            st.json(get_dataset_info(df))
        except Exception:
            st.write("get_dataset_info not available or raised error.")
    with st.expander("Descriptive statistics"):
        try:
            desc_num, desc_cat = descriptive_stats(df)
            st.write("Numeric summary")
            st.dataframe(desc_num)
            st.write("Categorical summary")
            st.dataframe(desc_cat)
        except Exception as e:
            st.write("descriptive_stats error:", e)

    # Filter panel
    with st.expander("Filter Panel"):
        selected_cat_col = st.selectbox("Category column", options=[None] + cat_cols, key="mod_cat")
        selected_num_col = st.selectbox("Numeric metric", options=[None] + num_cols, key="mod_num")
        selected_date_col = st.selectbox("Date column", options=[None] + date_cols, key="mod_date")
        filtered_df = df.copy()
        if selected_cat_col:
            vals = sorted(filtered_df[selected_cat_col].dropna().unique().tolist())
            sel_val = st.selectbox(f"Filter {selected_cat_col} by", options=["All"] + vals, key="mod_cat_val")
            if sel_val and sel_val != "All":
                filtered_df = filtered_df[filtered_df[selected_cat_col] == sel_val]
        if selected_date_col:
            try:
                min_d = pd.to_datetime(filtered_df[selected_date_col].min())
                max_d = pd.to_datetime(filtered_df[selected_date_col].max())
                start, end = st.date_input("Date range", value=[min_d, max_d], key="mod_date_range")
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df[selected_date_col]) >= pd.to_datetime(start)) &
                    (pd.to_datetime(filtered_df[selected_date_col]) <= pd.to_datetime(end))
                ]
            except Exception:
                st.warning("Date filtering failed for the selected column.")

    # Filtered KPIs
    st.markdown("### Filtered KPIs")
    f1, f2, f3 = st.columns(3)
    f1.metric("Total rows (filtered)", f"{filtered_df.shape[0]:,}")
    if selected_num_col:
        total_val = filtered_df[selected_num_col].sum()
        avg_val = filtered_df[selected_num_col].mean()
        f2.metric(f"Total {selected_num_col}", f"{total_val:,.2f}")
        f3.metric(f"Avg {selected_num_col}", f"{avg_val:,.2f}")
    else:
        f2.metric("Metric", "Not selected")
        f3.metric("Avg", "N/A")

        # Detect columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    from streamlit_lottie import st_lottie
    import requests

    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # 📈 Interactive Dashboard (Blinkit Style)
    # ---------------------------
    st.subheader("📈 Interactive Dashboard")
    with st.expander("📂 Filter Panel"):
        selected_cat_col = st.selectbox("Select Category Column", cat_cols if cat_cols else [None])
        selected_num_col = st.selectbox("Select Numeric Metric (e.g. Sales)", num_cols if num_cols else [None])
        selected_date_col = st.selectbox("Select Date Column", date_cols if date_cols else [None])

        filtered_df = df.copy()

        if selected_cat_col:
            unique_cats = df[selected_cat_col].dropna().unique().tolist()
            selected_cat_value = st.selectbox(f"Filter {selected_cat_col} by", options=["All"] + unique_cats)
            if selected_cat_value != "All":
                filtered_df = filtered_df[filtered_df[selected_cat_col] == selected_cat_value]

        if selected_date_col:
            try:
                min_date = filtered_df[selected_date_col].min()
                max_date = filtered_df[selected_date_col].max()
                start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])
                filtered_df = filtered_df[
                    (filtered_df[selected_date_col] >= pd.to_datetime(start_date)) &
                    (filtered_df[selected_date_col] <= pd.to_datetime(end_date))
                ]
            except Exception as e:
                st.warning(f"Date filtering error: {e}")

    # KPIs
    st.markdown("### 📊 Key Metrics (Filtered View)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{filtered_df.shape[0]:,}")
    if selected_num_col:
        col2.metric(f"Total {selected_num_col}", f"{filtered_df[selected_num_col].sum():,.2f}")
        col3.metric(f"Average {selected_num_col}", f"{filtered_df[selected_num_col].mean():,.2f}")
    else:
        col2.metric("Numeric Column", "Not selected")
        col3.metric("Average", "N/A")

    # Visualizations
    st.markdown("### 📊 Visualizations (Filtered Data)")
    if selected_cat_col and selected_num_col:
        agg_df = filtered_df.groupby(selected_cat_col)[selected_num_col].sum().reset_index().sort_values(by=selected_num_col, ascending=False)
        fig_bar = px.bar(agg_df, x=selected_cat_col, y=selected_num_col, title=f"{selected_num_col} by {selected_cat_col}")
        st.plotly_chart(fig_bar, use_container_width=True)

    if selected_date_col and selected_num_col:
        ts_df = filtered_df.groupby(selected_date_col)[selected_num_col].sum().reset_index()
        fig_line = px.line(ts_df, x=selected_date_col, y=selected_num_col, title=f"Time Series of {selected_num_col}", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    if selected_cat_col:
        pie_df = filtered_df[selected_cat_col].value_counts().reset_index()
        pie_df.columns = [selected_cat_col, "Count"]
        fig_pie = px.pie(pie_df, names=selected_cat_col, values="Count", title=f"Distribution of {selected_cat_col}")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------------------
    # (Rest of your features continue here unchanged)
    # ---------------------------
    # You can safely include Executive Dashboard, AI Assistant, Outlier Detection, etc. here.

    # ---------------------------
    # Executive Dashboard
    # ---------------------------
    st.subheader("🏢 Executive Dashboard - Premium View")

    rev_col = st.selectbox("Revenue Column", [None]+num_cols)
    cust_col = st.selectbox("Customer ID Column", [None]+df_clean.columns.tolist())
    date_col = st.selectbox("Date Column", [None]+date_cols)

    kpis = compute_kpis(df_clean, revenue_col=rev_col, customer_id_col=cust_col, date_col=date_col)

    # KPI Cards with Trend Sparkline
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(label="Total Rows", value=f"{kpis['total_rows']:,}")
    missing_trend = (kpis['missing_pct'] - df_clean.isna().mean().mean()*100)
    k2.metric(label="Missing %", value=f"{kpis['missing_pct']:.2f}%", delta=f"{missing_trend:.2f}%")
    if rev_col:
        rev_data = df_clean.groupby(date_col)[rev_col].sum() if date_col else df_clean[rev_col]
        fig_rev = px.line(rev_data, title="Revenue Trend", color_discrete_sequence=['#636EFA'])
        k3.metric(label="Total Revenue", value=f"₹{int(kpis['total_revenue']):,}")
        with st.expander("Revenue Trend Chart"):
            st.plotly_chart(fig_rev, use_container_width=True)
    else:
        k3.metric(label="Total Revenue", value="N/A")

    if cust_col and rev_col:
        clv_data = df_clean.groupby(cust_col)[rev_col].sum()
        fig_clv = px.histogram(clv_data, nbins=20, title="Customer Revenue Distribution", color_discrete_sequence=['#EF553B'])
        k4.metric(label="Avg Customer Value", value=f"₹{int(kpis['clv']):,}")
        with st.expander("Customer Revenue Distribution"):
            st.plotly_chart(fig_clv, use_container_width=True)
    else:
        k4.metric(label="Avg Customer Value", value="N/A")

    # Stakeholder Recommendations
    st.markdown("**📌 Stakeholder Recommendations**")
    cols_rec = st.columns(2)
    for idx, rec in enumerate(stakeholder_recommendations(df_clean, kpis, revenue_col=rev_col)):
        with cols_rec[idx % 2]:
            st.info(f"💡 {rec}")

    # Quick Insights Charts
    st.markdown("**🔍 Quick Insights Charts**")
    cols_chart = st.columns(3)
    if rev_col and date_col:
        with cols_chart[0]:
            st.markdown("**Revenue Over Time**")
            st.line_chart(df_clean.groupby(date_col)[rev_col].sum())
    if num_cols:
        with cols_chart[1]:
            st.markdown("**Numeric Columns Snapshot**")
            st.bar_chart(df_clean[num_cols].head(10))
    if cat_cols:
        with cols_chart[2]:
            st.markdown("**Top Categories**")
            top_cat = df_clean[cat_cols[0]].value_counts().head(10)
            st.bar_chart(top_cat)

    # ---------------------------
    # Descriptive Statistics
    # ---------------------------
    st.subheader("📐 Descriptive Statistics")
    desc_num, desc_cat = descriptive_stats(df_clean)
    st.write("**Numeric Summary**")
    st.dataframe(desc_num.style.background_gradient(cmap='Blues'))
    st.write("**Categorical Summary**")
    st.dataframe(desc_cat.style.background_gradient(cmap='Greens'))

    # ---------------------------
    # Visualizations
    # ---------------------------
    st.subheader("📊 Interactive Visualizations")
    color_hist = st.color_picker("Histogram Color", "#636EFA")
    color_box = st.color_picker("Boxplot Color", "#EF553B")
    cols_for_plots = st.multiselect("Select Numeric Columns", num_cols, default=num_cols[:2])
    for c in cols_for_plots:
        st.plotly_chart(plotly_hist(df_clean, c, color=color_hist)[1], use_container_width=True)
        st.plotly_chart(plotly_box(df_clean, c, color=color_box)[1], use_container_width=True)
    if cat_cols:
        col_c = st.selectbox("Categorical Column for Bar/Pie", cat_cols)
        st.plotly_chart(plotly_bar(df_clean, col_c)[1], use_container_width=True)
        st.plotly_chart(plotly_pie(df_clean, col_c)[1], use_container_width=True)

    # Heatmap
    heat_img = save_matplotlib_heatmap(df_clean)
    if heat_img:
        st.image(heat_img, caption="Correlation Heatmap", use_column_width=True)

    # ---------------------------
    # Outlier Detection
    # ---------------------------
    st.subheader("🚨 Outlier Detection")
    method = st.radio("Method", ["Z-score","IQR","Isolation Forest"])
    outlier_summary = {}
    for c in num_cols:
        if method=="Z-score":
            idx = zscore_outliers(df_clean, c)
        elif method=="IQR":
            idx = iqr_outliers(df_clean, c)
        else:
            idx = isolation_forest_outliers(df_clean, cols=num_cols, contamination=0.02)
        if idx:
            outlier_summary[c] = idx
    if outlier_summary:
        st.write("Outliers detected (sample):")
        for k,v in outlier_summary.items():
            st.write(f"{k}: {len(v)} rows, sample indices: {v[:5]}")

    # ---------------------------
    # AI Data Assistant
    # ---------------------------
    st.subheader("🤖 AI Data Assistant")
    use_ai = st.checkbox("Use Gemini/OpenRouter for advanced answers")
    user_q = st.text_input("Ask about the dataset")
    if st.button("Ask AI") and user_q.strip():
        if use_ai:
            numeric_info = ", ".join([f"{c} mean={df_clean[c].mean():.2f}" for c in num_cols])
            cat_info = ", ".join(cat_cols)
            summary_text = f"Columns: {', '.join(df_clean.columns)}\nNumeric: {numeric_info}\nCategorical: {cat_info}"
            prompt = f"{summary_text}\nQuestion: {user_q}\nReturn JSON or Python Plotly snippet"
            ai_response = ask_gemini(prompt)
            try:
                chart_json = json.loads(ai_response)
                if chart_json.get("type")=="bar":
                    fig = px.bar(x=chart_json["data"][0]["x"], y=chart_json["data"][0]["y"])
                    st.plotly_chart(fig)
                else:
                    st.write(ai_response)
            except:
                st.write(ai_response)
        else:
            chatbot = DataChatbot(df_clean)
            ans = chatbot.ask(user_q)
            st.write(ans)

    # ---------------------------
    # Auto Dashboard
    # ---------------------------
    st.markdown("---")
    st.header("🌈 Auto Dashboard")
    if num_cols:
        st.subheader("Numeric Columns")
        st.bar_chart(df_clean[num_cols])
    if cat_cols:
        st.subheader("Categorical Columns")
        for col in cat_cols:
            vc = df_clean[col].value_counts().reset_index()
            vc.columns = [col, 'count']
            fig = px.bar(vc, x=col, y='count', title=f"Distribution of {col}", color='count')
            st.plotly_chart(fig, use_container_width=True)

      # ---------------------------
# Export Reports
# ---------------------------
st.subheader("💾 Export Reports")

# Wrap everything inside the button block
if st.button("📊 Export Analytics Report"):
    buffer = io.BytesIO()

    # Recalculate all needed variables inside this block
    desc_num, desc_cat = descriptive_stats(df_clean)
    miss = missing_value_report(df_clean)

    kpis = compute_kpis(df_clean, revenue_col=rev_col, customer_id_col=cust_col, date_col=date_col)

    # Recompute outliers
    outlier_summary = {}
    method = "IQR"
    for c in num_cols:
        idx = iqr_outliers(df_clean, c)
        if idx:
            outlier_summary[c] = idx

    # AI response (optional)
    user_q = user_q if 'user_q' in locals() else ""
    ai_response = ai_response if 'ai_response' in locals() else ""

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # 1. Cleaned Data
        df_clean.to_excel(writer, sheet_name="Cleaned_Data", index=False)

        # 2. Descriptive Statistics
        desc_num.to_excel(writer, sheet_name="Numeric_Summary")
        desc_cat.to_excel(writer, sheet_name="Categorical_Summary")

        # 3. Missing Values
        miss.to_excel(writer, sheet_name="Missing_Values")

        # 4. KPIs
        kpi_df = pd.DataFrame(list(kpis.items()), columns=["Metric", "Value"])
        kpi_df.to_excel(writer, sheet_name="KPIs", index=False)

        # 5. Outlier Summary
        if outlier_summary:
            outlier_df = pd.DataFrame(
                [(col, len(rows), rows[:10]) for col, rows in outlier_summary.items()],
                columns=["Column", "Num_Outliers", "Sample_Indices"]
            )
            outlier_df.to_excel(writer, sheet_name="Outliers", index=False)

        # 6. Recommendations
        recs = stakeholder_recommendations(df_clean, kpis, revenue_col=rev_col)
        rec_df = pd.DataFrame({"Recommendations": recs})
        rec_df.to_excel(writer, sheet_name="Recommendations", index=False)

        # 7. (Optional) AI Assistant Logs
        if user_q.strip() and ai_response:
            ai_df = pd.DataFrame([[user_q, ai_response]], columns=["Question", "AI_Response"])
            ai_df.to_excel(writer, sheet_name="AI_Insights", index=False)

    buffer.seek(0)
    st.download_button(
        "📥 Download Analytics Report (Excel)", 
        data=buffer, 
        file_name="InsightX_Analytics_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ===========================================
# MODE 2: ULTIMATE SCAFFOLD
# ===========================================
elif app_mode == "🚀 Ultimate Scaffold":
    st.title("InsightX PRO — Ultimate Scaffold")
    st.markdown("This mode shows the demo scaffold for production forecasting, SHAP interpretability, layout saving, and simple auth.")

    # SIMPLE DEMO AUTH (uses your modules.auth)
    st.sidebar.markdown("### Demo Auth")
    if 'ult_username' not in st.session_state:
        st.session_state['ult_username'] = None

    if st.session_state['ult_username'] is None:
        ult_mode = st.sidebar.selectbox("Mode", ["Login", "Create account"], key="ult_auth_mode")
        ult_user = st.sidebar.text_input("Username", key="ult_user")
        ult_pass = st.sidebar.text_input("Password", type="password", key="ult_pass")
        if ult_mode == "Create account":
            if st.sidebar.button("Create account", key="ult_create"):
                try:
                    ok = create_user(ult_user, ult_pass)
                    st.sidebar.success("Account created. Please login.")
                except Exception as e:
                    st.sidebar.error(f"Create failed: {e}")
        else:
            if st.sidebar.button("Login", key="ult_login"):
                try:
                    ok = check_user(ult_user, ult_pass)
                    if ok:
                        st.session_state['ult_username'] = ult_user
                        st.sidebar.success(f"Logged in: {ult_user}")
                    else:
                        st.sidebar.error("Invalid credentials")
                except Exception as e:
                    st.sidebar.error(f"Login error: {e}")
    else:
        st.sidebar.success(f"Logged in as {st.session_state['ult_username']}")
        if st.sidebar.button("Logout", key="ult_logout"):
            st.session_state['ult_username'] = None
            st.experimental_rerun()

    # Upload or demo dataset
    ult_up = st.file_uploader("Upload dataset (CSV/XLSX/JSON) for Ultimate mode", type=["csv","xlsx","xls","json"], key="ult_up")
    if ult_up:
        try:
            df = load_file(ult_up)
            st.session_state['ult_df'] = df
        except Exception as e:
            st.error(f"Failed to read: {e}")
            st.stop()
    elif 'ult_df' not in st.session_state:
        st.info("Upload dataset for Ultimate demo or load the AirPassengers demo below.")
        if st.button("Load AirPassengers demo", key="ult_demo"):
            try:
                url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
                demo = pd.read_csv(url)
                demo.columns = ['ds','y']
                demo['ds'] = pd.to_datetime(demo['ds'])
                st.session_state['ult_df'] = demo
            except Exception:
                st.error("Failed to load demo.")
                st.stop()

    df = st.session_state.get('ult_df', None)
    if df is None:
        st.stop()

    st.write(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]:,} cols")

    # TOP KPIs
    st.subheader("Executive KPIs")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{df.shape[0]:,}")
    cols[1].metric("Columns", f"{df.shape[1]}")
    cols[2].metric("Missing %", f"{df.isna().mean().mean()*100:.2f}%")
    cols[3].metric("Numeric cols", len(df.select_dtypes(include=np.number).columns.tolist()))

    # Layout Builder (simple reorderable list)
    st.subheader("Dashboard Builder (simple reorder)")
    if 'ult_layout' not in st.session_state:
        st.session_state['ult_layout'] = [
            {'id':'kpis','title':'KPIs'},
            {'id':'timeseries','title':'Time Series'},
            {'id':'top_categories','title':'Top Categories'},
            {'id':'correlation','title':'Correlation Heatmap'}
        ]
    for idx, item in enumerate(st.session_state['ult_layout']):
        c0, c1, c2 = st.columns([4,1,1])
        c0.write(f"**{item['title']}**")
        if c1.button("Up", key=f"ult_up_{idx}") and idx>0:
            st.session_state['ult_layout'][idx-1], st.session_state['ult_layout'][idx] = st.session_state['ult_layout'][idx], st.session_state['ult_layout'][idx-1]
            st.experimental_rerun()
        if c2.button("Down", key=f"ult_down_{idx}") and idx < len(st.session_state['ult_layout'])-1:
            st.session_state['ult_layout'][idx+1], st.session_state['ult_layout'][idx] = st.session_state['ult_layout'][idx], st.session_state['ult_layout'][idx+1]
            st.experimental_rerun()

    # Save / load layouts (per-user)
    if st.session_state.get('ult_username'):
        layout_name = st.text_input("Layout name to save", value=f"default_{st.session_state['ult_username']}", key="ult_layout_name")
        if st.button("Save layout", key="ult_save_layout"):
            try:
                save_layout(st.session_state['ult_username'], layout_name, st.session_state['ult_layout'])
                st.success("Layout saved.")
            except Exception as e:
                # try alternate signature (module might expect path arg)
                try:
                    save_layout(str(LAYOUTS_DIR), st.session_state['ult_username'], layout_name, st.session_state['ult_layout'])
                    st.success("Saved with alternate call.")
                except Exception as e2:
                    st.error(f"Save failed: {e} / {e2}")
        # list & load
        try:
            user_layouts = load_layouts_for_user(st.session_state['ult_username'])
        except Exception:
            user_layouts = {}
        if user_layouts:
            sel = st.selectbox("Load layout", options=[None] + list(user_layouts.keys()), key="ult_load_layout")
            if sel:
                st.session_state['ult_layout'] = user_layouts[sel]['layout']
                st.success(f"Loaded {sel}")

    # Render layout
    st.subheader("Live Dashboard (Ultimate)")
    for item in st.session_state['ult_layout']:
        if item['id'] == 'kpis':
            st.markdown("### KPIs")
            st.metric("Rows", df.shape[0])
            st.metric("Cols", df.shape[1])
            st.metric("Missing %", f"{df.isna().mean().mean()*100:.2f}%")
        if item['id'] == 'timeseries':
            st.markdown("### Time Series & Forecasting")
            # try to find date and numeric columns automatically
            date_cols = df.select_dtypes(include=['datetime64[ns]','datetime64']).columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not date_cols:
                # try parsing typical date column names
                for cand in ['date','ds','order_date','Order_Date','Date']:
                    if cand in df.columns:
                        try:
                            df[cand] = pd.to_datetime(df[cand])
                            date_cols.append(cand)
                            break
                        except Exception:
                            continue
            if date_cols and numeric_cols:
                sel_date = st.selectbox("Date column", options=[None] + date_cols, key="ult_ts_date")
                sel_metric = st.selectbox("Metric", options=[None] + numeric_cols, key="ult_ts_metric")
                if sel_date and sel_metric:
                    ts = df[[sel_date, sel_metric]].dropna()
                    ts[sel_date] = pd.to_datetime(ts[sel_date])
                    agg = ts.groupby(sel_date).sum().reset_index()
                    st.plotly_chart(px.line(agg, x=sel_date, y=sel_metric, title=f"{sel_metric} over time"), use_container_width=True)

                    # Forecast selection
                    method = st.selectbox("Forecast method", ["Prophet", "LSTM"], key="ult_fc_method")
                    if method == "Prophet":
                        if 'prophet' not in globals() and not getattr(__import__('sys'), 'modules', None):
                            pass
                        if st.button("Run Prophet Forecast", key="ult_prophet"):
                            try:
                                # Expecting your modules.forecasting.prophet_forecast to return (plotly_fig, forecast_df)
                                fig, forecast_df = prophet_forecast(agg, sel_date, sel_metric, periods=30)
                                try:
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception:
                                    st.write(forecast_df.head())
                                if forecast_df is not None:
                                    csv = forecast_df.to_csv(index=False)
                                    st.download_button("Download forecast CSV", data=csv, file_name=f"forecast_{sel_metric}.csv")
                            except Exception as e:
                                st.error(f"Prophet forecasting failed: {e}")
                    else:
                        st.markdown("LSTM forecasting (seq2seq). Training may take time.")
                        lookback = st.number_input("Lookback", min_value=1, max_value=365, value=30, key="ult_lstm_lookback")
                        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10, key="ult_lstm_epochs")
                        if st.button("Run LSTM Forecast", key="ult_lstm_run"):
                            try:
                                fig, fut_df = lstm_forecast(agg, sel_date, sel_metric, lookback=lookback, epochs=epochs, future_steps=30)
                                try:
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception:
                                    st.write(fut_df.head())
                                if fut_df is not None:
                                    st.download_button("Download LSTM forecast", data=fut_df.to_csv(index=False), file_name=f"lstm_forecast_{sel_metric}.csv")
                            except Exception as e:
                                st.error(f"LSTM forecast failed: {e}")
            else:
                st.info("Need at least one date column and one numeric column for time series forecasting.")

        if item['id'] == 'top_categories':
            st.markdown("### Top Categories")
            cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
            if cat_cols:
                col = st.selectbox("Category column", cat_cols, key="ult_topcat")
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, 'count']
                st.plotly_chart(px.bar(vc.head(20), x=col, y='count'), use_container_width=True)
            else:
                st.info("No categorical columns available.")

        if item['id'] == 'correlation':
            st.markdown("### Correlation Heatmap")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation heatmap.")

    # Predictive + SHAP (auto)
    st.markdown("---")
    st.subheader("Auto Predictive + Interpretability (Demo)")
    if st.button("Run Auto Predictive + SHAP", key="ult_run_predict"):
        try:
            run_shap_explain(df)  # expecting your interpretability module handles detection/training & plotting
        except TypeError:
            # fallback signature: run_shap_explain(model, X, X_test) or run_shap_explain(df, numeric_cols, cat_cols)
            try:
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                run_shap_explain(df, num_cols)  # try another common signature
            except Exception as e:
                st.error(f"run_shap_explain failed: {e}")
        except Exception as e:
            st.error(f"Predictive+SHAP pipeline error: {e}")

    # Export area
    st.markdown("---")
    st.subheader("Export & Reports")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Excel Report (Ultimate)", key="ult_export_excel"):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="data", index=False)
            buf.seek(0)
            st.download_button("Download Excel", data=buf, file_name=f"InsightX_Ultimate_{datetime.now().strftime('%Y%m%d')}.xlsx")
    with col2:
        if st.button("Export Layout JSON", key="ult_export_layout"):
            layout_json = {'layout': st.session_state.get('ult_layout', [])}
            st.download_button("Download Layout JSON", data=json.dumps(layout_json), file_name=f"layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    st.markdown("---")
    st.info("Ultimate Scaffold is a demo. For production: run heavy jobs (LSTM/SHAP) on background workers and use SSO for auth.")



# End of app
# ===========================================
# MODE 3: ANALYTICS PIPELINE (Descriptive / Diagnostic / Predictive / Prescriptive)
# ===========================================
elif app_mode == "📈 Analytics Pipeline":
    st.title("InsightX PRO — Analytics Pipeline")
    
    # ---------------------------
    # Sidebar Navigation
    # ---------------------------
    analysis_type = st.radio("Choose Analysis Type",
                             ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])
    uploaded_file = st.file_uploader("Upload CSV/XLSX/JSON", type=['csv','xlsx','xls','json'])

    # ---------------------------
    # Load dataset
    # ---------------------------
    df = None
    if uploaded_file:
        name = uploaded_file.name
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif name.endswith('.json'):
            df = pd.read_json(uploaded_file)

    if df is not None:
        st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        df_clean = preprocess_dataset(df)
    else:
        st.info("Upload a dataset to start analysis.")
        st.stop()

    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object','category']).columns.tolist()
    date_cols = df_clean.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    fig = px.scatter(df, x="x", y="y", size="value", color="category",
                 animation_frame="time", animation_group="category")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 1️⃣ Descriptive Analysis
    # ---------------------------
    if analysis_type == "Descriptive":
        st.header("Descriptive Analysis")
        st.subheader("Numeric Summary")
        st.dataframe(df_clean[num_cols].describe().transpose())
        st.subheader("Categorical Summary")
        st.dataframe(df_clean[cat_cols].describe().transpose())

        st.subheader("Visualizations")

        # Histogram
        for col in num_cols[:3]:
            st.plotly_chart(
                plotly_hist(df_clean, col, color=st.session_state['viz_colors']['histogram']),
                use_container_width=True
            )

        # Bar chart
        for col in cat_cols[:3]:
            st.plotly_chart(
                plotly_bar(df_clean, col, color=st.session_state['viz_colors']['bar']),
                use_container_width=True
            )

        # Boxplot
        if len(num_cols) >= 2:
            st.plotly_chart(
                plotly_box(df_clean, num_cols[0], color=st.session_state['viz_colors']['box']),
                use_container_width=True
            )

        # Scatter
        if len(num_cols) >= 2:
            st.plotly_chart(
                plotly_scatter(df_clean, num_cols[0], num_cols[1], color=st.session_state['viz_colors']['scatter']),
                use_container_width=True
            )

        # Pie chart
        if cat_cols:
            st.plotly_chart(
                plotly_pie(df_clean, cat_cols[0], color=st.session_state['viz_colors']['pie']),
                use_container_width=True
            )

        # Line chart (if time column exists)
        if date_cols and num_cols:
            import plotly.express as px
            st.plotly_chart(
                px.line(
                    df_clean.sort_values(date_cols[0]),
                    x=date_cols[0], y=num_cols[0],
                    title=f"{num_cols[0]} over {date_cols[0]}",
                    color_discrete_sequence=[st.session_state['viz_colors']['line']]
                ),
                use_container_width=True
            )

    # ---------------------------
    # 2️⃣ Diagnostic Analysis
    # ---------------------------
    elif analysis_type == "Diagnostic":
        st.header("Diagnostic Analysis")
        st.subheader("Missing Values")
        st.dataframe(df_clean.isna().sum())

        st.subheader("Outlier Detection")
        method = st.radio("Method", ["Z-score","IQR","Isolation Forest"])
        outlier_results = {}
        for col in num_cols:
            if method == "Z-score":
                outlier_results[col] = zscore_outliers(df_clean, col)
            elif method == "IQR":
                outlier_results[col] = iqr_outliers(df_clean, col)
            else:
                outlier_results = isolation_forest_outliers(df_clean, cols=num_cols, contamination=0.02)
                break
        st.write(outlier_results)

        st.subheader("Correlation Heatmap")
        if len(num_cols) >= 2:
            import plotly.express as px
            corr = df_clean[num_cols].corr()
            st.plotly_chart(
                px.imshow(
                    corr, text_auto=True, aspect="auto",
                    color_continuous_scale=[st.session_state['viz_colors']['heatmap']] if 'heatmap' in st.session_state['viz_colors'] else "Viridis"
                ),
                use_container_width=True
            )

    # ---------------------------
    # 3️⃣ Predictive Analysis
    # ---------------------------
    elif analysis_type == "Predictive":
        st.header("Predictive Analysis")
        target = st.selectbox("Select target variable", options=df_clean.columns)
        task_type = "regression" if df_clean[target].dtype in [np.float64, np.int64] else "classification"
        
        st.subheader(f"{task_type.title()} Modeling")
        if st.button("Run Predictive Model"):
            if task_type == "regression":
                run_regression(df_clean, target)
            else:
                run_classification(df_clean, target)

        st.subheader("Forecasting (Time Series)")
        if date_cols and num_cols:
            sel_date = st.selectbox("Select Date Column", date_cols)
            sel_metric = st.selectbox("Select Metric to Forecast", num_cols)
            run_forecast(df_clean, sel_date, sel_metric)

    # ---------------------------
    # 4️⃣ Prescriptive Analysis
    # ---------------------------
    elif analysis_type == "Prescriptive":
        st.header("Prescriptive Analysis")
        recs = generate_recommendations(df_clean, num_cols, cat_cols)
        for r in recs:
            st.info(r)
