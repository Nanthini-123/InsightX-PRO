from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import streamlit as st

def run_regression(df, target):
    X = df.drop(columns=[target]).select_dtypes(include='number').fillna(0)
    y = df[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write("MAE:", mean_absolute_error(y_test, preds))

def run_classification(df, target):
    X = df.drop(columns=[target]).select_dtypes(include='number').fillna(0)
    y = df[target].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, preds))

def run_forecast(df, date_col, metric_col):
    import plotly.express as px
    ts = df[[date_col, metric_col]].dropna()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts_agg = ts.groupby(date_col)[metric_col].sum().reset_index()
    fig = px.line(ts_agg, x=date_col, y=metric_col, title=f"{metric_col} over time")
    st.plotly_chart(fig)
