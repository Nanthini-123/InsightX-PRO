# modules/visualizer.py
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Histogram
def plotly_hist(df, col, color='#636EFA'):
    fig = px.histogram(
        df, x=col, nbins=30,
        color_discrete_sequence=[color],
        title=f"Histogram of {col}"
    )
    fig.update_layout(bargap=0.05)
    return fig

# Bar chart
def plotly_bar(df, col, color='#EF553B'):
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, 'count']
    fig = px.bar(
        vc, x=col, y='count',
        color_discrete_sequence=[color],
        title=f"Bar Chart of {col}"
    )
    return fig

# Scatter plot
def plotly_scatter(df, x, y, color='#00CC96'):
    fig = px.scatter(
        df, x=x, y=y,
        color_discrete_sequence=[color],
        title=f"Scatter: {x} vs {y}"
    )
    return fig

# Box plot
def plotly_box(df, col, color='#FFA15A'):
    fig = px.box(
        df, y=col,
        color_discrete_sequence=[color],
        title=f"Boxplot of {col}"
    )
    return fig

# Pie chart
def plotly_pie(df, col, color='#AB63FA'):
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, 'count']
    fig = px.pie(
        vc, names=col, values="count",
        color_discrete_sequence=[color],
        title=f"Pie Chart of {col}"
    )
    return fig

# Line chart (time series)
def plotly_line(df, x, y, color='#19D3F3'):
    fig = px.line(
        df, x=x, y=y,
        color_discrete_sequence=[color],
        title=f"{y} over {x}"
    )
    return fig

# Heatmap (Plotly)
def plotly_heatmap(df, cols, color_scale="Viridis"):
    corr = df[cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=color_scale,
        title="Correlation Heatmap"
    )
    return fig

# Heatmap (Matplotlib + Seaborn, returns in-memory PNG for Streamlit)
def save_matplotlib_heatmap(df, cols=None, cmap="coolwarm"):
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf
