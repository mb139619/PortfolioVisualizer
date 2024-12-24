import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

st.set_page_config(
    page_title="Portfolio Visualizer",
    layout="wide"
)

st.title("Portfolio Visualizer")

# Default portfolio setup
default_portfolio = pd.DataFrame([{"Ticker": "AAPL", "Weight": 25}])

def load_portfolio(file, editor_data):
    """Load portfolio from file or editor input."""
    if file is not None:
        return pd.read_excel(file)
    return editor_data

def create_pie_chart(portfolio):
    """Generate a pie chart for portfolio allocation."""
    pie_chart = go.Figure(
        go.Pie(
            labels=portfolio.index,
            values=portfolio["Weight"],
            hole=0.4,
            hoverinfo="label+percent",
            textinfo="label+percent",
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']),
        )
    )
    pie_chart.update_layout(title="Portfolio Allocation", title_x=0.5, template="plotly_white")
    return pie_chart

def calculate_log_returns(close_prices):
    """Calculate log returns from closing prices."""
    return np.log(close_prices.shift(1) / close_prices)

def create_line_chart(dates, returns):
    """Generate a line chart for portfolio returns."""
    fig = go.Figure(
        go.Scatter(
            x=dates, y=returns, mode="lines", name="Log Returns", line=dict(color="blue")
        )
    )
    fig.update_layout(
        title="Portfolio Returns",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Log Returns"),
        template="plotly_white",
        showlegend=False
    )
    return fig

def create_dist_plot(returns, var_95, var_99):
    """Generate a distribution plot with VaR markers."""
    hist_data = [returns]
    group_labels = ["Log Returns"]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.005, show_rug=False)
    max_density = max(fig['data'][1]['y'])
    
    for value, color, name in [(var_95, "red", "VaR 95%"), (var_99, "darkred", "VaR 99%")]:
        fig.add_trace(
            go.Scatter(x=[value, value], y=[0, max_density], mode="lines",
                       line=dict(color=color, dash="dash" if "95" in name else "solid", width=2),
                       name=f"{name}: {value:.2%}")
        )
    fig.update_layout(
        title="Portfolio Returns Distribution",
        xaxis=dict(title="Log Returns (in %)", tickformat=".1%"),
        yaxis=dict(title="Density"),
        template="plotly_white",
        showlegend=True
    )
    return fig

def create_corr_matrix_plot(correlation_matrix):
    """Generate a correlation matrix heatmap."""
    fig = px.imshow(
        correlation_matrix, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, labels=dict(x="Assets", y="Assets", color="Correlation"),
        title="Correlation Matrix of Asset Returns"
    )
    fig.update_layout(xaxis=dict(tickangle=45), template="plotly_white")
    return fig

# Portfolio form
with st.form(key="User Portfolio"):
    st.text("Insert portfolio data manually...")
    portfolio_data = st.data_editor(default_portfolio, num_rows="dynamic", hide_index=True)
    file_upload = st.file_uploader("...or upload portfolio file (Excel format)")
    submit = st.form_submit_button("Submit")

if submit:
    portfolio = load_portfolio(file_upload, portfolio_data)
    portfolio.set_index("Ticker", inplace=True)
    portfolio.dropna(inplace=True)

    tickers = portfolio.index.tolist()
    weights = portfolio["Weight"] / 100
    data = yf.download(tickers=tickers, period="1y")
    log_returns = calculate_log_returns(data["Close"])

    # Portfolio returns
    portfolio_returns = log_returns.dot(weights).dropna()
    
    # VaR calculations
    var_95, var_99 = np.percentile(portfolio_returns, [5, 1])
    
    # Correlation matrix
    corr_matrix = log_returns.corr()
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_pie_chart(portfolio), use_container_width=True)
        st.plotly_chart(create_line_chart(portfolio_returns.index, portfolio_returns), use_container_width=True)

    with col2:
        st.plotly_chart(create_corr_matrix_plot(corr_matrix), use_container_width=True)
        st.plotly_chart(create_dist_plot(portfolio_returns, var_95, var_99), use_container_width=True)

    st.text("How about optimizing your portfolio?")