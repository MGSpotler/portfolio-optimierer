import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252

forecast_horizons = {
    "1 Tag": 1 / TRADING_DAYS,
    "3 Tage": 3 / TRADING_DAYS,
    "1 Woche": 5 / TRADING_DAYS,
    "2 Wochen": 10 / TRADING_DAYS,
    "1 Monat": 21 / TRADING_DAYS,
    "3 Monate": 63 / TRADING_DAYS,
    "6 Monate": 126 / TRADING_DAYS,
    "1 Jahr": 1,
    "5 Jahre": 5,
    "10 Jahre": 10
}


def parse_tickers(text):
    return [t.strip().upper() for t in text.split(",") if t.strip()]


def load_data(tickers, period_years):
    data = yf.download(
        tickers,
        period=f"{period_years}y",
        auto_adjust=True,
        progress=False
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna()
    return data


def simulate_portfolios(mean_returns, cov_matrix, risk_free_rate, num_portfolios):
    n_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns.values)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

        if portfolio_volatility == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
        results[3, i] = i

        weights_record.append(weights)

    return results, weights_record


def build_forecast_table(horizons, equal_r, min_r, sharpe_r):
    rows = []

    for label, years in horizons.items():
        rows.append({
            "Zeitraum": label,
            "Gleichgewichtet (1/n)": f"{((1 + equal_r) ** years - 1):.2%}",
            "Minimal-Risiko": f"{((1 + min_r) ** years - 1):.2%}",
            "Maximale Sharpe": f"{((1 + sharpe_r) ** years - 1):.2%}",
        })

    return pd.DataFrame(rows)


def build_forecast_table_numeric(horizons, equal_r, min_r, sharpe_r):
    rows = []

    for label, years in horizons.items():
        rows.append({
            "Zeitraum": label,
            "Gleichgewichtet (1/n)": ((1 + equal_r) ** years - 1) * 100,
            "Minimal-Risiko": ((1 + min_r) ** years - 1) * 100,
            "Maximale Sharpe": ((1 + sharpe_r) ** years - 1) * 100,
        })

    return pd.DataFrame(rows)


st.set_page_config(page_title="Portfolio Optimierer", layout="wide")
st.title("Portfolio Optimierer")
st.write("Markowitz-Portfolioanalyse mit optimistischer und realistischer Prognose über mehrere Zeiträume.")

with st.sidebar:
    st.header("Einstellungen")

    tickers_input = st.text_input(
        "Ticker (kommagetrennt)",
        value="IWDA.AS, AVGO, ARKK, UNP, EIMI.L"
    )

    period_years = st.selectbox(
        "Historischer Zeitraum",
        options=[1, 3, 5, 10],
        index=2
    )

    num_portfolios = st.slider(
        "Anzahl zufälliger Portfolios",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    risk_free_rate = st.number_input(
        "Risikofreier Zins",
        min_value=0.0,
        max_value=0.2,
        value=0.02,
        step=0.005,
        format="%.3f"
    )

    selected_horizons = st.multiselect(
        "Anzuzeigende Prognose-Zeiträume",
        options=list(forecast_horizons.keys()),
        default=list(forecast_horizons.keys())
    )

    start_button = st.button("Analyse starten")


if start_button:
    tickers = parse_tickers(tickers_input)

    if not selected_horizons:
        st.error("Bitte mindestens einen Prognose-Zeitraum auswählen.")
        st.stop()

    selected_forecast_horizons = {
        label: forecast_horizons[label] for label in selected_horizons
    }

    if len(tickers) < 2:
        st.error("Bitte mindestens 2 Ticker eingeben.")
        st.stop()

    try:
        data = load_data(tickers, period_years)
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        st.stop()

    if data.empty:
        st.error("Keine Daten geladen. Prüfe die Ticker.")
        st.stop()

    returns = data.pct_change().dropna()

    if returns.empty:
        st.error("Es konnten keine Renditen berechnet werden.")
        st.stop()

    mean_returns = returns.mean() * TRADING_DAYS
    cov_annual = returns.cov() * TRADING_DAYS

    results, weights_record = simulate_portfolios(
        mean_returns,
        cov_annual,
        risk_free_rate,
        num_portfolios
    )

    min_vol_idx = np.argmin(results[0])
    max_sharpe_idx = np.argmax(results[2])

    min_vol_weights = weights_record[min_vol_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]

    min_vol_return = results[1, min_vol_idx]
    max_sharpe_return = results[1, max_sharpe_idx]

    min_vol_risk = results[0, min_vol_idx]
    max_sharpe_risk = results[0, max_sharpe_idx]

    min_vol_sharpe = results[2, min_vol_idx]
    max_sharpe_sharpe = results[2, max_sharpe_idx]

    equal_weights = np.ones(len(tickers)) / len(tickers)
    equal_return = float(np.dot(equal_weights, mean_returns.values))
    equal_risk = float(np.sqrt(np.dot(equal_weights.T, np.dot(cov_annual.values, equal_weights))))
    equal_sharpe = (equal_return - risk_free_rate) / equal_risk if equal_risk != 0 else 0

    st.subheader("Geladene Kursdaten")
    st.line_chart(data)

    metrics_df = pd.DataFrame({
        "Portfolio": ["Gleichgewichtet (1/n)", "Minimal-Risiko", "Maximale Sharpe"],
        "Erwartete Jahresrendite": [equal_return, min_vol_return, max_sharpe_return],
        "Volatilität": [equal_risk, min_vol_risk, max_sharpe_risk],
        "Sharpe Ratio": [equal_sharpe, min_vol_sharpe, max_sharpe_sharpe]
    })

    metrics_df["Erwartete Jahresrendite"] = metrics_df["Erwartete Jahresrendite"].map(lambda x: f"{x:.2%}")
    metrics_df["Volatilität"] = metrics_df["Volatilität"].map(lambda x: f"{x:.2%}")
    metrics_df["Sharpe Ratio"] = metrics_df["Sharpe Ratio"].map(lambda x: f"{x:.2f}")

    st.subheader("Portfolio-Kennzahlen")
    st.dataframe(metrics_df, use_container_width=True)

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Gleichgewichtet (1/n)": equal_weights,
        "Minimal-Risiko": min_vol_weights,
        "Maximale Sharpe": max_sharpe_weights
    })

    for col in ["Gleichgewichtet (1/n)", "Minimal-Risiko", "Maximale Sharpe"]:
        weights_df[col] = weights_df[col].map(lambda x: f"{x:.2%}")

    st.subheader("Portfolio-Gewichte")
    st.dataframe(weights_df, use_container_width=True)

    forecast_opt = build_forecast_table(
        selected_forecast_horizons,
        equal_return,
        min_vol_return,
        max_sharpe_return
    )

    log_returns = np.log1p(returns)
    cagr_assets = np.exp(log_returns.mean() * TRADING_DAYS) - 1

    real_equal = float(np.dot(equal_weights, cagr_assets.values))
    real_min = float(np.dot(min_vol_weights, cagr_assets.values))
    real_sharpe = float(np.dot(max_sharpe_weights, cagr_assets.values))

    forecast_real = build_forecast_table(
        selected_forecast_horizons,
        real_equal,
        real_min,
        real_sharpe
    )

    st.subheader("Optimistische Zukunftsprognose")
    st.dataframe(forecast_opt, use_container_width=True)

    st.subheader("Realistische Zukunftsprognose (CAGR)")
    st.dataframe(forecast_real, use_container_width=True)

    plot_opt = build_forecast_table_numeric(
        selected_forecast_horizons,
        equal_return,
        min_vol_return,
        max_sharpe_return
    )

    plot_real = build_forecast_table_numeric(
        selected_forecast_horizons,
        real_equal,
        real_min,
        real_sharpe
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(plot_opt["Zeitraum"], plot_opt["Gleichgewichtet (1/n)"], marker="o", label="Optimistisch - Gleichgewichtet")
    ax.plot(plot_opt["Zeitraum"], plot_opt["Minimal-Risiko"], marker="o", label="Optimistisch - Minimal-Risiko")
    ax.plot(plot_opt["Zeitraum"], plot_opt["Maximale Sharpe"], marker="o", label="Optimistisch - Maximale Sharpe")

    ax.plot(plot_real["Zeitraum"], plot_real["Gleichgewichtet (1/n)"], marker="x", linestyle="--", label="Realistisch - Gleichgewichtet")
    ax.plot(plot_real["Zeitraum"], plot_real["Minimal-Risiko"], marker="x", linestyle="--", label="Realistisch - Minimal-Risiko")
    ax.plot(plot_real["Zeitraum"], plot_real["Maximale Sharpe"], marker="x", linestyle="--", label="Realistisch - Maximale Sharpe")

    ax.set_title("Vergleich der Portfolio-Prognosen über verschiedene Zeiträume")
    ax.set_xlabel("Zeitraum")
    ax.set_ylabel("Prognostizierte Rendite in %")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)
    ax.legend()

    st.subheader("Prognose-Diagramm")
    st.pyplot(fig)