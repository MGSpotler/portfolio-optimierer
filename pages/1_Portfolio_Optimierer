import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from utils.asset_resolver import resolve_assets

st.set_page_config(page_title="Portfolio-Optimierer", page_icon="📊", layout="wide")


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def percent(x: float) -> str:
    return f"{x * 100:.2f} %"


def euro(x: float) -> str:
    return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def download_close_prices(tickers, period="5y"):
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.sort_index()
    close = close.dropna(axis=1, how="all")
    close = close.ffill()

    return close


def compute_portfolio_metrics(weights, mean_annual, cov_annual):
    annual_return = float(np.dot(weights, mean_annual))
    annual_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights))))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    return annual_return, annual_vol, sharpe


def simulate_portfolios(returns, n_portfolios=10000):
    n_assets = len(returns.columns)
    mean_annual = returns.mean().values * 252
    cov_annual = returns.cov().values * 252

    weights = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

    portfolio_returns = weights @ mean_annual
    portfolio_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_annual, weights))
    sharpes = np.divide(
        portfolio_returns,
        portfolio_vols,
        out=np.zeros_like(portfolio_returns),
        where=portfolio_vols > 0
    )

    sim_df = pd.DataFrame({
        "return": portfolio_returns,
        "vol": portfolio_vols,
        "sharpe": sharpes
    })

    for i, col in enumerate(returns.columns):
        sim_df[col] = weights[:, i]

    return sim_df


def project_capital(start_value, annual_return, years):
    return float(start_value * ((1 + annual_return) ** years))


def weight_table(asset_names, weights, investment):
    df = pd.DataFrame({
        "Asset": asset_names,
        "Gewichtung": [percent(w) for w in weights],
        "Betrag": [euro(investment * w) for w in weights]
    })
    df = df.sort_values("Betrag", ascending=False)
    return df


def generate_recommendations(strategy_name, weights_series):
    suggestions = []

    assets_upper = [a.upper() for a in weights_series.index]
    world_etfs = {"URTH", "VT", "VWCE.DE", "VOO", "VTI", "EUNL.DE"}
    tech_assets = {"NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "QQQ", "XLK"}
    crypto_assets = {"BTC-USD", "ETH-USD"}

    top_weight = float(weights_series.max())
    top_asset = str(weights_series.idxmax())
    tech_weight = float(weights_series[[a for a in weights_series.index if a.upper() in tech_assets]].sum()) if len(weights_series) else 0.0
    crypto_weight = float(weights_series[[a for a in weights_series.index if a.upper() in crypto_assets]].sum()) if len(weights_series) else 0.0
    has_world_etf = any(a in world_etfs for a in assets_upper)

    if strategy_name == "Gleichgewichtet":
        suggestions.append(
            "Die gleichgewichtete Variante ist oft ein guter, robuster Startpunkt, wenn du keine einzelne starke Wette eingehen willst."
        )

    if strategy_name == "Minimales Risiko":
        suggestions.append(
            "Die Minimum-Risk-Variante priorisiert Stabilität. Sie ist besonders interessant, wenn du Schwankungen reduzieren möchtest."
        )

    if strategy_name == "Maximale Rendite":
        suggestions.append(
            "Die renditeorientierte Variante ist aggressiver und geht in der Regel mit höherem Risiko und stärkeren Schwankungen einher."
        )

    if top_weight > 0.45:
        suggestions.append(
            f"Die Position **{top_asset}** ist sehr dominant. Das erhöht das Klumpenrisiko. Eine breitere Streuung wäre stabiler."
        )

    if tech_weight < 0.15:
        suggestions.append(
            "Der Tech-Anteil ist eher niedrig. Eine moderate Beimischung von Tech-Werten oder einem Nasdaq-ETF könnte das Wachstumsprofil erhöhen."
        )

    if not has_world_etf:
        suggestions.append(
            "Ein breit gestreuter Welt-ETF könnte die Diversifikation verbessern und das Portfolio stabiler machen."
        )

    if crypto_weight == 0:
        suggestions.append(
            "Falls du chancenorientierter investieren möchtest, könnte eine kleine Krypto-Beimischung interessant sein."
        )

    if not suggestions:
        suggestions.append(
            "Die Struktur wirkt bereits recht ausgewogen. Hier wären eher kleinere Feinjustierungen sinnvoll."
        )

    return suggestions


def plot_pie(weights_series, title):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(
        weights_series.values,
        labels=weights_series.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    ax.set_title(title)
    return fig


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📊 Portfolio-Optimierer")
st.write(
    "Gib mehrere Assets ein und vergleiche drei Portfolio-Varianten: "
    "**Gleichgewichtet**, **Minimales Risiko** und **Maximale Rendite**."
)
st.caption("Hinweis: Die Auswertung basiert auf historischen Kursdaten und ist keine Finanzberatung.")

assets_input = st.text_area(
    "Assets eingeben (ein Wert pro Zeile, z. B. Nvidia, Amazon, Bitcoin, MSCI World)",
    value="Nvidia\nAmazon\nBitcoin",
    height=150
)

investment = st.number_input(
    "Gesamtbetrag in €",
    min_value=100.0,
    value=10000.0,
    step=500.0
)

period = st.selectbox(
    "Historische Datenbasis",
    ["1y", "3y", "5y"],
    index=1
)

if st.button("Portfolio berechnen", use_container_width=True):
    raw_assets = [a.strip() for a in assets_input.splitlines() if a.strip()]

    if len(raw_assets) < 2:
        st.warning("Bitte gib mindestens 2 Assets ein.")
        st.stop()

    resolution = resolve_assets(raw_assets)
    resolved_assets = resolution["resolved"]
    unresolved_assets = resolution["unresolved"]

    if unresolved_assets:
        st.warning(
            "Diese Eingaben konnten nicht eindeutig erkannt werden: "
            + ", ".join(unresolved_assets)
        )


    if len(resolved_assets) < 2:
        st.error("Es müssen mindestens 2 gültige Assets erkannt werden.")
        st.stop()

    with st.spinner("Marktdaten werden geladen und Portfolios berechnet..."):
        prices = download_close_prices(resolved_assets, period=period)

    if prices.empty:
        st.error("Für die eingegebenen Assets konnten keine Kursdaten geladen werden.")
        st.stop()

    valid_assets = [a for a in resolved_assets if a in prices.columns]
    invalid_assets = [a for a in resolved_assets if a not in valid_assets]

    if invalid_assets:
        st.warning(
            "Diese Assets konnten bei yfinance nicht verarbeitet werden und wurden ignoriert: "
            + ", ".join(invalid_assets)
        )

    if len(valid_assets) < 2:
        st.error("Es müssen mindestens 2 gültige Assets übrig bleiben.")
        st.stop()

    prices = prices[valid_assets].copy()
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill()

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.loc[:, returns.notna().any()]

    if returns.empty or len(returns.columns) < 2:
        st.error("Es konnten keine Renditen berechnet werden.")
        st.stop()

    asset_names = list(returns.columns)
    n_assets = len(asset_names)

    mean_annual = returns.mean().values * 252
    cov_annual = returns.cov().values * 252

    # 1) Gleichgewichtet
    equal_weights = np.array([1 / n_assets] * n_assets)
    eq_return, eq_vol, eq_sharpe = compute_portfolio_metrics(equal_weights, mean_annual, cov_annual)

    # 2) Simulation für min Risk / max Return
    sim_df = simulate_portfolios(returns, n_portfolios=12000)

    min_risk_row = sim_df.sort_values("vol", ascending=True).iloc[0]
    max_return_row = sim_df.sort_values("return", ascending=False).iloc[0]

    min_risk_weights = np.array([min_risk_row[a] for a in asset_names])
    max_return_weights = np.array([max_return_row[a] for a in asset_names])

    mr_return, mr_vol, mr_sharpe = compute_portfolio_metrics(min_risk_weights, mean_annual, cov_annual)
    mx_return, mx_vol, mx_sharpe = compute_portfolio_metrics(max_return_weights, mean_annual, cov_annual)

    comparison_df = pd.DataFrame([
        {
            "Strategie": "Gleichgewichtet",
            "Erwartete Jahresrendite": percent(eq_return),
            "Jahresvolatilität": percent(eq_vol),
            "Sharpe Ratio": f"{eq_sharpe:.2f}",
        },
        {
            "Strategie": "Minimales Risiko",
            "Erwartete Jahresrendite": percent(mr_return),
            "Jahresvolatilität": percent(mr_vol),
            "Sharpe Ratio": f"{mr_sharpe:.2f}",
        },
        {
            "Strategie": "Maximale Rendite",
            "Erwartete Jahresrendite": percent(mx_return),
            "Jahresvolatilität": percent(mx_vol),
            "Sharpe Ratio": f"{mx_sharpe:.2f}",
        },
    ])

    st.divider()
    st.subheader("Vergleich der drei Portfolio-Varianten")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    tabs = st.tabs(["Gleichgewichtet", "Minimales Risiko", "Maximale Rendite"])

    strategy_payload = [
        ("Gleichgewichtet", equal_weights, eq_return, eq_vol, eq_sharpe),
        ("Minimales Risiko", min_risk_weights, mr_return, mr_vol, mr_sharpe),
        ("Maximale Rendite", max_return_weights, mx_return, mx_vol, mx_sharpe),
    ]

    for tab, payload in zip(tabs, strategy_payload):
        strategy_name, weights, ann_return, ann_vol, sharpe = payload
        weights_series = pd.Series(weights, index=asset_names).sort_values(ascending=False)

        with tab:
            c1, c2 = st.columns([1, 1])

            with c1:
                st.subheader("Gewichtung")
                st.dataframe(
                    weight_table(asset_names, weights, investment),
                    use_container_width=True,
                    hide_index=True
                )

            with c2:
                st.subheader("Verteilung")
                fig = plot_pie(weights_series, strategy_name)
                st.pyplot(fig)

            k1, k2, k3 = st.columns(3)
            k1.metric("Erwartete Jahresrendite", percent(ann_return))
            k2.metric("Jahresvolatilität", percent(ann_vol))
            k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

            st.subheader("Prognose des Portfoliowerts")
            horizons = [1, 3, 5, 10]
            projection_df = pd.DataFrame({
                "Zeitraum": [f"{h} Jahr(e)" for h in horizons],
                "Portfoliowert": [euro(project_capital(investment, ann_return, h)) for h in horizons]
            })
            st.dataframe(projection_df, use_container_width=True, hide_index=True)

            chart_df = pd.DataFrame({
                "Jahre": horizons,
                "Portfoliowert": [project_capital(investment, ann_return, h) for h in horizons]
            }).set_index("Jahre")
            st.line_chart(chart_df)

            st.subheader("Kurzinterpretation")
            st.write(
                f"Die Variante **{strategy_name}** erzielt auf Basis der historischen Daten "
                f"eine geschätzte Jahresrendite von **{percent(ann_return)}** bei einer "
                f"Volatilität von **{percent(ann_vol)}**."
            )

            st.subheader("Mögliche Ergänzungen / Empfehlungen")
            recommendations = generate_recommendations(strategy_name, weights_series)
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"**{i}.** {rec}")
