import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


st.title("Asset Screener")
st.write("Suche und bewerte einzelne Assets nach Rendite, Sicherheit und Risiko-Rendite-Verhältnis.")


@st.cache_data
def load_universe():
    df = pd.read_csv("data/tr_universe_de.csv")

    if "traderepublic_de" in df.columns:
        df["traderepublic_de"] = (
            df["traderepublic_de"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "ja"])
        )

        df = df[df["traderepublic_de"]].copy()

    return df


@st.cache_data(show_spinner=False)
def load_prices(symbol, period, interval):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna()


def get_horizon_config(horizon):
    configs = {
        "1 Stunde": {"period": "7d", "interval": "60m", "bars": 1},
        "Halber Tag": {"period": "7d", "interval": "60m", "bars": 4},
        "1 Tag": {"period": "3mo", "interval": "1d", "bars": 1},
        "1 Monat": {"period": "2y", "interval": "1d", "bars": 21},
        "3 Monate": {"period": "3y", "interval": "1d", "bars": 63},
        "6 Monate": {"period": "5y", "interval": "1d", "bars": 126},
        "1 Jahr": {"period": "10y", "interval": "1d", "bars": 252},
        "5 Jahre": {"period": "10y", "interval": "1wk", "bars": 260},
        "10 Jahre": {"period": "20y", "interval": "1wk", "bars": 520},
        "50 Jahre": {"period": "max", "interval": "1mo", "bars": 600},
    }
    return configs[horizon]


def analyze_symbol(symbol, name, asset_type, horizon):
    cfg = get_horizon_config(horizon)
    prices = load_prices(symbol, cfg["period"], cfg["interval"])

    if prices.empty:
        return None

    bars = cfg["bars"]

    if len(prices) < bars + 1:
        return None

    start_price = prices.iloc[-(bars + 1)]
    end_price = prices.iloc[-1]

    if start_price == 0:
        return None

    horizon_return_pct = (end_price / start_price - 1) * 100

    interval_returns = prices.pct_change().dropna()
    if interval_returns.empty:
        return None

    volatility_pct = interval_returns.std() * 100

    risk_return_score = horizon_return_pct / max(volatility_pct, 0.01)
    safety_score = 100 / (1 + volatility_pct)
    aggressive_score = horizon_return_pct + (0.35 * volatility_pct)

    return {
        "Name": name,
        "Symbol": symbol,
        "Asset-Klasse": asset_type,
        "Startkurs": float(start_price),
        "Letzter Kurs": float(end_price),
        "Rendite %": float(horizon_return_pct),
        "Volatilität %": float(volatility_pct),
        "Risiko-Rendite-Score": float(risk_return_score),
        "Sicherheits-Score": float(safety_score),
        "Aggressiv-Score": float(aggressive_score),
    }


def sort_results(df, goal):
    if goal == "Höchste Rendite":
        return df.sort_values(["Rendite %", "Risiko-Rendite-Score"], ascending=[False, False])

    if goal == "Höchste Sicherheit":
        return df.sort_values(["Sicherheits-Score", "Rendite %"], ascending=[False, False])

    if goal == "Bestes Risiko-Rendite-Verhältnis":
        return df.sort_values(["Risiko-Rendite-Score", "Rendite %"], ascending=[False, False])

    if goal == "Hohe Rendite bei hohem Risiko":
        return df.sort_values(["Aggressiv-Score", "Rendite %"], ascending=[False, False])

    return df


try:
    universe = load_universe()
except FileNotFoundError:
    st.error("Die Datei data/tr_universe_de.csv wurde nicht gefunden.")
    st.stop()

required_cols = {"symbol", "name", "isin", "asset_type", "source", "traderepublic_de"}
if not required_cols.issubset(universe.columns):
    st.error("tr_universe_de.csv braucht die Spalten: symbol, name, isin, asset_type, source, traderepublic_de")
    st.stop()


with st.sidebar:
    st.header("Filter")

    asset_type_options = ["Alle"] + sorted(universe["asset_type"].dropna().unique().tolist())
    selected_asset_type = st.selectbox("Asset-Klasse", asset_type_options)

    search_query = st.text_input(
        "Name oder Ticker oder ISIN suchen",
        placeholder="z. B. Apple, Nvidia, BYD, BTC, US0378331005"
    )

    selected_horizon = st.selectbox(
        "Zeitraum",
        [
            "1 Stunde",
            "Halber Tag",
            "1 Tag",
            "1 Monat",
            "3 Monate",
            "6 Monate",
            "1 Jahr",
            "5 Jahre",
            "10 Jahre",
            "50 Jahre"
        ]
    )

    selected_goal = st.selectbox(
        "Ziel",
        [
            "Höchste Rendite",
            "Höchste Sicherheit",
            "Bestes Risiko-Rendite-Verhältnis",
            "Hohe Rendite bei hohem Risiko"
        ]
    )

    ranking_button = st.button("Ranking berechnen")


filtered = universe.copy()

if selected_asset_type != "Alle":
    filtered = filtered[filtered["asset_type"] == selected_asset_type]

if search_query:
    q = search_query.strip().lower()
    filtered = filtered[
        filtered["name"].str.lower().str.contains(q, na=False)
        | filtered["symbol"].str.lower().str.contains(q, na=False)
        | filtered["isin"].fillna("").str.lower().str.contains(q, na=False)
    ]

st.subheader("Trefferliste")

if filtered.empty:
    st.warning("Keine passenden Assets gefunden.")
    st.stop()

display_filtered = filtered[["name", "symbol", "isin", "asset_type", "source"]].rename(
    columns={
        "name": "Name",
        "symbol": "Symbol",
        "isin": "ISIN",
        "asset_type": "Asset-Klasse",
        "source": "Quelle"
    }
)

st.dataframe(display_filtered, use_container_width=True, hide_index=True)
st.caption("Zur Geschwindigkeit werden aktuell maximal 15 Treffer gleichzeitig bewertet.")

if ranking_button:
    candidates = filtered.head(15)

    with st.spinner("Marktdaten werden geladen und bewertet..."):
        results = []

        for _, row in candidates.iterrows():
            analyzed = analyze_symbol(
                symbol=row["symbol"],
                name=row["name"],
                asset_type=row["asset_type"],
                horizon=selected_horizon
            )

            if analyzed is not None:
                results.append(analyzed)

    if not results:
        st.error("Für die aktuelle Auswahl konnten keine auswertbaren Marktdaten geladen werden.")
        st.stop()

    results_df = pd.DataFrame(results)
    results_df = sort_results(results_df, selected_goal).reset_index(drop=True)

    best = results_df.iloc[0]

    st.subheader("Bestes Asset nach deinem Filter")
    st.success(f"{best['Name']} ({best['Symbol']}) — {best['Asset-Klasse']}")

    st.write(f"**Zeitraum:** {selected_horizon}")
    st.write(f"**Ziel:** {selected_goal}")
    st.write(f"**Rendite:** {best['Rendite %']:.2f}%")
    st.write(f"**Volatilität:** {best['Volatilität %']:.2f}%")
    st.write(f"**Risiko-Rendite-Score:** {best['Risiko-Rendite-Score']:.2f}")
    st.write(f"**Sicherheits-Score:** {best['Sicherheits-Score']:.2f}")

    ranking_display = results_df.copy()

    for col in [
        "Startkurs",
        "Letzter Kurs",
        "Rendite %",
        "Volatilität %",
        "Risiko-Rendite-Score",
        "Sicherheits-Score",
        "Aggressiv-Score",
    ]:
        ranking_display[col] = ranking_display[col].map(lambda x: f"{x:.2f}")

    st.subheader("Ranking")
    st.dataframe(ranking_display, use_container_width=True, hide_index=True)
