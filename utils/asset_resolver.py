import re
import difflib
from functools import lru_cache

import pandas as pd


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip().upper()
    value = value.replace("&", " AND ")
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def looks_like_yahoo_ticker(value: str) -> bool:
    """
    Strenger als vorher:
    - reine Buchstaben/Zahlen nur bis Länge 5
    - oder mit . / - wie BTC-USD, VWCE.DE, BRK-B
    """
    value = str(value).strip().upper()
    if not value:
        return False

    # Mit Punkt oder Bindestrich: typische Yahoo-Formen erlauben
    if "-" in value or "." in value:
        return re.match(r"^[A-Z0-9]{1,10}([\-\.][A-Z0-9]{1,10})+$", value) is not None

    # Reiner "klassischer" Ticker ohne Sonderzeichen: nur kurz erlauben
    return re.match(r"^[A-Z0-9]{1,5}$", value) is not None


@lru_cache(maxsize=1)
def load_tr_catalog() -> pd.DataFrame:
    path = "data/tr_universe_de.csv"

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception:
            return pd.DataFrame(columns=[
                "symbol", "name", "isin",
                "norm_symbol", "norm_name", "norm_isin"
            ])

    required_cols = ["symbol", "name", "isin"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame(columns=[
                "symbol", "name", "isin",
                "norm_symbol", "norm_name", "norm_isin"
            ])

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()

    df = df[df["symbol"] != ""]
    df = df[df["symbol"] != "NAN"]

    df["norm_symbol"] = df["symbol"].apply(normalize_text)
    df["norm_name"] = df["name"].apply(normalize_text)
    df["norm_isin"] = df["isin"].apply(normalize_text)

    df = df.drop_duplicates(subset=["symbol", "name", "isin"], keep="first")
    return df


@lru_cache(maxsize=1)
def load_yahoo_map() -> pd.DataFrame:
    path = "data/yahoo_symbol_map.csv"

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=[
            "name", "isin", "tr_symbol", "yahoo_symbol",
            "norm_name", "norm_isin", "norm_tr_symbol", "norm_yahoo_symbol"
        ])

    required_cols = ["name", "isin", "tr_symbol", "yahoo_symbol"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame(columns=[
                "name", "isin", "tr_symbol", "yahoo_symbol",
                "norm_name", "norm_isin", "norm_tr_symbol", "norm_yahoo_symbol"
            ])

    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["tr_symbol"] = df["tr_symbol"].astype(str).str.strip().str.upper()
    df["yahoo_symbol"] = df["yahoo_symbol"].astype(str).str.strip().str.upper()

    df = df[df["yahoo_symbol"] != ""]
    df = df[df["yahoo_symbol"] != "NAN"]

    df["norm_name"] = df["name"].apply(normalize_text)
    df["norm_isin"] = df["isin"].apply(normalize_text)
    df["norm_tr_symbol"] = df["tr_symbol"].apply(normalize_text)
    df["norm_yahoo_symbol"] = df["yahoo_symbol"].apply(normalize_text)

    df = df.drop_duplicates(subset=["name", "isin", "tr_symbol", "yahoo_symbol"], keep="first")
    return df


@lru_cache(maxsize=1)
def build_tr_lookup():
    catalog = load_tr_catalog()

    return {
        "catalog": catalog,
        "by_symbol": {row["norm_symbol"]: row for _, row in catalog.iterrows() if row["norm_symbol"]},
        "by_name": {row["norm_name"]: row for _, row in catalog.iterrows() if row["norm_name"]},
        "by_isin": {row["norm_isin"]: row for _, row in catalog.iterrows() if row["norm_isin"]},
    }


@lru_cache(maxsize=1)
def build_yahoo_lookup():
    df = load_yahoo_map()

    return {
        "df": df,
        "by_name": {row["norm_name"]: row["yahoo_symbol"] for _, row in df.iterrows() if row["norm_name"]},
        "by_isin": {row["norm_isin"]: row["yahoo_symbol"] for _, row in df.iterrows() if row["norm_isin"]},
        "by_tr_symbol": {row["norm_tr_symbol"]: row["yahoo_symbol"] for _, row in df.iterrows() if row["norm_tr_symbol"]},
        "by_yahoo_symbol": {row["norm_yahoo_symbol"]: row["yahoo_symbol"] for _, row in df.iterrows() if row["norm_yahoo_symbol"]},
    }


def fuzzy_match_name(query_norm: str, catalog: pd.DataFrame, cutoff: float = 0.84):
    if catalog.empty or not query_norm:
        return None

    names = catalog["norm_name"].dropna().unique().tolist()
    if not names:
        return None

    matches = difflib.get_close_matches(query_norm, names, n=1, cutoff=cutoff)
    if not matches:
        return None

    matched_name = matches[0]
    row = catalog[catalog["norm_name"] == matched_name].head(1)
    if row.empty:
        return None
    return row.iloc[0]


def fuzzy_match_yahoo_name(query_norm: str, df: pd.DataFrame, cutoff: float = 0.84):
    if df.empty or not query_norm:
        return None

    names = df["norm_name"].dropna().unique().tolist()
    if not names:
        return None

    matches = difflib.get_close_matches(query_norm, names, n=1, cutoff=cutoff)
    if not matches:
        return None

    matched_name = matches[0]
    row = df[df["norm_name"] == matched_name].head(1)
    if row.empty:
        return None
    return row.iloc[0]["yahoo_symbol"]


def map_tr_row_to_yahoo(row):
    if row is None:
        return None

    yahoo_lookup = build_yahoo_lookup()

    norm_isin = normalize_text(row.get("isin", ""))
    norm_symbol = normalize_text(row.get("symbol", ""))
    norm_name = normalize_text(row.get("name", ""))

    if norm_isin and norm_isin in yahoo_lookup["by_isin"]:
        return yahoo_lookup["by_isin"][norm_isin]

    if norm_symbol and norm_symbol in yahoo_lookup["by_tr_symbol"]:
        return yahoo_lookup["by_tr_symbol"][norm_symbol]

    if norm_name and norm_name in yahoo_lookup["by_name"]:
        return yahoo_lookup["by_name"][norm_name]

    return None


def resolve_asset(user_input: str):
    original = str(user_input).strip()
    query_norm = normalize_text(original)

    if not query_norm:
        return {
            "input": original,
            "resolved": None,
            "method": "empty",
            "success": False
        }

    yahoo_lookup = build_yahoo_lookup()

    # 1) DIREKT in der Yahoo-Mapping-Datei suchen
    if query_norm in yahoo_lookup["by_yahoo_symbol"]:
        return {
            "input": original,
            "resolved": yahoo_lookup["by_yahoo_symbol"][query_norm],
            "method": "exact_yahoo_symbol",
            "success": True
        }

    if query_norm in yahoo_lookup["by_tr_symbol"]:
        return {
            "input": original,
            "resolved": yahoo_lookup["by_tr_symbol"][query_norm],
            "method": "exact_tr_symbol_to_yahoo",
            "success": True
        }

    if query_norm in yahoo_lookup["by_name"]:
        return {
            "input": original,
            "resolved": yahoo_lookup["by_name"][query_norm],
            "method": "exact_name_to_yahoo",
            "success": True
        }

    if query_norm in yahoo_lookup["by_isin"]:
        return {
            "input": original,
            "resolved": yahoo_lookup["by_isin"][query_norm],
            "method": "exact_isin_to_yahoo",
            "success": True
        }

    # 2) Fuzzy direkt in der Yahoo-Mapping-Datei
    fuzzy_yahoo = fuzzy_match_yahoo_name(query_norm, yahoo_lookup["df"])
    if fuzzy_yahoo:
        return {
            "input": original,
            "resolved": fuzzy_yahoo,
            "method": "fuzzy_name_to_yahoo_direct",
            "success": True
        }

    # 3) Dann erst TR-Katalog
    tr_lookup = build_tr_lookup()

    if query_norm in tr_lookup["by_symbol"]:
        row = tr_lookup["by_symbol"][query_norm]
        yahoo_symbol = map_tr_row_to_yahoo(row)
        if yahoo_symbol:
            return {
                "input": original,
                "resolved": yahoo_symbol,
                "method": "tr_symbol_to_yahoo",
                "success": True
            }

    if query_norm in tr_lookup["by_name"]:
        row = tr_lookup["by_name"][query_norm]
        yahoo_symbol = map_tr_row_to_yahoo(row)
        if yahoo_symbol:
            return {
                "input": original,
                "resolved": yahoo_symbol,
                "method": "tr_name_to_yahoo",
                "success": True
            }

    if query_norm in tr_lookup["by_isin"]:
        row = tr_lookup["by_isin"][query_norm]
        yahoo_symbol = map_tr_row_to_yahoo(row)
        if yahoo_symbol:
            return {
                "input": original,
                "resolved": yahoo_symbol,
                "method": "tr_isin_to_yahoo",
                "success": True
            }

    fuzzy_row = fuzzy_match_name(query_norm, tr_lookup["catalog"])
    if fuzzy_row is not None:
        yahoo_symbol = map_tr_row_to_yahoo(fuzzy_row)
        if yahoo_symbol:
            return {
                "input": original,
                "resolved": yahoo_symbol,
                "method": "fuzzy_name_to_yahoo_via_tr",
                "success": True
            }

    # 4) Erst ganz am Ende enger Fallback
    if looks_like_yahoo_ticker(original):
        return {
            "input": original,
            "resolved": original.strip().upper(),
            "method": "raw_ticker_fallback",
            "success": True
        }

    return {
        "input": original,
        "resolved": None,
        "method": "not_found",
        "success": False
    }


def resolve_assets(asset_list):
    results = [resolve_asset(x) for x in asset_list]

    resolved = []
    unresolved = []

    for r in results:
        if r["success"] and r["resolved"]:
            resolved.append(r["resolved"])
        else:
            unresolved.append(r["input"])

    resolved = list(dict.fromkeys(resolved))

    return {
        "results": results,
        "resolved": resolved,
        "unresolved": unresolved
    }
