"""
High-performance CSV updater for price data.
- Auto loads existing CSV
- Downloads ONLY missing dates (append)
- Unified schema: Open, High, Low, Close, Volume
- Saves to data/{symbol}.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ---- Utility: load existing CSV if exists ----
def load_existing_csv(symbol: str) -> pd.DataFrame | None:
    csv_path = DATA_DIR / f"{symbol}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            df = df.sort_index()
            return df
        except Exception:
            print(f"⚠ CSV for {symbol} corrupted. Rebuilding...")
            return None
    return None


# ---- Utility: download missing part only ----
def download_missing(symbol: str, start_date: datetime) -> pd.DataFrame:
    end_date = datetime.today() + timedelta(days=1)
    print(f"⬇ Downloading {symbol} from {start_date.date()} to {end_date.date()} ...")

    df = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"❌ No new data for {symbol}")
        return df

    # flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"

    return df


# ---- Main update logic ----
def update_symbol(symbol: str):
    DATA_DIR.mkdir(exist_ok=True)

    existing = load_existing_csv(symbol)

    # If no existing file → full download
    if existing is None or existing.empty:
        print(f"📦 No existing CSV for {symbol}. Downloading FULL history...")
        df = yf.download(symbol, auto_adjust=False, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[REQUIRED_COLUMNS].copy()
        df.index.name = "Date"

        df.to_csv(DATA_DIR / f"{symbol}.csv")
        print(f"✅ Saved full CSV: data/{symbol}.csv ({len(df)} rows)")
        return

    # If CSV exists → append mode
    last_date = existing.index.max()
    fetch_from = last_date + timedelta(days=1)

    print(f"📄 Existing CSV for {symbol}: last date = {last_date.date()}")

    if fetch_from.date() > datetime.today().date():
        print(f"⏭ {symbol} already up to date.")
        return

    new_data = download_missing(symbol, fetch_from)

    if new_data.empty:
        print(f"⏭ No new rows for {symbol}.")
        return

    merged = pd.concat([existing, new_data])
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()

    merged.to_csv(DATA_DIR / f"{symbol}.csv")
    print(f"✅ Updated {symbol}: +{len(new_data)} rows")


# ---- CLI args ----
def parse_args():
    parser = argparse.ArgumentParser(description="Update price CSV files (append mode)")
    parser.add_argument("symbols", nargs="+", help="Symbols accepted by yfinance (e.g., 0050.TW)")
    return parser.parse_args()


# ---- Entry point ----
def main():
    args = parse_args()
    for sym in args.symbols:
        print(f"\n=== Processing {sym} ===")
        try:
            update_symbol(sym)
        except Exception as e:
            print(f"⚠ ERROR updating {sym}: {e}")
            continue


if __name__ == "__main__":
    main()
