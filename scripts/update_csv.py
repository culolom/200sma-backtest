import yfinance as yf
import pandas as pd
import os
from datetime import datetime

DATA_DIR = "data"
SYMBOL_FILE = "symbols.txt"


def normalize_symbol(s: str) -> str:
    """台股自動補 .TW，美股原樣"""
    s = s.upper().strip()
    if s.isdigit() or (s[:-1].isdigit() and s[-1].isalpha()):
        return s + ".TW"
    return s


def load_symbols():
    if not os.path.exists(SYMBOL_FILE):
        raise FileNotFoundError("找不到 symbols.txt")

    with open(SYMBOL_FILE, "r", encoding="utf-8") as f:
        syms = [line.strip() for line in f if line.strip()]
    return syms


def update_one(symbol: str):
    yf_symbol = normalize_symbol(symbol)
    print(f"更新 {symbol}（yfinance 代號：{yf_symbol}）...")

    df = yf.download(yf_symbol, period="max", auto_adjust=True)

    if df.empty:
        print(f"⚠ 無法下載 {symbol}，跳過")
        return

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # 統一欄位（你的 loader 會需要）
    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume"
    })

    out_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✔ 已更新：{out_path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    syms = load_symbols()
    for s in syms:
        update_one(s)

    print("\n🎉 所有商品更新完成！")


if __name__ == "__main__":
    main()
