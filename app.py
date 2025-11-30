# app.py — 200sma Strategy 回測系統（台股+美股統一使用 yfinance，含拆股調整 + 完整專業儀表板）

import os
import re
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# 字型設定
# ================================
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ================================
# Streamlit 頁面設定
# ================================
st.set_page_config(page_title="200sma Strategy 回測系統", page_icon="📈", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 200sma Strategy — SMA 回測系統</h1>", unsafe_allow_html=True)


# ================================
# 工具函式
# ================================
def is_taiwan_stock(raw_symbol: str) -> bool:
    """
    判斷是否當成台股處理：
    - 純數字或「數字+字母」(0050, 2330, 00878, 00631L...) 視為台股
    - 其它 (QQQ, SPY...) 視為海外商品
    """
    s = raw_symbol.strip().upper()
    return bool(re.match(r"^\d+[A-Z]*$", s))


def normalize_for_yfinance(raw_symbol: str) -> str:
    """
    給 yfinance 用的代號：
    - 台股：0050 -> 0050.TW
    - 其它：原樣回傳（QQQ, SPY...）
    """
    s = raw_symbol.strip().upper()
    if is_taiwan_stock(s):
        return s + ".TW"
    return s


@st.cache_data(show_spinner=False)
def fetch_yf_history(yf_symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    從 yfinance 下載歷史資料，保留常見欄位，並移除重複日期。
    優先使用 auto_adjust=True 的價格（含拆股與股利調整）。
    """
    df_raw = yf.download(yf_symbol, start=start, end=end, auto_adjust=True)

    # auto_adjust=True 時，回傳欄位通常是：Open, High, Low, Close, Volume
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty:
        return df_raw

    df_raw = df_raw.sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

    # 建一個 'Adj Close' 欄位 = Close（保險）
    if "Close" in df_raw.columns and "Adj Close" not in df_raw.columns:
        df_raw["Adj Close"] = df_raw["Close"]

    return df_raw


def adjust_for_splits(df: pd.DataFrame, price_col: str = "Adj Close", threshold: float = 0.3) -> pd.DataFrame:
    """
    即使 yfinance 已做 auto_adjust，仍保留這一層：
    - 若某天價格單日變動幅度 |r| >= threshold 且是「大跌」（ratio < 1）
      則視為拆股 / 價格重算，往前所有價格乘上 ratio，讓曲線連續。
    threshold 預設 0.3（單日跌 >30%）
    """
    if df.empty or price_col not in df.columns:
        return df

    df = df.copy()
    df["Price_raw"] = df[price_col].astype(float)
    df["Price_adj"] = df["Price_raw"].copy()

    pct = df["Price_raw"].pct_change()
    candidates = pct[abs(pct) >= threshold].dropna()

    for date, r in candidates.sort_index().items():
        ratio = 1.0 + r
        # 只處理「價格向下跳水」且 ratio > 0
        if ratio <= 0 or ratio >= 1:
            continue
        mask = df.index < date
        df.loc[mask, "Price_adj"] *= ratio

    return df


@st.cache_data(show_spinner=False)
def load_price_data(raw_symbol: str, yf_symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    回傳欄位至少包含：Price_raw / Price_adj
    """
    df_src = fetch_yf_history(yf_symbol, start, end)
    if df_src.empty:
        return df_src

    price_col = "Adj Close" if "Adj Close" in df_src.columns else "Close"
    df_adj = adjust_for_splits(df_src, price_col=price_col, threshold=0.3)
    return df_adj


@st.cache_data(show_spinner=False)
def get_available_range(yf_symbol: str):
    """
    從 yfinance 抓最完整歷史，回傳起訖日期。
    例：0050.TW 可從 2003-06 開始。
    """
    hist = yf.Ticker(yf_symbol).history(period="max", auto_adjust=True)
    if hist.empty:
        return pd.to_datetime("1990-01-01").date(), dt.date.today()
    hist = hist.sort_index()
    hist = hist[~hist.index.duplicated(keep="first")]
    return hist.index.min().date(), hist.index.max().date()


def calc_metrics(series: pd.Series):
    """
    計算：年化波動率、Sharpe、Sortino
    """
    daily = series.dropna()
    if len(daily) <= 1:
        return np.nan, np.nan, np.nan
    avg = daily.mean()
    std = daily.std()
    downside = daily[daily < 0].std()
    vol = std * np.sqrt(252)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
    sortino = (avg / downside) * np.sqrt(252) if downside > 0 else np.nan
    return vol, sharpe, sortino


def compute_rolling_stats(strategy_returns, benchmark_returns, equity_curve, window: int = 252):
    """
    計算滾動 Sharpe / MDD / CAGR / Beta
    """

    # Rolling Sharpe
    def roll_sharpe(x: pd.Series):
        std = x.std()
        return (x.mean() / std) * np.sqrt(252) if std > 0 else np.nan

    rolling_sharpe = strategy_returns.rolling(window).apply(roll_sharpe)

    # Rolling MDD
    def roll_mdd(x: pd.Series):
        series = pd.Series(x)
        return 1 - (series / series.cummax()).min()

    rolling_mdd = equity_curve.rolling(window).apply(roll_mdd, raw=False)

    # Rolling CAGR
    def roll_cagr(x: pd.Series):
        if len(x) <= 1 or x.iloc[0] <= 0:
            return np.nan
        years = len(x) / 252
        return (x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1

    rolling_cagr = equity_curve.rolling(window).apply(roll_cagr, raw=False)

    # Rolling Beta
    df_rb = pd.concat([strategy_returns, benchmark_returns], axis=1)
    df_rb.columns = ["S", "B"]

    rolling_cov = df_rb["S"].rolling(window).cov(df_rb["B"])
    rolling_var = df_rb["B"].rolling(window).var()
    rolling_beta = rolling_cov / rolling_var

    return rolling_sharpe, rolling_mdd, rolling_cagr, rolling_beta


def extract_drawdown_periods(equity: pd.Series):
    """
    回傳 drawdown episode 清單：包含起點、谷底、恢復日與對應天數。
    """
    if equity.empty:
        return []

    records = []
    peak_value = equity.iloc[0]
    peak_date = equity.index[0]
    trough_value = peak_value
    trough_date = peak_date
    in_drawdown = False

    for date, value in equity.iloc[1:].items():
        if value >= peak_value:
            if in_drawdown:
                recovery_date = date
                drawdown_pct = 1 - (trough_value / peak_value)
                records.append(
                    {
                        "開始": peak_date.date(),
                        "谷底": trough_date.date(),
                        "恢復": recovery_date.date(),
                        "最大回撤": drawdown_pct,
                        "跌幅天數": (trough_date - peak_date).days,
                        "修復天數": (recovery_date - trough_date).days,
                    }
                )
                in_drawdown = False
            peak_value = value
            peak_date = date
            trough_value = value
            trough_date = date
        else:
            in_drawdown = True
            if value < trough_value:
                trough_value = value
                trough_date = date

    if in_drawdown:
        drawdown_pct = 1 - (trough_value / peak_value)
        records.append(
            {
                "開始": peak_date.date(),
                "谷底": trough_date.date(),
                "恢復": None,
                "最大回撤": drawdown_pct,
                "跌幅天數": (trough_date - peak_date).days,
                "修復天數": None,
            }
        )

    return records


def run_monte_carlo_sim(returns: pd.Series, paths: int = 200, seed: int = 42):
    """使用日報酬做重抽樣，回傳各路徑的累積報酬陣列與分位數。"""
    rng = np.random.default_rng(seed)
    data = returns.fillna(0).values
    n = len(data)

    sims = np.empty((paths, n))
    for i in range(paths):
        sampled = rng.choice(data, size=n, replace=True)
        sims[i] = np.cumprod(1 + sampled)

    quantiles = {
        "p5": np.quantile(sims, 0.05, axis=0),
        "p50": np.quantile(sims, 0.50, axis=0),
        "p95": np.quantile(sims, 0.95, axis=0),
    }

    return sims, quantiles


def format_currency(value: float) -> str:
    """金額格式化（台幣，千分位）"""
    try:
        return f"{value:,.0f} 元"
    except Exception:
        return "—"


def format_percent(value: float, decimals: int = 2) -> str:
    """百分比格式化，並處理 NaN。"""
    try:
        if np.isnan(value):
            return "—"
        return f"{value:.{decimals}%}"
    except Exception:
        return "—"


def nz(x, default: float = 0.0):
    """把 NaN 轉成 0（或自訂值），避免圖表炸裂。"""
    return float(np.nan_to_num(x, nan=default))


# ================================
# 介面：使用者輸入
# ================================
col1, col2, col3 = st.columns(3)
with col1:
    raw_symbol = st.text_input("輸入代號（例：0050, 2330, 00878, QQQ）", "0050")

yf_symbol = normalize_for_yfinance(raw_symbol)

# 若使用者更換代號，自動偵測日期範圍
if "last_yf_symbol" not in st.session_state or st.session_state.last_yf_symbol != yf_symbol:
    st.session_state.last_yf_symbol = yf_symbol
    min_start, max_end = get_available_range(yf_symbol)
    st.session_state.min_start = min_start
    st.session_state.max_end = max_end
else:
    min_start = st.session_state.min_start
    max_end = st.session_state.max_end

st.info(f"🔎 {yf_symbol} 可用資料區間：{min_start} ~ {max_end}")

with col2:
    start = st.date_input(
        "開始日期",
        value=max(min_start, pd.to_datetime("2013-01-01").date()),
        min_value=min_start,
        max_value=max_end,
        format="YYYY/MM/DD",
    )
with col3:
    end = st.date_input(
        "結束日期",
        value=max_end,
        min_value=min_start,
        max_value=max_end,
        format="YYYY/MM/DD",
    )

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA"], index=0, disabled=True)
with col5:
    window = st.slider("均線天數", 10, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 1000, 1_000_000, 10000, step=1000)


# ================================
# 主程式：回測 + 視覺化
# ================================
if st.button("開始回測 🚀"):
    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)

    with st.spinner("資料下載與整理中…（自動多抓一年暖機資料 + 拆股調整）"):
        df_all = load_price_data(raw_symbol, yf_symbol, start_early.date(), end)

    if df_all.empty:
        st.error(f"⚠️ 無法取得 {yf_symbol} 的歷史資料，請確認代號或時間區間。")
        st.stop()

    # --- 準備資料 ---
    df = df_all.copy()
    df = df[(df.index >= pd.to_datetime(start_early)) & (df.index <= pd.to_datetime(end))]
    df = df.sort_index()
    df["Price"] = df["Price_adj"]

    # 均線
    df["MA"] = df["Price"].rolling(window=window).mean()

    df = df.dropna(subset=["MA"])
    if len(df) == 0:
        st.error("資料不足，請調整日期區間或均線天數。")
        st.stop()

    # 訊號：第一天強制多頭，之後用均線穿越
    df["Signal"] = 0
    df.iloc[0, df.columns.get_loc("Signal")] = 1
    for i in range(1, len(df)):
        if df["Price"].iloc[i] > df["MA"].iloc[i] and df["Price"].iloc[i - 1] <= df["MA"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("Signal")] = 1
        elif df["Price"].iloc[i] < df["MA"].iloc[i] and df["Price"].iloc[i - 1] >= df["MA"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("Signal")] = -1
        else:
            df.iloc[i, df.columns.get_loc("Signal")] = 0

    # 持倉
    position, current = [], 1
    for sig in df["Signal"]:
        if sig == 1:
            current = 1
        elif sig == -1:
            current = 0
        position.append(current)
    df["Position"] = position

    # 報酬
    df["Return"] = df["Price"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    # 資金曲線（以1為起點）
    df["Equity_200sma"] = 1.0
    for i in range(1, len(df)):
        if df["Position"].iloc[i - 1] == 1:
            df.iloc[i, df.columns.get_loc("Equity_200sma")] = df["Equity_200sma"].iloc[i - 1] * (1 + df["Return"].iloc[i])
        else:
            df.iloc[i, df.columns.get_loc("Equity_200sma")] = df["Equity_200sma"].iloc[i - 1]

    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # 重新裁切使用者區間，歸一化
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["Equity_200sma"] /= df["Equity_200sma"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    df["Capital_200sma"] = df["Equity_200sma"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital

    # 買賣點
    buy_points = [(df.index[i], df["Price"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == 1]
    sell_points = [(df.index[i], df["Price"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == -1]
    buy_count, sell_count = len(buy_points), len(sell_points)

    # 指標
    final_return_200sma = df["Equity_200sma"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = (df.index[-1] - df.index[0]).days / 365
    cagr_200sma = (1 + final_return_200sma) ** (1 / years_len) - 1 if years_len > 0 else np.nan
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1 if years_len > 0 else np.nan
    mdd_200sma = 1 - (df["Equity_200sma"] / df["Equity_200sma"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    vol_200sma, sharpe_200sma, sortino_200sma = calc_metrics(df["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = calc_metrics(df["Return"])

    equity_200sma_final = df["Capital_200sma"].iloc[-1]
    equity_bh_final = df["BH_Capital"].iloc[-1]

    # ================================
    # 視覺化總覽：圖表 + KPI 卡片
    # ================================
    st.markdown("<h2 style='margin-top:1em;'>📈 策略績效視覺化</h2>", unsafe_allow_html=True)

    tabs = st.tabs(["價格/資金曲線", "回撤比較", "風險報酬雷達", "日報酬分佈"])

    # 主要價格與資金曲線
    with tabs[0]:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("收盤價與均線（含買賣點）", "資金曲線：200sma vs Buy&Hold"),
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["Price"], name="收盤價", line=dict(color="#1f77b4", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="#f5a623", width=2, dash="dash")),
            row=1,
            col=1,
        )

        if buy_points:
            bx, by = zip(*buy_points)
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="markers",
                    name="買進",
                    marker=dict(color="#2ecc71", symbol="triangle-up", size=9, line=dict(color="#145a32", width=1)),
                ),
                row=1,
                col=1,
            )
        if sell_points:
            sx, sy = zip(*sell_points)
            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="markers",
                    name="賣出",
                    marker=dict(color="#e74c3c", symbol="x", size=9, line=dict(color="#922b21", width=1)),
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["Equity_200sma"], name="200sma 策略", line=dict(color="#2ecc71", width=3)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Equity_BuyHold"],
                name="Buy & Hold",
                line=dict(color="#7f8c8d", width=2, dash="dot"),
                fill="tozeroy",
                fillcolor="rgba(127,140,141,0.08)",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=820, showlegend=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # 回撤對比
    with tabs[1]:
        dd_200sma = (df["Equity_200sma"] / df["Equity_200sma"].cummax() - 1) * 100
        dd_bh = (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax() - 1) * 100

        fig_dd_compare = go.Figure()
        fig_dd_compare.add_trace(
            go.Scatter(
                x=df.index,
                y=dd_200sma,
                mode="lines",
                name="200sma 回撤",
                line=dict(color="#e67e22", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(230,126,34,0.08)",
            )
        )
        fig_dd_compare.add_trace(
            go.Scatter(
                x=df.index,
                y=dd_bh,
                mode="lines",
                name="Buy & Hold 回撤",
                line=dict(color="#4a90e2", width=2, dash="dot"),
                fill=None,
            )
        )

        fig_dd_compare.update_layout(
            height=460,
            template="plotly_white",
            yaxis_title="回撤 (%)",
            xaxis_title="日期",
            legend=dict(y=1.02, orientation="h"),
        )

        st.plotly_chart(fig_dd_compare, use_container_width=True)

    # 雷達圖：風險報酬關鍵指標
    with tabs[2]:
        radar_categories = ["CAGR", "Sharpe", "Sortino", "-MDD", "波動率(反轉)"]
        radar_200sma = [
            nz(cagr_200sma),
            nz(sharpe_200sma),
            nz(sortino_200sma),
            nz(-mdd_200sma),
            nz(-vol_200sma),
        ]
        radar_bh = [
            nz(cagr_bh),
            nz(sharpe_bh),
            nz(sortino_bh),
            nz(-mdd_bh),
            nz(-vol_bh),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(r=radar_200sma, theta=radar_categories, fill="toself", name="200sma", line=dict(color="#27ae60"))
        )
        fig_radar.add_trace(
            go.Scatterpolar(r=radar_bh, theta=radar_categories, fill="toself", name="Buy&Hold", line=dict(color="#7f8c8d"))
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, showline=False, gridcolor="rgba(0,0,0,0.1)")),
            template="plotly_white",
            height=520,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 報酬分佈
    with tabs[3]:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=df["Strategy_Return"] * 100,
                nbinsx=50,
                name="200sma 日報酬",
                marker_color="#2ecc71",
                opacity=0.7,
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=df["Return"] * 100,
                nbinsx=50,
                name="Buy&Hold 日報酬",
                marker_color="#95a5a6",
                opacity=0.6,
            )
        )
        fig_hist.update_layout(
            barmode="overlay",
            template="plotly_white",
            height=520,
            xaxis_title="日報酬 (%)",
            yaxis_title="次數",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ================================
    # KPI Summary Cards（200sma vs Buy&Hold）
    # ================================
    asset_gap_pct = ((equity_200sma_final / equity_bh_final) - 1) * 100 if equity_bh_final != 0 else 0.0
    cagr_delta_pct = (cagr_200sma - cagr_bh) * 100 if (not np.isnan(cagr_200sma) and not np.isnan(cagr_bh)) else 0.0
    vol_delta_pct = (vol_200sma - vol_bh) * 100 if (not np.isnan(vol_200sma) and not np.isnan(vol_bh)) else 0.0
    mdd_delta_pct = (mdd_200sma - mdd_bh) * 100 if (not np.isnan(mdd_200sma) and not np.isnan(mdd_bh)) else 0.0

    st.markdown("<h3 style='margin-top:1em;'>🎯 核心指標對比</h3>", unsafe_allow_html=True)

    row_strategy = st.columns(4)
    with row_strategy[0]:
        st.metric(label="最終資產（200sma）", value=format_currency(equity_200sma_final), delta=f"較 Buy&Hold {asset_gap_pct:+.2f}%")
    with row_strategy[1]:
        st.metric(label="年化報酬（CAGR, 200sma）", value=format_percent(cagr_200sma), delta=f"較 Buy&Hold {cagr_delta_pct:+.2f}%")
    with row_strategy[2]:
        st.metric(label="年化波動率（200sma）", value=format_percent(vol_200sma), delta=f"較 Buy&Hold {vol_delta_pct:+.2f}%", delta_color="inverse")
    with row_strategy[3]:
        st.metric(label="最大回撤（200sma）", value=format_percent(mdd_200sma), delta=f"較 Buy&Hold {mdd_delta_pct:+.2f}%", delta_color="inverse")

    row_bh = st.columns(4)
    with row_bh[0]:
        st.metric(label="最終資產（Buy&Hold）", value=format_currency(equity_bh_final), delta=f"較 200sma {-asset_gap_pct:+.2f}%", delta_color="inverse")
    with row_bh[1]:
        st.metric(label="年化報酬（CAGR, Buy&Hold）", value=format_percent(cagr_bh), delta=f"較 200sma {-cagr_delta_pct:+.2f}%", delta_color="inverse")
    with row_bh[2]:
        st.metric(label="年化波動率（Buy&Hold）", value=format_percent(vol_bh), delta=f"較 200sma {-vol_delta_pct:+.2f}%", delta_color="inverse")
    with row_bh[3]:
        st.metric(label="最大回撤（Buy&Hold）", value=format_percent(mdd_bh), delta=f"較 200sma {-mdd_delta_pct:+.2f}%", delta_color="inverse")

    # 進一步的對比表格 + 條形圖
    st.markdown("<h3 style='margin-top:1em;'>📊 指標總覽</h3>", unsafe_allow_html=True)
    summary_df = pd.DataFrame(
        {
            "策略": ["200sma", "Buy & Hold"],
            "CAGR": [cagr_200sma, cagr_bh],
            "年化波動": [vol_200sma, vol_bh],
            "Sharpe": [sharpe_200sma, sharpe_bh],
            "Sortino": [sortino_200sma, sortino_bh],
            "最大回撤": [mdd_200sma, mdd_bh],
            "交易次數": [buy_count + sell_count, 0],
            "期末資產": [equity_200sma_final, equity_bh_final],
        }
    )
    summary_df_display = summary_df.copy()
    summary_df_display["CAGR"] = summary_df_display["CAGR"].apply(format_percent)
    summary_df_display["年化波動"] = summary_df_display["年化波動"].apply(format_percent)
    summary_df_display["Sharpe"] = summary_df_display["Sharpe"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "—")
    summary_df_display["Sortino"] = summary_df_display["Sortino"].map(lambda x: f"{x:.2f}" if not np.isnan(x) else "—")
    summary_df_display["最大回撤"] = summary_df_display["最大回撤"].apply(format_percent)
    summary_df_display["期末資產"] = summary_df_display["期末資產"].apply(format_currency)

    st.dataframe(summary_df_display, use_container_width=True, hide_index=True)

    metric_fig = go.Figure()
    metric_fig.add_trace(go.Bar(x=["CAGR", "Sharpe", "Sortino"], y=[cagr_200sma * 100, sharpe_200sma, sortino_200sma], name="200sma", marker_color="#27ae60"))
    metric_fig.add_trace(go.Bar(x=["CAGR", "Sharpe", "Sortino"], y=[cagr_bh * 100, sharpe_bh, sortino_bh], name="Buy&Hold", marker_color="#7f8c8d"))
    metric_fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=420,
        yaxis_title="指標值（CAGR 為 %）",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(metric_fig, use_container_width=True)
    # ================================
    # 指標說明區塊（極簡風）
    # ================================
    st.markdown("""
    <style>
    .saas-card {
        margin-top: 28px;
        padding: 26px 30px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        font-size: 15.5px;
        line-height: 1.75;
        color: #e6e6e6;
    }
    
    .saas-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 18px;
        color: #ffffff;
    }
    
    /* 雙欄布局 */
    .saas-grid {
        display: grid;
        grid-template-columns: 180px 1fr;
        row-gap: 10px;
        column-gap: 24px;
        align-items: start;
    }
    
    .saas-key {
        font-weight: 600;
        color: #ffffff;
    }
    
    .saas-value {
        color: #dcdcdc;
    }
    </style>
    
    <div class="saas-card">
        <div class="saas-title">📘 指標說明（SaaS 專業版）</div>
    
        <div class="saas-grid">
            <div class="saas-key">CAGR</div>
            <div class="saas-value">越大越好（代表年化報酬越高）</div>
    
            <div class="saas-key">年化波動</div>
            <div class="saas-value">越小越好（數值越低越穩定）</div>
    
            <div class="saas-key">Sharpe Ratio</div>
            <div class="saas-value">越大越好（每承擔 1 單位風險可換多少報酬）</div>
    
            <div class="saas-key">Sortino Ratio</div>
            <div class="saas-value">越大越好（只計算下跌風險，更能反映策略穩定度）</div>
    
            <div class="saas-key">最大回撤（MDD）</div>
            <div class="saas-value">越小越好（越抗跌、越安全）</div>
    
            <div class="saas-key">交易次數</div>
            <div class="saas-value">中性指標（多＝敏感、少＝省心）</div>
    
            <div class="saas-key">期末資產</div>
            <div class="saas-value">越多越好（策略最終成果）</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
