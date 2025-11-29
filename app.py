# app.py — LRS 回測系統（台股+美股統一使用 yfinance，含拆股調整 + 美化報表）

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

# === 字型設定 ===
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="LRS 回測系統", page_icon="📈", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — SMA/EMA 回測系統</h1>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# 公用工具
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# yfinance 歷史資料（台股+美股統一）
# ---------------------------------------------------------------------
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

    # 為了和舊版邏輯一致，建一個 'Adj Close' 欄位 = Close
    if "Close" in df_raw.columns and "Adj Close" not in df_raw.columns:
        df_raw["Adj Close"] = df_raw["Close"]

    return df_raw


# ---------------------------------------------------------------------
# 額外的「拆股/斷崖」偵測與平滑（在 yfinance auto_adjust 之上再保險一次）
# ---------------------------------------------------------------------
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

    # 若完全沒有異常，就直接把 Price_adj=Price_raw
    if "Price_adj" not in df.columns:
        df["Price_adj"] = df["Price_raw"]

    return df


# ---------------------------------------------------------------------
# 統一的價格載入函式（全部用 yfinance）
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_data(raw_symbol: str, yf_symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    回傳欄位至少包含：Price_raw / Price_adj
    """
    df_src = fetch_yf_history(yf_symbol, start, end)
    if df_src.empty:
        return df_src

    # 優先用 Adj Close，如果沒有就用 Close
    price_col = "Adj Close" if "Adj Close" in df_src.columns else "Close"
    df_adj = adjust_for_splits(df_src, price_col=price_col, threshold=0.3)

    return df_adj


# ---------------------------------------------------------------------
# 取得可用日期區間（全部以 yfinance 真實最早日期為準）
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 介面：使用者輸入
# ---------------------------------------------------------------------
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
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 10, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 1000, 1_000_000, 10000, step=1000)


# ---------------------------------------------------------------------
# 主程式：回測 + 視覺化
# ---------------------------------------------------------------------
if st.button("開始回測 🚀"):
    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)

    with st.spinner("資料下載與整理中…（自動多抓一年暖機資料 + 拆股調整）"):
        df_all = load_price_data(raw_symbol, yf_symbol, start_early.date(), end)

    if df_all.empty:
        st.error(f"⚠️ 無法取得 {yf_symbol} 的歷史資料，請確認代號或時間區間。")
        st.stop()

    # 用拆股調整後價格當作「策略判斷與績效」的基礎價格
    df = df_all.copy()
    df = df[(df.index >= pd.to_datetime(start_early)) & (df.index <= pd.to_datetime(end))]
    df = df.sort_index()

    df["Price"] = df["Price_adj"]

    # === 均線 ===
    if ma_type == "SMA":
        df["MA"] = df["Price"].rolling(window=window).mean()
    else:
        df["MA"] = df["Price"].ewm(span=window, adjust=False).mean()

    # 若暖機區間不足導致前面都是 NaN，就直接丟掉
    df = df.dropna(subset=["MA"])

    # === 生成訊號（第一天強制買入） ===
    df["Signal"] = 0
    if len(df) == 0:
        st.error("資料不足，請調整日期區間或均線天數。")
        st.stop()

    df.iloc[0, df.columns.get_loc("Signal")] = 1
    for i in range(1, len(df)):
        if df["Price"].iloc[i] > df["MA"].iloc[i] and df["Price"].iloc[i - 1] <= df["MA"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("Signal")] = 1
        elif df["Price"].iloc[i] < df["MA"].iloc[i] and df["Price"].iloc[i - 1] >= df["MA"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("Signal")] = -1
        else:
            df.iloc[i, df.columns.get_loc("Signal")] = 0

    # === 持倉 ===
    position, current = [], 1
    for sig in df["Signal"]:
        if sig == 1:
            current = 1
        elif sig == -1:
            current = 0
        position.append(current)
    df["Position"] = position

    # === 報酬（用拆股調整後價格） ===
    df["Return"] = df["Price"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    # === 真實資金曲線 ===
    df["Equity_LRS"] = 1.0
    for i in range(1, len(df)):
        if df["Position"].iloc[i - 1] == 1:
            df.iloc[i, df.columns.get_loc("Equity_LRS")] = df["Equity_LRS"].iloc[i - 1] * (1 + df["Return"].iloc[i])
        else:
            df.iloc[i, df.columns.get_loc("Equity_LRS")] = df["Equity_LRS"].iloc[i - 1]

    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # 只保留使用者選定區間，並從第一天重新歸一化
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["Equity_LRS"] /= df["Equity_LRS"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    df["LRS_Capital"] = df["Equity_LRS"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital

    # === 買賣點 ===
    buy_points = [(df.index[i], df["Price"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == 1]
    sell_points = [(df.index[i], df["Price"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == -1]
    buy_count, sell_count = len(buy_points), len(sell_points)

    # === 指標 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = (df.index[-1] - df.index[0]).days / 365
    cagr_lrs = (1 + final_return_lrs) ** (1 / years_len) - 1 if years_len > 0 else np.nan
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1 if years_len > 0 else np.nan
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    def calc_metrics(series):
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

    vol_lrs, sharpe_lrs, sortino_lrs = calc_metrics(df["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = calc_metrics(df["Return"])

    equity_lrs_final = df["LRS_Capital"].iloc[-1]
    equity_bh_final = df["BH_Capital"].iloc[-1]

    def format_currency(value: float) -> str:
        """Format currency values safely even when NaN."""
        return "-" if not np.isfinite(value) else f"{value:,.0f} 元"

    def format_plain_currency(value: float) -> str:
        return "-" if not np.isfinite(value) else f"{value:,.0f}"

    # === 圖表 ===
    st.markdown("<h2 style='margin-top:1em;'>📈 策略績效視覺化</h2>", unsafe_allow_html=True)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("收盤價與均線（含買賣點）", "資金曲線：LRS vs Buy&Hold"),
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["Price"], name="收盤價", line=dict(color="blue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")),
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
                marker=dict(color="green", symbol="triangle-up", size=8),
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
                marker=dict(color="red", symbol="x", size=8),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略", line=dict(color="green")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Equity_BuyHold"], name="Buy & Hold", line=dict(color="gray", dash="dot")),
        row=2,
        col=1,
    )
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ================================
# 📌 1）KPI Summary Cards
# ================================
st.markdown("## 📌 回測總覽 Summary")

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.metric(
        label="最終資產（LRS）",
        value=format_currency(equity_lrs_final),
        delta=f"{final_return_lrs:.2%}"
    )

with kpi_col2:
    st.metric(
        label="年化報酬（CAGR）",
        value=f"{cagr_lrs:.2%}",
        delta=f"{(cagr_lrs - cagr_bh) * 100:.2f}%"  # 比 BH 多多少
    )

with kpi_col3:
    st.metric(
        label="最大回撤（LRS）",
        value=f"{mdd_lrs:.2%}",
        delta=f"{(mdd_bh - mdd_lrs) * 100:.2f}%",
        delta_color="inverse"
    )


# ================================
# 📌 2）Heatmap 指標比較表（LRS vs BH）
# ================================
st.markdown("## 📊 指標比較（LRS vs Buy & Hold）")

report_df = pd.DataFrame([
    ["最終資產", format_plain_currency(equity_lrs_final), format_plain_currency(equity_bh_final)],
    ["總報酬", f"{final_return_lrs:.2%}", f"{final_return_bh:.2%}"],
    ["年化報酬", f"{cagr_lrs:.2%}", f"{cagr_bh:.2%}"],
    ["最大回撤", f"{mdd_lrs:.2%}", f"{mdd_bh:.2%}"],
    ["年化波動率", f"{vol_lrs:.2%}", f"{vol_bh:.2%}"],
    ["夏普值", f"{sharpe_lrs:.2f}", f"{sharpe_bh:.2f}"],
    ["索提諾值", f"{sortino_lrs:.2f}", f"{sortino_bh:.2f}"],
], columns=["指標名稱", "LRS 策略", "Buy & Hold"])


# === 職業級 Heatmap（Dark/Light Mode 自適應） ===
styled = (
    report_df.style
        .set_properties(subset=["指標名稱"], **{
            "font-weight": "bold"
        })
        .set_properties(**{
            "text-align": "center",
            "border": "1px solid rgba(180,180,180,0.1)"
        })
        .background_gradient(
            cmap="Blues",
            subset=["LRS 策略", "Buy & Hold"]
        )
)

st.dataframe(styled, use_container_width=True)


# ================================
# 📌 3）交易統計（小卡片）
# ================================
st.markdown("## 📈 交易統計")

trade_col1, trade_col2 = st.columns(2)

with trade_col1:
    st.metric(label="📥 買進次數", value=buy_count)

with trade_col2:
    st.metric(label="📤 賣出次數", value=sell_count)
# ==========================================
# 📌 5）策略 vs 指數：風險雷達圖（Radar Chart）
# ==========================================
st.markdown("## 🛡️ 策略 vs 指數 — 風險雷達圖")

# 雷達圖需要的指標
radar_categories = ["年化報酬", "最大回撤", "波動率", "夏普值", "索提諾值"]

# 雷達值（注意：最大回撤要轉成「負值越大越差」，所以用 (1 - MDD) 來正規化）
radar_lrs = [
    float(cagr_lrs),
    float(1 - mdd_lrs),
    float(1 - vol_lrs),     # 波動越低越好
    float(sharpe_lrs),
    float(sortino_lrs),
]

radar_bh = [
    float(cagr_bh),
    float(1 - mdd_bh),
    float(1 - vol_bh),
    float(sharpe_bh),
    float(sortino_bh),
]

import plotly.graph_objects as go

radar_fig = go.Figure()

radar_fig.add_trace(go.Scatterpolar(
    r=radar_lrs,
    theta=radar_categories,
    fill='toself',
    name='LRS 策略',
    line=dict(color='green')
))

radar_fig.add_trace(go.Scatterpolar(
    r=radar_bh,
    theta=radar_categories,
    fill='toself',
    name='Buy & Hold',
    line=dict(color='gray')
))

radar_fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True)
    ),
    showlegend=True,
    height=500
)

st.plotly_chart(radar_fig, use_container_width=True)



# ==========================================
# 📌 6）Portfolio Summary（最高資產、最低資產、最佳月、最差月）
# ==========================================
st.markdown("## 📦 Portfolio Summary — 資產摘要")

# === 計算最高 / 最低資產 ===
highest_value = df["LRS_Capital"].max()
lowest_value = df["LRS_Capital"].min()

# === 月報酬計算 ===
df_monthly = df["Equity_LRS"].resample("M").last().pct_change()

best_month = df_monthly.max()
worst_month = df_monthly.min()

summ_col1, summ_col2, summ_col3, summ_col4 = st.columns(4)

with summ_col1:
    st.metric(
        label="💰 最高資產",
        value=f"{highest_value:,.0f} 元"
    )

with summ_col2:
    st.metric(
        label="📉 最低資產",
        value=f"{lowest_value:,.0f} 元"
    )

with summ_col3:
    st.metric(
        label="📈 最佳月份報酬",
        value=f"{best_month:.2%}"
    )

with summ_col4:
    st.metric(
        label="📉 最差月份報酬",
        value=f"{worst_month:.2%}",
        delta_color="inverse"
    )


# ================================
# 📌 4）回測完成訊息
# ================================
st.success("✅ 回測完成！所有資料已產生（含專業儀表板呈現）")



