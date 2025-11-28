###############################################################
# app.py — 台股 LRS 回測（0050 / 006208 + 正2 槓桿ETF）
# 版本說明：
# 1. 使用 yfinance「收盤價 Close」計算報酬（不再做拆股調整）
# 2. LRS 策略從一開始「空手」，只在訊號出現才進場
# 3. 實際進出場使用槓桿 ETF 收盤價，邏輯與手算一致
# 4. 圖形 1：只畫原型 ETF + 200SMA，買賣點畫在 0050 上，
#    hover 顯示槓桿 ETF價格，買進綠色空心圓、賣出紅色空心圓
# 5. 所有線皆使用實線
###############################################################

import os
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib
import matplotlib.font_manager as fm
import plotly.graph_objects as go

###############################################################
# 字型設定
###############################################################

font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei", "PingFang TC", "Heiti TC"
    ]

matplotlib.rcParams["axes.unicode_minus"] = False

###############################################################
# Streamlit 頁面設定
###############################################################

st.set_page_config(page_title="台股 LRS 回測系統", page_icon="📈", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 台股 LRS 槓桿策略回測</h1>", unsafe_allow_html=True)

st.markdown(
    """
<b>本工具比較三種策略：</b><br>
1️⃣ 原型 ETF Buy & Hold（0050 / 006208）<br>
2️⃣ 槓桿 ETF Buy & Hold（00631L / 00663L / 00675L / 00685L）<br>
3️⃣ 槓桿 ETF LRS（訊號來自原型 ETF 的 200 日 SMA，實際進出槓桿 ETF）<br>
""",
    unsafe_allow_html=True,
)

###############################################################
# ETF 名稱清單
###############################################################

BASE_ETFS = {
    "0050 元大台灣50": "0050.TW",
    "006208 富邦台50": "006208.TW",
}

LEV_ETFS = {
    "00631L 元大台灣50正2": "00631L.TW",
    "00663L 國泰台灣加權正2": "00663L.TW",
    "00675L 富邦台灣加權正2": "00675L.TW",
    "00685L 群益台灣加權正2": "00685L.TW",
}

WINDOW = 200  # 固定使用 200 日 SMA

###############################################################
# yfinance 下載資料（使用收盤價 Close）
###############################################################

@st.cache_data(show_spinner=False)
def fetch_history(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    只抓 yfinance 的「收盤價 Close」，不做 auto_adjust、不調整拆股。
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated()]  # 去重複日期

    # 若沒有 Close，就退而求其次用 Adj Close
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    return df


@st.cache_data(show_spinner=False)
def load_price(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    回傳只有一欄 Price（即 yfinance 的收盤價 Close）
    """
    df = fetch_history(symbol, start, end)
    if df.empty:
        return df

    price_col = "Close" if "Close" in df.columns else "Adj Close"
    df = df[[price_col]].rename(columns={price_col: "Price"})
    return df


@st.cache_data(show_spinner=False)
def get_full_range(base_symbol: str, lev_symbol: str):
    """
    估算兩檔 ETF 可共同回測的期間（只用來抓日期，不影響價格計算）
    """
    b = yf.Ticker(base_symbol).history(period="max", auto_adjust=False)
    l = yf.Ticker(lev_symbol).history(period="max", auto_adjust=False)
    if b.empty or l.empty:
        return dt.date(2012, 1, 1), dt.date.today()
    b = b.sort_index()
    l = l.sort_index()
    start = max(b.index.min(), l.index.min()).date()
    end = min(b.index.max(), l.index.max()).date()
    return start, end


###############################################################
# UI
###############################################################

col1, col2 = st.columns(2)
with col1:
    base_label = st.selectbox("原型 ETF（訊號來源）", list(BASE_ETFS.keys()))
    base_symbol = BASE_ETFS[base_label]
with col2:
    lev_label = st.selectbox("槓桿 ETF（實際進出場標的）", list(LEV_ETFS.keys()))
    lev_symbol = LEV_ETFS[lev_label]

s_min, s_max = get_full_range(base_symbol, lev_symbol)
st.info(f"📌 可回測區間：{s_min} ~ {s_max}")

col3, col4, col5 = st.columns(3)
with col3:
    start = st.date_input(
        "開始日期",
        value=max(s_min, s_max - dt.timedelta(days=5 * 365)),
        min_value=s_min,
        max_value=s_max,
    )
with col4:
    end = st.date_input("結束日期", value=s_max, min_value=s_min, max_value=s_max)
with col5:
    capital = st.number_input("投入本金（元）", 1000, 5_000_000, 100_000, step=10_000)

###############################################################
# 主程式
###############################################################

if st.button("開始回測 🚀"):

    if start >= end:
        st.error("⚠️ 開始日期需早於結束日期")
        st.stop()

    # 為了計算 200SMA，多抓一年前資料
    start_early = start - dt.timedelta(days=365)

    with st.spinner("下載資料中…"):
        df_base_raw = load_price(base_symbol, start_early, end)
        df_lev_raw = load_price(lev_symbol, start_early, end)

    if df_base_raw.empty:
        st.error(f"⚠️ 無法取得 {base_symbol} 價格資料")
        st.stop()
    if df_lev_raw.empty:
        st.error(f"⚠️ 無法取得 {lev_symbol} 價格資料")
        st.stop()

    # 合併兩檔 ETF 價格（取交集交易日）
    df = pd.DataFrame(index=df_base_raw.index)
    df["Price_base"] = df_base_raw["Price"]
    df = df.join(df_lev_raw["Price"].rename("Price_lev"), how="inner")
    df = df.sort_index()

    # 限制在 start_early ~ end
    df = df[
        (df.index >= pd.to_datetime(start_early))
        & (df.index <= pd.to_datetime(end))
    ]

    # 計算 200SMA（原型 ETF）
    df["MA_200"] = df["Price_base"].rolling(WINDOW).mean()
    df = df.dropna(subset=["MA_200"]).copy()

    # 再切一次在使用者輸入期間
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    if df.empty:
        st.error("⚠️ 有效資料不足，請調整期間")
        st.stop()

    # 日報酬（原型 / 槓桿）
    df["Return_base"] = df["Price_base"].pct_change().fillna(0)
    df["Return_lev"] = df["Price_lev"].pct_change().fillna(0)

    ###############################################################
    # 產生 LRS 訊號（用原型 ETF 價格與 200SMA）
    # 訊號只在「穿越」當天產生：
    # p > m 且 前一天 p0 <= m0 → 買進訊號（1）
    # p < m 且 前一天 p0 >= m0 → 賣出訊號（-1）
    ###############################################################

    df["Signal"] = 0

    for i in range(1, len(df)):
        p, m = df["Price_base"].iloc[i], df["MA_200"].iloc[i]
        p0, m0 = df["Price_base"].iloc[i - 1], df["MA_200"].iloc[i - 1]

        if p > m and p0 <= m0:
            df.iloc[i, df.columns.get_loc("Signal")] = 1  # 進場訊號
        elif p < m and p0 >= m0:
            df.iloc[i, df.columns.get_loc("Signal")] = -1  # 出場訊號

    # 由 Signal 累積出 Position：
    # 初始為空手（0），收到 1 → 變成持有（1），收到 -1 → 變回 0
    pos = []
    current = 0  # 一開始「空手」，與手算一致
    for sig in df["Signal"]:
        if sig == 1:
            current = 1
        elif sig == -1:
            current = 0
        pos.append(current)
    df["Position"] = pos

    ###############################################################
    # LRS 資金曲線（用槓桿 ETF 收盤價計算，邏輯與手算一致）
    #
    # 規則：
    # 1. Position = 1 → 持有槓桿 ETF，Equity_t = Equity_{t-1} * (P_t / P_{t-1})
    # 2. Position = 0 → 空手，Equity_t = Equity_{t-1}
    # 3. 買進當天：當日訊號發生在收盤價，實際曝險從「下一個交易日」開始
    #    （數學上與你用「進場價 → 出場價」算報酬是等價的）
    ###############################################################

    equity_lrs = [1.0]
    for i in range(1, len(df)):
        if df["Position"].iloc[i] == 1 and df["Position"].iloc[i - 1] == 1:
            # 持有期間，照每日收盤價漲跌計算
            r = df["Price_lev"].iloc[i] / df["Price_lev"].iloc[i - 1]
            equity_lrs.append(equity_lrs[-1] * r)
        else:
            # 空手或剛切換部位的當天：資金維持不變
            equity_lrs.append(equity_lrs[-1])

    df["Equity_LRS"] = equity_lrs
    df["Return_LRS"] = df["Equity_LRS"].pct_change().fillna(0)

    # 兩種 Buy & Hold
    df["Equity_BH_Base"] = (1 + df["Return_base"]).cumprod()
    df["Equity_BH_Lev"] = (1 + df["Return_lev"]).cumprod()

    # 報酬率（圖用）
    df["Pct_Base"] = df["Equity_BH_Base"] - 1
    df["Pct_Lev"] = df["Equity_BH_Lev"] - 1
    df["Pct_LRS"] = df["Equity_LRS"] - 1

    # 買賣點（用來畫 marker）
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]

    ###############################################################
    # 圖 1：原型 ETF 價格 + 200SMA + 買賣點（乾淨版）
    # 買賣點畫在「原型 ETF 價格」上，hover 顯示「槓桿 ETF 價格」
    # 買進：綠色空心圓；賣出：紅色空心圓；所有線為實線
    ###############################################################

    st.markdown(
        "<h3>📈 原型 ETF 價格 & 200SMA（買賣訊號來自這裡，hover 顯示槓桿 ETF 價格）</h3>",
        unsafe_allow_html=True,
    )

    fig_price = go.Figure()

    # 原型 ETF 收盤價（實線）
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Price_base"],
            name=f"{base_label} 收盤價",
            mode="lines",
            line=dict(color="#1f77b4", width=2),  # 實線
        )
    )

    # 200 日 SMA（實線）
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA_200"],
            name="200 日 SMA",
            mode="lines",
            line=dict(color="#7f7f7f", width=2),  # 實線，不加 dash
        )
    )

    # 買進點：綠色空心圓，畫在「原型 ETF 價格」上，hover 顯示槓桿 ETF 買進價
    if not buys.empty:
        fig_price.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Price_base"],
                mode="markers",
                name="買進 Buy",
                marker=dict(
                    symbol="circle-open",
                    size=12,
                    line=dict(width=2, color="#2ca02c"),  # 綠色空心圓
                ),
                customdata=buys["Price_lev"],
                hovertemplate=(
                    "📈 <b>買進訊號（來自原型 ETF）</b><br>"
                    "日期: %{x|%Y-%m-%d}<br>"
                    + f"{base_label} 價格: "
                    + "%{y:.2f}<br>"
                    + f"{lev_label} 買進價: "
                    + "%{customdata:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # 賣出點：紅色空心圓，畫在「原型 ETF 價格」上，hover 顯示槓桿 ETF 賣出價
    if not sells.empty:
        fig_price.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Price_base"],
                mode="markers",
                name="賣出 Sell",
                marker=dict(
                    symbol="circle-open",
                    size=12,
                    line=dict(width=2, color="#d62728"),  # 紅色空心圓
                ),
                customdata=sells["Price_lev"],
                hovertemplate=(
                    "📉 <b>賣出訊號（來自原型 ETF）</b><br>"
                    "日期: %{x|%Y-%m-%d}<br>"
                    + f"{base_label} 價格: "
                    + "%{y:.2f}<br>"
                    + f"{lev_label} 賣出價: "
                    + "%{customdata:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # 圖右上角標註：原型 / 槓桿 ETF 身份
    fig_price.add_annotation(
        xref="paper",
        yref="paper",
        x=1.01,
        y=1.0,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        text=f"{base_label}：訊號來源<br>{lev_label}：實際交易標的",
        font=dict(size=12, color="#555555"),
    )

    fig_price.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=80, t=40, b=40),
        legend=dict(orientation="h"),
        xaxis=dict(title="日期"),
        yaxis=dict(title="價格"),
    )

    st.plotly_chart(fig_price, use_container_width=True)

    ###############################################################
    # 圖 2：三種策略資金曲線（報酬率）— 全部實線
    ###############################################################

    st.markdown("<h3>📊 三種策略資金曲線（報酬率）</h3>", unsafe_allow_html=True)

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Pct_Base"],
            mode="lines",
            name=f"{base_label} BH",
            line=dict(width=2),  # 實線
        )
    )
    fig_equity.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Pct_Lev"],
            mode="lines",
            name=f"{lev_label} BH",
            line=dict(width=2),  # 實線
        )
    )
    fig_equity.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Pct_LRS"],
            mode="lines",
            name=f"{lev_label} LRS",
            line=dict(width=2),  # 實線（可考慮改顏色，但不加 dash）
        )
    )

    fig_equity.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h"),
        xaxis=dict(title="日期"),
        yaxis=dict(title="報酬率", tickformat=".0%"),
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    ###############################################################
    # 指標計算
    ###############################################################

    def calc_metrics(eq: pd.Series, ret: pd.Series):
        final = eq.iloc[-1]
        total_ret = final - 1
        years = (eq.index[-1] - eq.index[0]).days / 365
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
        mdd = 1 - (eq / eq.cummax()).min()

        daily = ret.dropna()
        if len(daily) <= 1:
            vol = sharpe = sortino = np.nan
        else:
            avg = daily.mean()
            std = daily.std()
            vol = std * np.sqrt(252)
            sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
            downside = daily[daily < 0].std()
            sortino = (avg / downside) * np.sqrt(252) if downside > 0 else np.nan

        return final, total_ret, cagr, mdd, vol, sharpe, sortino

    m_base = calc_metrics(df["Equity_BH_Base"], df["Return_base"])
    m_lev = calc_metrics(df["Equity_BH_Lev"], df["Return_lev"])
    m_lrs = calc_metrics(df["Equity_LRS"], df["Return_LRS"])

    ###############################################################
    # 美化表格
    ###############################################################

    st.markdown(
        """
    <style>
    .custom-table { width:100%; border-collapse:collapse; margin-top:1.2em; }
    .custom-table th {
        background:#f4f4f4; padding:10px; font-weight:700;
        border-bottom:2px solid #ddd;
    }
    .custom-table td {
        text-align:center; padding:8px;
        border-bottom:1px solid #eee; font-size:14px;
    }
    .custom-table tr:nth-child(even) td { background-color:#fafafa; }
    .custom-table tr:hover td { background-color:#f1f7ff; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    html_table = f"""
<table class="custom-table">
  <thead>
    <tr>
      <th>指標</th>
      <th>{base_label} BH</th>
      <th>{lev_label} BH</th>
      <th>{lev_label} LRS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>總報酬</td>
      <td>{m_base[1]:.2%}</td>
      <td>{m_lev[1]:.2%}</td>
      <td>{m_lrs[1]:.2%}</td>
    </tr>
    <tr>
      <td>年化報酬 (CAGR)</td>
      <td>{m_base[2]:.2%}</td>
      <td>{m_lev[2]:.2%}</td>
      <td>{m_lrs[2]:.2%}</td>
    </tr>
    <tr>
      <td>最大回撤 (MDD)</td>
      <td>{m_base[3]:.2%}</td>
      <td>{m_lev[3]:.2%}</td>
      <td>{m_lrs[3]:.2%}</td>
    </tr>
    <tr>
      <td>年化波動率</td>
      <td>{m_base[4]:.2%}</td>
      <td>{m_lev[4]:.2%}</td>
      <td>{m_lrs[4]:.2%}</td>
    </tr>
    <tr>
      <td>Sharpe Ratio</td>
      <td>{m_base[5]:.2f}</td>
      <td>{m_lev[5]:.2f}</td>
      <td>{m_lrs[5]:.2f}</td>
    </tr>
    <tr>
      <td>Sortino Ratio</td>
      <td>{m_base[6]:.2f}</td>
      <td>{m_lev[6]:.2f}</td>
      <td>{m_lrs[6]:.2f}</td>
    </tr>
    <tr>
      <td>買進次數（LRS）</td>
      <td colspan="3">{len(buys)}</td>
    </tr>
    <tr>
      <td>賣出次數（LRS）</td>
      <td colspan="3">{len(sells)}</td>
    </tr>
  </tbody>
</table>
"""

    st.markdown(html_table, unsafe_allow_html=True)
    st.success("✅ 回測完成！已顯示三種策略的資金曲線與績效指標。")
