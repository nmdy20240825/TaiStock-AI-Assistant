import streamlit as st
import requests
import yfinance as yf
import pandas as pd

# 1. 您的真實庫存 (未來可改為讀取 Google Sheets)
MY_PORTFOLIO = {
    "2317": {"name": "鴻海", "cost": 171.0, "shares": 1000},
    "2330": {"name": "台積電", "cost": 990.0, "shares": 500}
}

st.set_page_config(page_title="持股損益監控", layout="wide")
st.title("💼 智慧庫存監控面板 V50.0")

# 2. 自動計算損益與停損監控
data = []
for code, info in MY_PORTFOLIO.items():
    ticker = yf.Ticker(f"{code}.TW")
    price = ticker.history(period="1d")['Close'].iloc[-1]
    profit = (price - info['cost']) * info['shares']
    profit_pct = (price - info['cost']) / info['cost'] * 100
    stop_loss = info['cost'] * 0.92
    
    data.append({
        "股票": info['name'],
        "現價": price,
        "成本": info['cost'],
        "市值": price * info['shares'],
        "損益%": f"{profit_pct:.2f}%",
        "距離停損": f"{(price - stop_loss) / info['cost'] * 100:.2f}%"
    })

df_portfolio = pd.DataFrame(data)

# 3. 顯示庫存表格
st.subheader("庫存總覽")
st.table(df_portfolio)

# 4. 決策分析區 (保留原本功能)
st.divider()
st.subheader("單股 AI 深度診斷")
# ... (這裡放入您原本 V49.0 的選股與 AI 分析邏輯)
