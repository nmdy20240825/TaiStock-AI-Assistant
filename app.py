import streamlit as st
import requests
import yfinance as yf
import pandas as pd

# 1. 股票資料庫
STOCK_DATABASE = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "3017": "奇鋐", "3037": "欣興", "3443": "創意"}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("📈 波段決策儀表板 V44.0 (技術指標版)")

# 2. 數據獲取與指標計算函數
def calculate_indicators(df):
    # 計算季線
    df['MA60'] = df['Close'].rolling(window=60).mean()
    # 計算 MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # 計算 KD (簡單版)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    return df

# 3. 側邊欄與資料處理
with st.sidebar:
    selected_code = st.selectbox("分析標的:", [f"{c} {n}" for c, n in STOCK_DATABASE.items()])
    code = selected_code.split(" ")[0]
    cost = st.number_input("持股成本價:", value=171.0)

ticker = yf.Ticker(f"{code}.TW")
df = ticker.history(period="1y") # 取得一年數據以利計算指標
df = calculate_indicators(df)

curr = df.iloc[-1]
stop_loss = cost * 0.92

# 4. 儀表板排版
col1, col2, col3, col4 = st.columns(4)
col1.metric("現價", f"{curr['Close']:.2f}")
col2.metric("MACD", f"{curr['MACD']:.2f}", delta=f"{curr['MACD']-curr['Signal']:.2f}")
col3.metric("K值", f"{curr['K']:.2f}")
col4.metric("D值", f"{curr['D']:.2f}")

# 5. AI 指標分析
if st.button("✨ 執行技術指標 AI 深度診斷"):
    prompt = f"""
    分析 {selected_code}：
    - 現價: {curr['Close']:.2f}, 成本: {cost}, 停損: {stop_loss:.2f}
    - 季線: {curr['MA60']:.2f}
    - MACD: {curr['MACD']:.2f}, Signal: {curr['Signal']:.2f}
    - KD值: K={curr['K']:.2f}, D={curr['D']:.2f}
    
    請綜合上述指標，判斷目前是「起漲點」、「過熱修正」、「還是箱型整理」，並提供具體操作策略。
    """
    # (省略 API 呼叫結構，同 V43.0，將 prompt 放入 payload 即可)
    # 此處請您保持 payload 呼叫邏輯一致
