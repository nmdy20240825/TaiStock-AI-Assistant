import streamlit as st
import requests
import yfinance as yf

st.title("波段決策系統 V40.0 (決策輔助版)")

# 1. 基礎設定
stock_map = {"2317": "2317.TW", "2330": "2330.TW", "2382": "2382.TW"}
s = st.selectbox("選擇股票", list(stock_map.keys()))

# 2. 加入成本輸入
cost = st.number_input("請輸入您的持股成本價:", min_value=0.0, value=200.0)

if st.button("開始計算與分析"):
    ticker_symbol = stock_map[s]
    
    # 取得即時數據
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="6mo") # 取得近半年數據
    current_price = hist['Close'].iloc[-1]
    ma60 = hist['Close'].rolling(window=60).mean().iloc[-1] # 簡單季線
    
    stop_loss = cost * 0.92
    
    # 顯示數據分析
    col1, col2, col3 = st.columns(3)
    col1.metric("即時股價", f"{current_price:.2f}")
    col2.metric("8% 停損點", f"{stop_loss:.2f}")
    col3.metric("季線(MA60)", f"{ma60:.2f}")
    
    # AI 教練建議
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    prompt = f"""
    分析股票 {s}，目前股價 {current_price:.2f}，您的持有成本 {cost}，8%停損價 {stop_loss:.2f}，季線 {ma60:.2f}。
    請分析目前股價相對於停損與季線的位置，並給出明確的交易操作建議。
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload).json()
    
    if "candidates" in response:
        st.write(response["candidates"][0]["content"]["parts"][0]["text"])
