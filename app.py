import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板")

# --- 檔案讀寫 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

# --- 指標計算引擎 ---
def calculate_all_indicators(df):
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # KD指標
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # 簡易AI評分與建議
    score = 0
    if df['Close'].iloc[-1] > df['MA60'].iloc[-1]: score += 15
    if df['K'].iloc[-1] > df['D'].iloc[-1]: score += 10
    
    advice = "續抱" if score >= 20 else "停損/減碼"
    target = df['MA60'].iloc[-1] * 1.1 # 簡易停利價
    stop = df['MA60'].iloc[-1] * 0.95  # 簡易停損價
    
    return score, advice, target, stop, df.iloc[-1]

# --- 主程式 ---
portfolio = load_portfolio()
# (側邊欄管理與之前相同，略過以節省篇幅)

for code, info in portfolio.items():
    name, cost = info
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if len(df) < 60: continue
    
    score, advice, target, stop, last = calculate_all_indicators(df)
    price = float(last['Close'])
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"{name} ({code})")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("AI評分", f"{score}分", delta=advice)
        
        if st.button(f"顯示 {name} 詳細分析", key=f"btn_{code}"):
            st.write(f"--- 📊 技術指標 ---")
            st.write(f"MA10:{last['MA10']:.1f} | MA20:{last['MA20']:.1f} | MA60:{last['MA60']:.1f}")
            st.write(f"KD: K={last['K']:.1f}, D={last['D']:.1f} | RSI:{last['RSI']:.1f}")
            st.write(f"MACD柱狀: {(last['MACD']-last['Signal']):.3f}")
            st.write(f"--- 💡 AI 決策建議 ---")
            st.write(f"停利價: {target:.1f} | 停損價: {stop:.1f} | 加碼建議: 跌至 MA60 時")
