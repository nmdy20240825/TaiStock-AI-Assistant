import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板")

def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def calculate_all_indicators(df):
    # 確保資料為浮點數
    df = df.astype(float)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # 取最後一筆
    last = df.iloc[-1]
    ma60_val = float(last['MA60'])
    close_val = float(last['Close'])
    k_val = float(last['K'])
    d_val = float(last['D'])
    
    score = (15 if close_val > ma60_val else 0) + (10 if k_val > d_val else 0)
    advice = "續抱" if score >= 20 else "停損/減碼"
    
    return score, advice, ma60_val * 1.1, ma60_val * 0.95, last

portfolio = load_portfolio()
for code, info in portfolio.items():
    name, cost = info
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if df is None or len(df) < 60: continue
    
    score, advice, target, stop, last = calculate_all_indicators(df)
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"{name} ({code})")
        col2.metric("現價", f"{float(last['Close']):.2f}")
        col3.metric("AI評分", f"{score}分", delta=advice)
        
        if st.button(f"查看詳細指標 {code}"):
            st.write(f"MA10: {float(last['MA10']):.1f} | MA60: {float(last['MA60']):.1f}")
            st.write(f"KD: K={float(last['K']):.1f}, D={float(last['D']):.1f}")
            st.write(f"停利: {target:.1f} | 停損: {stop:.1f}")
