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

# --- 絕對穩定指標計算 ---
def calculate_all_indicators(df):
    # 確保資料為數值，非數值轉為 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 計算
    ma10 = df['Close'].rolling(10).mean()
    ma60 = df['Close'].rolling(60).mean()
    
    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 0.001) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    
    # 取得最新一筆數值 (強制轉為 float)
    last_ma10 = float(ma10.iloc[-1])
    last_ma60 = float(ma60.iloc[-1])
    last_k = float(k.iloc[-1])
    last_d = float(d.iloc[-1])
    last_close = float(df['Close'].iloc[-1])
    
    score = (15 if last_close > last_ma60 else 0) + (10 if last_k > last_d else 0)
    advice = "續抱" if score >= 20 else "停損/減碼"
    
    return score, advice, last_ma10, last_ma60, last_k, last_d, last_close

# --- 主程式 ---
portfolio = load_portfolio()
for code, info in portfolio.items():
    name, cost = info
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if df is None or len(df) < 60: continue
    
    score, advice, ma10, ma60, k, d, price = calculate_all_indicators(df)
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"{name} ({code})")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("AI評分", f"{score}分", delta=advice)
        
        if st.button(f"顯示 {name} 詳細數據", key=f"btn_{code}"):
            st.write(f"MA10: {ma10:.1f} | MA60: {ma60:.1f}")
            st.write(f"KD指標: K={k:.1f}, D={d:.1f}")
            st.write(f"停利建議: {ma60*1.1:.1f} | 停損建議: {ma60*0.95:.1f}")
