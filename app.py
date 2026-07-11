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

# --- 指標計算引擎 (已修正 NaN 錯誤) ---
def calculate_all_indicators(df):
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # KD指標
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # 數值清理：將空值補0並確保是單一數值
    last_row = df.iloc[-1].fillna(0)
    ma60_val = float(last_row['MA60'])
    
    # AI評分 (使用安全數值比較)
    score = 0
    if float(last_row['Close']) > float(last_row['MA60']): score += 15
    if float(last_row['K']) > float(last_row['D']): score += 10
    
    advice = "續抱" if score >= 20 else "停損/減碼"
    return score, advice, ma60_val * 1.1, ma60_val * 0.95, last_row

# --- 主程式 ---
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
        
        if st.button(f"顯示 {name} 詳細指標", key=f"btn_{code}"):
            st.write(f"MA10:{last['MA10']:.1f} | MA60:{last['MA60']:.1f}")
            st.write(f"KD: K={last['K']:.1f}, D={last['D']:.1f}")
            st.write(f"停利:{target:.1f} | 停損:{stop:.1f}")
