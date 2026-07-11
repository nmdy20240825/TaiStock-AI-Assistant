import streamlit as st
import pandas as pd
import yfinance as yf

# --- [核心引擎模組] ---
def calculate_score(price, ma10, ma60):
    score = 0
    if price > ma10 and ma10 > ma60: score += 50  # 權重：趨勢
    if price > ma60: score += 30                  # 權重：支撐
    return score

# --- [UI 介面層] ---
st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板")

# 庫存資料 (代號: 成本)
portfolio = {"2317": 171.0, "2330": 990.0}

for code, cost in portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    price = float(df['Close'].iloc[-1].item())
    ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
    ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
    
    score = calculate_score(price, ma10, ma60)
    profit_pct = ((price - cost) / cost) * 100
    
    # 傑夫大大要求的卡片式顯示 (含損益)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"代號: {code}")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("損益", f"{profit_pct:.1f}%", delta=f"成本: {cost:.1f}")
        
        st.write(f"AI評分: {score} 分")
        st.progress(score / 100)
