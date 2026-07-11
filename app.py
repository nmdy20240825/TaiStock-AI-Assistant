import streamlit as st
import pandas as pd
import yfinance as yf

# 設定頁面格式
st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板")

# --- 核心引擎：技術指標計算 ---
def calculate_advanced_score(df):
    price = float(df['Close'].iloc[-1].item())
    ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
    ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
    
    ema12 = df['Close'].ewm(span=12).mean().iloc[-1].item()
    ema26 = df['Close'].ewm(span=26).mean().iloc[-1].item()
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).rolling(9).mean().iloc[-1]
    
    score = 0
    if price > ma10 and ma10 > ma60: score += 25
    elif price > ma60: score += 10
    if macd_line > signal_line: score += 10
    if abs(price - ma60) / ma60 < 0.03: score += 10
    return score

# --- AI 教練邏輯 ---
def get_ai_coach_advice(score):
    if score >= 35:
        return "續抱", "強勢多頭，技術指標完美"
    elif score >= 20:
        return "觀察", "趨勢震盪，建議守穩支撐"
    else:
        return "停損/減碼", "跌破關鍵均線，風險增加"

# --- 市場雷達 ---
def check_volume_breakout(df):
    vol_today = df['Volume'].iloc[-1].item()
    vol_avg_5 = df['Volume'].rolling(5).mean().iloc[-1].item()
    if vol_today > (vol_avg_5 * 1.5):
        return True, "🚀 量增突破"
    return False, "無特別訊號"

# --- 儀表板顯示 ---
portfolio = {"2317": 171.0, "2330": 990.0}

for code, cost in portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    price = float(df['Close'].iloc[-1].item())
    score = calculate_advanced_score(df)
    is_breakout, status = check_volume_breakout(df)
    decision, reason = get_ai_coach_advice(score)
    profit_pct = ((price - cost) / cost) * 100
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"代號: {code}")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("損益", f"{profit_pct:.1f}%", delta=f"成本: {cost:.1f}")
        
        st.write(f"**AI評分**: {score} | **建議**: {decision}")
        st.caption(f"理由: {reason} | 雷達: {status}")
        st.progress(min(score / 45, 1.0))
