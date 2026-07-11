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
    
    # MACD 簡易計算
    ema12 = df['Close'].ewm(span=12).mean().iloc[-1].item()
    ema26 = df['Close'].ewm(span=26).mean().iloc[-1].item()
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).rolling(9).mean().iloc[-1]
    
    score = 0
    # 權重：趨勢 (25%)
    if price > ma10 and ma10 > ma60: score += 25
    elif price > ma60: score += 10
    
    # 權重：MACD (10%)
    if macd_line > signal_line: score += 10
    
    # 權重：支撐 (10%)
    if abs(price - ma60) / ma60 < 0.03: score += 10
    
    return score

# --- 市場雷達：爆量篩選邏輯 ---
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
    
    # 取得必要數值
    price = float(df['Close'].iloc[-1].item())
    score = calculate_advanced_score(df)
    is_breakout, status = check_volume_breakout(df)
    profit_pct = ((price - cost) / cost) * 100
    
    # 卡片顯示
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"代號: {code}")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("損益", f"{profit_pct:.1f}%", delta=f"成本: {cost:.1f}")
        
        st.write(f"AI評分: {score} 分 | 雷達狀態: {status}")
        st.progress(min(score / 45, 1.0)) # 滿分約 45 分，調整進度條比例
