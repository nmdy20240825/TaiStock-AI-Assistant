import streamlit as st
import pandas as pd
import yfinance as yf
import json

# 設定頁面格式
st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板 (戰術優化版)")

# --- 核心引擎：技術與籌碼加權 ---
def calculate_advanced_score(df):
    price = float(df['Close'].iloc[-1].item())
    ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
    ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
    
    # 模擬籌碼得分
    chip_score = 15 
    
    score = 0
    if price > ma10 and ma10 > ma60: score += 25
    elif price > ma60: score += 10
    score += chip_score 
    
    return score

# --- 雙策略比對邏輯 ---
def get_dual_strategy_advice(score, df):
    vol_today = df['Volume'].iloc[-1].item()
    vol_avg_5 = df['Volume'].rolling(5).mean().iloc[-1].item()
    short_term = "積極搶短" if vol_today > (vol_avg_5 * 1.5) else "靜待時機"
    long_term = "長線續抱" if score >= 30 else "風險控管"
    
    return short_term, long_term

# --- 儀表板顯示 ---
# 自動從 JSON 檔案讀取持股清單
try:
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
except FileNotFoundError:
    st.error("找不到 portfolio.json，請確保檔案已上傳至專案根目錄。")
    portfolio = {}

for code, info in portfolio.items():
    name, cost = info
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    
    # 檢查資料是否為空
    if df.empty:
        st.write(f"代號 {code} 無法取得數據")
        continue
        
    price = float(df['Close'].iloc[-1].item())
    score = calculate_advanced_score(df)
    short_adv, long_adv = get_dual_strategy_advice(score, df)
    profit_pct = ((price - cost) / cost) * 100
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"{name} ({code})")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("損益", f"{profit_pct:.1f}%", delta=f"成本: {cost:.1f}")
        
        st.write(f"**AI綜合評分**: {score} 分")
        col_s, col_l = st.columns(2)
        col_s.info(f"短線策略: {short_adv}")
        col_l.success(f"長線策略: {long_adv}")
        st.progress(min(score / 50, 1.0))
