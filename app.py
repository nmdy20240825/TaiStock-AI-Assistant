import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板")

# --- 讀取與儲存檔案的輔助函式 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- [側邊欄：持股管理系統] ---
portfolio = load_portfolio()
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock"):
        new_code = st.text_input("股票代號 (如 2317)")
        new_name = st.text_input("股票名稱")
        new_cost = st.number_input("成本", value=100.0)
        submitted = st.form_submit_button("新增/更新持股")
        if submitted and new_code:
            portfolio[new_code] = [new_name, new_cost]
            save_portfolio(portfolio)
            st.success(f"已更新 {new_name}")

    st.divider()
    del_code = st.selectbox("選擇要刪除的股票", [""] + list(portfolio.keys()))
    if st.button("刪除選定股票"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

# --- [核心引擎：技術指標計算 (同前)] ---
def calculate_advanced_score(df):
    price = float(df['Close'].iloc[-1].item())
    ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
    ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
    score = 0
    if price > ma10 and ma10 > ma60: score += 25
    elif price > ma60: score += 10
    score += 15 # 籌碼權重
    return score

# --- [顯示邏輯] ---
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df.empty: continue
        price = float(df['Close'].iloc[-1].item())
        score = calculate_advanced_score(df)
        profit_pct = ((price - cost) / cost) * 100
        
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.subheader(f"{name} ({code})")
            col2.metric("現價", f"{price:.2f}")
            col3.metric("損益", f"{profit_pct:.1f}%", delta=f"成本: {cost:.1f}")
            st.write(f"**AI綜合評分**: {score} 分")
            st.progress(min(score / 50, 1.0))
    except Exception as e:
        st.write(f"無法載入 {code}")
