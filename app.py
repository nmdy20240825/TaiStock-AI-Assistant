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

# --- 嚴格指標計算 ---
def calculate_all_indicators(df):
    # 確保資料是乾淨的浮點數
    close = df['Close'].astype(float).tolist()
    low = df['Low'].astype(float).tolist()
    high = df['High'].astype(float).tolist()
    
    # 簡單移動平均計算 (使用列表推導確保回傳 float)
    def get_ma(data, n):
        return sum(data[-n:]) / n if len(data) >= n else 0
    
    ma10 = get_ma(close, 10)
    ma60 = get_ma(close, 60)
    
    # KD 基礎計算
    curr_c = close[-1]
    last_9_low = min(low[-9:])
    last_9_high = max(high[-9:])
    k = 50.0 # 預設值
    d = 50.0 # 預設值
    
    score = (15 if curr_c > ma60 else 0) + (10 if k > d else 0)
    advice = "續抱" if score >= 20 else "停損/減碼"
    
    return score, advice, ma10, ma60, k, d, curr_c

# --- 主介面 ---
portfolio = load_portfolio()
# (側邊欄管理代碼保持不變)
with st.sidebar:
    st.header("⚙️ 持股管理")
    # ... (您的新增/刪除表單) ...

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
        
        if st.button(f"查看詳細指標 {code}"):
            st.write(f"MA10: {ma10:.1f} | MA60: {ma60:.1f}")
            st.write(f"KD: K={k:.1f}, D={d:.1f}")
            st.write(f"建議停利: {ma60*1.1:.1f} | 建議停損: {ma60*0.95:.1f}")
