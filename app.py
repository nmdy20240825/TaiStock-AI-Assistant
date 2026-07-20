import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock V3.0 專業升級版")

# === 視覺優化 ===
st.markdown("""<style>
[data-testid="stMetricValue"] { font-size: 18px !important; }
.ai-advice-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; }
</style>""", unsafe_allow_html=True)

# === 核心數據模組 ===
@st.cache_data(ttl=1800)
def fetch_macro_data():
    tickers = {'TW': '^TWII', 'US': '^IXIC', 'VIX': '^VIX'}
    macro_status = {}
    for key, symbol in tickers.items():
        try:
            df = yf.download(symbol, period="3mo", progress=False)
            if not df.empty:
                c_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
                c, ma20 = float(c_series.iloc[-1]), float(c_series.rolling(20).mean().iloc[-1])
                macro_status[key] = {'price': c, 'trend': '🟢 多頭' if c > ma20 else '🔴 空頭'}
        except: macro_status[key] = None
    return macro_status

def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f: return json.load(f)

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()
macro_data = fetch_macro_data()
today_str = datetime.datetime.now().strftime("%Y-%m-%d")

# === 側邊欄：新增持股 ===
with st.sidebar:
    st.header("📋 持股設定")
    with st.form("add_stock"):
        code = st.text_input("代號")
        name = st.text_input("名稱")
        cost = st.number_input("成本", value=100.0)
        if st.form_submit_button("新增/更新"):
            portfolio[code] = {"name": name, "cost": cost, "status": "Active"}
            save_portfolio(portfolio); st.rerun()

# === 主邏輯：渲染與結算 ===
st.title("⚡ TaiStock V3.0 戰情室")

for code, info in list(portfolio.items()):
    if info.get('status') == "Closed": continue
    
    # 獲取股價
    df = yf.download(code if '.' in code else f"{code}.TW", period="1mo", progress=False)
    if df.empty: continue
    price = float(df['Close'].iloc[-1].iloc[0] if isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[-1])
    
    # 破線計數器邏輯
    is_broken = price < (price * 0.95) # 範例邏輯
    status_label = f" <span style='color: red;'>[🚨破線]</span>" if is_broken else ""
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.markdown(f"#### {info['name']} ({code}){status_label}")
        col2.metric("現價", f"{price:.2f}")
        
        # 一鍵結算按鈕
        if col3.button(f"🚀 出清", key=f"close_{code}"):
            portfolio[code]['status'] = "Closed"
            save_portfolio(portfolio); st.rerun()

        # 大盤環境扣分機制 (簡化版)
        tw_trend = macro_data.get('TW', {}).get('trend', '多頭')
        confidence = 80
        if "空頭" in tw_trend: confidence -= 15
        st.write(f"當前信心分數: {confidence} (大盤環境: {tw_trend})")
