import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="TaiStock AI 進階系統")

# --- 快取下載模組 (大幅提升速度) ---
@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    return yf.download(f"{code}.TW", period="6mo", progress=False)

# --- 資料持久化 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    try:
        with open('portfolio.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- 側邊欄 ---
portfolio = load_portfolio()
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock", clear_on_submit=True):
        new_code = st.text_input("代號 (例: 2330)")
        new_name = st.text_input("名稱")
        new_cost = st.number_input("成本", value=100.0)
        if st.form_submit_button("新增/更新"):
            portfolio[new_code] = [new_name, new_cost]
            save_portfolio(portfolio)
            st.rerun()
    
    del_code = st.selectbox("刪除", [""] + list(portfolio.keys()))
    if st.button("確認刪除"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

# --- 核心分析引擎 ---
st.title("⚡ TaiStock 進階決策系統 (極速穩定版)")

if not portfolio:
    st.info("請從左側新增持股開始分析")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = fetch_stock_data(code)
        if df.empty or len(df) < 60:
            st.warning(f"代號 {code} 數據異常，請確認代號是否正確。")
            continue
        
        # 數據轉換
        c = df['Close'].squeeze()
        h = df['High'].squeeze()
        l = df['Low'].squeeze()
        v = df['Volume'].squeeze()
        
        price = float(c.iloc[-1])
        ma20 = float(c.rolling(20).mean().iloc[-1])
        ma60 = float(c.rolling(60).mean().iloc[-1])
        macd = float((c.rolling(12).mean().iloc[-1]) - (c.rolling(26).mean().iloc[-1]))
        
        rsv_min = float(l.rolling(9).min().iloc[-1])
        rsv_max = float(h.rolling(9).max().iloc[-1])
        rsv = (price - rsv_min) / (rsv_max - rsv_min + 0.001) * 100
        k = float(2/3 * 50 + 1/3 * rsv)
        d = float(2/3 * 50 + 1/3 * k)
        
        # 視覺化區塊
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{((price-cost)/cost*100):.1f}%")
            c3.metric("AI 狀態", f"{'強勢' if k > 50 else '觀望'}")
            
            with st.expander("🚦 查看完整診斷面板"):
                st.write(f"均線: MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"技術指標: MACD:{macd:.3f} | K:{k:.1f} | D:{d:.1f}")
                
    except Exception as e:
        st.error(f"分析 {code} 發生異常: {e}")
