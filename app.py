import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import json
import os

st.set_page_config(layout="wide", page_title="TaiStock AI 專業決策版")

# --- 法人數據爬蟲 ---
@st.cache_data(ttl=3600)
def get_institutional_data(code):
    try:
        url = f"https://statementdog.com/api/v1/taiwan_stocks/{code}/institutional_investors"
        # 模擬簡單法人抓取邏輯 (註: 實際環境需依目標網站結構調整)
        return {"buy_sell": 1500, "days": 3, "trend": "連3買"}
    except: return {"buy_sell": 0, "days": 0, "trend": "無"}

# --- 既有邏輯 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    return yf.download(f"{code}.TW", period="6mo", progress=False)

def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    try:
        with open('portfolio.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()
# (側邊欄管理程式碼同前)

st.title("⚡ TaiStock 進階決策系統 (法人版)")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = fetch_stock_data(code)
        inst = get_institutional_data(code)
        
        c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        price, volume = float(c.iloc[-1]), float(v.iloc[-1])
        
        # 指標計算
        ma20 = float(c.rolling(20).mean().iloc[-1])
        k, d = 50.0, 45.0 # 簡化指標範例
        
        # 加碼邏輯：連續買超 >= 3 且為起漲股
        coeff = price / ma20
        can_add = (inst['days'] >= 3 and coeff > 1.15)
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("法人動能", inst['trend'], delta=f"{inst['buy_sell']}張")
            c2.metric("股性判別", "🚀 起漲股" if coeff > 1.15 else "📊 一般股")
            c3.metric("AI 建議", "🎯 可加碼" if can_add else "🛡️ 風險控管")
            c4.metric("現價", f"{price:.2f}")
            
            with st.expander("🚦 查看完整數據"):
                st.write(f"成交量: {volume:,.0f} | 均線: MA20:{ma20:.1f}")
                st.write(f"法人狀況: {inst['trend']} (連續 {inst['days']} 天)")
                st.write(f"加碼判斷: {'符合條件' if can_add else '未達加碼門檻'}")
                st.divider()
                st.caption("• 加碼標準：法人連買 3 天以上 且 股性係數 > 1.15")
    except Exception as e:
        st.error(f"分析 {code} 發生異常: {e}")

