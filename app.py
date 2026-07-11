import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 決策儀表板")

def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

portfolio = load_portfolio()

# --- 核心數據獲取 (極簡化，不使用欄位名稱) ---
for code, info in portfolio.items():
    name, cost = info
    try:
        data = yf.download(f"{code}.TW", period="6mo", progress=False)
        if data is None or len(data) < 60: continue
        
        # 轉換為純數值列表 (不依賴任何欄位名稱)
        values = data.values.tolist()
        close_list = [v[3] for v in values] # 通常 Close 是第 4 個欄位
        price = float(close_list[-1])
        
        # 計算簡單均線
        ma10 = sum(close_list[-10:]) / 10
        ma60 = sum(close_list[-60:]) / 60
        
        # 顯示
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.subheader(f"{name} ({code})")
            col2.metric("現價", f"{price:.2f}")
            col3.metric("損益", f"{((price-cost)/cost*100):.1f}%")
            
            if st.button(f"查看詳細數據 {code}"):
                st.write(f"10日均線: {ma10:.1f} | 60日均線: {ma60:.1f}")
                st.write(f"短線評估: {'強勢' if price > ma10 else '弱勢'}")
                
    except Exception as e:
        st.error(f"無法載入 {code}: 請稍後再試")
