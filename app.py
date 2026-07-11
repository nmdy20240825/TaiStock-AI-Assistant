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

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock"):
        new_code = st.text_input("股票代號 (如 2317)")
        new_name = st.text_input("股票名稱")
        new_cost = st.number_input("成本", value=100.0)
        if st.form_submit_button("新增/更新持股"):
            portfolio[new_code] = [new_name, new_cost]
            save_portfolio(portfolio)
            st.rerun()
    st.divider()
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("刪除選定股票"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df.empty: continue
        price = float(df['Close'].iloc[-1].item())
        
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.subheader(f"{name} ({code})")
            col2.metric("現價", f"{price:.2f}")
            col3.metric("損益", f"{((price-cost)/cost*100):.1f}%", delta=f"成本:{cost:.1f}")
            
            # 使用更穩定的顯示方式，不直接渲染圖表
            if st.checkbox(f"顯示 {name} 詳細數據", key=f"check_{code}"):
                ma10 = df['Close'].rolling(10).mean().iloc[-1]
                ma60 = df['Close'].rolling(60).mean().iloc[-1]
                st.write(f"• 10日均線: {ma10:.2f}")
                st.write(f"• 60日均線: {ma60:.2f}")
                st.write(f"• 乖離率: {((price-ma60)/ma60*100):.2f}%")
    except Exception: continue

st.divider()
st.subheader("📊 AI 評分標準說明")
st.write("- 25 分: 完美多頭 (價格 > MA10 > MA60)，建議續抱。\n- 15 分: 籌碼穩健，趨勢轉強。\n- 10 分: 均線支撐，震盪區。\n- 5 分: 弱勢，注意跌破風險。")
