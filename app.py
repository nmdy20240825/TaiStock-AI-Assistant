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
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock"):
        new_code = st.text_input("股票代號")
        new_name = st.text_input("股票名稱")
        new_cost = st.number_input("成本", value=100.0)
        if st.form_submit_button("新增/更新持股"):
            portfolio[new_code] = [new_name, new_cost]
            with open('portfolio.json', 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, ensure_ascii=False, indent=4)
            st.rerun()
    st.divider()
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("刪除選定股票"):
        if del_code in portfolio:
            del portfolio[del_code]
            with open('portfolio.json', 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, ensure_ascii=False, indent=4)
            st.rerun()

@st.cache_data(ttl=300)
def get_stock_data(code):
    return yf.download(f"{code}.TW", period="6mo", progress=False)

for code, info in portfolio.items():
    name, cost = info
    df = get_stock_data(code)
    if df is None or len(df) < 60: continue
    
    price = float(df['Close'].iloc[-1].item())
    profit_pct = ((price - cost) / cost) * 100
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.subheader(f"{name} ({code})")
        col2.metric("現價", f"{price:.2f}")
        col3.metric("損益", f"{profit_pct:.1f}%")
        
        # 使用 Session State 控制開關，避免崩潰
        if f"show_{code}" not in st.session_state: st.session_state[f"show_{code}"] = False
        if st.button(f"顯示 {name} 詳細指標", key=f"btn_{code}"):
            st.session_state[f"show_{code}"] = not st.session_state[f"show_{code}"]
            
        if st.session_state[f"show_{code}"]:
            # 強制轉換並檢查數值
            ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
            ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
            diff_rate = ((price - ma60) / ma60 * 100)
            st.write(f"• 10日均線: {ma10:.2f} | 60日均線: {ma60:.2f}")
            st.write(f"• 乖離率: {diff_rate:.2f}%")

st.divider()
st.write("📊 評分標準：25分(強勢) / 15分(穩健) / 10分(震盪) / 5分(轉弱)")
