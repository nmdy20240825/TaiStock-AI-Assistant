import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="TaiStock 系統修正版")

@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    # 強制獲取完整數據
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    return df

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
        new_code = st.text_input("代號 (如: 2330)")
        new_name = st.text_input("名稱")
        new_cost = st.number_input("成本", value=100.0, step=0.1)
        if st.form_submit_button("新增/更新持股"):
            portfolio[new_code] = [new_name, new_cost]
            save_portfolio(portfolio)
            st.rerun()
    
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("確認刪除"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

st.title("⚡ TaiStock 進階決策系統 (成交量與成本修正版)")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = fetch_stock_data(code)
        if df.empty: continue
        
        # 確保成交量正確抓取
        c = df['Close'].squeeze()
        v = df.get('Volume', pd.Series(0, index=df.index)).squeeze()
        
        price = float(c.iloc[-1])
        volume = float(v.iloc[-1])
        ma20 = float(c.rolling(20).mean().iloc[-1])
        
        # 股性判斷與顯示
        coeff = price / ma20
        stock_type = "🚀 起漲股" if coeff > 1.15 else "📊 一般股"
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("成交量", f"{volume:,.0f}")
            c3.metric("AI 狀態", "強勢" if coeff > 1.0 else "觀望")
            c4.metric("股性判別", stock_type)
            
            with st.expander("🚦 查看完整決策報告"):
                st.write(f"均線 MA20: {ma20:.1f} | 股性係數: {coeff:.3f}")
                st.write(f"加碼建議: {'🎯 可加碼' if coeff > 1.15 else '🛡️ 風險控管'}")
    except Exception as e:
        st.error(f"分析 {code} 失敗: {e}")
