import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 初始化 Session State
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        "2317": {"name": "鴻海", "cost": 171.0, "shares": 1000}
    }

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("📈 波段決策儀表板 V55.0 (動態管理版)")

# 2. 側邊欄：新增與刪除庫存
with st.sidebar:
    st.header("庫存管理")
    with st.expander("新增股票"):
        new_code = st.text_input("股票代號 (如 2330):")
        new_cost = st.number_input("持有成本:", value=100.0)
        new_shares = st.number_input("股數:", value=1000)
        if st.button("加入庫存"):
            # 簡單獲取名稱
            ticker = yf.Ticker(f"{new_code}.TW")
            name = ticker.info.get('shortName', '未知')
            st.session_state.portfolio[new_code] = {"name": name, "cost": new_cost, "shares": new_shares}
            st.rerun()

    with st.expander("刪除股票"):
        del_code = st.selectbox("選擇要刪除的標的:", list(st.session_state.portfolio.keys()))
        if st.button("確認刪除"):
            del st.session_state.portfolio[del_code]
            st.rerun()

# 3. 顯示庫存表格
data = []
for code, info in st.session_state.portfolio.items():
    ticker = yf.Ticker(f"{code}.TW")
    price = ticker.history(period="1d")['Close'].iloc[-1]
    data.append({
        "股票資訊": f"{code} {info['name']}",
        "現價": round(price, 2),
        "成本": info['cost'],
        "損益": round((price - info['cost']) * info['shares'], 0)
    })

df = pd.DataFrame(data)
st.subheader("💼 我的持股總覽")
st.table(df)

# 4. AI 診斷區 (保持穩定)
st.divider()
st.subheader("✨ 智慧診斷")
target = st.selectbox("選擇診斷標的:", df['股票資訊'].tolist())
if st.button("執行 AI 深度診斷"):
    st.info(f"正在分析 {target}...")
    # (此處可延續您原本的 AI 診斷邏輯)
