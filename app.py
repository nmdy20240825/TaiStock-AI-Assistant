import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 初始化資料結構
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        "2317": {"name": "鴻海", "cost": 171.0, "shares": 1000},
        "2330": {"name": "台積電", "cost": 990.0, "shares": 500}
    }

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("📈 波段決策儀表板 V57.0 (完整數據版)")

# 2. 側邊欄：新增/刪除功能
with st.sidebar:
    st.header("庫存管理")
    new_code = st.text_input("新增代號 (如 2308):")
    new_cost = st.number_input("成本:", value=0.0)
    new_shares = st.number_input("股數:", value=1000)
    if st.button("加入庫存"):
        name = yf.Ticker(f"{new_code}.TW").info.get('shortName', '新標的')
        st.session_state.portfolio[new_code] = {"name": name, "cost": new_cost, "shares": new_shares}
        st.rerun()

# 3. 數據計算與表格顯示
data = []
for code, info in st.session_state.portfolio.items():
    ticker = yf.Ticker(f"{code}.TW")
    price = ticker.history(period="1d")['Close'].iloc[-1]
    profit = (price - info['cost']) * info['shares']
    data.append({
        "代號": code, "名稱": info['name'], "現價": round(price, 2),
        "成本": info['cost'], "股數": info['shares'], "損益": round(profit, 0)
    })

df = pd.DataFrame(data)
st.table(df)

# 4. 獨立運行的 AI 診斷
st.divider()
st.subheader("✨ AI 深度診斷")
target = st.selectbox("請選擇診斷標的:", df['代號'].tolist())

if st.button("執行 AI 深度診斷"):
    st.info(f"正在分析 {target}...")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    # 針對特定代號進行技術診斷
    prompt = f"分析 {target} 技術面，現價為 {df[df['代號']==target]['現價'].values[0]}。"
    
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15).json()
        st.write(res["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        st.error("分析回應逾時，請再次點擊嘗試。")
