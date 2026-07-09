import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 設置快取：避免重複抓取導致卡死
@st.cache_data(ttl=600)
def get_stock_price(code):
    ticker = yf.Ticker(f"{code}.TW")
    return ticker.history(period="1d")['Close'].iloc[-1]

# 2. 初始化
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"name": "鴻海", "cost": 171.0, "shares": 1000}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("📈 波段決策儀表板 V56.0 (效能優化版)")

# 3. 側邊欄管理
with st.sidebar:
    st.header("庫存管理")
    new_code = st.text_input("輸入股票代號 (如 2330):")
    if st.button("加入監控"):
        if new_code not in st.session_state.portfolio:
            st.session_state.portfolio[new_code] = {"name": "新標的", "cost": 0.0, "shares": 1}
            st.rerun()

# 4. 表格顯示 (使用快取抓取價格)
data = []
for code, info in st.session_state.portfolio.items():
    try:
        price = get_stock_price(code)
        data.append({"代號": code, "名稱": info['name'], "現價": round(price, 2)})
    except:
        continue

df = pd.DataFrame(data)
st.table(df)

# 5. 獨立診斷區 (避免卡頓)
st.divider()
st.subheader("✨ AI 獨立診斷通道")
target = st.selectbox("請選擇診斷標的:", df['代號'].tolist() if not df.empty else [])

if st.button("執行 AI 診斷"):
    # 這裡確保只有在按下按鈕時才呼叫 AI
    st.info(f"正在分析 {target}...")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    # 簡化 prompt 確保回應速度
    prompt = f"請快速評估 {target} 目前技術面趨勢。"
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10).json()
        st.write(res["candidates"][0]["content"]["parts"][0]["text"])
    except:
        st.error("分析回應逾時，請稍後再試。")
