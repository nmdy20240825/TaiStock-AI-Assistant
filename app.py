import streamlit as st
import requests
import yfinance as yf

# 1. 股票資料庫
STOCK_DATABASE = {
    "2317": "鴻海", "2330": "台積電", "2382": "廣達", 
    "3017": "奇鋐", "3037": "欣興", "3443": "創意"
}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("🚀 波段決策儀表板 V43.0")

# 2. 側邊欄設定
with st.sidebar:
    st.header("自選股設定")
    selected_codes = st.multiselect("選擇要監控的股票:", list(STOCK_DATABASE.keys()), default=["2317", "2330", "2382"])
    
    if not selected_codes:
        st.warning("請至少選擇一支股票")
        st.stop()
        
    current_code = st.selectbox("切換分析對象:", selected_codes)
    cost = st.number_input(f"{STOCK_DATABASE[current_code]} 的成本價:", min_value=0.0, value=990.0)

# 3. 數據獲取
ticker = yf.Ticker(f"{current_code}.TW")
hist = ticker.history(period="6mo")
current_price = hist['Close'].iloc[-1]
ma60 = hist['Close'].rolling(window=60).mean().iloc[-1]
stop_loss = cost * 0.92

# 4. 儀表板視覺化
st.subheader(f"分析標的: {current_code} {STOCK_DATABASE[current_code]}")
col1, col2, col3 = st.columns(3)
col1.metric("即時股價", f"{current_price:.2f}")
col2.metric("8% 停損點", f"{stop_loss:.2f}")
col3.metric("季線(MA60)", f"{ma60:.2f}", delta=f"{current_price - ma60:.2f}")

# 5. AI 深度分析模組
if st.button("✨ 開始 AI 深度分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    prompt = f"""
    請針對 {STOCK_DATABASE[current_code]} ({current_code}) 進行波段操作分析。
    數據如下：
    - 現價: {current_price:.2f}
    - 您的持股成本: {cost}
    - 計算停損價: {stop_loss:.2f}
    - 季線支撐: {ma60:.2f}
    
    請提供：
    1. 風險評估：目前股價與季線、停損點的相對位置分析。
    2. 操作建議：在不追高的前提下，波段交易的策略為何？
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    with st.spinner('AI 教練正在運算中...'):
        response = requests.post(url, json=payload).json()
        if "candidates" in response:
            st.write("---")
            st.write(response["candidates"][0]["content"]["parts"][0]["text"])
