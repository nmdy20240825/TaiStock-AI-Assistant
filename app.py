import streamlit as st
import requests
import yfinance as yf

# 股票資料庫 (未來可持續擴充)
STOCK_DATABASE = {
    "2317": "鴻海", "2330": "台積電", "2382": "廣達", 
    "3017": "奇鋐", "3037": "欣興", "3443": "創意"
}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("🚀 波段決策儀表板 V41.0")

# 1. 側邊欄設定
with st.sidebar:
    st.header("設定")
    selected_code = st.selectbox("選擇分析股票:", [f"{code} {name}" for code, name in STOCK_DATABASE.items()])
    code = selected_code.split(" ")[0]
    cost = st.number_input("持股成本價:", min_value=0.0, value=171.0)

# 2. 數據獲取與計算
ticker = yf.Ticker(f"{code}.TW")
hist = ticker.history(period="6mo")
current_price = hist['Close'].iloc[-1]
ma60 = hist['Close'].rolling(window=60).mean().iloc[-1]
stop_loss = cost * 0.92

# 3. 視覺化儀表板排版
col1, col2, col3 = st.columns(3)
col1.metric("即時股價", f"{current_price:.2f}")
col2.metric("8% 停損點", f"{stop_loss:.2f}")
col3.metric("季線(MA60)", f"{ma60:.2f}", delta=f"{current_price - ma60:.2f}")

# 風險警示邏輯
if current_price < ma60:
    st.warning("⚠️ 目前股價位於季線之下，請留意中線修正風險。")
else:
    st.success("✅ 目前股價位於季線之上，趨勢偏多。")

# 4. AI 分析
if st.button("更新 AI 深度分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    prompt = f"針對 {selected_code}，現價 {current_price}，成本 {cost}，停損 {stop_loss}，季線 {ma60}。請提供簡潔的波段操作建議。"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    response = requests.post(url, json=payload).json()
    if "candidates" in response:
        st.write("---")
        st.write(response["candidates"][0]["content"]["parts"][0]["text"])
