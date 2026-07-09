import streamlit as st
import requests
import yfinance as yf

# 1. 股票資料庫
STOCK_DATABASE = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "3017": "奇鋐", "3037": "欣興", "3443": "創意"}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("📈 波段決策儀表板 V45.0 (完全整合版)")

# 2. 側邊欄與運算
with st.sidebar:
    selected_code = st.selectbox("分析標的:", [f"{c} {n}" for c, n in STOCK_DATABASE.items()])
    code = selected_code.split(" ")[0]
    cost = st.number_input("持股成本價:", value=171.0)

# 獲取數據與計算
ticker = yf.Ticker(f"{code}.TW")
df = ticker.history(period="1y")
# 計算指標
df['MA60'] = df['Close'].rolling(window=60).mean()
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
low_min = df['Low'].rolling(window=9).min()
high_max = df['High'].rolling(window=9).max()
rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
df['K'] = rsv.ewm(com=2).mean()
df['D'] = df['K'].ewm(com=2).mean()

curr = df.iloc[-1]
stop_loss = cost * 0.92

# 3. 儀表板面板 (恢復原本的 Metric)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("即時股價", f"{curr['Close']:.2f}")
c2.metric("8% 停損點", f"{stop_loss:.2f}")
c3.metric("季線(MA60)", f"{curr['MA60']:.2f}")
c4.metric("MACD", f"{curr['MACD']:.2f}")
c5.metric("KD(K/D)", f"{curr['K']:.1f}/{curr['D']:.1f}")

# 4. AI 深度診斷按鈕
if st.button("✨ 執行技術指標 AI 深度診斷"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    prompt = f"分析 {selected_code}：現價 {curr['Close']:.2f}, 成本 {cost}, 停損 {stop_loss:.2f}, 季線 {curr['MA60']:.2f}, MACD {curr['MACD']:.2f}, K值 {curr['K']:.2f}, D值 {curr['D']:.2f}。請綜合判斷趨勢並提供具體策略。"
    
    with st.spinner('AI 教練正在分析指標中...'):
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        if "candidates" in res:
            st.write("---")
            st.write(res["candidates"][0]["content"]["parts"][0]["text"])
        else:
            st.error(f"分析失敗: {res}")
