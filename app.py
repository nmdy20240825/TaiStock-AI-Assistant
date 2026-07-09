import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 股票資料庫
STOCK_DATABASE = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "3017": "奇鋐", "3037": "欣興", "3443": "創意"}

# 2. 初始化 Session State
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        "2317": {"name": "鴻海", "cost": 171.0, "shares": 1000},
        "2330": {"name": "台積電", "cost": 990.0, "shares": 500}
    }
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = ""

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("📈 波段決策儀法寶 V54.0")

# 3. 處理庫存數據
data = []
for code, info in st.session_state.portfolio.items():
    ticker = yf.Ticker(f"{code}.TW")
    hist = ticker.history(period="1d")
    price = hist['Close'].iloc[-1]
    profit_amt = (price - info['cost']) * info['shares']
    profit_pct = (price - info['cost']) / info['cost'] * 100
    data.append({
        "股票資訊": f"{code} {info['name']}",
        "代號": code,
        "現價": round(price, 2),
        "成本": info['cost'],
        "帳面損益": round(profit_amt, 0),
        "損益%": round(profit_pct, 2)
    })

df = pd.DataFrame(data)

# 4. 顯示儀表板
st.subheader("💼 庫存總覽")
st.table(df)

# 5. 技術指標分析與 AI 診斷
st.divider()
st.subheader("✨ 智慧診斷通道")
selected_entry = st.selectbox("選擇要診斷的標的:", df['股票資訊'].tolist())
target_code = selected_entry.split(" ")[0]

if st.button(f"對 {selected_entry} 進行 AI 深度診斷"):
    # 獲取一年數據算指標
    ticker = yf.Ticker(f"{target_code}.TW")
    df_tech = ticker.history(period="1y")
    df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
    df_tech['MA60'] = df_tech['Close'].rolling(window=60).mean()
    # MACD & KD
    ema12 = df_tech['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_tech['Close'].ewm(span=26, adjust=False).mean()
    df_tech['MACD'] = ema12 - ema26
    curr = df_tech.iloc[-1]
    
    # AI 請求
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    prompt = f"分析 {selected_entry}：現價 {curr['Close']:.2f}, 月線 {curr['MA20']:.2f}, 季線 {curr['MA60']:.2f}, MACD {curr['MACD']:.2f}。請給出明確策略。"
    
    with st.spinner('AI 教練正在處理數據...'):
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        if "candidates" in res:
            st.session_state.analysis_result = res["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.session_state.analysis_result = "連線失敗，請檢查金鑰。"

if st.session_state.analysis_result:
    st.write(st.session_state.analysis_result)
