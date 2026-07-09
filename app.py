import streamlit as st
import requests
import yfinance as yf

# 1. 股票資料庫
STOCK_DATABASE = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "3017": "奇鋐", "3037": "欣興", "3443": "創意"}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("📈 波段決策儀表板 V48.0 (加入月線 MA20)")

# 2. 側邊欄與運算
with st.sidebar:
    selected_code = st.selectbox("分析標的:", [f"{c} {n}" for c, n in STOCK_DATABASE.items()])
    code = selected_code.split(" ")[0]
    cost = st.number_input("持股成本價:", value=171.0)

ticker = yf.Ticker(f"{code}.TW")
df = ticker.history(period="1y")

# 計算技術指標
df['MA20'] = df['Close'].rolling(window=20).mean() # 月線
df['MA60'] = df['Close'].rolling(window=60).mean() # 季線
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
low_min = df['Low'].rolling(window=9).min()
high_max = df['High'].rolling(window=9).max()
rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
df['K'] = rsv.ewm(com=2).mean()
df['D'] = df['K'].ewm(com=2).mean()

curr = df.iloc[-1]
prev = df.iloc[-2]
stop_loss = cost * 0.92

# 3. 儀表板面板
# 分兩列顯示，讓 Metric 更寬敞
c1, c2, c3 = st.columns(3)
c1.metric("即時股價", f"{curr['Close']:.2f}")
c2.metric("8% 停損點", f"{stop_loss:.2f}")
c3.metric("月線(MA20)", f"{curr['MA20']:.2f}", delta=f"{curr['MA20'] - prev['MA20']:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("季線(MA60)", f"{curr['MA60']:.2f}", delta=f"{curr['MA60'] - prev['MA60']:.3f}")
c5.metric("MACD", f"{curr['MACD']:.2f}", delta=f"{curr['MACD'] - prev['MACD']:.3f}")
c6.metric("KD(K/D)", f"{curr['K']:.1f}/{curr['D']:.1f}")

# 4. AI 深度診斷按鈕
if st.button("✨ 執行技術指標 AI 深度診斷"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
    
    prompt = f"""
    請針對 {selected_code} 進行專業波段分析。
    當前數據：現價 {curr['Close']:.2f}, 成本 {cost}, 停損 {stop_loss:.2f}。
    趨勢指標：月線 {curr['MA20']:.2f}, 季線 {curr['MA60']:.2f}。
    動能指標：MACD {curr['MACD']:.2f}, K={curr['K']:.1f}, D={curr['D']:.1f}。
    
    分析重點：
    1. 黃金/死亡交叉判斷：目前股價位置相對於月線與季線的關係。
    2. 動能評估：綜合 MACD 與 KD 判斷是否為短線買點或獲利了結點。
    """
    
    with st.spinner('AI 教練正在處理月線與季線動能...'):
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        if "candidates" in res:
            st.write("---")
            st.write(res["candidates"][0]["content"]["parts"][0]["text"])
