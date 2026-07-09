import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 設定與金鑰檢查
st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V67.0 (分離運作版)")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}

# 2. 側邊欄維護庫存
with st.sidebar:
    st.header("庫存清單")
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
        st.rerun()

# 3. 穩定顯示區 (僅計算基本數據，速度最快)
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        data.append({"代號": code, "現價": round(c, 2), "帳面損益": round((c - info['cost']) * info['shares'], 0)})
st.table(pd.DataFrame(data))

# 4. 完全獨立的 AI 診斷區 (避免相互影響)
st.divider()
st.subheader("🤖 AI 智能診斷分析")
target = st.text_input("輸入要診斷的代號:")
if st.button("執行 AI 分析"):
    st.info(f"啟動診斷: {target}...")
    try:
        # 單獨抓取該標的數據給 AI
        df = yf.download(f"{target}.TW", period="1y", progress=False)
        curr = float(df['Close'].iloc[-1].item())
        ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
        ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
        
        prompt = f"分析標的:{target},現價:{curr},10MA:{ma10},60MA:{ma60}。請給出明確買進/賣出/加碼/停利建議。"
        
        api_key = st.secrets.get("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15).json()
        st.markdown(res["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        st.error(f"分析失敗，請檢查 API Key 或代號是否正確: {str(e)}")
