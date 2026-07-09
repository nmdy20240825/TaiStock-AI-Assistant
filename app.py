import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 離線對照表與初始設定
NAME_MAP = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "2308": "台達電", "3711": "日月光投控", "6409": "旭隼"}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}, "2330": {"cost": 990.0, "shares": 47}}
if 'analysis' not in st.session_state:
    st.session_state.analysis = ""

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V63.0 (穩定診斷版)")

# 2. 側邊欄
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        if code:
            st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
            st.rerun()

# 3. 數據運算
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = df['Close'].iloc[-1].item()
        ma10 = df['Close'].rolling(10).mean().iloc[-1].item()
        ma60 = df['Close'].rolling(60).mean().iloc[-1].item()
        
        data.append({
            "代號": code, "名稱": NAME_MAP.get(code, code), "現價": round(float(c), 2),
            "成本": round(float(info['cost']), 2), "帳面損益": round(float((c - info['cost']) * info['shares']), 0),
            "10MA": round(float(ma10), 2), "60MA": round(float(ma60), 2)
        })

df = pd.DataFrame(data)
st.table(df)

# 4. 改進的 AI 診斷 (使用 st.empty 避免重繪錯誤)
st.divider()
target = st.selectbox("選擇診斷標的:", df['代號'].tolist() if not df.empty else [])
status_text = st.empty()

if st.button("執行 AI 深度決策診斷"):
    row = df[df['代號'] == target].iloc[0]
    prompt = f"標的:{target},現價:{row['現價']},成本:{row['成本']},10MA:{row['10MA']},60MA:{row['60MA']}。請根據技術面與成本位置，給出加碼/減碼/停利/停損的明確建議。"
    
    status_text.info(f"正在與 AI 連線分析 {target}...")
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={st.secrets.get('GEMINI_API_KEY')}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15).json()
        st.session_state.analysis = res["candidates"][0]["content"]["parts"][0]["text"]
        status_text.empty()
    except:
        status_text.error("分析逾時，請檢查連線或 API 金鑰權限。")

if st.session_state.analysis:
    st.markdown(st.session_state.analysis)
