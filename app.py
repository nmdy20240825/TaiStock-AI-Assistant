import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 股票名稱強制中文化字典
NAME_MAP = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "3711": "日月光投控", "2308": "台達電"}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("📈 波段決策儀表板 V58.0 (專業數據版)")

# 2. 側邊欄：手動管理庫存
with st.sidebar:
    st.header("個人庫存維護")
    code = st.text_input("代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1)
    if st.button("加入/更新"):
        st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
        st.rerun()

# 3. 數據運算
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.Ticker(f"{code}.TW").history(period="1y")
    curr = df.iloc[-1]
    # 計算指標
    ma10 = df['Close'].rolling(window=10).mean().iloc[-1]
    ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma60 = df['Close'].rolling(window=60).mean().iloc[-1]
    
    data.append({
        "代號": code, "名稱": NAME_MAP.get(code, "未知"), "現價": round(curr['Close'], 2),
        "成本": info['cost'], "股數": info['shares'],
        "帳面損益": round((curr['Close'] - info['cost']) * info['shares'], 0),
        "損益%": round((curr['Close'] - info['cost']) / info['cost'] * 100, 2),
        "10MA": round(ma10, 2), "20MA": round(ma20, 2), "60MA": round(ma60, 2)
    })

df = pd.DataFrame(data)
st.table(df)

# 4. 極簡化 AI 診斷 (大幅提升速度)
st.divider()
target = st.selectbox("選擇診斷標的:", df['代號'].tolist())
if st.button("✨ 執行 AI 決策診斷"):
    row = df[df['代號'] == target].iloc[0]
    prompt = f"標的:{target},現價:{row['現價']},成本:{row['成本']},MA10:{row['10MA']},MA20:{row['20MA']},MA60:{row['60MA']}。請提供明確策略:加碼/減碼/停利/停損。"
    
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10).json()
        st.write(res["candidates"][0]["content"]["parts"][0]["text"])
    except:
        st.error("分析延遲，請再次點擊。")
