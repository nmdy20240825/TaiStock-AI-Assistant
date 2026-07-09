import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 離線對照表 (您可以隨時補充)
NAME_MAP = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "2308": "台達電", "3035": "智原", "3443": "創意"}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V59.0 (極速穩定版)")

# 2. 側邊欄：手動管理
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
        st.rerun()

# 3. 極速計算 (只抓收盤價，不抓 info)
data = []
for code, info in st.session_state.portfolio.items():
    # 使用 download 只抓必要數據，速度最快
    df = yf.download(f"{code}.TW", period="1y", progress=False)
    if not df.empty:
        curr = df['Close'].iloc[-1]
        ma10 = df['Close'].rolling(window=10).mean().iloc[-1]
        ma60 = df['Close'].rolling(window=60).mean().iloc[-1]
        
        data.append({
            "代號": code, 
            "名稱": NAME_MAP.get(code, code), # 沒對應到就直接顯示代號
            "現價": float(curr), 
            "成本": info['cost'],
            "帳面損益": (float(curr) - info['cost']) * info['shares'],
            "10MA": float(ma10), "60MA": float(ma60)
        })

df = pd.DataFrame(data)
# 格式化顯示 (小數點二位)
st.table(df.style.format("{:.2f}", subset=['現價', '成本', '帳面損益', '10MA', '60MA']))

# 4. 極簡 AI 診斷
st.divider()
target = st.selectbox("選擇標的:", df['代號'].tolist())
if st.button("執行 AI 診斷"):
    row = df[df['代號'] == target].iloc[0]
    prompt = f"代號{target},現價{row['現價']:.2f},成本{row['成本']:.2f},10MA{row['10MA']:.2f},60MA{row['60MA']:.2f}。請給策略。"
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={st.secrets.get('GEMINI_API_KEY')}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8).json()
        st.write(res["candidates"][0]["content"]["parts"][0]["text"])
    except:
        st.error("分析中斷，請再次點擊。")
