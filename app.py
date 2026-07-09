import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 離線對照表
NAME_MAP = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "2308": "台達電", "3711": "日月光投控", "6409": "旭隼"}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V62.0 (強韌修正版)")

# 2. 側邊欄
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        if code:
            st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
            st.rerun()

# 3. 數據計算 (全面使用 .item() 進行類型轉換)
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="1y", progress=False)
    if not df.empty:
        # 使用 .item() 提取純數值
        val_close = df['Close'].iloc[-1].item()
        val_ma10 = df['Close'].rolling(10).mean().iloc[-1].item()
        val_ma60 = df['Close'].rolling(60).mean().iloc[-1].item()
        
        data.append({
            "代號": code, 
            "名稱": NAME_MAP.get(code, code), 
            "現價": round(float(val_close), 2), 
            "成本": round(float(info['cost']), 2),
            "損益": round(float((val_close - info['cost']) * info['shares']), 0),
            "10MA": round(float(val_ma10), 2), 
            "60MA": round(float(val_ma60), 2)
        })

df = pd.DataFrame(data)
st.table(df)

# 4. AI 診斷
st.divider()
target = st.selectbox("選擇診斷標的:", df['代號'].tolist() if not df.empty else [])
if st.button("執行 AI 診斷"):
    row = df[df['代號'] == target].iloc[0]
    prompt = f"代號{target},現價{row['現價']},成本{row['成本']},10MA{row['10MA']},60MA{row['60MA']}。請給決策建議。"
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={st.secrets.get('GEMINI_API_KEY')}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10).json()
        st.write(res["candidates"][0]["content"]["parts"][0]["text"])
    except:
        st.error("分析延遲，請再次點擊。")
