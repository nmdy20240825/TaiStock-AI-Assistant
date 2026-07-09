import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V69.0 (最終穩定版)")

# 1. 庫存管理
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    if st.button("更新庫存"): st.session_state.portfolio[code] = {"cost": cost, "shares": 1000}; st.rerun()

# 2. 顯示表格
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        data.append({"代號": code, "現價": round(c, 2), "成本": info['cost'], "損益": round((c - info['cost']) * 1000, 0)})
st.table(pd.DataFrame(data))

# 3. 穩定 AI 診斷 (使用 gemini-pro)
st.divider()
target = st.text_input("輸入要診斷的代號:")
if st.button("執行 AI 診斷"):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={st.secrets.get('GEMINI_API_KEY')}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": f"分析 {target} 技術面建議"}]}]}, timeout=15).json()
        st.markdown(res["candidates"][0]["content"]["parts"][0]["text"])
    except:
        st.error("模型切換後仍失敗，請確認您的 API Key 是否在 Google AI Studio 有啟用 Gemini Pro 權限。")
