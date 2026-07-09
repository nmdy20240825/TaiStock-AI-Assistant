import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V72.0 (全兼容版)")

# 1. 庫存管理
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"2317": {"cost": 171.0}}
with st.sidebar:
    code = st.text_input("代號:")
    cost = st.number_input("成本:", value=0.0)
    if st.button("更新庫存"): st.session_state.portfolio[code] = {"cost": cost}; st.rerun()

# 2. 數據顯示
data = [{"代號": k, "現價": round(float(yf.download(f"{k}.TW", period="1d", progress=False)['Close'].iloc[-1].item()), 2), "成本": v['cost']} for k, v in st.session_state.portfolio.items()]
st.table(pd.DataFrame(data))

# 3. 診斷通道：切換為 gemini-pro (這是最老牌且最穩定的模型名稱)
st.divider()
target = st.text_input("輸入要診斷的代號:")
if st.button("執行 AI 深度診斷"):
    api_key = st.secrets.get("GEMINI_API_KEY")
    # 注意：這裡刻意使用 gemini-pro，這是 Google API 中最不容易報 404 的名稱
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": f"請以專家角度分析 {target} 技術面建議"}]}]}, timeout=10)
        if res.status_code == 200:
            st.markdown(res.json()["candidates"][0]["content"]["parts"][0]["text"])
        else:
            st.error(f"連線細節錯誤: {res.text}")
    except Exception as e:
        st.error(f"連線失敗: {str(e)}")
