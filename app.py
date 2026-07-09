import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(layout="wide")
st.title("⚡ 波段決策儀表板 V75.0 (最終修正版)")

# 1. 庫存管理
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"2317": {"cost": 171.0}}
with st.sidebar:
    code = st.text_input("代號:")
    cost = st.number_input("成本:", value=0.0)
    if st.button("更新"): st.session_state.portfolio[code] = {"cost": cost}; st.rerun()

# 2. 表格
data = [{"代號": k, "現價": round(float(yf.download(f"{k}.TW", period="1d", progress=False)['Close'].iloc[-1].item()), 2), "成本": v['cost']} for k, v in st.session_state.portfolio.items()]
st.table(pd.DataFrame(data))

# 3. 診斷 (修正後的 URL 格式)
target = st.text_input("輸入要診斷的代號:")
if st.button("執行 AI 診斷"):
    api_key = st.secrets.get("GEMINI_API_KEY")
    # 這是正確的 URL 拼接方式，確保 model 名稱放在 path 中
    model_name = "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": f"分析股票 {target} 的技術面建議"}]}]}, timeout=15)
        if res.status_code == 200:
            st.markdown(res.json()["candidates"][0]["content"]["parts"][0]["text"])
        else:
            st.error(f"API 回應錯誤 (代碼 {res.status_code}): {res.json()}")
    except Exception as e:
        st.error(f"連線失敗: {str(e)}")
