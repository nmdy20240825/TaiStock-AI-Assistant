import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V68.0 (金鑰診斷版)")

# 1. 強制檢查 API Key
api_key = st.secrets.get("GEMINI_API_KEY")
st.sidebar.subheader("API 金鑰診斷")
if api_key:
    st.sidebar.success(f"金鑰已載入: {api_key[:4]}****")
else:
    st.sidebar.error("❌ 找不到 GEMINI_API_KEY！請檢查 Streamlit Cloud 的 Secrets 設定。")

# 2. 診斷功能
target = st.text_input("輸入要診斷的代號:")
if st.button("執行 AI 分析") and api_key:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        # 故意簡化到最原始的格式，檢查回應內容
        res = requests.post(url, json={"contents": [{"parts": [{"text": "你好"}]}]}, timeout=10)
        
        st.write("伺服器完整回應內容：")
        st.json(res.json()) # 這一行能直接把 Google 給您的錯誤訊息顯現出來
        
    except Exception as e:
        st.error(f"連線細節錯誤: {str(e)}")
