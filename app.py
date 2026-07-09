import streamlit as st
import requests

st.title("波段決策系統 V36.0")
s = st.selectbox("選股", ["2317", "2330", "2382"])

if st.button("AI 教練分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("請確認 Secrets")
    else:
        # 強制使用 gemini-1.5-flash，這是目前最穩定的版本
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        payload = {"contents": [{"parts": [{"text": f"分析股票 {s}"}]}]}
        
        try:
            response = requests.post(url, json=payload).json()
            if "candidates" in response:
                st.success(response["candidates"][0]["content"]["parts"][0]["text"])
            else:
                st.error(f"模型回應錯誤: {response}")
        except Exception as e:
            st.error(f"連線失敗: {e}")
