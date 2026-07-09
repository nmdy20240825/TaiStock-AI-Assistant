import streamlit as st
import requests

st.title("波段決策系統 V37.0")
s = st.selectbox("選股", ["2317", "2330", "2382", "3017", "3037"])

if st.button("AI 教練分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("請確認 Secrets")
    else:
        # 使用 v1beta 路徑搭配 gemini-pro，這是 Google AI Studio 最通用的穩定組合
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"分析股票 {s}"}]}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            
            if "candidates" in result:
                st.success(result["candidates"][0]["content"]["parts"][0]["text"])
            else:
                # 輸出詳細錯誤以便診斷
                st.error(f"API 回應: {result}")
        except Exception as e:
            st.error(f"系統錯誤: {e}")
