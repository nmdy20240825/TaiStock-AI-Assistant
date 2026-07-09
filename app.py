import streamlit as st
import requests

st.title("波段決策系統 V35.0")
s = st.selectbox("選股", ["2317", "2330", "2382", "3017", "3037"])

if st.button("AI 教練分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("請檢查 Secrets")
    else:
        # 使用通用路徑，讓 API 自動尋找帳號下可用的模型
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
                st.error(f"錯誤代碼: {result.get('error', {}).get('code', '未知')} - {result.get('error', {}).get('message', '請嘗試更換模型名稱')}")
        except Exception as e:
            st.error(f"連線細節: {e}")
