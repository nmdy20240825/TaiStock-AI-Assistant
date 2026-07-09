import streamlit as st
import requests

st.title("波段決策系統 V34.0")
s = st.selectbox("選股", ["2317", "2330", "2382", "3017", "3037"])

if st.button("AI 教練分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("請確認 Secrets 中是否有 GEMINI_API_KEY")
    else:
        # 【關鍵修復】：使用最標準的通用路徑
        # 移除了所有不必要的路徑版本參數，讓 Google 伺服器自動導向
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"你是專業台股教練，請為股票 {s} 提供波段交易建議"}]}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            
            if "candidates" in result:
                st.success(result["candidates"][0]["content"]["parts"][0]["text"])
            else:
                st.error(f"AI 回應失敗 (狀態: {result})")
        except Exception as e:
            st.error(f"系統錯誤: {e}")
