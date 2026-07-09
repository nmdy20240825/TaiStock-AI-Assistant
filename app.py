import streamlit as st
import requests

st.set_page_config(page_title="TaiStock 最終版", layout="wide")

# 1. 直接讀取 API Key (請確保 Secrets 名稱完全正確)
api_key = st.secrets.get("GEMINI_API_KEY", "")

st.title("波段決策系統 V30.0")
s = st.selectbox("選股", ["2317", "2330", "2382", "3017", "3037"])

if st.button("AI 教練分析"):
    if not api_key:
        st.error("請確認 Secrets 中是否有 GEMINI_API_KEY")
    else:
        # 使用 REST API 直接存取正式版路徑，完全避開 SDK 的 v1beta 衝突
        # 這是您在 Playground 測試成功的方式
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"你是專業台股教練，請為股票 {s} 提供具體的波段交易建議，包含：股性判斷、加碼區、停利停損區。"}]}]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            if "candidates" in result:
                st.success(result["candidates"][0]["content"]["parts"][0]["text"])
            else:
                st.error(f"AI 回應失敗，訊息: {result}")
        except Exception as e:
            st.error(f"系統錯誤: {e}")
