import streamlit as st
import requests

st.set_page_config(page_title="TaiStock 最終穩定版", layout="wide")

api_key = st.secrets.get("GEMINI_API_KEY", "")

st.title("波段決策系統 V31.0")
s = st.selectbox("選股", ["2317", "2330", "2382", "3017", "3037"])

if st.button("AI 教練分析"):
    if not api_key:
        st.error("請確認 Secrets 中是否有 GEMINI_API_KEY")
    else:
        # 【關鍵修復】：將網址路徑中的 v1beta 改為 v1
        # 這會強制呼叫正式環境，不再去測試環境找模型
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        
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
            st.error(f"連線細節: {e}")
