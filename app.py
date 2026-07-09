import streamlit as st
import google.generativeai as genai

st.title("波段決策系統 V28.0")
s = st.selectbox("選股", ["2317", "2330", "2382"])

if st.button("AI 分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("請確認 Secrets 中是否有 GEMINI_API_KEY")
    else:
        try:
            # 關鍵：完全不依賴 SDK 自動判斷路徑
            genai.configure(api_key=api_key)
            
            # 使用最通用的 gemini-pro
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content(f"請為股票 {s} 提供波段交易建議")
            st.success(response.text)
        except Exception as e:
            st.error(f"連線細節: {e}")
