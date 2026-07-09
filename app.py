import streamlit as st
import google.generativeai as genai

# 強制重新初始化
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    # 關鍵修正：直接指定 API 版本，避開系統預設的 v1beta
    genai.configure(api_key=api_key, api_version='v1')
    model = genai.GenerativeModel('gemini-1.5-flash')

st.title("波段決策系統 V26.0")
s = st.selectbox("選股", ["2317", "2330", "2382"])

if st.button("分析"):
    if not api_key:
        st.error("請檢查 Secrets 中的 GEMINI_API_KEY")
    else:
        try:
            # 直接進行呼叫，不再透過複雜函式
            response = model.generate_content(f"請為股票 {s} 提供波段交易建議")
            st.success(response.text)
        except Exception as e:
            st.error(f"連線細節: {e}")
