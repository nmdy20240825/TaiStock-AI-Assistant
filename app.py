import streamlit as st
import google.generativeai as genai

# 1. 初始化設定
api_key = st.secrets.get("GEMINI_API_KEY", "")

if api_key:
    # 這裡只保留最基本的 api_key 參數，移除所有額外參數以避開 TypeError
    genai.configure(api_key=api_key)
    # 使用正確的 model 物件初始化方式
    model = genai.GenerativeModel('gemini-1.5-flash')

st.title("波段決策系統 V27.0")
s = st.selectbox("選股", ["2317", "2330", "2382"])

if st.button("AI 分析"):
    if not api_key:
        st.error("請在 Secrets 中設定 GEMINI_API_KEY")
    else:
        try:
            # 發送請求
            response = model.generate_content(f"請為股票 {s} 提供波段交易建議")
            st.success(response.text)
        except Exception as e:
            st.error(f"AI 執行錯誤: {e}")
