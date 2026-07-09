import streamlit as st
import google.generativeai as genai
import yfinance as yf

# 1. 直接設定金鑰
api_key = st.secrets.get("GEMINI_API_KEY", "")
genai.configure(api_key=api_key)

# 2. 核心修正：使用 generativeai 的最新穩定版本呼叫
def get_ai_response(prompt):
    # 改用 get_model 而不是 generate_content(model_name=...)
    # 這能避開版本路徑衝突
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

st.title("波段決策系統 V25.0")
s = st.selectbox("選股", ["2317", "2330", "2382"])
if st.button("分析"):
    try:
        # 這裡只傳送文字，避免傳送複雜的字典物件導致傳輸錯誤
        response = get_ai_response(f"分析股票 {s} 的波段策略")
        st.success(response)
    except Exception as e:
        st.error(f"連線細節: {e}")
