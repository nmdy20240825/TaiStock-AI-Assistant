import streamlit as st
import requests

st.title("波段決策系統 V38.0 (診斷模式)")
s = st.selectbox("選股", ["2317", "2330", "2382"])

if st.button("AI 教練分析"):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    # 改用列出所有模型端點，這不會報 NOT_FOUND，而是會告訴你帳號有多少權限
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        response = requests.get(url).json()
        if "models" in response:
            model_names = [m["name"] for m in response["models"]]
            st.write("您的帳號目前可存取的模型列表：")
            st.write(model_names)
        else:
            st.error(f"無法取得模型列表: {response}")
    except Exception as e:
        st.error(f"連線細節: {e}")
