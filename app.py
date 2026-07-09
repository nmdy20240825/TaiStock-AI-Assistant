import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("⚡ 波段決策儀表板 V73.0 (金鑰權限診斷)")

api_key = st.secrets.get("GEMINI_API_KEY")

if st.button("檢測我的金鑰可用模型"):
    # 這是 Google API 官方獲取模型列表的接口
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        res = requests.get(url, timeout=10).json()
        model_names = [m['name'] for m in res.get('models', [])]
        st.success("以下是您的金鑰目前真正能使用的模型名稱，請挑選一個填入程式碼：")
        st.write(model_names)
    except Exception as e:
        st.error(f"連線失敗，請檢查金鑰：{str(e)}")
