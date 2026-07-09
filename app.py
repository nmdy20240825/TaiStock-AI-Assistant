import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

# 1. 頁面設定
st.set_page_config(page_title="波段助手 V24.0", layout="wide")

# 2. API 強制寫入與設定
# 這裡我們手動處理，確保程式不會因為抓不到 Secrets 而崩潰
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
else:
    api_key = st.sidebar.text_input("🔑 手動輸入 API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    # 不指定模型版本，讓 SDK 自動尋找可用的預設模型
    model = genai.GenerativeModel('gemini-pro')

# 3. 簡單數據分析 (確保絕對不會報型別錯誤)
def get_data(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="1mo")
        return round(float(df['Close'].iloc[-1]), 1)
    except: return None

# 4. 介面
st.title("波段決策系統 V24.0")
s = st.selectbox("選股", ["2317", "2330", "2382", "3017", "3037"])

if st.button("AI 教練分析"):
    price = get_data(s)
    if not api_key:
        st.error("請在左側輸入 API Key")
    elif not price:
        st.error("數據獲取失敗")
    else:
        try:
            # 這是最簡單的 prompt，確保不傳遞複雜字典數據，避免參數錯誤
            prompt = f"我是波段交易者，股票 {s} 現價 {price}。請給出操作建議。"
            response = model.generate_content(prompt)
            st.success(response.text)
        except Exception as e:
            st.error(f"AI 連線失敗 (錯誤碼: {e})")
