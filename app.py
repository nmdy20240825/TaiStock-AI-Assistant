import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

# 1. 強制設定
st.set_page_config(page_title="波段助手 V21.0", layout="wide")

# 2. API 設定 - 使用最簡單的配置方式
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("🔑 輸入 API 金鑰", type="password")

if api_key:
    genai.configure(api_key=api_key)

# 3. 資料處理核心
def get_stock_data(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="3mo").dropna()
        c = df['Close'].iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        return {
            "現價": round(c, 1),
            "月線": round(ma20, 1),
            "動能": "🚀 急漲" if df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] else "🚶 一般"
        }
    except: return None

# 4. 介面
page = st.sidebar.radio("功能", ["📊 數據儀表板", "💬 AI 教練"])

if page == "📊 數據儀表板":
    st.title("📊 市場監控")
    for s in ["2317", "2330", "2382", "3017", "3037"]:
        data = get_stock_data(s)
        if data:
            st.write(f"**代號 {s}** | 現價: {data['現價']} | {data['動能']}")

elif page == "💬 AI 教練":
    st.title("💬 AI 教練決策")
    sym = st.selectbox("標的", ["2317", "2330", "2382", "3017", "3037"])
    if st.button("分析"):
        data = get_stock_data(sym)
        if data:
            st.write(f"數據: {data}")
            if api_key:
                try:
                    # 改用最通用的 gemini-pro
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"你是操盤教練，分析 {sym}。現價 {data['現價']}，月線 {data['月線']}。請回覆：1.加減碼建議 2.停損停利點。"
                    st.success(model.generate_content(prompt).text)
                except Exception as e:
                    st.error(f"AI 連線設定錯誤，請檢查金鑰或重新申請: {e}")
