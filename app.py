import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

st.set_page_config(page_title="波段助手 V23.0", layout="wide")

# 1. API 強制設定 (移除所有模型版本限制)
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

# 2. 核心分析
def get_analysis(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="3mo").dropna()
        if df.empty: return None
        c = float(df['Close'].iloc[-1])
        return {"現價": round(c, 1), "趨勢": "上漲" if c > df['Close'].rolling(20).mean().iloc[-1] else "下跌"}
    except: return None

# 3. 介面
page = st.sidebar.radio("功能", ["💼 持股", "💬 AI 教練"])

if page == "💼 持股":
    st.title("💼 持股管理")
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame([{"代號": "2317", "名稱": "鴻海"}])
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

elif page == "💬 AI 教練":
    st.title("💬 AI 教練 (通用模式)")
    sel = st.selectbox("標的", st.session_state.portfolio['代號'].tolist())
    if st.button("🗣️ 分析"):
        data = get_analysis(sel)
        if not api_key: st.error("請輸入金鑰")
        else:
            try:
                # 使用最通用的模型名稱 gemini-pro，避開模型不存在錯誤
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"分析台股代號 {sel}，現價 {data['現價']}。請直接給出建議。"
                response = model.generate_content(prompt)
                st.success(response.text)
            except Exception as e:
                st.error(f"連線失敗，請檢查 API Key 是否有權限或嘗試重新申請: {e}")
