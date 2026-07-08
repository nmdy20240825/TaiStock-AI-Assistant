import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

st.set_page_config(page_title="台股波段助手 V19.0", layout="wide")

# 1. API 金鑰與設定
api_key = st.secrets.get("GEMINI_API_KEY", st.sidebar.text_input("🔑 輸入 API 金鑰", type="password"))
if api_key: genai.configure(api_key=api_key)

# 2. 持久化數據存儲 (解決持股與自選股遺失問題)
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([{"代號": "2317", "名稱": "鴻海", "均價": 180.0, "股數": 1000}])
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = pd.DataFrame([{"代號": "3037", "名稱": "欣興"}])

# 3. 核心功能函式
def get_analysis(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="6mo").dropna()
        if df.empty: return None
        close = df['Close'].iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
        return {
            "現價": round(close, 1),
            "股性": "🚀 急漲" if vol_ratio > 1.5 else "🚶 一般",
            "起漲加碼": f"{round(close*0.96, 1)}~{round(close*0.99, 1)}",
            "短線停利": f"{round(close*1.04, 1)}~{round(close*1.07, 1)}"
        }
    except: return None

# 4. 選單邏輯
page = st.sidebar.radio("功能選單", ["🌅 晨報", "📡 雷達", "💼 我的持股", "⭐ 自選股", "💬 AI 教練"])

if page == "🌅 晨報":
    st.title("🌅 晨報")
    for _, row in st.session_state.portfolio.iterrows():
        data = get_analysis(row['代號'])
        if data: st.write(f"**{row['名稱']}**: 現價 {data['現價']} | {data['股性']}")

elif page == "📡 雷達":
    st.title("📡 全面雷達掃描")
    all_syms = list(set(st.session_state.portfolio['代號'].tolist() + st.session_state.watchlist['代號'].tolist()))
    if st.button("🚀 掃描所有標的"):
        for s in all_syms:
            data = get_analysis(s)
            if data: st.write(f"代號 {s} | {data}")

elif page == "💼 我的持股":
    st.title("💼 持股管理")
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

elif page == "⭐ 自選股":
    st.title("⭐ 自選股追蹤")
    st.session_state.watchlist = st.data_editor(st.session_state.watchlist, num_rows="dynamic")

elif page == "💬 AI 教練":
    st.title("🏋️‍♂️ AI 教練")
    all_s = [f"{r['代號']} {r['名稱']}" for _, r in st.session_state.portfolio.iterrows()] + \
            [f"{r['代號']} {r['名稱']}" for _, r in st.session_state.watchlist.iterrows()]
    sel = st.selectbox("選股", list(set(all_s)))
    sym = sel.split(" ")[0]
    if st.button("🗣️ 分析"):
        data = get_analysis(sym)
        if data:
            st.metric("起漲加碼", data['起漲加碼'])
            st.metric("短線停利", data['短線停利'])
            if api_key:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    st.success(model.generate_content(f"分析股票 {sym}，現價 {data['現價']}，區間 {data['起漲加碼']}。給我策略。").text)
                except Exception as e: st.error(f"連線異常: {e}")
