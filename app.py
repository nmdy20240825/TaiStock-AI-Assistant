import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

st.set_page_config(page_title="TaiStock 波段助手", layout="wide")

# --- 1. API 與模型初始化 ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)
    # 強制使用穩定的 1.5-flash 模型
    model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. 數據存儲 ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([{"代號": "2317", "名稱": "鴻海", "成本": 180.0, "股數": 1000}])

# --- 3. 分析引擎 ---
def get_analysis(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="3mo").dropna()
        if df.empty: return None
        c = float(df['Close'].iloc[-1])
        ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])
        return {
            "現價": round(c, 1),
            "月線": round(ma20, 1),
            "股性": "🚀 急漲" if vol_ratio > 1.5 else "🚶 一般",
            "起漲": f"{round(c*0.96, 1)}~{round(c*0.99, 1)}",
            "停利": f"{round(c*1.05, 1)}~{round(c*1.08, 1)}"
        }
    except: return None

# --- 4. 介面規劃 ---
page = st.sidebar.radio("功能導航", ["📊 Dashboard", "💼 持股管理", "📡 AI 雷達", "💬 AI 教練"])

if page == "📊 Dashboard":
    st.title("📊 市場總覽")
    for _, row in st.session_state.portfolio.iterrows():
        res = get_analysis(row['代號'])
        if res: st.metric(row['名稱'], res['現價'], res['股性'])

elif page == "💼 持股管理":
    st.title("💼 持股管理")
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

elif page == "📡 AI 雷達":
    st.title("📡 掃描強勢股")
    if st.button("🚀 執行掃描"):
        for _, row in st.session_state.portfolio.iterrows():
            res = get_analysis(row['代號'])
            if res: st.write(f"**{row['名稱']}** | 現價:{res['現價']} | {res['股性']}")

elif page == "💬 AI 教練":
    st.title("💬 AI 教練決策")
    sel = st.selectbox("選股", st.session_state.portfolio['名稱'].tolist())
    sym = st.session_state.portfolio.loc[st.session_state.portfolio['名稱']==sel, '代號'].values[0]
    if st.button("🗣️ AI 分析"):
        res = get_analysis(sym)
        if not res: st.error("無數據")
        elif not api_key: st.error("請在雲端後台設定 GEMINI_API_KEY")
        else:
            try:
                prompt = f"分析 {sel}({sym})：現價{res['現價']}，月線{res['月線']}，股性{res['股性']}。請給出具體買賣策略。"
                st.success(model.generate_content(prompt).text)
            except Exception as e: st.error(f"AI 錯誤: {e}")
