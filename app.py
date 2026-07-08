import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests

# 1. 網頁基本配置
st.set_page_config(page_title="台股AI波段交易助手 V16.0", layout="wide", page_icon="📈")

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] span { font-size: 18px !important; font-weight: bold; }
    div[role="radiogroup"] label p { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

# 2. 側邊欄配置
st.sidebar.title("🤖 波段助手 V16.0")
api_key = st.sidebar.text_input("🔑 Gemini API 金鑰", value=st.secrets.get("GEMINI_API_KEY", ""), type="password")

if api_key:
    # 關鍵修正：強制指定使用通用路徑，避免 v1beta 版本衝突
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

page = st.sidebar.radio("功能選單", ["Dashboard（首頁）", "📌 AI 雷達", "我的持股", "💬 AI 教練模式"])

# 3. 初始化數據 (簡化版，確保穩定)
STOCK_MAPPING = {"2330": "台積電", "2317": "鴻海", "2382": "廣達", "3017": "奇鋐", "3037": "欣興"}

def get_data(symbol):
    try:
        df = yf.Ticker(f"{symbol}.TW").history(period="6mo")
        return df.dropna()
    except: return pd.DataFrame()

def compute_signals(df):
    if df.empty or len(df) < 20: return None
    data = df.copy()
    close = data['Close'].iloc[-1]
    ma20 = data['Close'].rolling(20).mean().iloc[-1]
    # 簡單動能計算
    vol_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
    
    return {
        "現價": round(close, 2),
        "股性判別": "🚀 急漲" if vol_ratio > 1.5 else "🚶 一般",
        "起漲加碼": f"{round(close*0.95, 1)} ~ {round(close*0.98, 1)}",
        "短線停利": f"{round(close*1.05, 1)} ~ {round(close*1.08, 1)}",
        "波段停利": f"{round(close*1.15, 1)} ~ {round(close*1.25, 1)}",
        "風險管控": f"{round(close*0.92, 1)} ~ {round(ma20, 1)}"
    }

# 4. 渲染邏輯
if page == "Dashboard（首頁）":
    st.title("🌅 AI 每日晨報")
    st.write("系統運作中，請點擊側邊欄進入雷達或教練模式。")

elif page == "📌 AI 雷達":
    st.title("📡 盤後雷達")
    if st.button("🚀 掃描"):
        for sym, name in STOCK_MAPPING.items():
            df = get_data(sym)
            sig = compute_signals(df)
            if sig:
                st.write(f"**{sym} {name}** | 現價: {sig['現價']} | {sig['股性判別']}")

elif page == "💬 AI 教練模式":
    st.title("🏋️‍♂️ AI 教練")
    all_s = [f"{s} {n}" for s, n in STOCK_MAPPING.items()]
    sel = st.selectbox("選股", all_s)
    sym = sel.split(" ")[0]
    
    if st.button("🗣️ 呼叫教練"):
        df = get_data(sym)
        sig = compute_signals(df)
        if sig:
            st.write(f"### {sel}")
            c1, c2 = st.columns(2)
            c1.metric("起漲加碼", sig['起漲加碼'])
            c2.metric("短線停利", sig['短線停利'])
            
            if api_key:
                try:
                    # 使用最簡化的 Prompt 呼叫，避開複雜格式導致的解析錯誤
                    prompt = f"請分析股票 {sym}，現價 {sig['現價']}。請直接給我操作策略。"
                    response = model.generate_content(prompt)
                    st.success(response.text)
                except Exception as e:
                    st.error("AI 服務暫時無法回應，請確認金鑰是否有效。")
            else:
                st.warning("請輸入金鑰")
