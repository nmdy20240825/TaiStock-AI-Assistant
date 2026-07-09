import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

# 設定頁面，針對手機端優化
st.set_page_config(page_title="TaiStock AI 助手", layout="centered")

# --- 設定 API ---
api_key = st.secrets.get("GEMINI_API_KEY", st.sidebar.text_input("🔑 API Key", type="password"))
if api_key: genai.configure(api_key=api_key)

# --- 核心資料存儲 ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([{"代號": "2317", "名稱": "鴻海", "成本": 180.0, "股數": 1000}])
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = pd.DataFrame([{"代號": "3037", "名稱": "欣興"}])

# --- 功能函式 ---
def get_data(sym):
    try: return yf.Ticker(f"{sym}.TW").history(period="3mo").dropna()
    except: return pd.DataFrame()

def analyze(sym):
    df = get_data(sym)
    if df.empty: return None
    c = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    # 簡易量化模型
    return {
        "現價": round(c, 1),
        "股性": "🚀 急漲" if df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] * 1.5 else "🚶 一般",
        "加碼區": f"{round(c*0.96, 1)}~{round(c*0.99, 1)}",
        "停利區": f"{round(c*1.05, 1)}~{round(c*1.10, 1)}",
        "風控區": f"{round(c*0.92, 1)}~{round(ma20, 1)}"
    }

# --- 側邊選單 ---
page = st.sidebar.radio("Navigation", ["🌅 Dashboard", "💼 持股", "📡 AI 雷達", "💬 教練"])

# --- 頁面內容 ---
if page == "🌅 Dashboard":
    st.title("🌅 TaiStock 儀表板")
    st.info("目標：5分鐘內完成決策")
    for _, row in st.session_state.portfolio.iterrows():
        res = analyze(row['代號'])
        if res: st.write(f"**{row['名稱']}** (現價 {res['現價']}) | {res['股性']}")

elif page == "💼 持股":
    st.title("💼 持股管理")
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic")

elif page == "📡 AI 雷達":
    st.title("📡 掃描強勢股")
    if st.button("🚀 啟動掃描"):
        for s in list(set(st.session_state.portfolio['代號'].tolist() + st.session_state.watchlist['代號'].tolist())):
            res = analyze(s)
            if res: st.write(f"代號 {s}: {res['股性']} | 加碼: {res['加碼區']}")

elif page == "💬 教練":
    st.title("💬 AI 教練決策")
    sel = st.selectbox("選擇標的", list(set(st.session_state.portfolio['代號'].tolist())))
    if st.button("🗣️ AI 輸出操作指令"):
        res = analyze(sel)
        if api_key:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"分析股票 {sel}, 現價 {res['現價']}, 股性 {res['股性']}, 加碼區 {res['加碼區']}, 風控區 {res['風控區']}。請給出：加碼/減碼/續抱/停利/停損 指令。"
            st.success(model.generate_content(prompt).text)
