import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

st.set_page_config(page_title="波段助手 V17.0", layout="wide")

# API 金鑰設定
api_key = st.secrets.get("GEMINI_API_KEY", st.sidebar.text_input("🔑 輸入 API 金鑰", type="password"))
if api_key:
    genai.configure(api_key=api_key)

# 股票清單
STOCK_MAPPING = {"2330": "台積電", "2317": "鴻海", "2382": "廣達", "3017": "奇鋐", "3037": "欣興"}

def get_data(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="3mo")
        return df.dropna()
    except: return pd.DataFrame()

def compute(df):
    close = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    return {
        "現價": round(close, 1),
        "起漲": f"{round(close*0.95, 1)} ~ {round(close*0.98, 1)}",
        "短停": f"{round(close*1.05, 1)} ~ {round(close*1.08, 1)}"
    }

page = st.sidebar.radio("選單", ["📊 雷達掃描", "💬 AI 教練"])

if page == "📊 雷達掃描":
    st.title("📡 盤後雷達")
    if st.button("🚀 開始掃描"):
        for s, n in STOCK_MAPPING.items():
            df = get_data(s)
            if not df.empty:
                res = compute(df)
                st.write(f"**{s} {n}** | 現價: {res['現價']}")
                st.info(f"建議區間: 加碼{res['起漲']} | 停利{res['短停']}")

elif page == "💬 AI 教練":
    st.title("🏋️‍♂️ AI 教練")
    sel = st.selectbox("選股", [f"{s} {n}" for s, n in STOCK_MAPPING.items()])
    sym = sel.split(" ")[0]
    
    if st.button("🗣️ 呼叫教練"):
        if not api_key: st.error("請先在左側欄輸入或設定 API 金鑰")
        else:
            df = get_data(sym)
            if df.empty: st.error("無數據")
            else:
                res = compute(df)
                st.write(f"現價: {res['現價']}")
                try:
                    # 使用最穩定路徑呼叫模型
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"你是操盤手。分析股票 {sym}，現價 {res['現價']}，建議加碼區 {res['起漲']}，建議短線停利區 {res['短停']}。請直接給我操作策略。"
                    response = model.generate_content(prompt)
                    st.success(response.text)
                except Exception as e:
                    st.error(f"AI 連線錯誤: {e}")
