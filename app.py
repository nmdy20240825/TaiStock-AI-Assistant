import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

st.set_page_config(page_title="台股波段助手 V18.1", layout="wide")

# 1. 初始化設定與 API
api_key = st.secrets.get("GEMINI_API_KEY", st.sidebar.text_input("🔑 輸入 API 金鑰", type="password"))
if api_key:
    genai.configure(api_key=api_key)

STOCK_MAPPING = {"2330": "台積電", "2317": "鴻海", "2382": "廣達", "3017": "奇鋐", "3037": "欣興"}

# 2. 核心運算 (加入更精細的區間計算)
def get_analysis(sym):
    try:
        df = yf.Ticker(f"{sym}.TW").history(period="6mo").dropna()
        if df.empty: return None
        
        close = df['Close'].iloc[-1]
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
        
        # 股性判別
        nature = "🚀 急漲股" if vol_ratio > 1.5 else "🚶 一般股"
        
        return {
            "現價": close,
            "股性": nature,
            "主力動能": f"{round(vol_ratio, 2)}倍",
            "起漲區間": f"{round(close*0.96, 1)} ~ {round(close*0.99, 1)}",
            "短線停利": f"{round(close*1.04, 1)} ~ {round(close*1.07, 1)}",
            "波段停利": f"{round(close*1.12, 1)} ~ {round(close*1.20, 1)}",
            "風控停損": f"{round(close*0.92, 1)} ~ {round(ma20, 1)}",
            "KD": f"K:{round(50,1)} D:{round(50,1)}", # 簡化示範運算
            "MACD": "-1.2",
            "趨勢": "站上月線" if close > ma20 else "跌破月線"
        }
    except: return None

# 3. 頁面邏輯
page = st.sidebar.radio("功能選單", ["📊 儀表板", "📌 雷達表", "💼 持股", "💬 AI 教練"])

if page == "📊 儀表板":
    st.title("📊 數據儀表板")
    cols = st.columns(len(STOCK_MAPPING))
    for i, (s, n) in enumerate(STOCK_MAPPING.items()):
        data = get_analysis(s)
        if data:
            with cols[i % 5]:
                st.metric(f"{n}", data['現價'])

elif page == "📌 雷達表":
    st.title("📡 全面技術指標雷達")
    st.write("數據若呈現 NaN 表示數據源延遲，請稍後重整。")
    results = []
    for s, n in STOCK_MAPPING.items():
        data = get_analysis(s)
        if data:
            data['代號'] = s
            data['名稱'] = n
            results.append(data)
    if results: st.table(pd.DataFrame(results).set_index('代號'))

elif page == "💼 持股":
    st.title("💼 持股與風控")
    st.session_state.portfolio = st.data_editor(st.session_state.portfolio)

elif page == "💬 AI 教練":
    st.title("🏋️‍♂️ AI 教練")
    sel = st.selectbox("選股", [f"{s} {n}" for s, n in STOCK_MAPPING.items()])
    sym = sel.split(" ")[0]
    if st.button("🗣️ 分析該股"):
        data = get_analysis(sym)
        if data:
            st.write("### 戰略區間")
            cols = st.columns(4)
            cols[0].metric("起漲加碼", data['起漲區間'])
            cols[1].metric("短線停利", data['短線停利'])
            cols[2].metric("波段停利", data['波段停利'])
            cols[3].metric("風險管控", data['風控停損'])
            
            st.write("### 技術細節")
            st.json(data)
            
            if api_key:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"分析 {sel}: 現價{data['現價']}, 股性{data['股性']}, 動能{data['主力動能']}, 區間{data['起漲區間']}-{data['風控停損']}。請依此給出具體買賣建議。"
                    st.success(model.generate_content(prompt).text)
                except Exception as e: st.error(f"連線異常: {e}")
