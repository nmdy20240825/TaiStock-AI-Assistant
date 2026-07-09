import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="專業波段系統", layout="wide")
st.title("⚡ 波段決策儀表板 V76.0 (決策增強版)")

# 1. 庫存管理
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"2317": {"cost": 171.0}}
with st.sidebar:
    code = st.text_input("代號:")
    cost = st.number_input("成本:", value=0.0)
    if st.button("更新"): st.session_state.portfolio[code] = {"cost": cost}; st.rerun()

# 2. 專業分析引擎
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
        ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
        
        # 專業技術指標：乖離率 (Bias) 與 趨勢判讀
        bias = ((c - ma60) / ma60) * 100
        if c > ma10 and ma10 > ma60: trend = "強力多頭"
        elif c < ma60: trend = "空頭修正"
        else: trend = "震盪整理"
        
        data.append({
            "代號": code, "現價": round(c, 2), "月乖離": f"{bias:.1f}%",
            "技術趨勢": trend, "操作建議": "過熱減碼" if bias > 15 else "長線布局" if bias < -10 else "持有觀望"
        })

st.table(pd.DataFrame(data))
st.info("💡 系統已啟動無限制技術指標引擎。若需 AI 分析，請稍待 24 小時等待 Google 釋放您的 API 額度。")
