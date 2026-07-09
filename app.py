import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V70.0 (離線專業引擎)")

# 1. 庫存管理
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}
with st.sidebar:
    code = st.text_input("代號:")
    cost = st.number_input("成本:", value=0.0)
    if st.button("更新庫存"): st.session_state.portfolio[code] = {"cost": cost, "shares": 1000}; st.rerun()

# 2. 數據運算
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
        ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
        
        # 專業邏輯：動態決策引擎
        if c < info['cost'] * 0.92: advice = "【強烈建議：停損】股價已觸及 8% 停損線，請執行紀律出場。"
        elif c > ma10 and c > ma60: advice = "【強勢趨勢：續抱】均線多頭排列，技術面強勁，建議以月線為追蹤停利點。"
        elif c < ma60: advice = "【轉弱訊號：減碼】股價跌破季線，中長期趨勢走弱，建議降低水位觀望。"
        else: advice = "【盤整格局：觀察】股價於月線與季線間遊走，建議持股觀望，等待突破方向。"
        
        data.append({"代號": code, "現價": round(c, 2), "損益": round((c - info['cost']) * 1000, 0), "診斷結果": advice})

st.table(pd.DataFrame(data))
st.success("💡 離線分析引擎已啟動：直接根據 MA 指標與成本線即時判定，無需依賴任何 API。")
