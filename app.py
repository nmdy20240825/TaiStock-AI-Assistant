import streamlit as st
import pandas as pd
import yfinance as yf

# 1. 設置
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V65.0 (自動化判讀版)")

# 2. 側邊欄維護
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
        st.rerun()

# 3. 自動決策邏輯 (無須 API)
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
        ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
        
        # 自動化判斷邏輯
        if c < ma60 * 0.92: decision = "🚨 跌破停損(8%)"
        elif c < ma60: decision = "⚠️ 修正中(季線下)"
        elif c > ma10: decision = "✅ 多頭強勢"
        else: decision = "⚖️ 箱型整理"
            
        data.append({
            "代號": code, "現價": round(c, 2), "成本": round(info['cost'], 2),
            "損益": round((c - info['cost']) * info['shares'], 0),
            "10MA": round(ma10, 2), "60MA": round(ma60, 2), "決策建議": decision
        })

# 4. 顯示結果
st.table(pd.DataFrame(data))
st.info("💡 系統已自動判讀季線與停損位置，無需額外連線即可執行。")
