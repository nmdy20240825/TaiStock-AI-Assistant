import streamlit as st
import pandas as pd
import yfinance as yf

# 1. 設置
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}, "2330": {"cost": 990.0, "shares": 47}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V66.0 (決策邏輯透明版)")

# 2. 側邊欄
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
        st.rerun()

# 3. 邏輯判讀引擎
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        ma10 = float(df['Close'].rolling(10).mean().iloc[-1].item())
        ma60 = float(df['Close'].rolling(60).mean().iloc[-1].item())
        
        # 核心判斷邏輯與原因說明
        if c < info['cost'] * 0.92:
            decision, reason = "🚨 停損", "股價跌破買進成本 8% 風險線"
        elif c < ma60:
            decision, reason = "⚠️ 修正", "股價低於季線(MA60)，轉為中線整理格局"
        elif c > ma10:
            decision, reason = "✅ 強勢", "站上月線(MA10)且趨勢向上，動能強勁"
        else:
            decision, reason = "⚖️ 整理", "股價位於月線與季線之間，等待方向確認"
            
        data.append({
            "代號": code, "現價": round(c, 2), "帳面損益": round((c - info['cost']) * info['shares'], 0),
            "10MA": round(ma10, 2), "60MA": round(ma60, 2), "建議": decision, "診斷依據": reason
        })

# 4. 顯示結果
st.table(pd.DataFrame(data))
st.info("💡 系統邏輯說明：採用『月線強弱』與『季線支撐』雙指標，配合您的成本進行即時自動診斷。")
