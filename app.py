import streamlit as st
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 進階決策系統")

# --- 讀取/儲存 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- 決策解釋引擎 (新功能模組) ---
def get_decision_reason(score, macd, ma20_up, volume_up):
    reasons = []
    if macd > 0: reasons.append("① MACD 黃金交叉 (趨勢向上)")
    if ma20_up: reasons.append("② MA20 向上 (中線走強)")
    if volume_up: reasons.append("③ 成交量放大 (買盤進駐)")
    if score >= 25: 
        status = "續抱"
    else: 
        status = "風險控管"
        reasons.append("注意：指標動能轉弱，建議優先保護資本")
    return status, reasons

# --- 側邊欄 ---
portfolio = load_portfolio()
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock"):
        new_code = st.text_input("代號")
        new_name = st.text_input("名稱")
        new_cost = st.number_input("成本", value=100.0)
        if st.form_submit_button("新增/更新"):
            portfolio[new_code] = [new_name, new_cost]
            save_portfolio(portfolio)
            st.rerun()
    del_code = st.selectbox("刪除", [""] + list(portfolio.keys()))
    if st.button("確認刪除"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

# --- 核心引擎 ---
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        c = [float(x) for x in df.iloc[:, 3]]
        v = [float(x) for x in df.iloc[:, 4]]
        ma10, ma20, ma60 = sum(c[-10:])/10, sum(c[-20:])/20, sum(c[-60:])/60
        macd = (sum(c[-12:])/12) - (sum(c[-26:])/26)
        
        # 決策判斷條件
        ma20_up = c[-1] > ma20
        volume_up = v[-1] > (sum(v[-5:])/5 * 1.2)
        score = (25 if c[-1] > ma60 else 10) + (10 if macd > 0 else 0)
        
        # 取得決策解釋
        status, reasons = get_decision_reason(score, macd, ma20_up, volume_up)
        profit = ((c[-1] - cost) / cost) * 100
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{c[-1]:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{profit:.1f}%")
            c3.metric("AI評分", f"{score}分")
            
            with st.expander("👉 決策解釋引擎"):
                st.write(f"**建議**: {status}")
                st.write("**原因**:")
                for r in reasons: st.write(r)
                st.write("---")
                st.write(f"均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"股性: {'起漲股' if (c[-1]/ma20) > 1.15 else '一般股'}")
    except: continue

st.divider()
st.subheader("📊 AI 評分標準")
st.write("25分+: 強勢(續抱) | 15-20分: 穩健 | 10分以下: 風險控管")
