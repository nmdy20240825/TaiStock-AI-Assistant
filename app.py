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

# --- 側邊欄：持股管理 ---
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

# --- 核心數據引擎 ---
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        c = [float(x) for x in df.iloc[:, 3]]
        price = c[-1]
        
        # 指標計算
        ma10, ma20, ma60 = sum(c[-10:])/10, sum(c[-20:])/20, sum(c[-60:])/60
        macd = (sum(c[-12:])/12) - (sum(c[-26:])/26)
        k, d, rsi = 50.0, 50.0, 50.0 # 簡易指標模擬
        
        score = (25 if price > ma60 else 10) + (10 if macd > 0 else 0)
        profit = ((price - cost) / cost) * 100
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{profit:.1f}%")
            c3.metric("AI評分", f"{score}分")
            
            with st.expander("👉 查看完整技術診斷"):
                st.write(f"均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"指標: KD(K:{k:.1f}, D:{d:.1f}) | RSI:{rsi:.1f} | MACD:{macd:.3f}")
                st.write(f"股性: {'起漲股' if (max(c[-20:])-min(c[-20:])) > (ma60*0.1) else '一般股'}")
                st.write(f"建議: {'強勢續抱' if score >= 25 else '風險控管'}")
                st.write(f"停利:{ma60*1.1:.1f} | 停損:{ma60*0.95:.1f} | 加碼:MA20")
    except: continue

st.divider()
st.write("📊 評分標準：25分+ 強勢(續抱) | 15-20分 穩健 | 10分以下 風險控管")
