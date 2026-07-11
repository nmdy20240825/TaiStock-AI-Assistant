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

# --- 核心引擎 (無依賴版本) ---
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        # 轉換為列表
        c = [float(x) for x in df.iloc[:, 3]]
        h = [float(x) for x in df.iloc[:, 1]]
        l = [float(x) for x in df.iloc[:, 2]]
        
        # 指標計算
        ma10, ma20, ma60 = sum(c[-10:])/10, sum(c[-20:])/20, sum(c[-60:])/60
        # MACD (DIF 簡易版)
        macd = (sum(c[-12:])/12) - (sum(c[-26:])/26)
        # KD (RSV)
        rsv = (c[-1] - min(l[-9:])) / (max(h[-9:]) - min(l[-9:]) + 0.001) * 100
        k = 50 if len(c) < 10 else (2/3 * 50 + 1/3 * rsv)
        d = 50 if len(c) < 10 else (2/3 * 50 + 1/3 * k)
        
        profit = ((c[-1] - cost) / cost) * 100
        score = (25 if c[-1] > ma60 else 10) + (10 if macd > 0 else 0)
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{c[-1]:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{profit:.1f}%")
            c3.metric("AI評分", f"{score}分")
            
            with st.expander("👉 查看完整技術診斷"):
                st.write(f"均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"指標: K:{k:.1f} | D:{d:.1f} | MACD:{macd:.3f}")
                st.write(f"建議: {'強勢續抱' if score >= 25 else '風險控管'}")
                st.write(f"停利:{ma60*1.1:.1f} | 停損:{ma60*0.95:.1f} | 加碼:觸及MA20")
    except: continue
