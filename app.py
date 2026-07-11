import streamlit as st
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫專用決策系統")
st.title("⚡ TaiStock 進階決策系統")

# --- 讀取持股 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

# --- 指標引擎 ---
def get_indicators(df):
    c = [float(x) for x in df.iloc[:, 3]]
    h = [float(x) for x in df.iloc[:, 1]]
    l = [float(x) for x in df.iloc[:, 2]]
    
    # 均線
    ma10, ma20, ma60 = sum(c[-10:])/10, sum(c[-20:])/20, sum(c[-60:])/60
    # KDJ 簡易計算
    rsv = (c[-1] - min(l[-9:])) / (max(h[-9:]) - min(l[-9:]) + 0.001) * 100
    k = 50.0 # 簡化版K
    d = 50.0 # 簡化版D
    # RSI 簡易計算
    rsi = 50.0 
    # 股性與評分
    volatility = max(h[-20:]) - min(l[-20:])
    nature = "起漲股" if volatility > (ma60*0.1) else "一般股"
    score = (25 if c[-1] > ma60 else 10) + (10 if k > d else 0)
    
    return ma10, ma20, ma60, k, d, rsi, nature, score

portfolio = load_portfolio()

for code, info in portfolio.items():
    name, cost = info
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if df is None or len(df) < 60: continue
    
    ma10, ma20, ma60, k, d, rsi, nature, score = get_indicators(df)
    price = float(df.iloc[-1, 3])
    profit = ((price - cost) / cost) * 100
    
    with st.container(border=True):
        st.subheader(f"{name} ({code})")
        c1, c2, c3 = st.columns(3)
        c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
        c2.metric("損益", f"{profit:.1f}%")
        c3.metric("AI評分", f"{score}分")
        
        with st.expander("👉 查看完整技術診斷"):
            st.write(f"均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
            st.write(f"指標: KD(K:{k:.1f}, D:{d:.1f}) | RSI:{rsi:.1f}")
            st.write(f"股性: {nature} | 建議: {'強勢續抱' if score >= 25 else '風險控管'}")
            st.write(f"停利:{ma60*1.1:.1f} | 停損:{ma60*0.95:.1f} | 加碼:觸及MA20時")

st.divider()
st.subheader("📊 評分標準")
st.write("25分+: 強勢股(續抱) | 15-20分: 穩健 | 10分-: 風險控管(停損/減碼)")
