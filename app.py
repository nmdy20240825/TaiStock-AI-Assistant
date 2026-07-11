import streamlit as st
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫進階策略系統")
st.title("⚡ TaiStock 進階診斷報告")

# --- 讀取/儲存 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- 核心邏輯 (保留所有原功能 + 疊加決策引擎) ---
portfolio = load_portfolio()
# (側邊欄管理省略，保持不變)

for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        c = [float(x) for x in df.iloc[:, 3]]
        h = [float(x) for x in df.iloc[:, 1]]
        l = [float(x) for x in df.iloc[:, 2]]
        
        # 1. 既有指標計算
        ma10, ma20, ma60 = sum(c[-10:])/10, sum(c[-20:])/20, sum(c[-60:])/60
        macd = (sum(c[-12:])/12) - (sum(c[-26:])/26)
        rsv = (c[-1] - min(l[-9:])) / (max(h[-9:]) - min(l[-9:]) + 0.001) * 100
        k, d = (2/3 * 50 + 1/3 * rsv), (2/3 * 50 + 1/3 * (2/3 * 50 + 1/3 * rsv))
        rsi = 50.0 # 暫留
        
        # 2. 決策解釋邏輯
        reasons = []
        if macd > 0: reasons.append("② MACD黃金交叉")
        if c[-1] > ma20: reasons.append("③ MA20向上支撐")
        if (c[-1]/ma20) > 1.15: reasons.append("④ 股價強勢起漲")
        
        score = (25 if c[-1] > ma60 else 10) + (10 if macd > 0 else 0)
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{c[-1]:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{((c[-1]-cost)/cost*100):.1f}%")
            c3.metric("AI評分", f"{score}分")
            
            with st.expander("📝 查看詳細決策報告"):
                st.write("**[完整技術數據]**")
                st.write(f"均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"指標: K:{k:.1f} | D:{d:.1f} | RSI:{rsi:.1f} | MACD:{macd:.3f}")
                st.write("**[決策解釋引擎]**")
                st.write(f"建議: {'續抱' if score >= 25 else '風險控管'}")
                st.write("原因:")
                for r in reasons: st.write(r)
                st.write(f"股性: {'起漲股' if (c[-1]/ma20) > 1.15 else '一般股'}")
    except: continue
