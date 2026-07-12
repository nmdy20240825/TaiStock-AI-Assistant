import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="TaiStock AI 進階系統")

@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    return yf.download(f"{code}.TW", period="6mo", progress=False)

def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    try:
        with open('portfolio.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock", clear_on_submit=True):
        new_code = st.text_input("代號 (例: 2330)")
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

st.title("⚡ TaiStock 進階決策系統 (全功能版)")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = fetch_stock_data(code)
        if df.empty or len(df) < 60: continue
        
        c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
        price = float(c.iloc[-1])
        
        # 指標計算
        ma10, ma20, ma60 = float(c.rolling(10).mean().iloc[-1]), float(c.rolling(20).mean().iloc[-1]), float(c.rolling(60).mean().iloc[-1])
        macd = float((c.rolling(12).mean().iloc[-1]) - (c.rolling(26).mean().iloc[-1]))
        rsv = (price - float(l.rolling(9).min().iloc[-1])) / (float(h.rolling(9).max().iloc[-1]) - float(l.rolling(9).min().iloc[-1]) + 0.001) * 100
        k, d = float(2/3 * 50 + 1/3 * rsv), float(2/3 * 50 + 1/3 * (2/3 * 50 + 1/3 * rsv))
        
        delta = c.diff()
        up, down = delta.clip(lower=0).rolling(14).mean().iloc[-1], -1 * delta.clip(upper=0).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (up / (down + 0.001))))
        
        # 判斷邏輯
        coeff = price / ma20
        stock_type = "🚀 起漲股" if coeff > 1.15 else "📊 一般股"
        ai_status = "強勢" if k > 50 else "觀望"
        
        bias = ((price - ma60) / ma60) * 100
        tr_list = [max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]
        atr = sum(tr_list) / 14
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{((price-cost)/cost*100):.1f}%")
            c3.metric("AI 狀態", ai_status)
            c4.metric("股性判別", stock_type)
            
            with st.expander("🚦 查看完整決策診斷報告"):
                cols = st.columns(3)
                cols[0].markdown(f"### {'🟢' if k > d else '🔴'} KD {'向上' if k > d else '交叉向下'}")
                cols[1].markdown(f"### {'🟢' if macd > 0 else '🔴'} MACD {'多頭' if macd > 0 else '空頭'}")
                cols[2].markdown(f"### {'🟢' if stock_type == '🚀 起漲股' else '🟡'} 動能 {'起漲中' if stock_type == '🚀 起漲股' else '盤整中'}")
                
                st.divider()
                st.write("**[完整技術數據]**")
                st.write(f"均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"指標: K:{k:.1f} | D:{d:.1f} | RSI:{rsi:.1f} | MACD:{macd:.3f}")
                
                st.write("**[動態交易策略]**")
                st.write(f"💡 ATR 動態停損: {price - (atr * 2):.1f} | 📈 乖離率: {bias:.1f}% {'(過熱)' if bias > 10 else '(穩)'}")
                st.write(f"🎯 建議: {'加碼關注' if stock_type == '🚀 起漲股' else '風險控管'} | 波段停利: {(price * 1.1):.1f} | 強制停損: {price * 0.95:.1f}")
                
                st.divider()
                st.caption("**[判別標準說明]**")
                st.caption("• 股性係數：收盤價 ÷ 20MA > 1.15 為起漲股")
                st.caption("• AI 狀態：強勢 (K > 50) | 觀望 (K <= 50)")
                st.caption("• KD/MACD：🟢 多頭/向上 | 🔴 空頭/向下")
                st.caption("• 乖離率 (BIAS)：乖離 > 10% 觸發過熱警示")
    except Exception as e:
        st.error(f"分析 {code} 發生異常: {e}")
