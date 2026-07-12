import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="TaiStock AI 進階系統")

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

# --- 核心分析引擎 ---
st.title("⚡ TaiStock 進階決策系統 (完整整合版)")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
        price = float(c.iloc[-1])
        
        # 1. 既有指標計算
        ma20 = c.rolling(20).mean().iloc[-1]
        ma60 = c.rolling(60).mean().iloc[-1]
        macd = (c.rolling(12).mean().iloc[-1]) - (c.rolling(26).mean().iloc[-1])
        rsv = (price - l.rolling(9).min().iloc[-1]) / (h.rolling(9).max().iloc[-1] - l.rolling(9).min().iloc[-1] + 0.001) * 100
        k, d = (2/3 * 50 + 1/3 * rsv), (2/3 * 50 + 1/3 * (2/3 * 50 + 1/3 * rsv))
        
        # RSI 真實計算
        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -1 * delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - (100 / (1 + (up.iloc[-1] / (down.iloc[-1] + 0.001))))
        
        # 2. 專家級指標 (BIAS, ATR, 量價)
        bias = ((price - ma60) / ma60) * 100
        tr = [max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]
        atr = sum(tr) / 14
        is_bullish = v.iloc[-1] > (v.rolling(5).mean().iloc[-1] * 1.2) and price > c.iloc[-2]
        
        # 3. 顯示區
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("損益", f"{((price-cost)/cost*100):.1f}%")
            c3.metric("AI 評分", f"{'強勢' if k > 50 else '觀望'}")
            
            with st.expander("🚦 AI 決策紅綠燈與專業診斷"):
                # 紅綠燈面板
                cols = st.columns(3)
                cols[0].markdown(f"### {'🟢' if k > d else '🔴'} KD {'向上' if k > d else '交叉向下'}")
                cols[1].markdown(f"### {'🟢' if macd > 0 else '🔴'} MACD {'多頭' if macd > 0 else '空頭'}")
                cols[2].markdown(f"### {'🟢' if is_bullish else '🟡'} 動能 {'量價齊揚' if is_bullish else '盤整中'}")
                
                st.divider()
                
                # ATR 與 BIAS 策略
                bias_color = "🔴" if bias > 10 else "🟢"
                st.write(f"**💡 ATR 動態停損**: {price - (atr * 2):.1f}")
                st.markdown(f"**📈 乖離率狀態**: {bias_color} {bias:.1f}% {'(⚠️ 短線過熱，注意風險)' if bias > 10 else '(✅ 區間穩定)'}")
                
                # 原始數據保留 (不遺漏)
                st.write("---")
                st.write(f"均線: MA20:{ma20:.1f} | MA60:{ma60:.1f} | 指標: RSI:{rsi:.1f} | MACD:{macd:.3f}")
    except: continue
