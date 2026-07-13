import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np # 引入 numpy 處理數值異常

st.set_page_config(layout="wide", page_title="TaiStock 最終修復版")

@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    return yf.download(f"{code}.TW", period="6mo", progress=False)

def get_institutional_data(code):
    # 此處未來可對接真實 API
    data_map = {"3711": {"buy_sell": 1500, "days": 3, "trend": "連3買"}}
    return data_map.get(code, {"buy_sell": 0, "days": 0, "trend": "盤整"})

def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()

st.title("⚡ TaiStock 進階決策系統 (最終修正版)")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = fetch_stock_data(code)
        if df.empty or len(df) < 60: continue
        
        # 數據提取
        c, h, l = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
        v = df.get('Volume', pd.Series(0, index=df.index)).squeeze()
        price, volume = float(c.iloc[-1]), float(v.iloc[-1])
        
        # 1. 變數強制初始化 (防止 name 'k' is not defined)
        k, d, macd, rsi = 50.0, 50.0, 0.0, 50.0
        
        # 2. 技術指標計算 (加入 np.nan_to_num 處理)
        ma10 = float(c.rolling(10).mean().iloc[-1])
        ma20 = float(c.rolling(20).mean().iloc[-1])
        ma60 = float(c.rolling(60).mean().iloc[-1])
        macd = float((c.rolling(12).mean().iloc[-1]) - (c.rolling(26).mean().iloc[-1]))
        
        rsv_val = (price - float(l.rolling(9).min().iloc[-1])) / (float(h.rolling(9).max().iloc[-1]) - float(l.rolling(9).min().iloc[-1]) + 0.001) * 100
        k = float(2/3 * 50 + 1/3 * np.nan_to_num(rsv_val))
        d = float(2/3 * 50 + 1/3 * k)
        
        # RSI 計算保護
        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        down = -1 * delta.clip(upper=0).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (np.nan_to_num(up) / (np.nan_to_num(down) + 0.001))))
        
        # 3. 策略指標
        coeff = price / ma20
        inst = get_institutional_data(code)
        bias = ((price - ma60) / ma60) * 100
        atr = sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]) / 14
        
        # 4. 面板顯示
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("法人動能", inst['trend'], delta=f"{inst['buy_sell']}張")
            c3.metric("AI 狀態", "強勢" if k > 50 else "觀望")
            c4.metric("股性判別", "🚀 起漲股" if coeff > 1.15 else "📊 一般股")
            
            with st.expander("🚦 查看完整決策診斷報告"):
                cols = st.columns(3)
                cols[0].markdown(f"### {'🟢' if k > d else '🔴'} KD {'向上' if k > d else '交叉向下'}")
                cols[1].markdown(f"### {'🟢' if macd > 0 else '🔴'} MACD {'多頭' if macd > 0 else '空頭'}")
                cols[2].markdown(f"### {'🟢' if coeff > 1.15 else '🟡'} 動能 {'起漲中' if coeff > 1.15 else '盤整中'}")
                st.divider()
                st.write(f"成交量: {volume:,.0f} | 均線: MA20:{ma20:.1f} | 乖離率: {bias:.1f}%")
                st.write(f"指標: K:{k:.1f} | D:{d:.1f} | RSI:{rsi:.1f} | MACD:{macd:.3f}")
                st.write(f"💡 ATR 停損: {price - (atr * 2):.1f} | 🎯 波段停利: {(price * 1.1):.1f}")
                
    except Exception as e:
        st.error(f"分析 {code} 發生錯誤: {e}")
