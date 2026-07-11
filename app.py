import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
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
# ... (側邊欄管理表單代碼保持不變) ...

# --- 真實指標計算引擎 (引入 pandas_ta) ---
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        # 使用 pandas_ta 計算真實指標
        df.ta.macd(append=True) # MACD_12_26_9
        df.ta.stoch(append=True) # STOCH_k, STOCH_d
        df.ta.rsi(append=True)   # RSI_14
        
        last = df.iloc[-1]
        price = float(last['Close'])
        # 抓取真實指標 (名稱依據 pandas_ta 自動產生)
        macd = float(last['MACD_12_26_9'])
        k = float(last['STOCHk_14_3_3'])
        d = float(last['STOCHd_14_3_3'])
        rsi = float(last['RSI_14'])
        
        # 均線
        ma10, ma20, ma60 = df['Close'].rolling(10).mean().iloc[-1], df['Close'].rolling(20).mean().iloc[-1], df['Close'].rolling(60).mean().iloc[-1]
        
        score = (25 if price > ma60 else 10) + (10 if macd > 0 else 0)
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            # ... (顯示邏輯) ...
            with st.expander("👉 查看完整技術診斷"):
                st.write(f"均線: MA10:{ma10:.1f} | MA60:{ma60:.1f}")
                st.write(f"指標: KD(K:{k:.1f}, D:{d:.1f}) | RSI:{rsi:.1f} | MACD:{macd:.3f}")
    except Exception as e:
        continue
