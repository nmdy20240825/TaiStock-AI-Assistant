import streamlit as st
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫進階策略系統")
st.title("📈 傑夫策略儀表板 (ATR版)")

# --- 核心數據處理 (ATR + 量價) ---
def get_advanced_diagnosis(df):
    # 計算基礎數值
    c = [float(x) for x in df.iloc[:, 3]] # Close
    h = [float(x) for x in df.iloc[:, 1]] # High
    l = [float(x) for x in df.iloc[:, 2]] # Low
    v = [float(x) for x in df.iloc[:, 4]] # Volume
    
    # ATR 計算 (動態停損)
    tr = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    atr = sum(tr[-14:]) / 14
    
    # 量價分析 (判斷起漲訊號)
    vol_avg = sum(v[-5:]) / 5
    is_volume_up = v[-1] > vol_avg * 1.5
    is_price_up = c[-1] > c[-2]
    nature = "量價齊揚(起漲中)" if (is_volume_up and is_price_up) else "盤整/一般"
    
    # 策略數據
    stop_loss = c[-1] - (atr * 2) # 動態停損：ATR*2
    take_profit = c[-1] + (atr * 4) # 動態停利：ATR*4
    
    return nature, stop_loss, take_profit, atr

# --- 主程式 ---
portfolio = json.load(open('portfolio.json', 'r', encoding='utf-8'))
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        nature, stop, profit_target, atr = get_advanced_diagnosis(df)
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            st.write(f"**診斷分析**: {nature} | ATR波動: {atr:.2f}")
            st.write(f"**策略建議**: 停損價 {stop:.1f} | 停利目標 {profit_target:.1f}")
            with st.expander("👉 詳細數據"):
                st.write(f"原始損益: {((df.iloc[-1, 3]-cost)/cost*100):.1f}%")
                st.write("量價分析：使用 5 日成交量均值與 ATR 波幅進行動態校正")
    except: continue
