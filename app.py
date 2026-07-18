import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import json
import os

st.set_page_config(page_title="TaiStock V2.6 雙軌穩定版", layout="wide")
DATA_FILE = "history.json"

# ==========================================
# 資料讀取 (模擬您的投資組合)
# ==========================================
def load_user_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f: 
                return json.load(f)
        except: 
            pass
    # 預設測試資料 (區分 TW 與 US)
    return {
        "2317": {"market": "TW", "name": "鴻海", "cost_price": 175.0},
        "3035": {"market": "TW", "name": "智原", "cost_price": 300.0},
        "NVDA": {"market": "US", "name": "NVIDIA", "cost_price": 125.5},
        "TSM": {"market": "US", "name": "TSMC", "cost_price": 150.0}
    }

# ==========================================
# [模組 A] 台股核心 (V2.5 封裝區 - 絕對穩定)
# ==========================================
def process_tw_stock(ticker, info):
    """
    這裡請直接貼上您原本 V2.5 版本的台股運算邏輯。
    包含 FinMind API 串接、籌碼分析、40分權重計算等。
    """
    try:
        # [在此替換您的 V2.5 原始程式碼]
        # 範例模擬回傳格式：
        return {
            "status": "success",
            "market": "TW",
            "ticker": ticker,
            "name": info["name"],
            "cost_price": info["cost_price"],
            "current_price": 0.0, # 請由 V2.5 邏輯填入
            "atr": 0.0,           # 請由 V2.5 邏輯填入
            "score": 35           # 請由 V2.5 邏輯填入
        }
    except Exception as e:
        # 台股運算異常防護
        return {"status": "error", "market": "TW", "ticker": ticker, "name": info["name"], "error_msg": str(e)}

# ==========================================
# [模組 B] 美股外掛 (安全隔離區 - 失敗不當機)
# ==========================================
@st.cache_data(ttl=900)
def process_us_stock(ticker, info):
    """
    獨立的美股 yfinance 抓取區。
    具備嚴格的 try-except，網路斷線或代號錯誤皆不會引發 KeyError。
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df.empty:
            return {"status": "error", "market": "US", "ticker": ticker, "name": info["name"], "error_msg": "無歷史資料"}
            
        close = df['Close']
        ma60 = close.rolling(60).mean()
        atr = np.maximum(
            df['High']-df['Low'], 
            np.maximum(abs(df['High']-close.shift()), abs(df['Low']-close.shift()))
        ).rolling(20).mean()
        
        return {
            "status": "success",
            "market": "US",
            "ticker": ticker,
            "name": info["name"],
            "cost_price": info["cost_price"],
            "current_price": round(close.iloc[-1], 2),
            "atr": round(atr.iloc[-1], 2),
            "bias_ma60": round(((close.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]) * 100, 2),
            "score": 30 # 美股暫定預設動能分數
        }
    except Exception as e:
        # 美股專屬錯誤攔截，絕對不會干擾台股運行
        return {"status": "error", "market": "US", "ticker": ticker, "name": info["name"], "error_msg": "API 抓取失敗"}

# ==========================================
# [模組 C] 介面渲染 (嚴格遵守無縮排防破圖原則)
# ==========================================
def render_stock_card(data):
    if data["status"] == "error":
        # 錯誤狀態顯示卡片
        html_error = f"""<div style="background:#FFEbee; padding:15px; border-radius:8px; margin-bottom:15px; border-left:5px solid #F44336;">
<h4 style="margin:0; color:#B71C1C;">{data['ticker']} {data['name']} - 資料載入異常</h4>
<p style="margin:5px 0 0 0; font-size:0.9rem; color:#D32F2F;">請檢查網路連線或代號設定 ({data.get('error_msg', '未知錯誤')})</p>
</div>"""
        st.markdown(html_error, unsafe_allow_html=True)
        return

    # 正常狀態顯示卡片 (HTML 標籤緊貼左側，杜絕 Streamlit 解析錯誤)
    color = "#4A90E2" if data['market'] == "US" else "#D32F2F"
    html_content = f"""<div style="background:#fff; padding:20px; border-radius:12px; margin-bottom:20px; border-left:5px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
<div style="display:flex; justify-content:space-between; margin-bottom:15px;">
<h3 style="margin:0; color:#333;">{data['ticker']} {data['name']}</h3>
<div><span style="background:#f0f0f0; padding:4px 8px; border-radius:4px; color:#555;">均價: {data['cost_price']}</span></div>
</div>
<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px;">
<div><small style="color:#777;">現價</small><div style="font-size:1.2rem; font-weight:bold; color:#000;">{data['current_price']}</div></div>
<div><small style="color:#777;">防守 (2ATR)</small><div style="font-size:1.2rem; color:#000;">{round(data['cost_price'] - (2*data['atr']), 2)}</div></div>
<div><small style="color:#777;">動能</small><div style="font-size:1.2rem; color:#000;">{data.get('score', 0)}/40</div></div>
<div><small style="color:#777;">目標 (10%)</small><div style="font-size:1.2rem; color:green;">{round(data['cost_price']*1.1, 2)}</div></div>
</div>
</div>"""
    st.markdown(html_content, unsafe_allow_html=True)

# ==========================================
# 主流程 (任務分流與彙整)
# ==========================================
def main():
    st.title("📈 TaiStock V2.6 (雙軌穩定版)")
    st.markdown("---")
    
    user_portfolio = load_user_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇹🇼 台股主力陣列 (V2.5 核心)")
        for ticker, info in user_portfolio.items():
            if info["market"] == "TW":
                # 呼叫台股專屬模組
                res = process_tw_stock(ticker, info)
                render_stock_card(res)
                
    with col2:
        st.subheader("🇺🇸 美股科技巨頭 (獨立外掛)")
        for ticker, info in user_portfolio.items():
            if info["market"] == "US":
                # 呼叫美股專屬模組
                res = process_us_stock(ticker, info)
                render_stock_card(res)

if __name__ == "__main__":
    main()
