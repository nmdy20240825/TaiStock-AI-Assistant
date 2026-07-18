import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import json
import os
from datetime import datetime

# ==========================================
# 模組 1：系統環境與資料庫設定
# ==========================================
st.set_page_config(page_title="TaiStock V2.6 完全體", layout="wide")
DATA_FILE = "history.json"

# 預設追蹤清單與初始成本 (若無 history.json 則載入此預設值)
DEFAULT_PORTFOLIO = {
    "NVDA": {"market": "US", "name": "NVIDIA", "cost_price": 125.5},
    "TSM": {"market": "US", "name": "TSMC ADR", "cost_price": 182.0},
    "2317.TW": {"market": "TW", "name": "鴻海", "cost_price": 175.0},
    "3035.TW": {"market": "TW", "name": "智原", "cost_price": 340.0}
}

def load_user_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return DEFAULT_PORTFOLIO
    return DEFAULT_PORTFOLIO

def save_user_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ==========================================
# 模組 2：量化指標與即時報價引擎 (yfinance)
# ==========================================
def calculate_indicators(df):
    if df.empty or len(df) < 60:
        return None
    
    # 計算 ATR (20日)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_20'] = true_range.rolling(20).mean()
    
    # 計算 60日均線 (季線) 與乖離率
    df['MA_60'] = df['Close'].rolling(60).mean()
    df['Bias_60'] = ((df['Close'] - df['MA_60']) / df['MA_60']) * 100
    
    return df

@st.cache_data(ttl=900) # 快取 15 分鐘避免頻繁呼叫 API
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        if not df.empty:
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            return {
                "current_price": round(latest['Close'], 2),
                "atr": round(latest['ATR_20'], 2),
                "bias_ma60": round(latest['Bias_60'], 2)
            }
    except Exception as e:
        return None
    return None

def calculate_risk(cost_price, atr):
    if cost_price <= 0 or atr <= 0:
        return 0.0, 0.0
    stop_loss = round(cost_price - (2 * atr), 2)
    take_profit = round(cost_price * 1.10, 2)
    return stop_loss, take_profit

# ==========================================
# 模組 3：動態 CSS 樣式 (已修復渲染 Bug)
# ==========================================
def inject_custom_css():
    st.markdown("""
    <style>
    .stock-card { background: #ffffff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 20px; border-left: 5px solid #e0e0e0; }
    .stock-card.us-market { border-left-color: #4A90E2; }
    .stock-card.tw-market { border-left-color: #D32F2F; }
    .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f0f0f0; padding-bottom: 12px; margin-bottom: 16px; }
    .stock-title { margin: 0; font-size: 1.2rem; color: #333; }
    .stock-name { font-size: 0.9rem; color: #777; margin-left: 8px; }
    .tags-container .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-left: 8px; }
    .badge-market { background: #E3F2FD; color: #1976D2; }
    .badge-cost { background: #F5F5F5; color: #616161; }
    .badge-status { background: #E8F5E9; color: #2E7D32; }
    .badge-alert { background: #FFEBEE; color: #C62828; }
    .data-grid-4col { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
    .data-col { display: flex; flex-direction: column; }
    .highlight-col { background: #F8FBFF; padding: 8px; border-radius: 8px; }
    .col-title { font-size: 0.85rem; color: #757575; margin-bottom: 4px; }
    .col-value { font-size: 1.4rem; font-weight: bold; color: #212121; }
    .col-sub { font-size: 0.85rem; color: #9E9E9E; margin-top: 2px; }
    .price-up { color: #D32F2F; }
    .total-score { color: #F57C00; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 模組 4：前端視覺化渲染卡片
# ==========================================
def render_stock_card(data):
    market_class = "us-market" if data['market'] == "US" else "tw-market"
    market_tag = "US-Tech" if data['market'] == "US" else "TW-Tech"
    
    status_class = "badge-status"
    status_text = "續抱"
    if data['current_price'] >= data['take_profit']:
        status_text = "🟢 達標"
    elif data['current_price'] <= data['stop_loss']:
        status_class = "badge-alert"
        status_text = "🔴 破線"

    # 動態產生分數 (簡化模擬計分，實戰可替換為更複雜演算法)
    base_score = 60 if data['bias_ma60'] > 0 else 40
    engine_score = min(40, int(20 + data['bias_ma60']))
    tech_score = min(30, int(15 + (data['current_price'] / data['cost_price'] * 5)))
    total_score = base_score + engine_score + tech_score
    
    engine_title = "動能引擎 (40%)" if data['market'] == "US" else "籌碼引擎 (40%)"

    # HTML 區塊嚴格規定：不可縮排，避免 Streamlit 判斷為程式碼區塊
    html_content = f"""<div class="stock-card {market_class}">
<div class="card-header">
    <h3 class="stock-title">{data['ticker'].replace('.TW', '')} <span class="stock-name">{data['name']}</span></h3>
    <div class="tags-container">
        <span class="badge badge-market">{market_tag}</span>
        <span class="badge badge-cost">均價: {data['cost_price']}</span>
        <span class="badge {status_class}">{status_text}</span>
    </div>
</div>
<div class="data-grid-4col">
    <div class="data-col">
        <span class="col-title">報價與風控</span>
        <div class="col-value price-up">{data['current_price']}</div>
        <div class="col-sub">ATR (20): {data['atr']}</div>
        <div class="col-sub">防守: {data['stop_loss']}</div>
    </div>
    <div class="data-col highlight-col">
        <span class="col-title">{engine_title}</span>
        <div class="col-value">{engine_score} / 40</div>
        <div class="col-sub">系統運算中</div>
    </div>
    <div class="data-col">
        <span class="col-title">技術共振 (30%)</span>
        <div class="col-value">{tech_score} / 30</div>
        <div class="col-sub">季線乖離: {data['bias_ma60']}%</div>
    </div>
    <div class="data-col">
        <span class="col-title">總戰力分數</span>
        <div class="col-value total-score">{total_score} 分</div>
        <div class="col-sub">波段目標: {data['take_profit']}</div>
    </div>
</div>
</div>"""
    st.markdown(html_content, unsafe_allow_html=True)

# ==========================================
# 主程式執行 (Main Loop)
# ==========================================
def main():
    inject_custom_css()
    st.title("📈 TaiStock V2.6 跨市場量化追蹤系統 (全功能版)")
    st.markdown("---")
    
    # 載入歷史資料與成本
    user_portfolio = load_user_data()
    processed_data = []
    
    # 即時抓取與運算
    with st.spinner('即時連線交易所抓取報價與運算指標中...'):
        for ticker, info in user_portfolio.items():
            market_data = fetch_stock_data(ticker)
            if market_data:
                sl, tp = calculate_risk(info['cost_price'], market_data['atr'])
                combined_info = {
                    "ticker": ticker,
                    "market": info['market'],
                    "name": info['name'],
                    "cost_price": info['cost_price'],
                    "current_price": market_data['current_price'],
                    "atr": market_data['atr'],
                    "bias_ma60": market_data['bias_ma60'],
                    "stop_loss": sl,
                    "take_profit": tp
                }
                processed_data.append(combined_info)
            else:
                st.warning(f"無法取得 {ticker} 即時資料，請確認網路連線。")

    # SOP 檢核清單邏輯
    sop_action = []
    sop_monitor = []
    for data in processed_data:
        ticker_display = data['ticker'].replace('.TW', '')
        if data['current_price'] <= data['stop_loss']:
            sop_action.append(f"🔴 {ticker_display} 跌破防守線！現價 {data['current_price']} <= 停損 {data['stop_loss']}")
        elif data['current_price'] >= data['take_profit']:
            sop_action.append(f"🟢 {ticker_display} 達標停利！現價 {data['current_price']} >= 目標 {data['take_profit']}")
        else:
            sop_monitor.append(f"⚪ {ticker_display} 狀態穩定 (現價 {data['current_price']}，防守 {data['stop_loss']})")

    st.header("📋 今日 SOP 操作檢核清單")
    col1, col2 = st.columns(2)
    with col1:
        st.error("🚨 需立即處置 (Action Required)")
        if not sop_action: st.write("無觸發條件。")
        for item in sop_action: st.write(item)
    with col2:
        st.success("🛡️ 安全監控中 (Hold & Monitor)")
        if not sop_monitor: st.write("無持股。")
        for item in sop_monitor: st.write(item)

    st.markdown("---")

    # 跨市場介面分頁
    st.header("📊 跨市場部位深度解析")
    tab_us, tab_tw = st.tabs(["🇺🇸 美股科技巨頭", "🇹🇼 台股主力陣列"])
    
    with tab_us:
        for data in processed_data:
            if data['market'] == "US":
                render_stock_card(data)
                
    with tab_tw:
        for data in processed_data:
            if data['market'] == "TW":
                render_stock_card(data)
                
    # 右側邊欄：更新持股成本介面 (對應 history.json 寫入)
    st.sidebar.header("⚙️ 成本管理中心")
    selected_ticker = st.sidebar.selectbox("選擇標的", list(user_portfolio.keys()))
    new_cost = st.sidebar.number_input(f"更新 {selected_ticker} 均價", value=float(user_portfolio[selected_ticker]['cost_price']))
    if st.sidebar.button("💾 儲存至資料庫"):
        user_portfolio[selected_ticker]['cost_price'] = new_cost
        save_user_data(user_portfolio)
        st.sidebar.success("✅ 成本已更新！請重整網頁。")

if __name__ == "__main__":
    main()
