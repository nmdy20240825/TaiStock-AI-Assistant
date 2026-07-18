import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime

# ==========================================
# 模組 1：系統參數與 CSS 樣式初始化
# ==========================================
st.set_page_config(page_title="TaiStock V2.6", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
    .stock-card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #e0e0e0;
    }
    .stock-card.us-market { border-left-color: #4A90E2; }
    .stock-card.tw-market { border-left-color: #D32F2F; }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .stock-title { margin: 0; font-size: 1.2rem; color: #333; }
    .stock-name { font-size: 0.9rem; color: #777; margin-left: 8px; }
    
    .tags-container .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .badge-market { background: #E3F2FD; color: #1976D2; }
    .badge-cost { background: #F5F5F5; color: #616161; }
    .badge-status { background: #E8F5E9; color: #2E7D32; }
    .badge-alert { background: #FFEBEE; color: #C62828; }
    
    .data-grid-4col {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
    }
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
# 模組 2：核心運算與風控邏輯
# ==========================================
def calculate_risk_levels(cost_price, atr_20):
    if cost_price <= 0 or atr_20 <= 0:
        return 0.0, 0.0
    stop_loss = round(cost_price - (2 * atr_20), 2)
    take_profit = round(cost_price * 1.10, 2)
    return stop_loss, take_profit

def generate_sop_checklist(portfolio_data):
    checklist = {
        "action_required": [],
        "hold_monitor": [],
        "new_entry": []
    }
    for stock in portfolio_data:
        ticker = stock['ticker']
        current_price = stock['current_price']
        cost_price = stock['cost_price']
        
        if cost_price == 0:
            if stock['total_score'] >= 80:
                checklist["new_entry"].append(f"{ticker} (戰力 {stock['total_score']} 分：強勢訊號浮現)")
            continue
            
        if current_price <= stock['stop_loss']:
            checklist["action_required"].append(f"🔴 {ticker} 跌破防守線！現價 {current_price} <= 停損 {stock['stop_loss']}")
        elif current_price >= stock['take_profit']:
            checklist["action_required"].append(f"🟢 {ticker} 達標停利！現價 {current_price} >= 目標 {stock['take_profit']}")
        else:
            checklist["hold_monitor"].append(f"⚪ {ticker} 狀態穩定 (現價 {current_price}，防守 {stock['stop_loss']})")
    return checklist

# ==========================================
# 模組 3：前端視覺化渲染卡片
# ==========================================
def render_stock_card(data):
    market_class = "us-market" if data['market'] == "US" else "tw-market"
    market_tag = "US-Tech" if data['market'] == "US" else "TW-Tech"
    
    # 判斷狀態標籤
    status_class = "badge-status"
    status_text = "續抱"
    if data['current_price'] >= data['take_profit']:
        status_text = "🟢 達標"
    elif data['current_price'] <= data['stop_loss']:
        status_class = "badge-alert"
        status_text = "🔴 破線"

    html_content = f"""
    <div class="stock-card {market_class}">
        <div class="card-header">
            <h3 class="stock-title">{data['ticker']} <span class="stock-name">{data['name']}</span></h3>
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
                <span class="col-title">{"動能引擎 (40%)" if data['market'] == "US" else "籌碼引擎 (40%)"}</span>
                <div class="col-value">{data['engine_score']} / 40</div>
                <div class="col-sub">{data['engine_sub_1']}</div>
                <div class="col-sub">{data['engine_sub_2']}</div>
            </div>
            
            <div class="data-col">
                <span class="col-title">技術共振 (30%)</span>
                <div class="col-value">{data['tech_score']} / 30</div>
                <div class="col-sub">季線乖離: {data['bias_ma60']}%</div>
                <div class="col-sub">技術面: 多頭</div>
            </div>
            
            <div class="data-col">
                <span class="col-title">總戰力分數</span>
                <div class="col-value total-score">{data['total_score']} 分</div>
                <div class="col-sub">波段目標: {data['take_profit']}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

# ==========================================
# 主程式執行 (Main Loop)
# ==========================================
def main():
    inject_custom_css()
    st.title("📈 TaiStock V2.6 跨市場量化追蹤系統")
    st.markdown("---")
    
    # 模擬從資料庫與 API 取得的即時與歷史運算資料
    # (實務上這裡會串接 yfinance 迴圈與 JSON 歷史檔)
    portfolio = [
        {
            "market": "US", "ticker": "NVDA", "name": "NVIDIA", "current_price": 135.20, "cost_price": 125.50, 
            "atr": 3.45, "total_score": 88, "engine_score": 38, "tech_score": 25,
            "engine_sub_1": "趨勢(20): 20", "engine_sub_2": "爆發(20): 18", "bias_ma60": 8.5
        },
        {
            "market": "US", "ticker": "TSM", "name": "TSMC ADR", "current_price": 178.50, "cost_price": 182.00, 
            "atr": 4.10, "total_score": 75, "engine_score": 28, "tech_score": 20,
            "engine_sub_1": "趨勢(20): 15", "engine_sub_2": "爆發(20): 13", "bias_ma60": 2.1
        },
        {
            "market": "TW", "ticker": "2317", "name": "鴻海", "current_price": 205.00, "cost_price": 175.00, 
            "atr": 4.50, "total_score": 92, "engine_score": 39, "tech_score": 28,
            "engine_sub_1": "外資連買: 5天", "engine_sub_2": "投信部位: 增", "bias_ma60": 12.0
        },
        {
            "market": "TW", "ticker": "3035", "name": "智原", "current_price": 315.00, "cost_price": 340.00, 
            "atr": 12.50, "total_score": 60, "engine_score": 15, "tech_score": 18,
            "engine_sub_1": "外資動向: 賣超", "engine_sub_2": "投信部位: 平", "bias_ma60": -5.5
        }
    ]

    # 計算每一檔的風控點位
    for stock in portfolio:
        sl, tp = calculate_risk_levels(stock['cost_price'], stock['atr'])
        stock['stop_loss'] = sl
        stock['take_profit'] = tp

    # 產生每日檢核清單
    sop = generate_sop_checklist(portfolio)

    # 介面佈局：上半部 SOP 檢核區
    st.header("📋 今日 SOP 操作檢核清單")
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("🚨 需立即處置 (Action Required)")
        if not sop["action_required"]:
            st.write("無觸發條件。")
        for item in sop["action_required"]:
            st.write(item)
            
    with col2:
        st.success("🛡️ 安全監控中 (Hold & Monitor)")
        if not sop["hold_monitor"]:
            st.write("無持股。")
        for item in sop["hold_monitor"]:
            st.write(item)

    st.markdown("---")

    # 介面佈局：下半部跨市場卡片區
    st.header("📊 跨市場部位深度解析")
    tab_us, tab_tw = st.tabs(["🇺🇸 美股科技巨頭", "🇹🇼 台股主力陣列"])
    
    with tab_us:
        for stock in portfolio:
            if stock['market'] == "US":
                render_stock_card(stock)
                
    with tab_tw:
        for stock in portfolio:
            if stock['market'] == "TW":
                render_stock_card(stock)

if __name__ == "__main__":
    main()
