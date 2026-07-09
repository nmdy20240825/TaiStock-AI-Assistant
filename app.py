import streamlit as st
import requests
import yfinance as yf

# 1. 完整股票資料庫
STOCK_DATABASE = {
    "2317": "鴻海", "2330": "台積電", "2382": "廣達", 
    "3017": "奇鋐", "3037": "欣興", "3443": "創意",
    "2303": "聯電", "2308": "台達電", "2357": "華碩"
}

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("🚀 波段決策儀表板 V42.0")

# 2. 側邊欄：自選股與多選
with st.sidebar:
    st.header("自選股設定")
    # 讓使用者自定義選股清單
    selected_codes = st.multiselect(
        "選擇要監控的股票:", 
        list(STOCK_DATABASE.keys()), 
        default=["2317", "2330"] # 預設勾選這兩支
    )
    
    if not selected_codes:
        st.warning("請至少選擇一支股票")
        st.stop()
        
    # 切換當前分析對象
    current_code = st.selectbox("切換分析對象:", selected_codes)
    cost = st.number_input(f"{STOCK_DATABASE[current_code]} 的成本價:", min_value=0.0, value=171.0)

# 3. 數據獲取 (與之前邏輯相同)
ticker = yf.Ticker(f"{current_code}.TW")
hist = ticker.history(period="6mo")
current_price = hist['Close'].iloc[-1]
ma60 = hist['Close'].rolling(window=60).mean().iloc[-1]
stop_loss = cost * 0.92

# 4. 儀表板視覺化
st.subheader(f"分析標的: {current_code} {STOCK_DATABASE[current_code]}")
col1, col2, col3 = st.columns(3)
col1.metric("即時股價", f"{current_price:.2f}")
col2.metric("8% 停損點", f"{stop_loss:.2f}")
col3.metric("季線(MA60)", f"{ma60:.2f}", delta=f"{current_price - ma60:.2f}")

# ... (AI 分析部分維持不變)
