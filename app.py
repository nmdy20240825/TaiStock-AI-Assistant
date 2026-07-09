import streamlit as st
import pandas as pd
import yfinance as yf

# 初始化庫存資料
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        "2317": {"name": "鴻海", "cost": 171.0, "shares": 1000},
        "2330": {"name": "台積電", "cost": 990.0, "shares": 500}
    }

st.set_page_config(page_title="波段決策儀表板", layout="wide")
st.title("💼 智慧庫存監控面板 V53.0")

# 處理數據
data = []
for code, info in st.session_state.portfolio.items():
    price = yf.Ticker(f"{code}.TW").history(period="1d")['Close'].iloc[-1]
    profit_amt = (price - info['cost']) * info['shares']
    profit_pct = (price - info['cost']) / info['cost'] * 100
    stop_loss = info['cost'] * 0.92
    
    data.append({
        "股票資訊": f"{code} {info['name']}",  # 合併代號與名稱
        "現價": round(price, 2), 
        "成本": info['cost'],
        "帳面損益": round(profit_amt, 0),
        "損益%": round(profit_pct, 2), 
        "離停損%": round((price - stop_loss) / info['cost'] * 100, 2)
    })

df = pd.DataFrame(data)

# 顯示互動表格
st.subheader("庫存總覽")
st.dataframe(
    df.style.format({"帳面損益": "{:,.0f}"}).background_gradient(subset=['帳面損益'], cmap='RdYlGn'), 
    use_container_width=True
)

# 互動診斷區
st.divider()
st.subheader("快速診斷通道")
# 下拉選單也同步顯示 "代號 名稱"
selected_entry = st.selectbox("選擇診斷標的:", df['股票資訊'].tolist())
target_code = selected_entry.split(" ")[0] # 取出代號部分

if st.button(f"執行 {selected_entry} AI 深度診斷"):
    st.info(f"正在分析 {selected_entry}，請稍候...")
    # (此處延續 V49.0 的 AI 分析邏輯)
