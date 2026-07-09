import streamlit as st
import pandas as pd
import yfinance as yf
import requests

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V71.0 (權限檢測版)")

# 1. 庫存維護
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"2317": {"cost": 171.0}}
with st.sidebar:
    code = st.text_input("代號:")
    cost = st.number_input("成本:", value=0.0)
    if st.button("更新庫存"): st.session_state.portfolio[code] = {"cost": cost}; st.rerun()

# 2. 顯示監控
data = [{"代號": k, "現價": round(float(yf.download(f"{k}.TW", period="1d", progress=False)['Close'].iloc[-1].item()), 2), "成本": v['cost']} for k, v in st.session_state.portfolio.items()]
st.table(pd.DataFrame(data))

# 3. 關鍵診斷通道 (我們使用通用模型名稱)
st.divider()
target = st.text_input("輸入要診斷的代號:")
if st.button("執行 AI 深度診斷"):
    st.info(f"正在連線 Gemini 伺服器...")
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    # 使用 Google 官方推薦最穩定的模型名稱
    model_name = "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": f"請分析股票 {target} 的技術面趨勢與交易建議"}]}]}, timeout=10)
        
        if res.status_code == 200:
            st.markdown(res.json()["candidates"][0]["content"]["parts"][0]["text"])
        else:
            # 直接顯示 Google 回傳的錯誤，這能告訴我們到底是哪裡鎖住了
            st.error(f"連線失敗 (錯誤代碼 {res.status_code}): {res.text}")
            
    except Exception as e:
        st.error(f"程式連線錯誤: {str(e)}")
