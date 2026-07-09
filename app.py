import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 離線對照表 (補充中文名)
NAME_MAP = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "2308": "台達電", "3711": "日月光投控", "6409": "旭隼"}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V60.0 (修正穩定版)")

# 2. 側邊欄：手動管理
with st.sidebar:
    code = st.text_input("輸入代號:")
    cost = st.number_input("成本:", value=0.0)
    shares = st.number_input("股數:", value=1000)
    if st.button("更新庫存"):
        if code:
            st.session_state.portfolio[code] = {"cost": cost, "shares": shares}
            st.rerun()

# 3. 數據計算 (修正取值邏輯)
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="1y", progress=False)
    if not df.empty:
        # 確保抓取的是純數值，排除 DataFrame/Series 結構干擾
        curr = float(df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].iloc[-1])
        ma10 = float(df['Close'].rolling(window=10).mean().iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].rolling(window=10).mean().iloc[-1])
        ma20 = float(df['Close'].rolling(window=20).mean().iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].rolling(window=20).mean().iloc[-1])
        ma60 = float(df['Close'].rolling(window=60).mean().iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].rolling(window=60).mean().iloc[-1])
        
        data.append({
            "代號": code, 
            "名稱": NAME_MAP.get(code, code), 
            "現價": curr, 
            "成本": info['cost'],
            "帳面損益": (curr - info['cost']) * info['shares'],
            "10MA": ma10, "20MA": ma20, "60MA": ma60
        })

df = pd.DataFrame(data)
# 顯示表格，統一格式化小數點二位
st.table(df.style.format("{:.2f}"))

# 4. 極簡 AI 診斷 (優化逾時與連線)
st.divider()
target = st.selectbox("選擇診斷標的:", df['代號'].tolist() if not df.empty else [])
if st.button("執行 AI 診斷"):
    row = df[df['代號'] == target].iloc[0]
    prompt = f"代號{target},現價{row['現價']:.2f},成本{row['成本']:.2f},10MA{row['10MA']:.2f},20MA{row['20MA']:.2f},60MA{row['60MA']:.2f}。請給予明確決策建議:加碼/減碼/停利/停損。"
    
    try:
        # 使用更穩定的 API 呼叫路徑
        api_key = st.secrets.get("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=10).json()
        st.markdown(res["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        st.error("分析中斷，請確認 API Key 或網路環境。")
