import streamlit as st
import pandas as pd
import yfinance as yf
import requests

# 1. 設定與離線對照
NAME_MAP = {"2317": "鴻海", "2330": "台積電", "2382": "廣達", "2308": "台達電", "3711": "日月光投控", "6409": "旭隼"}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {"2317": {"cost": 171.0, "shares": 1000}, "2330": {"cost": 990.0, "shares": 47}}

st.set_page_config(page_title="波段決策系統", layout="wide")
st.title("⚡ 波段決策儀表板 V64.0 (除錯版)")

# 2. 顯示監控表格
data = []
for code, info in st.session_state.portfolio.items():
    df = yf.download(f"{code}.TW", period="6mo", progress=False)
    if not df.empty:
        c = float(df['Close'].iloc[-1].item())
        data.append({
            "代號": code, "現價": round(c, 2), "成本": round(float(info['cost']), 2),
            "帳面損益": round((c - info['cost']) * info['shares'], 0)
        })
st.table(pd.DataFrame(data))

# 3. 檢查 API Key 是否存在 (除錯關鍵)
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("系統偵測到 GEMINI_API_KEY 為空！請確認您已在 Streamlit Cloud 的 Secrets 設定中加入 KEY。")

# 4. 極簡 AI 診斷
st.divider()
target = st.selectbox("診斷標的:", [d['代號'] for d in data])
if st.button("執行 AI 診斷"):
    if api_key:
        try:
            # 使用最簡單的請求方式
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": f"分析 {target} 技術面決策"}]}]}
            
            with st.spinner("連線中..."):
                res = requests.post(url, json=payload, timeout=10).json()
                st.markdown(res["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            st.error(f"連線異常: {str(e)}")
    else:
        st.error("金鑰未設定")
