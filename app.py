import streamlit as st
import requests

# 頁面設定
st.set_page_config(page_title="波段決策系統", layout="centered")

st.title("波段決策系統 V39.0")
s = st.selectbox("請選擇要分析的股票代號:", ["2317", "2330", "2382", "3017", "3037"])

if st.button("開始 AI 教練分析"):
    # 從 Secrets 安全讀取金鑰
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if not api_key:
        st.error("錯誤：請在 Streamlit Secrets 設定中確認 GEMINI_API_KEY 是否已填入。")
    else:
        # 使用經過清單驗證的模型名稱
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": f"你是專業台股教練，請針對股票 {s} 進行波段交易分析，包含趨勢判斷、潛在支撐壓力區、以及交易策略建議。"}]
            }]
        }
        
        with st.spinner('AI 教練正在分析中，請稍候...'):
            try:
                response = requests.post(url, headers=headers, json=payload)
                result = response.json()
                
                if "candidates" in result:
                    analysis_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    st.success("分析結果如下：")
                    st.write(analysis_text)
                else:
                    st.error(f"API 回應異常，請確認金鑰權限: {result}")
            except Exception as e:
                st.error(f"連線過程中發生錯誤: {e}")
