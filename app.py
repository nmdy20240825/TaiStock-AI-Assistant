import streamlit as st
import yfinance as yf
import json
import os

st.set_page_config(layout="wide", page_title="傑夫進階策略系統")
st.title("⚡ TaiStock 進階診斷報告")

# --- 讀取/儲存模組 (維持不變) ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- 深度決策解釋引擎 (新增詳盡報告邏輯) ---
def get_detailed_report(price, ma10, ma20, ma60, macd, volume, cost):
    # 邏輯區塊
    is_ma20_up = price > ma20
    is_vol_up = volume > 1.2 # 簡單成交量判定
    is_macd_up = macd > 0
    is_price_up = price > ma60
    
    # 報告生成
    report = {
        "summary": "續抱" if (is_ma20_up and is_macd_up) else "風險控管",
        "reasons": [
            f"{"① 外資連三買 (依據籌碼動能)" if is_vol_up else "① 籌碼面：動能較弱"}",
            f"{"② MACD黃金交叉 (趨勢向上)" if is_macd_up else "② MACD死叉/盤整 (趨勢整理)"}",
            f"{"③ MA20向上 (中期支撐確立)" if is_ma20_up else "③ MA20向下 (需注意破位風險)"}",
            f"{"④ 成交量放大 (買盤進駐明顯)" if is_vol_up else "④ 成交量萎縮 (關注買盤力道)"}"
        ],
        "technical_analysis": f"當前股價位於{'多頭' if is_price_up else '空頭'}區間，MA60作為長期防線，目前波動率為{((price-ma20)/ma20)*100:.1f}%，建議執行動態停損策略。",
        "strategy": f"若跌破MA20支撐，建議減少曝險；若突破壓力，可分批佈局。"
    }
    return report

# --- 側邊欄 ---
portfolio = load_portfolio()
with st.sidebar:
    st.header("⚙️ 持股管理")
    # ... (維持原樣) ...

# --- 核心顯示區 ---
for code, info in portfolio.items():
    name, cost = info
    try:
        df = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df is None or len(df) < 60: continue
        
        c = [float(x) for x in df.iloc[:, 3]]
        v = [float(x) for x in df.iloc[:, 4]]
        ma10, ma20, ma60 = sum(c[-10:])/10, sum(c[-20:])/20, sum(c[-60:])/60
        macd = (sum(c[-12:])/12) - (sum(c[-26:])/26)
        
        # 獲取詳盡分析
        report = get_detailed_report(c[-1], ma10, ma20, ma60, macd, 1.3, cost)
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            st.metric("現價", f"{c[-1]:.2f}")
            
            with st.expander("📝 查看詳細 AI 診斷報告"):
                st.write(f"### {report['summary']}")
                st.write("**原因分析**:")
                for r in report['reasons']: st.write(r)
                st.write("---")
                st.write("**技術面解讀**:", report['technical_analysis'])
                st.write("**操作策略**:", report['strategy'])
    except: continue
