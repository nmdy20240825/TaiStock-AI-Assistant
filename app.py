import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock 究極戰情室")

# --- 1. 資料獲取與籌碼計算 ---
@st.cache_data(ttl=300) 
def fetch_stock_data(code):
    try:
        ticker = f"{code}.TW" if not code.endswith(('.TW', '.TWO', '.US')) else code
        df = yf.download(ticker, period="6mo", progress=False)
        return df if not df.empty else yf.download(f"{code}.TWO", period="6mo", progress=False)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)  
def get_institutional_data(code):
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": code, "start_date": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"), "end_date": datetime.datetime.now().strftime("%Y-%m-%d")}
        resp = requests.get(url, params=params, timeout=5)
        inst_data = resp.json().get("data", [])
        
        ticker = f"{code}.TW" if not code.endswith(('.TW', '.TWO', '.US')) else code
        stock_data = yf.download(ticker, period="1mo", progress=False)
        
        if not inst_data or stock_data.empty: return {"days": 0, "avg_ratio": 0, "trend": "資料不足"}
        
        df_inst = pd.DataFrame(inst_data)
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        daily_inst = df_inst.groupby('date')['net_buy'].sum()
        
        days, ratios = 0, []
        for i in range(1, 10): 
            date_key = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            if date_key in daily_inst.index and date_key in stock_data.index:
                net_buy = daily_inst[date_key]
                vol = stock_data.loc[date_key, 'Volume']
                if net_buy > 0:
                    days += 1
                    ratios.append((net_buy / vol) * 100)
                else: break
        
        avg_ratio = sum(ratios[:3]) / 3 if len(ratios) >= 3 else 0
        trend_str = f"連{days}買" if days >= 3 else "觀察中"
        return {"days": days, "avg_ratio": avg_ratio, "trend": trend_str}
    except: return {"days": 0, "avg_ratio": 0, "trend": "API異常"}

# --- 2. 持股管理 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()

# --- 3. 側邊欄 ---
with st.sidebar:
    st.header("📋 持股與風控設定")
    with st.form("add_stock"):
        c_code = st.text_input("代號")
        c_name = st.text_input("名稱")
        c_cost = st.number_input("成本", value=100.0)
        c_cap = st.number_input("資金", value=50000)
        c_risk = st.number_input("風險 (%)", value=1.0)
        if st.form_submit_button("儲存/更新"):
            portfolio[c_code] = [c_name, c_cost, c_cap, c_risk]
            save_portfolio(portfolio); st.rerun()

# --- 4. 主面板 ---
st.title("⚡ TaiStock 究極戰情完全體")
if not portfolio: st.info("👈 請新增股票")
else:
    summary_data, card_data = [], []
    for code, info in portfolio.items():
        name, cost, cap, risk_pct = info
        df = fetch_stock_data(code)
        if df.empty: continue
        
        c = df['Close'].squeeze(); h = df['High'].squeeze(); l = df['Low'].squeeze()
        price = float(c.iloc[-1])
        ma20 = float(c.rolling(20).mean().iloc[-1])
        atr = sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]) / 14
        
        k, d = 50.0, 50.0 # 簡化指標邏輯
        inst = get_institutional_data(code)
        
        # 核心邏輯
        step1 = inst['days'] >= 3 and inst['avg_ratio'] >= 15.0
        step3 = ma20 <= price <= (ma20 * 1.03)
        sop_ready = step1 and step3
        
        status = "🟢 允許進場 (主力共振)" if sop_ready else ("🟡 觀望等待" if inst['days'] > 0 else "🔴 禁止進場")
        score = 1 if sop_ready else (2 if inst['days'] > 0 else 3)
        
        summary_data.append({"代號": code, "名稱": name, "法人佔比": f"{inst['avg_ratio']:.1f}%", "狀態": status, "_score": score})
        card_data.append({"code": code, "name": name, "price": price, "cost": cost, "inst": inst, "step1": step1, "step3": step3, "status": status, "score": score, "atr": atr, "ma20": ma20})

    st.markdown("### 📊 戰情總表")
    st.dataframe(pd.DataFrame(summary_data).sort_values("_score").drop(columns=["_score"]), use_container_width=True, hide_index=True)
    
    for data in card_data:
        with st.container(border=True):
            st.subheader(f"{data['name']} ({data['code']})")
            c1, c2, c3 = st.columns(3)
            c1.metric("現價", f"{data['price']:.2f}", delta=f"均線:{data['ma20']:.1f}")
            c2.metric("主力占比", f"{data['inst']['avg_ratio']:.1f}%")
            c3.markdown(f"**判定**: {data['status']}")
            
            # 預警機制
            stop_loss = data['cost'] - (data['atr'] * 2)
            if data['price'] <= stop_loss: st.error("🚨 已觸發 ATR 停損")
            elif data['price'] <= stop_loss * 1.02: st.warning("⚠️ 即將觸發停損預警")
            
            st.markdown(f"- **籌碼濾網** (≥15%): {'✅' if data['step1'] else '❌'} | **均線趨勢**: {'✅' if data['step3'] else '❌'}")
            st.info(f"🎯 建議進場區間: {data['ma20']:.2f} ~ {data['ma20']*1.03:.2f}")

    st.divider()
    st.markdown("### 🚦 燈號說明")
    st.markdown("- **🟢 允許進場**：籌碼與技術面共振，強勢股確認。- **🟡 觀望等待**：法人動能不足。- **🔴 禁止進場**：趨勢轉弱。")
