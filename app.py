import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock 專業決策系統")

# --- 1. 報價與技術資料抓取 (支援上市/上櫃) ---
@st.cache_data(ttl=300) 
def fetch_stock_data(code):
    try:
        if code.endswith('.TW') or code.endswith('.TWO') or code.endswith('.US'):
            return yf.download(code, period="6mo", progress=False)
            
        df_tw = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df_tw is not None and not df_tw.empty and len(df_tw) > 0:
            return df_tw
            
        df_two = yf.download(f"{code}.TWO", period="6mo", progress=False)
        return df_two
    except Exception:
        return pd.DataFrame()

# --- 2. 真實三大法人籌碼抓取 (FinMind API) ---
@st.cache_data(ttl=3600)  
def get_institutional_data(code):
    try:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": code,
            "start_date": start_date,
            "end_date": end_date,
        }
        resp = requests.get(url, params=parameter, timeout=5)
        data = resp.json()
        
        if data.get("msg") != "success" or not data.get("data"):
            return {"buy_sell": 0, "days": 0, "trend": "資料不足"}
            
        df_inst = pd.DataFrame(data["data"])
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        
        daily_net = df_inst.groupby('date')['net_buy'].sum().reset_index()
        daily_net = daily_net.sort_values('date', ascending=False).reset_index(drop=True)
        
        if daily_net.empty:
            return {"buy_sell": 0, "days": 0, "trend": "近期無交易"}
            
        latest_net = int(daily_net.iloc[0]['net_buy'] / 1000)
        
        days = 0
        is_buy = latest_net > 0
        
        for val in daily_net['net_buy']:
            if is_buy and val > 0:
                days += 1
            elif not is_buy and val < 0:
                days += 1
            else:
                break 
                
        if days == 0:
            trend_str = "盤整"
        else:
            trend_str = f"連{days}買" if is_buy else f"連{days}賣"
            
        return {"buy_sell": latest_net, "days": days, "trend": trend_str}
        
    except Exception:
        return {"buy_sell": 0, "days": 0, "trend": "API異常"}

# --- 3. 持股檔案管理 ---
def load_portfolio():
    if not os.path.exists('portfolio.json'): return {}
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()

# --- 4. 側邊欄 UI ---
with st.sidebar:
    st.header("⚙️ 持股管理")
    with st.form("add_stock"):
        new_code = st.text_input("代號")
        new_name = st.text_input("名稱")
        new_cost = st.number_input("成本", value=100.0, step=0.1)
        if st.form_submit_button("儲存/更新"):
            fetch_stock_data.clear() 
            get_institutional_data.clear()
            portfolio[new_code] = [new_name, new_cost]
            save_portfolio(portfolio)
            st.rerun()
            
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("確認刪除"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

# --- 5. 主面板運算與顯示 ---
st.title("⚡ TaiStock 進階決策系統 (第一階段：嚴格風控版)")

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號與成本！")

for code, info in portfolio.items():
    name, cost = info
    try:
        df = fetch_stock_data(code)
        
        if df is None or df.empty or len(df) < 60: 
            st.warning(f"⚠️ {name} ({code}) 歷史資料不足或 API 暫時無回應，請稍後再試。")
            continue
        
        # 基本數據
        c, h, l = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
        v = df.get('Volume', pd.Series(0, index=df.index)).squeeze()
        price, volume = float(c.iloc[-1]), float(v.iloc[-1])
        
        # 技術指標
        ma10 = float(c.rolling(10).mean().iloc[-1])
        ma20 = float(c.rolling(20).mean().iloc[-1])
        ma60 = float(c.rolling(60).mean().iloc[-1])
        macd = float((c.rolling(12).mean().iloc[-1]) - (c.rolling(26).mean().iloc[-1]))
        
        rsv_val = (price - float(l.rolling(9).min().iloc[-1])) / (float(h.rolling(9).max().iloc[-1]) - float(l.rolling(9).min().iloc[-1]) + 0.001) * 100
        k = float(2/3 * 50 + 1/3 * np.nan_to_num(rsv_val))
        d = float(2/3 * 50 + 1/3 * k)
        
        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        down = -1 * delta.clip(upper=0).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (np.nan_to_num(up) / (np.nan_to_num(down) + 0.001))))
        
        atr = sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]) / 14
        bias = ((price - ma60) / ma60) * 100
        coeff = price / ma20
        inst = get_institutional_data(code)
        
        # --- 模組 A：嚴格進場 SOP 檢核邏輯 ---
        step1_pass = inst['buy_sell'] > 0 and inst['days'] >= 3
        step2_pass = (k > d) and (rsi > 50)
        # 突破 MA20 且乖離不超過 3% (確保不追高)
        step3_pass = ma20 <= price <= (ma20 * 1.03) 
        
        sop_ready = step1_pass and step2_pass and step3_pass
        sop_status_text = "🟢 **強烈進場訊號** (符合嚴格 SOP)" if sop_ready else "⏳ 條件未齊，持續觀察"
        
        # --- 模組 B：動態風控警示邏輯 (基於輸入成本) ---
        atr_stop_price = cost - (atr * 2)
        take_profit_price = cost * 1.10
        
        with st.container(border=True):
            st.subheader(f"{name} ({code})")
            
            # --- 風控警示橫幅 ---
            if cost > 0: # 確保有輸入成本才計算
                if price <= atr_stop_price:
                    st.error(f"🚨 **風控警報**：現價 ({price:.2f}) 已跌破 ATR 停損點 ({atr_stop_price:.2f})！請嚴格執行紀律。")
                elif price >= take_profit_price:
                    st.success(f"🎉 **停利提醒**：現價 ({price:.2f}) 已達 10% 波段停利目標 ({take_profit_price:.2f})！可考慮分批了結。")
            
            # --- 核心看板 ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{price:.2f}", delta=f"成本:{cost:.1f}")
            c2.metric("法人動能", inst['trend'], delta=f"{inst['buy_sell']}張")
            c3.metric("AI 狀態", "強勢" if k > 50 else "觀望")
            c4.metric("股性判別", "🚀 起漲股" if coeff > 1.15 else "📊 一般股")
            
            # --- 嚴格 SOP 檢核表 ---
            st.markdown("##### 📋 嚴格進場 SOP 檢核")
            st.markdown(f"- **Step 1**: 法人連續買超 ≥ 3 天 ➔ {'✅' if step1_pass else '❌'}")
            st.markdown(f"- **Step 2**: KD 交叉向上 且 RSI > 50 ➔ {'✅' if step2_pass else '❌'}")
            st.markdown(f"- **Step 3**: 收盤價突破 MA20 (±3% 內) ➔ {'✅' if step3_pass else '❌'}")
            st.markdown(f"**最終判定**: {sop_status_text}")
            
            with st.expander("🚦 查看完整決策診斷報告"):
                cols = st.columns(3)
                cols[0].markdown(f"### {'🟢' if k > d else '🔴'} KD {'向上' if k > d else '向下'}")
                cols[1].markdown(f"### {'🟢' if macd > 0 else '🔴'} MACD {'多頭' if macd > 0 else '空頭'}")
                cols[2].markdown(f"### {'🟢' if coeff > 1.15 else '🟡'} 動能 {'起漲中' if coeff > 1.15 else '盤整中'}")
                st.divider()
                st.write("**[完整技術數據]**")
                st.write(f"成交量: {volume:,.0f} | 均線: MA10:{ma10:.1f} | MA20:{ma20:.1f} | MA60:{ma60:.1f}")
                st.write(f"指標: K:{k:.1f} | D:{d:.1f} | RSI:{rsi:.1f} | MACD:{macd:.3f}")
                st.write("**[專屬交易策略]**")
                st.write(f"💡 基準 ATR 停損: {atr_stop_price:.1f} | 🎯 10% 波段停利: {take_profit_price:.1f} | 📈 季線乖離率: {bias:.1f}%")
                st.caption("• 系統自動依據輸入成本計算停損與停利提醒")
                
    except Exception as e:
        st.error(f"分析 {code} 發生錯誤: {e}")
