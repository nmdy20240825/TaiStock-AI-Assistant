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

# --- 4. 側邊欄 UI (新增資金與部位管理) ---
with st.sidebar:
    st.header("⚙️ 資金與部位管理")
    total_capital = st.number_input("總操作資金 (台幣)", value=100000, step=10000)
    risk_pct = st.number_input("單筆風險承受度 (%)", value=1.0, step=0.1, help="建議單筆虧損控制在總資金的 1%~2%")
    risk_amount = total_capital * (risk_pct / 100)
    st.info(f"單筆最大停損金額: **{risk_amount:,.0f} 元**")
    st.divider()

    st.header("📋 持股名單")
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
st.title("⚡ TaiStock 進階決策系統 (全功能完全體)")

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號與成本！")
else:
    summary_data = []
    card_data = []

    for code, info in portfolio.items():
        name, cost = info
        try:
            df = fetch_stock_data(code)
            if df is None or df.empty or len(df) < 60: 
                continue
            
            c, h, l = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze()
            v = df.get('Volume', pd.Series(0, index=df.index)).squeeze()
            price, volume = float(c.iloc[-1]), float(v.iloc[-1])
            
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
            
            # --- SOP 檢核邏輯 ---
            step1_pass = inst['buy_sell'] > 0 and inst['days'] >= 3
            step2_pass = (k > d) and (rsi > 50)
            step3_pass = ma20 <= price <= (ma20 * 1.03) 
            sop_ready = step1_pass and step2_pass and step3_pass
            
            # --- 零股部位精算邏輯 ---
            # 計算公式：可承受風險金額 / ATR真實波幅 = 建議買進股數
            suggested_shares = 0
            if atr > 0:
                raw_shares = int(risk_amount / atr)
                # 防呆機制：確保買進總額不會超過您的總操作資金
                max_affordable_shares = int(total_capital / price)
                suggested_shares = min(raw_shares, max_affordable_shares)
            
            # 存入總表清單
            summary_data.append({
                "代號": code,
                "名稱": name,
                "現價": round(price, 2),
                "成本": round(cost, 2),
                "法人動能": inst['trend'],
                "AI狀態": "強勢" if k > 50 else "觀望",
                "建議部位": f"{suggested_shares} 股" if sop_ready else "-",
                "SOP判定": "🟢 強烈進場" if sop_ready else "⏳ 觀察中"
            })
            
            # 存入卡片清單
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi,
                "atr": atr, "bias": bias, "coeff": coeff, "inst": inst,
                "step1": step1_pass, "step2": step2_pass, "step3": step3_pass, "sop_ready": sop_ready,
                "shares": suggested_shares
            })
            
        except Exception as e:
            st.error(f"分析 {code} 發生錯誤: {e}")
            
    # --- 繪製多股戰情總表 ---
    if summary_data:
        st.markdown("### 📊 持股戰情總表")
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        st.divider()

    # --- 繪製個別診斷卡片 ---
    for data in card_data:
        atr_stop_price = data['cost'] - (data['atr'] * 2)
        take_profit_price = data['cost'] * 1.10
        
        if data['sop_ready']:
            sop_status_text = f"🟢 **強烈進場訊號** (建議買進: {data['shares']} 股)"
        else:
            sop_status_text = "⏳ 條件未齊，持續觀察"
        
        with st.container(border=True):
            st.subheader(f"{data['name']} ({data['code']})")
            
            if data['cost'] > 0: 
                if data['price'] <= atr_stop_price:
                    st.error(f"🚨 **風控警報**：現價 ({data['price']:.2f}) 已跌破 ATR 停損點 ({atr_stop_price:.2f})！請嚴格執行紀律。")
                elif data['price'] >= take_profit_price:
                    st.success(f"🎉 **停利提醒**：現價 ({data['price']:.2f}) 已達 10% 波段停利目標 ({take_profit_price:.2f})！可考慮分批了結。")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{data['price']:.2f}", delta=f"成本:{data['cost']:.1f}")
            c2.metric("法人動能", data['inst']['trend'], delta=f"{data['inst']['buy_sell']}張")
            c3.metric("AI 狀態", "強勢" if data['k'] > 50 else "觀望")
            c4.metric("建議部位", f"{data['shares']} 股" if data['sop_ready'] else "等待訊號")
            
            st.markdown("##### 📋 嚴格進場 SOP 檢核")
            st.markdown(f"- **Step 1**: 法人連續買超 ≥ 3 天 ➔ {'✅' if data['step1'] else '❌'}")
            st.markdown(f"- **Step 2**: KD 交叉向上 且 RSI > 50 ➔ {'✅' if data['step2'] else '❌'}")
            st.markdown(f"- **Step 3**: 收盤價突破 MA20 (±3% 內) ➔ {'✅' if data['step3'] else '❌'}")
            st.markdown(f"**最終判定**: {sop_status_text}")
            
            with st.expander("🚦 查看完整決策診斷報告"):
                cols = st.columns(3)
                cols[0].markdown(f"### {'🟢' if data['k'] > data['d'] else '🔴'} KD {'向上' if data['k'] > data['d'] else '向下'}")
                cols[1].markdown(f"### {'🟢' if data['macd'] > 0 else '🔴'} MACD {'多頭' if data['macd'] > 0 else '空頭'}")
                cols[2].markdown(f"### {'🟢' if data['coeff'] > 1.15 else '🟡'} 動能 {'起漲中' if data['coeff'] > 1.15 else '盤整中'}")
                st.divider()
                st.write("**[完整技術數據]**")
                st.write(f"成交量: {data['volume']:,.0f} | 均線: MA10:{data['ma10']:.1f} | MA20:{data['ma20']:.1f} | MA60:{data['ma60']:.1f}")
                st.write(f"指標: K:{data['k']:.1f} | D:{data['d']:.1f} | RSI:{data['rsi']:.1f} | MACD:{data['macd']:.3f}")
                st.write("**[專屬交易策略]**")
                st.write(f"💡 基準 ATR 停損: {atr_stop_price:.1f} | 🎯 10% 波段停利: {take_profit_price:.1f} | 📈 季線乖離率: {data['bias']:.1f}%")
