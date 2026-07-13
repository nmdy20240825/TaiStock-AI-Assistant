import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock 專業決策系統")

# --- 1. 報價與技術資料抓取 ---
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

# --- 2. 真實三大法人籌碼抓取 (新增累積金額計算) ---
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
            return {"buy_sell": 0, "days": 0, "trend": "資料不足", "accumulated_shares": 0}
            
        df_inst = pd.DataFrame(data["data"])
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        
        daily_net = df_inst.groupby('date')['net_buy'].sum().reset_index()
        daily_net = daily_net.sort_values('date', ascending=False).reset_index(drop=True)
        
        if daily_net.empty:
            return {"buy_sell": 0, "days": 0, "trend": "近期無交易", "accumulated_shares": 0}
            
        latest_net = int(daily_net.iloc[0]['net_buy'] / 1000)
        
        days = 0
        is_buy = latest_net > 0
        accumulated_shares = 0
        
        for val in daily_net['net_buy']:
            if is_buy and val > 0:
                days += 1
                accumulated_shares += val
            elif not is_buy and val < 0:
                days += 1
                accumulated_shares += val
            else:
                break 
                
        if days == 0:
            trend_str = "盤整"
        else:
            trend_str = f"連{days}買" if is_buy else f"連{days}賣"
            
        return {"buy_sell": latest_net, "days": days, "trend": trend_str, "accumulated_shares": accumulated_shares}
        
    except Exception:
        return {"buy_sell": 0, "days": 0, "trend": "API異常", "accumulated_shares": 0}

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
    st.header("📋 持股與專屬風控設定")
    with st.form("add_stock"):
        new_code = st.text_input("代號")
        new_name = st.text_input("名稱")
        new_cost = st.number_input("成本價", value=100.0, step=0.1)
        st.divider()
        new_cap = st.number_input("分配操作資金 (台幣)", value=50000, step=5000)
        new_risk = st.number_input("單筆風險承受度 (%)", value=1.0, step=0.1)
        
        if st.form_submit_button("儲存/更新設定"):
            fetch_stock_data.clear() 
            get_institutional_data.clear()
            portfolio[new_code] = [new_name, new_cost, new_cap, new_risk]
            save_portfolio(portfolio)
            st.rerun()
            
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("確認刪除"):
        if del_code in portfolio:
            del portfolio[del_code]
            save_portfolio(portfolio)
            st.rerun()

# --- 5. 主面板運算與顯示 ---
st.title("⚡ TaiStock 進階決策系統 (全功能究極版)")

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號、成本與專屬資金！")
else:
    summary_data = []
    card_data = []

    for code, info in portfolio.items():
        if len(info) == 2:
            name, cost = info
            cap, risk_pct = 50000.0, 1.0 
        elif len(info) == 4:
            name, cost, cap, risk_pct = info
        else:
            continue
            
        risk_amount = cap * (risk_pct / 100)

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
            
            # --- 升級 1：計算法人連續買賣超金額 (億元) ---
            inst_amount_e = (inst['accumulated_shares'] * price) / 100000000
            if inst_amount_e > 0:
                inst_trend_display = f"{inst['trend']} (流入 {inst_amount_e:.1f}億)"
            elif inst_amount_e < 0:
                inst_trend_display = f"{inst['trend']} (流出 {abs(inst_amount_e):.1f}億)"
            else:
                inst_trend_display = inst['trend']
            
            step1_pass = inst['buy_sell'] > 0 and inst['days'] >= 3
            step2_pass = (k > d) and (rsi > 50)
            step3_pass = ma20 <= price <= (ma20 * 1.03) 
            sop_ready = step1_pass and step2_pass and step3_pass
            
            atr_stop_price = cost - (atr * 2) if cost > 0 else 0
            
            # --- 升級 4：終極判定燈號整合 ---
            if cost > 0 and price <= atr_stop_price:
                final_status = "🔴 禁止進場 (已破停損)"
                status_score = 3
            elif price < ma20 * 0.95:
                final_status = "🔴 禁止進場 (嚴重破線)"
                status_score = 3
            elif sop_ready:
                final_status = "🟢 允許進場 (SOP 齊備)"
                status_score = 1
            else:
                final_status = "🟡 觀望等待"
                status_score = 2
            
            suggested_shares = 0
            if atr > 0:
                raw_shares = int(risk_amount / atr)
                max_affordable_shares = int(cap / price)
                suggested_shares = min(raw_shares, max_affordable_shares)
            
            summary_data.append({
                "代號": code, "名稱": name, "現價": round(price, 2), "成本": round(cost, 2),
                "分配資金": f"{cap:,.0f}",
                "法人動態": inst_trend_display, 
                "建議部位": f"{suggested_shares} 股" if status_score == 1 else "-",
                "終極判定": final_status,
                "_score": status_score # 用於排序的隱藏欄位
            })
            
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi,
                "atr": atr, "bias": bias, "coeff": coeff, "inst": inst, "inst_trend_display": inst_trend_display,
                "cap": cap, "risk_pct": risk_pct, "risk_amount": risk_amount,
                "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "final_status": final_status, "status_score": status_score, "shares": suggested_shares,
                "atr_stop_price": atr_stop_price
            })
            
        except Exception as e:
            st.error(f"分析 {code} 發生錯誤: {e}")
            
    # --- 升級 2：多股戰情總表 (依 SOP 狀態排序) ---
    if summary_data:
        st.markdown("### 📊 持股戰情總表")
        df_summary = pd.DataFrame(summary_data)
        # 依據評分排序：綠燈(1) -> 黃燈(2) -> 紅燈(3)
        df_summary = df_summary.sort_values(by="_score").drop(columns=["_score"])
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        st.divider()

    # --- 繪製個別診斷卡片 ---
    for data in card_data:
        take_profit_price = data['cost'] * 1.10 if data['cost'] > 0 else 0
        
        with st.container(border=True):
            st.subheader(f"{data['name']} ({data['code']}) - 專屬資金: {data['cap']:,.0f} 元")
            
            # --- 升級 3：±2% 預警防護機制 ---
            if data['cost'] > 0: 
                if data['price'] <= data['atr_stop_price']:
                    st.error(f"🚨 **風控警報**：現價 ({data['price']:.2f}) 已跌破停損 ({data['atr_stop_price']:.2f})！請執行紀律。")
                elif data['price'] <= data['atr_stop_price'] * 1.02:
                    st.warning(f"⚠️ **停損預警**：現價 ({data['price']:.2f}) 距離停損點不到 2%，請準備執行紀律。")
                
                if data['price'] >= take_profit_price:
                    st.success(f"🎉 **停利提醒**：現價 ({data['price']:.2f}) 已達 10% 波段目標 ({take_profit_price:.2f})！")
                elif data['price'] >= take_profit_price * 0.98:
                    st.warning(f"⚠️ **停利預警**：現價 ({data['price']:.2f}) 距離 10% 停利目標不到 2%，可考慮分批了結。")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{data['price']:.2f}", delta=f"成本:{data['cost']:.1f}")
            c2.metric("法人動能", data['inst_trend_display'])
            c3.metric("單筆容損", f"{data['risk_amount']:,.0f} 元", delta=f"{data['risk_pct']}% 風險")
            c4.metric("建議部位", f"{data['shares']} 股" if data['status_score'] == 1 else "等待訊號")
            
            st.markdown("##### 📋 嚴格進場 SOP 檢核")
            st.markdown(f"- **Step 1**: 法人連買 ≥ 3 天 ➔ {'✅' if data['step1'] else '❌'}")
            st.markdown(f"- **Step 2**: KD 向上且 RSI > 50 ➔ {'✅' if data['step2'] else '❌'}")
            st.markdown(f"- **Step 3**: 收盤價突破 MA20 (±3% 內) ➔ {'✅' if data['step3'] else '❌'}")
            
            buy_zone_bottom = data['ma20']
            buy_zone_top = data['ma20'] * 1.03
            st.info(f"🎯 **建議進場區間 (20MA 突破)**：{buy_zone_bottom:.2f} ~ {buy_zone_top:.2f} 元")
            
            # 整合後的終極判定
            st.markdown(f"**最終判定**: {data['final_status']}")
            
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
                st.write(f"💡 基準 ATR 停損: {data['atr_stop_price']:.1f} | 🎯 10% 波段停利: {take_profit_price:.1f} | 📈 季線乖離: {data['bias']:.1f}%")
