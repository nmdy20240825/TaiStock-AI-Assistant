import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock V2 專業決策系統")

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

# --- 2. 真實法人籌碼抓取 ---
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
        
        ticker = f"{code}.TW" if not code.endswith(('.TW', '.TWO', '.US')) else code
        stock_data = yf.download(ticker, period="1mo", progress=False)
        
        if data.get("msg") != "success" or not data.get("data") or stock_data.empty:
            return {"buy_sell": 0, "days": 0, "trend": "資料不足", "avg_ratio": 0, "accumulated_shares": 0}
            
        df_inst = pd.DataFrame(data["data"])
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        daily_net = df_inst.groupby('date')['net_buy'].sum().sort_index(ascending=False)
        
        days = 0
        ratios = []
        accumulated_shares = 0
        
        for date_key in daily_net.index:
            if date_key in stock_data.index:
                net_buy_raw = daily_net[date_key]
                if isinstance(net_buy_raw, pd.Series):
                    net_buy_raw = net_buy_raw.iloc[0]
                net_buy = float(net_buy_raw)
                
                volume_raw = stock_data.loc[date_key, 'Volume']
                if isinstance(volume_raw, pd.Series):
                    volume_raw = volume_raw.iloc[0]
                volume = float(volume_raw)
                
                if net_buy > 0:
                    days += 1
                    if volume > 0:
                        ratios.append((net_buy / volume) * 100)
                    accumulated_shares += net_buy
                elif net_buy <= 0 and days > 0:
                    break
                elif net_buy < 0 and days == 0:
                    for sell_date in daily_net.index:
                        val_raw = daily_net[sell_date]
                        if isinstance(val_raw, pd.Series): val_raw = val_raw.iloc[0]
                        val = float(val_raw)
                        
                        if val < 0:
                            days -= 1
                            accumulated_shares += val
                        else:
                            break
                    break
                    
        avg_ratio = sum(ratios[:3]) / 3 if len(ratios) >= 3 else (sum(ratios) / len(ratios) if ratios else 0)
        
        if days == 0:
            trend_str = "盤整"
        elif days > 0:
            trend_str = f"連{days}買"
        else:
            trend_str = f"連{abs(days)}賣"
            
        latest_buy_sell = daily_net.iloc[0] if not daily_net.empty else 0
        if isinstance(latest_buy_sell, pd.Series): latest_buy_sell = latest_buy_sell.iloc[0]
            
        return {"buy_sell": float(latest_buy_sell), "days": days, "trend": trend_str, "avg_ratio": float(avg_ratio), "accumulated_shares": float(accumulated_shares)}
        
    except Exception:
        return {"buy_sell": 0, "days": 0, "trend": "API異常", "avg_ratio": 0, "accumulated_shares": 0}

# --- 3. 持股檔案管理 ---
def load_portfolio():
    default_portfolio = {
        "3711": ["日月光投控", 158.0, 20000, 5.0],
        "6414": ["樺漢", 339.2, 20000, 5.0],
        "2317": ["鴻海", 210.0, 20000, 5.0]
    }
    if not os.path.exists('portfolio.json'): 
        return default_portfolio
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return default_portfolio

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()

# --- 4. 側邊欄 UI ---
with st.sidebar:
    st.header("📋 持股與專屬風控設定")
    with st.form("add_stock"):
        new_code = st.text_input("代號")
        new_name = st.text_input("名稱 (可留白)")
        new_cost = st.number_input("成本價", value=100.0, step=0.1)
        st.divider()
        new_cap = st.number_input("分配操作資金 (台幣)", value=20000, step=5000)
        new_risk = st.number_input("單筆風險承受度 (%)", value=5.0, step=0.1)
        
        if st.form_submit_button("儲存/更新設定"):
            if new_code:
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
st.title("⚡ TaiStock V2 進階決策系統")

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號！")
else:
    summary_data = []
    card_data = []

    for code, info in portfolio.items():
        if len(info) == 2:
            name, cost = info
            cap, risk_pct = 20000.0, 5.0 
        elif len(info) == 4:
            name, cost, cap, risk_pct = info
        else:
            continue
            
        risk_amount = cap * (risk_pct / 100)

        try:
            df = fetch_stock_data(code)
            if df is None or df.empty or len(df) < 60: continue
            
            c = df['Close'].squeeze()
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            h = df['High'].squeeze()
            if isinstance(h, pd.DataFrame): h = h.iloc[:, 0]
            l = df['Low'].squeeze()
            if isinstance(l, pd.DataFrame): l = l.iloc[:, 0]
            v = df.get('Volume', pd.Series(0, index=df.index)).squeeze()
            if isinstance(v, pd.DataFrame): v = v.iloc[:, 0]
                
            price = float(c.iloc[-1])
            volume = float(v.iloc[-1])
            
            # --- 新增：計算 5日均量 ---
            vol_ma5 = float(v.rolling(5).mean().iloc[-1])
            
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
            rsi_raw = 100 - (100 / (1 + (np.nan_to_num(up) / (np.nan_to_num(down) + 0.001))))
            rsi = float(rsi_raw) if not isinstance(rsi_raw, pd.Series) else float(rsi_raw.iloc[-1])
            
            atr = sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]) / 14
            atr = float(atr)
            bias = float(((price - ma60) / ma60) * 100)
            
            inst = get_institutional_data(code)
            
            # --- AI 決策引擎 2.0 (階梯式計分與動能檢核) ---
            ai_score = 0
            
            # S1 籌碼面 (滿分 30分) - 導入階梯式給分
            if inst['days'] >= 3: ai_score += 15
            elif inst['days'] == 2: ai_score += 10
            elif inst['days'] == 1: ai_score += 5
            
            if inst['avg_ratio'] >= 15.0: ai_score += 15
            elif inst['avg_ratio'] >= 10.0: ai_score += 10
            elif inst['avg_ratio'] >= 5.0: ai_score += 5
                
            # S2 技術指標 (滿分 30分) - 導入量能檢核
            if k > d: ai_score += 10
            if rsi > 50: ai_score += 10
            if volume > vol_ma5: ai_score += 10  # 新增：量能突破
            
            # S3 趨勢與防禦 (滿分 40分) - 導入多頭排列保護
            if price > ma20: ai_score += 15
            if price <= (ma20 * 1.03): ai_score += 15
            if ma10 > ma20 and ma20 > ma60: ai_score += 10 # 新增：多頭排列
            
            # 確保分數不超過 100
            ai_score = min(ai_score, 100)
            
            # 終極過濾：是否完全達標最核心的進場條件
            step1_pass = inst['days'] >= 3 and inst['avg_ratio'] >= 15.0
            step2_pass = k > d and rsi > 50 and volume > vol_ma5
            step3_pass = price > ma20 and price <= (ma20 * 1.03)
            sop_ready = step1_pass and step2_pass and step3_pass
            
            atr_stop_price = cost - (atr * 2) if cost > 0 else 0
            take_profit_price = cost * 1.10 if cost > 0 else 0
            
            # --- 手機版 UI 極簡化狀態文字 ---
            if cost > 0 and price <= atr_stop_price:
                final_status = "🔴 破損"
            elif price < ma20 * 0.95:
                final_status = "🔴 破線"
            elif sop_ready:
                final_status = "🟢 進場"
            else:
                final_status = "🟡 觀望"
            
            suggested_shares = 0
            if atr > 0:
                raw_shares = int(risk_amount / atr)
                max_affordable_shares = int(cap / price)
                suggested_shares = min(raw_shares, max_affordable_shares)
            
            risk_str = f"{atr_stop_price:.1f} / {take_profit_price:.1f}" if cost > 0 else "- / -"
            
            summary_data.append({
                "代號": code, 
                "名稱": name, 
                "現價": round(price, 2), 
                "成本": round(cost, 2),
                "AI分數": ai_score,
                "風控點(損/利)": risk_str,
                "終極判定": final_status
            })
            
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume, "vol_ma5": vol_ma5,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi,
                "atr": atr, "bias": bias, "inst": inst,
                "cap": cap, "risk_pct": risk_pct, "risk_amount": risk_amount,
                "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "ai_score": ai_score, "final_status": final_status, "shares": suggested_shares,
                "atr_stop_price": atr_stop_price, "take_profit_price": take_profit_price
            })
            
        except Exception as e:
            st.error(f"分析 {code} 發生錯誤: {e}")
            
    # --- V2.0 戰情儀表板 (Dashboard) ---
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values(by="AI分數", ascending=False).reset_index(drop=True)
        
        st.markdown("### 🎯 盤前決策儀表板")
        
        best_stock = df_summary.iloc[0]
        worst_stock = df_summary.iloc[-1]
        ready_count = len(df_summary[df_summary["AI分數"] >= 80])
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="🏆 最佳標的", value=f"{best_stock['名稱']} ({best_stock['代號']})", delta=f"戰力: {best_stock['AI分數']} 分")
        with c2:
            st.metric(label="⚠️ 弱勢警告", value=f"{worst_stock['名稱']} ({worst_stock['代號']})", delta=f"戰力: {worst_stock['AI分數']} 分", delta_color="inverse")
        with c3:
            st.metric(label="🟢 潛力檔數", value=f"{ready_count} 檔", delta="可佈局" if ready_count > 0 else "耐心等待", delta_color="normal" if ready_count > 0 else "off")
            
        st.divider()

    # --- 繪製個別完整診斷卡片 ---
    card_data = sorted(card_data, key=lambda x: x['ai_score'], reverse=True)
    
    st.markdown("### 📊 AI 深度解析清單")
    for data in card_data:
        with st.container(border=True):
            # 引入戰力條設計，直觀顯示 AI 分數
            st.markdown(f"#### {data['name']} ({data['code']}) - 綜合戰力：{data['ai_score']} 分")
            st.progress(data['ai_score'] / 100)
            
            if data['cost'] > 0: 
                if data['price'] <= data['atr_stop_price']:
                    st.error(f"🚨 **風控警報**：現價 ({data['price']:.2f}) 已跌破停損 ({data['atr_stop_price']:.2f})！")
                
                if data['price'] >= data['take_profit_price']:
                    st.success(f"🎉 **停利提醒**：現價 ({data['price']:.2f}) 已達 10% 波段目標 ({data['take_profit_price']:.2f})！")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("現價", f"{data['price']:.2f}")
            col_b.metric("法人", f"{data['inst']['trend']}")
            # 使用精簡字串解決截斷問題
            col_c.metric("判定", data['final_status'])
            col_d.metric("部位", f"{data['shares']} 股" if data['final_status'] == "🟢 進場" else "-")
            
            st.markdown("##### 🔬 戰力因子拆解")
            st.markdown(f"- **S1 籌碼動能**: 法人買超天數與比例 {'🟢' if data['inst']['days'] > 0 else '⚪'} ")
            st.markdown(f"- **S2 技術量能**: KD向上 / RSI>50 / 放量 {'🟢' if data['step2'] else '⚪'} ")
            st.markdown(f"- **S3 趨勢防護**: MA20防守 / 多頭排列 {'🟢' if data['step3'] else '⚪'} ")
            
            buy_zone_bottom = data['ma20']
            buy_zone_top = data['ma20'] * 1.03
            st.info(f"🎯 **打擊防守區間 (20MA 突破)**：{buy_zone_bottom:.2f} ~ {buy_zone_top:.2f} 元")
            
            with st.expander("🚦 查看底層技術數據與風控點"):
                st.write(f"**今日量**: {data['volume']:,.0f} | **5日均量**: {data['vol_ma5']:,.0f}")
                st.write(f"**MA10**: {data['ma10']:.1f} | **MA20**: {data['ma20']:.1f} | **MA60**: {data['ma60']:.1f}")
                st.write(f"**K**: {data['k']:.1f} | **D**: {data['d']:.1f} | **RSI**: {data['rsi']:.1f}")
                st.write(f"**基準 ATR 停損**: {data['atr_stop_price']:.1f} | **10% 停利**: {data['take_profit_price']:.1f}")
