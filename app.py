import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock V2.5 全自動紀律決策系統")

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

# --- 2. 籌碼資料抓取 ---
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
        
        default_res = {
            "buy_sell": 0, "days": 0, "trend": "資料不足", "avg_ratio": 0, 
            "accumulated_shares": 0, "foreign_trend": "無資料", "trust_trend": "無資料",
            "f_days": 0, "t_days": 0
        }
        
        if data.get("msg") != "success" or not data.get("data") or stock_data.empty:
            return default_res
            
        df_inst = pd.DataFrame(data["data"])
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        
        daily_net = df_inst.groupby('date')['net_buy'].sum().sort_index(ascending=False)
        
        foreign_mask = df_inst['name'].str.contains('外資|Foreign', case=False, na=False)
        trust_mask = df_inst['name'].str.contains('投信|Investment', case=False, na=False)
        
        df_foreign = df_inst[foreign_mask].groupby('date')['net_buy'].sum().sort_index(ascending=False)
        df_trust = df_inst[trust_mask].groupby('date')['net_buy'].sum().sort_index(ascending=False)
        
        def calc_trend(series):
            if series.empty: return 0, "無資料"
            days = 0
            for val in series:
                v = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
                if v > 0 and days >= 0: days += 1
                elif v < 0 and days <= 0: days -= 1
                else: break
            if days > 0: return days, f"連{days}買"
            elif days < 0: return days, f"連{abs(days)}賣"
            else: return 0, "盤整"

        f_days, f_trend = calc_trend(df_foreign)
        t_days, t_trend = calc_trend(df_trust)
        
        days = 0
        ratios = []
        accumulated_shares = 0
        
        for date_key in daily_net.index:
            if date_key in stock_data.index:
                net_buy_raw = daily_net[date_key]
                net_buy = float(net_buy_raw.iloc[0]) if isinstance(net_buy_raw, pd.Series) else float(net_buy_raw)
                
                volume_raw = stock_data.loc[date_key, 'Volume']
                volume = float(volume_raw.iloc[0]) if isinstance(volume_raw, pd.Series) else float(volume_raw)
                
                if net_buy > 0:
                    days += 1
                    if volume > 0: ratios.append((net_buy / volume) * 100)
                    accumulated_shares += net_buy
                elif net_buy <= 0 and days > 0: break
                elif net_buy < 0 and days == 0:
                    for sell_date in daily_net.index:
                        val_raw = daily_net[sell_date]
                        val = float(val_raw.iloc[0]) if isinstance(val_raw, pd.Series) else float(val_raw)
                        if val < 0:
                            days -= 1
                            accumulated_shares += val
                        else: break
                    break
                    
        avg_ratio = sum(ratios[:3]) / 3 if len(ratios) >= 3 else (sum(ratios) / len(ratios) if ratios else 0)
        
        if days == 0: trend_str = "盤整"
        elif days > 0: trend_str = f"連{days}買"
        else: trend_str = f"連{abs(days)}賣"
            
        latest_buy_sell = daily_net.iloc[0] if not daily_net.empty else 0
        latest_buy_sell = float(latest_buy_sell.iloc[0]) if isinstance(latest_buy_sell, pd.Series) else float(latest_buy_sell)
            
        return {
            "buy_sell": latest_buy_sell, "days": days, "trend": trend_str, "avg_ratio": float(avg_ratio), 
            "accumulated_shares": float(accumulated_shares), "foreign_trend": f_trend, "trust_trend": t_trend,
            "f_days": f_days, "t_days": t_days
        }
    except Exception:
        return default_res

# --- 3. 持股檔案與歷史軌跡管理 ---
def load_portfolio():
    default_portfolio = {
        "3711": ["日月光投控", 158.0, 20000, 5.0],
        "6414": ["樺漢", 339.2, 20000, 5.0],
        "2317": ["鴻海", 210.0, 20000, 5.0]
    }
    if not os.path.exists('portfolio.json'): return default_portfolio
    with open('portfolio.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return default_portfolio

def save_portfolio(data):
    with open('portfolio.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_history():
    if not os.path.exists('history.json'): return {}
    with open('history.json', 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_history(data):
    with open('history.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

portfolio = load_portfolio()
system_history = load_history()
today_str = datetime.datetime.now().strftime("%Y-%m-%d")

# --- 4. 側邊欄 UI ---
with st.sidebar:
    st.header("📋 持股與風控設定")
    with st.form("add_stock"):
        new_code = st.text_input("代號")
        new_name = st.text_input("名稱 (可留白)")
        new_cost = st.number_input("成本價", value=100.0, step=0.1)
        st.divider()
        new_cap = st.number_input("分配資金 (台幣)", value=20000, step=5000)
        new_risk = st.number_input("單筆風險 (%)", value=5.0, step=0.1)
        
        if st.form_submit_button("更新設定"):
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
            if del_code in system_history:
                del system_history[del_code]
                save_history(system_history)
            st.rerun()

# --- 5. 主面板運算 ---
st.title("⚡ TaiStock V2.5 全自動紀律決策系統")

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
        else: continue
            
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
            
            atr_stop_price = cost - (atr * 2) if cost > 0 else 0
            take_profit_price = cost * 1.10 if cost > 0 else 0
            
            # ===== V2.5 核心戰力公式 =====
            score_inst = min(inst['days'] * 5, 20)
            accumulated_amount = inst['accumulated_shares'] * price
            if accumulated_amount >= 3000000000: score_inst += 20
            elif accumulated_amount >= 1000000000: score_inst += 10
            elif accumulated_amount >= 500000000: score_inst += 5
            
            score_tech = 0
            if k > d: score_tech += 10
            if rsi > 50: score_tech += 10
            if price > ma20: score_tech += 10
            
            score_vol = min((volume / vol_ma5) * 10, 15) if vol_ma5 > 0 else 0
            
            score_risk = 0
            if cost > 0:
                if price > atr_stop_price:
                    score_risk += 10
                    if price >= take_profit_price: score_risk += 5
                    elif price >= cost * 1.05: score_risk += 5
            else:
                score_risk = 15 
                
            ai_score = int(score_inst + score_tech + score_vol + score_risk)
            if cost > 0 and price <= atr_stop_price: ai_score = 0
            ai_score = min(ai_score, 100)
            
            is_bull_aligned = (ma10 > ma20 and ma20 > ma60)
            step1_pass = inst['days'] >= 3 or accumulated_amount >= 1000000000
            step2_pass = k > d and rsi > 50 and volume > vol_ma5
            step3_pass = price > ma20 and is_bull_aligned
            
            if cost > 0 and price <= atr_stop_price: 
                final_status = "🔴 破損"
                ai_explanation = f"🚨 已觸發基準停損 ({atr_stop_price:.1f})，戰力歸零，務必嚴格執行紀律退場。"
            elif cost > 0 and price >= take_profit_price:
                final_status = "🟢 達標"
                ai_explanation = f"🎉 股價已達波段停利目標 ({take_profit_price:.1f})，建議分批獲利了結。"
            elif cost > 0 and price >= cost * 1.05:
                final_status = "🟡 接近停利"
                ai_explanation = f"⚠️ 獲利已拉開空間，接近停利目標，可考慮將停損點上調至成本價確保不敗。"
            elif price < ma20 * 0.95: 
                final_status = "🔴 破線"
                ai_explanation = "股價跌破月線防守區，短線趨勢轉弱，建議優先收回資金觀望。"
            elif ai_score >= 70: 
                final_status = "🟢 進場"
                ai_explanation = f"🚀 綜合戰力極強 ({ai_score}分)，法人與技術指標共振，為當日優質潛力標的。"
            else: 
                final_status = "🟡 觀望"
                ai_explanation = f"⏳ 綜合戰力 {ai_score} 分，條件尚未完全齊備，建議密切盯盤等待突破契機。"
            
            suggested_shares = min(int(risk_amount / atr), int(cap / price)) if atr > 0 else 0
            
            tags = []
            if inst.get('t_days', 0) >= 3: tags.append("🔥投信作帳")
            if inst.get('f_days', 0) >= 3: tags.append("🌊外資波段")
            if is_bull_aligned and price > ma20: tags.append("🚀多頭起漲")
            elif price < ma60 and ma20 < ma60: tags.append("❄️弱勢空頭")
            if not tags: tags.append("⏳區間震盪")
            
            if code not in system_history:
                system_history[code] = {}
            system_history[code][today_str] = {
                "score": ai_score,
                "status": final_status,
                "price": price
            }
            if len(system_history[code]) > 10:
                oldest_date = sorted(system_history[code].keys())[0]
                del system_history[code][oldest_date]
            
            summary_data.append({
                "代號": code, "名稱": name, "現價": round(price, 2), "成本": round(cost, 2),
                "AI分數": ai_score, "股性標籤": " | ".join(tags[:2]),
                "風控點": f"{atr_stop_price:.1f}/{take_profit_price:.1f}" if cost > 0 else "-/-",
                "判定": final_status
            })
            
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume, "vol_ma5": vol_ma5,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi,
                "atr": atr, "bias": bias, "inst": inst, "tags": tags,
                "cap": cap, "risk_amount": risk_amount,
                "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "ai_score": ai_score, "final_status": final_status, "shares": suggested_shares,
                "atr_stop_price": atr_stop_price, "take_profit_price": take_profit_price,
                "ai_explanation": ai_explanation
            })
            
        except Exception as e:
            st.error(f"分析 {code} 發生錯誤: {e}")
            
    save_history(system_history)
            
    if summary_data:
        df_summary = pd.DataFrame(summary_data).sort_values(by="AI分數", ascending=False).reset_index(drop=True)
        st.markdown("### 🎯 盤前決策儀表板")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🏆 最佳標的", f"{df_summary.iloc[0]['名稱']}", f"戰力: {df_summary.iloc[0]['AI分數']}分")
        with c2: st.metric("⚠️ 弱勢警告", f"{df_summary.iloc[-1]['名稱']}", f"戰力: {df_summary.iloc[-1]['AI分數']}分", delta_color="inverse")
        with c3: st.metric("🟢 潛力檔數", f"{len(df_summary[df_summary['AI分數']>=70])} 檔", "可佈局" if len(df_summary[df_summary['AI分數']>=70]) > 0 else "耐心等待", delta_color="normal" if len(df_summary[df_summary['AI分數']>=70]) > 0 else "off")
        st.divider()

    # --- 新增：每日紀律檢核清單 (SOP) ---
    if card_data:
        st.markdown("### ✅ 每日紀律檢核清單 (SOP)")
        with st.expander("展開今日操作任務", expanded=True):
            action_sell = [] 
            action_buy = [] 
            action_watch = [] 
            
            for data in card_data:
                if data['final_status'] == "🔴 破損":
                    action_sell.append(f"🚨 **停損退場**：{data['name']} ({data['code']}) 現價 {data['price']} 跌破防守點 {data['atr_stop_price']:.1f}，收回資金。")
                elif data['final_status'] == "🟢 達標":
                    action_sell.append(f"🎉 **獲利了結**：{data['name']} ({data['code']}) 達波段目標 {data['take_profit_price']:.1f}，執行分批停利。")
                elif data['final_status'] == "🟢 進場":
                    action_buy.append(f"🎯 **進場佈局**：{data['name']} ({data['code']}) 戰力達 {data['ai_score']} 分，建議部位：{data['shares']} 股。")
                elif data['final_status'] == "🟡 接近停利":
                    action_watch.append(f"⚠️ **防守上調**：{data['name']} ({data['code']}) 獲利脫離成本，將停損設為成本價。")
                elif data['final_status'] == "🔴 破線":
                    action_watch.append(f"📉 **弱勢預警**：{data['name']} ({data['code']}) 跌破月線，確認是否減碼。")

            st.markdown("#### 🟥 優先執行 (風控與停利)")
            if not action_sell: st.write("✅ 今日無急迫停損/停利需求")
            for i, task in enumerate(action_sell): st.checkbox(task, key=f"sell_{i}")
            
            st.markdown("#### 🟩 佈局清單 (高勝率機會)")
            if not action_buy: st.write("⏸️ 今日無符合標準的進場標的，耐心等待")
            for i, task in enumerate(action_buy): st.checkbox(task, key=f"buy_{i}")
            
            st.markdown("#### 🟨 觀察追蹤 (防守與調整)")
            if not action_watch: st.write("👀 目前無特別需要調整的持股")
            for i, task in enumerate(action_watch): st.checkbox(task, key=f"watch_{i}")
        st.divider()

    st.markdown("### 📊 AI 深度解析清單")
    
    card_data = sorted(card_data, key=lambda x: x['ai_score'], reverse=True)
    
    for data in card_data:
        with st.container(border=True):
            hist_records = system_history.get(data['code'], {})
            sorted_dates = sorted(hist_records.keys(), reverse=True)
            delta_str = ""
            if len(sorted_dates) > 1:
                yesterday_score = hist_records[sorted_dates[1]]['score']
                diff = data['ai_score'] - yesterday_score
                if diff > 0: delta_str = f" (🔺+{diff})"
                elif diff < 0: delta_str = f" (🔻{diff})"
                else: delta_str = " (➖ 持平)"

            st.markdown(f"#### {data['name']} ({data['code']}) - {' '.join(data['tags'][:2])} {delta_str}")
            
            st.markdown(f"<div style='font-size: 0.9em; margin-bottom: 5px; color: #cbd5e1;'>SOP 檢核：籌碼 {'🟢' if data['step1'] else '⚪'} | 量能 {'🟢' if data['step2'] else '⚪'} | 趨勢 {'🟢' if data['step3'] else '⚪'}</div>", unsafe_allow_html=True)
            
            st.progress(data['ai_score'] / 100)
            
            if data['cost'] > 0 and data['price'] <= data['atr_stop_price']:
                st.error(f"🚨 風控警報：已跌破停損 ({data['atr_stop_price']:.2f})！")
            elif data['cost'] > 0 and data['price'] >= data['take_profit_price']:
                st.success(f"🎉 停利提醒：已達波段目標 ({data['take_profit_price']:.2f})！")
            elif data['cost'] > 0 and data['price'] >= data['cost'] * 1.05:
                st.warning(f"🟡 接近停利：目前獲利空間已拉開，請留意出場時機。")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("現價", f"{data['price']:.2f}")
            
            cost_str = f"{data['cost']:.2f}" if data['cost'] > 0 else "-"
            col_a.markdown(
                f"<div style='margin-top: -15px;'><span style='font-size: 0.85em; color: #94a3b8; background-color: #334155; padding: 2px 6px; border-radius: 4px;'>成本 {cost_str}</span></div>", 
                unsafe_allow_html=True
            )
            
            col_b.metric("總法人", f"{data['inst']['trend']}")
            col_c.metric("判定", data['final_status'])
            col_d.metric("部位", f"{data['shares']}股" if data['final_status'] == "🟢 進場" else "-")
            
            st.write("") 
            
            tab1, tab2, tab3, tab4 = st.tabs(["⚙️ SOP與籌碼", "📉 技術數據", "🛡️ 風控點位", "⏳ 決策時間軸"])
            
            with tab1:
                st.info(f"**🤖 AI 總結**：{data['ai_explanation']}")
                st.markdown(f"- **外資動向**: {data['inst']['foreign_trend']} | **投信動向**: {data['inst']['trust_trend']}")
                st.markdown(f"- **S1 籌碼**: 法人買超與比例 {'🟢' if data['step1'] else '⚪'}")
                st.markdown(f"- **S2 量能**: KD向上 / RSI>50 / 放量 {'🟢' if data['step2'] else '⚪'}")
                st.markdown(f"- **S3 趨勢**: MA20防守 / 多頭排列 {'🟢' if data['step3'] else '⚪'}")
                
            with tab2:
                c_t1, c_t2 = st.columns(2)
                c_t1.write(f"**今日量**: {data['volume']:,.0f} | **5日均量**: {data['vol_ma5']:,.0f}")
                c_t1.write(f"**K**: {data['k']:.1f} | **D**: {data['d']:.1f} | **RSI**: {data['rsi']:.1f}")
                c_t2.write(f"**MA10**: {data['ma10']:.1f}")
                c_t2.write(f"**MA20**: {data['ma20']:.1f}")
                c_t2.write(f"**MA60**: {data['ma60']:.1f}")
                
            with tab3:
                st.info(f"🎯 **打擊防守區間 (20MA 突破)**：{data['ma20']:.2f} ~ {data['ma20']*1.03:.2f} 元")
                st.write(f"**基準 ATR 停損**: {data['atr_stop_price']:.1f}")
                st.write(f"**10% 波段停利**: {data['take_profit_price']:.1f}")
                
            with tab4:
                st.markdown("##### 📅 近期戰力與決策軌跡")
                if len(sorted_dates) == 0:
                    st.write("尚無歷史資料，系統將自今日起開始記錄。")
                else:
                    for d in sorted_dates[:5]: 
                        h_score = hist_records[d]['score']
                        h_status = hist_records[d]['status']
                        h_price = hist_records[d].get('price', 0)
                        score_badge = "🟢" if h_score >= 70 else ("🟡" if h_score >= 40 else "🔴")
                        st.markdown(f"- **{d}**：戰力 {h_score} 分 {score_badge} | 判定：{h_status} | 收盤價：{h_price:.2f}")
