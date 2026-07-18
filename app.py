import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock V2 進階決策系統")

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

# --- 2. Phase 2 升級：真實法人籌碼 (外資/投信 獨立拆解) ---
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
        
        # 總和計算
        daily_net = df_inst.groupby('date')['net_buy'].sum().sort_index(ascending=False)
        
        # 拆解外資與投信 (導入中英文雙重辨識容錯)
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

# --- 3. 持股檔案管理 ---
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

portfolio = load_portfolio()

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
            st.rerun()

# --- 5. 主面板運算 ---
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
            
            ai_score = 0
            if inst['days'] >= 3: ai_score += 15
            elif inst['days'] == 2: ai_score += 10
            elif inst['days'] == 1: ai_score += 5
            
            if inst['avg_ratio'] >= 15.0: ai_score += 15
            elif inst['avg_ratio'] >= 10.0: ai_score += 10
            elif inst['avg_ratio'] >= 5.0: ai_score += 5
                
            if k > d: ai_score += 10
            if rsi > 50: ai_score += 10
            if volume > vol_ma5: ai_score += 10
            
            if price > ma20: ai_score += 15
            if price <= (ma20 * 1.03): ai_score += 15
            is_bull_aligned = (ma10 > ma20 and ma20 > ma60)
            if is_bull_aligned: ai_score += 10 
            
            ai_score = min(ai_score, 100)
            
            step1_pass = inst['days'] >= 3 and inst['avg_ratio'] >= 15.0
            step2_pass = k > d and rsi > 50 and volume > vol_ma5
            step3_pass = price > ma20 and price <= (ma20 * 1.03)
            sop_ready = step1_pass and step2_pass and step3_pass
            
            atr_stop_price = cost - (atr * 2) if cost > 0 else 0
            take_profit_price = cost * 1.10 if cost > 0 else 0
            
            if cost > 0 and price <= atr_stop_price: final_status = "🔴 破損"
            elif price < ma20 * 0.95: final_status = "🔴 破線"
            elif sop_ready: final_status = "🟢 進場"
            else: final_status = "🟡 觀望"

            # --- Phase 2: 白話文 AI 決策解釋邏輯 ---
            if final_status == "🟢 進場":
                ai_explanation = f"技術面已站穩月線防守區，且多方指標齊聚，配合法人籌碼優勢 ({inst['trend']})，建議可依風控比例啟動首批試單。"
            elif final_status == "🔴 破線":
                ai_explanation = "股價已跌破 20MA (月線) 關鍵防守區，短線趨勢轉弱，建議優先收回資金、退場觀望。"
            elif final_status == "🔴 破損":
                ai_explanation = f"已觸發 ATR 基準停損警戒線 ({atr_stop_price:.1f})，請務必嚴格執行停損紀律，鎖住單筆虧損風險。"
            else:
                if ai_score >= 70:
                    ai_explanation = f"綜合戰力偏高 ({ai_score}分)，法人籌碼已見進駐，惟技術面 SOP 尚未完全達標，建議密切盯盤等待突破契機。"
                elif is_bull_aligned:
                    ai_explanation = "目前均線維持多頭排列，但短期缺乏強勁籌碼或量能推升，呈現區間震盪整理，建議保持耐心等待。"
                else:
                    ai_explanation = "籌碼動能與技術面均未見明顯反轉跡象，處於弱勢或整理格局，目前不宜貿然進場。"
            
            suggested_shares = min(int(risk_amount / atr), int(cap / price)) if atr > 0 else 0
            
            tags = []
            if inst.get('t_days', 0) >= 3: tags.append("🔥投信作帳")
            if inst.get('f_days', 0) >= 3: tags.append("🌊外資波段")
            if is_bull_aligned and price > ma20: tags.append("🚀多頭起漲")
            elif price < ma60 and ma20 < ma60: tags.append("❄️弱勢空頭")
            if not tags: tags.append("⏳區間震盪")
            
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
            
    # --- 戰情儀表板 ---
    if summary_data:
        df_summary = pd.DataFrame(summary_data).sort_values(by="AI分數", ascending=False).reset_index(drop=True)
        st.markdown("### 🎯 盤前決策儀表板")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🏆 最佳標的", f"{df_summary.iloc[0]['名稱']}", f"戰力: {df_summary.iloc[0]['AI分數']}分")
        with c2: st.metric("⚠️ 弱勢警告", f"{df_summary.iloc[-1]['名稱']}", f"戰力: {df_summary.iloc[-1]['AI分數']}分", delta_color="inverse")
        with c3: st.metric("🟢 潛力檔數", f"{len(df_summary[df_summary['AI分數']>=80])} 檔", "可佈局" if len(df_summary[df_summary['AI分數']>=80]) > 0 else "耐心等待", delta_color="normal" if len(df_summary[df_summary['AI分數']>=80]) > 0 else "off")
        st.divider()

    # --- Phase 2: AI 深度解析清單 ---
    st.markdown("### 📊 AI 深度解析清單")
    
    card_data = sorted(card_data, key=lambda x: x['ai_score'], reverse=True)
    
    for data in card_data:
        with st.container(border=True):
            st.markdown(f"#### {data['name']} ({data['code']}) - {' '.join(data['tags'][:2])}")
            st.progress(data['ai_score'] / 100)
            
            if data['cost'] > 0 and data['price'] <= data['atr_stop_price']:
                st.error(f"🚨 風控警報：已跌破停損 ({data['atr_stop_price']:.2f})！")
            elif data['cost'] > 0 and data['price'] >= data['take_profit_price']:
                st.success(f"🎉 停利提醒：已達波段目標 ({data['take_profit_price']:.2f})！")
            
            # --- 恢復 4 欄位，並將成本設計為標籤放在現價下方 ---
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("現價", f"{data['price']:.2f}")
            
            # 使用 HTML/CSS 縮排向上推，並建立質感標籤
            cost_str = f"{data['cost']:.2f}" if data['cost'] > 0 else "-"
            col_a.markdown(
                f"<div style='margin-top: -15px;'><span style='font-size: 0.85em; color: #94a3b8; background-color: #334155; padding: 2px 6px; border-radius: 4px;'>成本 {cost_str}</span></div>", 
                unsafe_allow_html=True
            )
            
            col_b.metric("總法人", f"{data['inst']['trend']}")
            col_c.metric("判定", data['final_status'])
            col_d.metric("部位", f"{data['shares']}股" if data['final_status'] == "🟢 進場" else "-")
            
            st.write("") # 增加微小間距避免擠壓下方 Tabs
            
            tab1, tab2, tab3 = st.tabs(["⚙️ SOP與籌碼", "📉 技術數據", "🛡️ 風控點位"])
            
            with tab1:
                # --- 白話文 AI 決策解釋 ---
                st.info(f"**🤖 AI 總結**：{data['ai_explanation']}")
                
                st.markdown(f"- **外資動向**: {data['inst']['foreign_trend']} | **投信動向**: {data['inst']['trust_trend']}")
                st.markdown(f"- **S1 籌碼**: 法人買超與比例 {'🟢' if data['inst']['days']>0 else '⚪'}")
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
