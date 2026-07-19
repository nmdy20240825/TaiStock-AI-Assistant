import streamlit as st
import pandas as pd
import yfinance as yf
import json
import os
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock V2.7-P2 全自動紀律決策系統")

# ===== UI 視覺與字體優化模組 =====
st.markdown("""
<style>
/* 強制縮小指標數值與標籤的字體，適應手機版面，並強制允許換行以防截斷 */
[data-testid="stMetricValue"] {
    font-size: 18px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    white-space: normal !important;
    word-break: break-word !important;
}
/* 縮小標題間距，讓畫面更緊湊 */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
/* 增強 AI 建議清單的排版 */
.ai-advice-box {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #3b82f6;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- 1. 報價與技術資料抓取 ---
@st.cache_data(ttl=300) 
def fetch_stock_data(code):
    try:
        if code.isalpha() or code.endswith('.US'):
            ticker = code.replace('.US', '')
            return yf.download(ticker, period="6mo", progress=False)
            
        if code.endswith('.TW') or code.endswith('.TWO'):
            return yf.download(code, period="6mo", progress=False)
            
        df_tw = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df_tw is not None and not df_tw.empty and len(df_tw) > 0:
            return df_tw
            
        df_two = yf.download(f"{code}.TWO", period="6mo", progress=False)
        return df_two
    except Exception:
        return pd.DataFrame()

# --- 2. 籌碼資料抓取 (自動略過美股以防報錯) ---
@st.cache_data(ttl=3600)  
def get_institutional_data(code):
    default_res = {
        "buy_sell": 0, "days": 0, "trend": "資料不足", "avg_ratio": 0, 
        "accumulated_shares": 0, "foreign_trend": "無資料", "trust_trend": "無資料",
        "f_days": 0, "t_days": 0
    }
    
    if code.isalpha() or code.endswith('.US'):
        default_res["trend"] = "美股無籌碼"
        return default_res

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
        
        ticker = f"{code}.TW" if not code.endswith(('.TW', '.TWO')) else code
        stock_data = yf.download(ticker, period="1mo", progress=False)
        
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
        "3035": ["智原", 300.0, 20000, 5.0],
        "2317": ["鴻海", 210.0, 20000, 5.0],
        "NVDA": ["輝達", 125.0, 20000, 5.0]
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
        new_code = st.text_input("代號 (台股數字 / 美股字母)")
        new_name = st.text_input("名稱 (可留白)")
        new_cost = st.number_input("成本價", value=100.0, step=0.1)
        st.divider()
        new_cap = st.number_input("分配資金", value=20000, step=5000)
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

# --- 獨立渲染卡片函數 ---
def render_stock_card(data, system_history):
    with st.container(border=True):
        hist_records = system_history.get(data['code'], {})
        sorted_dates = sorted(hist_records.keys(), reverse=True)
        delta_str = ""
        # 歷史變化追蹤
        if len(sorted_dates) > 1:
            yesterday_score = hist_records[sorted_dates[1]]['score']
            diff = data['ai_score'] - yesterday_score
            if diff > 0: delta_str = f" <span style='color: #4ade80;'>(🔺+{diff})</span>"
            elif diff < 0: delta_str = f" <span style='color: #f87171;'>(🔻{diff})</span>"
            else: delta_str = " <span style='color: #94a3b8;'>(➖ 持平)</span>"

        st.markdown(f"#### {data['name']} ({data['code']}) - {' '.join(data['tags'][:2])}{delta_str}", unsafe_allow_html=True)
        
        s1_name = "動能" if data['is_us'] else "籌碼"
        st.markdown(f"<div style='font-size: 0.9em; margin-bottom: 5px; color: #cbd5e1;'>SOP 檢核：{s1_name} {'🟢' if data['step1'] else '⚪'} | 量能 {'🟢' if data['step2'] else '⚪'} | 趨勢 {'🟢' if data['step3'] else '⚪'}</div>", unsafe_allow_html=True)
        
        st.progress(data['ai_score'] / 100)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("現價", f"{data['price']:.2f}")
        
        cost_str = f"{data['cost']:.2f}" if data['cost'] > 0 else "-"
        col_a.markdown(
            f"<div style='margin-top: -15px;'><span style='font-size: 0.85em; color: #94a3b8; background-color: #334155; padding: 2px 6px; border-radius: 4px;'>成本 {cost_str}</span></div>", 
            unsafe_allow_html=True
        )
        
        # 多空分水嶺 (Phase 2 升級：直接顯示於卡片主視覺)
        pivot_color = "normal" if data['pivot_status'] == "🟢 站上" else "inverse"
        col_b.metric("多空分水嶺", f"{data['pivot_point']:.2f}", data['pivot_status'], delta_color=pivot_color)
        
        col_c.metric("判定", data['final_status'])
        col_d.metric("部位", f"{data['shares']}股" if data['final_status'] == "🟢 進場" else "-")
        
        st.write("") 
        
        tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs(["⚙️ AI決策與SOP", "📉 技術數據", "🛡️ 風控點位", "⏳ 決策時間軸"])
        
        with tab_c1:
            # 具體可執行的 AI 結論與信心 (Phase 2 升級)
            advice_html = f"""
            <div class='ai-advice-box'>
                <div style='font-size: 1.1em; font-weight: bold; margin-bottom: 8px;'>🤖 AI 執行建議：</div>
                {''.join([f"<div style='margin-bottom: 4px;'>{item}</div>" for item in data['ai_advice']])}
            </div>
            """
            st.markdown(advice_html, unsafe_allow_html=True)
            
            st.markdown(f"**🧠 AI 戰力拆解 (總分 {data['ai_score']})**")
            st.code(f"籌碼/長線: +{data['score_inst']:.0f} | 趨勢技術: +{data['score_tech']:.0f} | 量能指標: +{data['score_vol']:.0f} | 風控狀態: +{data['score_risk']:.0f}", language="text")

            if not data['is_us']:
                st.markdown(f"- **外資動向**: {data['inst']['foreign_trend']} | **投信動向**: {data['inst']['trust_trend']}")
                st.markdown(f"- **S1 籌碼**: 法人買超與比例 {'🟢' if data['step1'] else '⚪'}")
            else:
                st.markdown(f"- **S1 動能**: 季線之上且 MACD 翻正 {'🟢' if data['step1'] else '⚪'}")
            st.markdown(f"- **S2 量能**: KD向上 / RSI>50 / 放量 {'🟢' if data['step2'] else '⚪'}")
            st.markdown(f"- **S3 趨勢**: MA20防守 / 多頭排列 {'🟢' if data['step3'] else '⚪'}")
            
        with tab_c2:
            c_t1, c_t2 = st.columns(2)
            c_t1.write(f"**今日量**: {data['volume']:,.0f} | **5日均量**: {data['vol_ma5']:,.0f}")
            c_t1.write(f"**K**: {data['k']:.1f} | **D**: {data['d']:.1f} | **RSI**: {data['rsi']:.1f}")
            c_t2.write(f"**MA10**: {data['ma10']:.2f} | **MA20**: {data['ma20']:.2f}")
            c_t2.write(f"**MA60**: {data['ma60']:.2f} | **季線乖離**: {data['bias']:.2f}%")
            
        with tab_c3:
            st.write(f"**設定成本**: {data['cost']:.2f}")
            st.write(f"**動態防守/停損**: {data['atr_stop_price']:.2f}")
            st.write(f"**波段動能目標**: {data['take_profit_price']:.2f}")
            st.write(f"**建議投入資金/股數**: {data['risk_amount']:,.0f} 元 / {data['shares']} 股")
            
        with tab_c4:
            st.write("近期決策軌跡:")
            for dt in sorted_dates[:5]:
                st.write(f"- {dt}: {hist_records[dt]['status']} (戰力: {hist_records[dt]['score']})")


# --- 5. 主面板運算 ---
st.title("⚡ TaiStock V2.7 全自動紀律決策系統")

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
            
            # 多空分水嶺計算 (Phase 2: Pivot Point = 前高+前低+前收 / 3)
            if len(h) >= 2:
                pivot_point = (float(h.iloc[-2]) + float(l.iloc[-2]) + float(c.iloc[-2])) / 3
            else:
                pivot_point = price
            pivot_status = "🟢 站上" if price > pivot_point else "🔴 未站上"

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
            
            if cost > 0 and price > cost * 1.10:
                atr_stop_price = max(cost, ma20)
                take_profit_price = cost * 2.0  
            else:
                atr_stop_price = cost - (atr * 2) if cost > 0 else 0
                take_profit_price = cost * 1.10 if cost > 0 else 0
            
            is_us_stock = code.isalpha() or code.endswith('.US')
            
            if not is_us_stock:
                score_inst = min(inst['days'] * 5, 20)
                accumulated_amount = inst['accumulated_shares'] * price
                if accumulated_amount >= 3000000000: score_inst += 20
                elif accumulated_amount >= 1000000000: score_inst += 10
                elif accumulated_amount >= 500000000: score_inst += 5
            else:
                score_inst = 0
                if price > ma60: score_inst += 20  
                if macd > 0: score_inst += 10      
                if 0 < bias < 20: score_inst += 10 
                
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
            
            # 決策信心計算 (Phase 2)
            confidence_base = ai_score * 0.8
            if is_bull_aligned: confidence_base += 10
            if price > pivot_point: confidence_base += 5
            confidence = min(99, max(10, int(confidence_base)))

            if is_us_stock:
                step1_pass = price > ma60 and macd > 0
            else:
                step1_pass = inst['days'] >= 3 or inst['accumulated_shares'] * price >= 1000000000
                
            step2_pass = k > d and rsi > 50 and volume > vol_ma5
            step3_pass = price > ma20 and is_bull_aligned
            
            # ===== 具體 AI 結論與行動清單 (Phase 2 升級) =====
            ai_advice = []
            if cost > 0 and price <= atr_stop_price: 
                if price > cost:
                    final_status = "🔵 停利退場"
                    ai_advice = [
                        "✓ 建議：立即執行紀律停利",
                        f"✓ 依據：股價跌破動態防守線 ({atr_stop_price:.1f})",
                        "✓ 狀態：已成功鎖住波段利潤，全數收回資金",
                        f"🎯 決策信心：{confidence}%"
                    ]
                else:
                    final_status = "🔴 破損"
                    ai_advice = [
                        "✓ 建議：執行基準停損，絕不凹單",
                        f"✓ 依據：觸發初始防守點 ({atr_stop_price:.1f})",
                        "✓ 狀態：戰力歸零，優先保護本金",
                        f"🎯 決策信心：{confidence}%"
                    ]
            elif cost > 0 and price >= cost * 1.10:
                final_status = "🔥 利潤奔跑"
                ai_advice = [
                        "✓ 建議：獲利續抱，不預設高點",
                        f"✓ 依據：已啟動動態防守，目前防守點上調至月線 ({atr_stop_price:.1f})",
                        "✓ 狀態：獲利脫離成本區超過 10%",
                        f"🎯 決策信心：{confidence}% (趨勢保護中)"
                    ]
            elif cost > 0 and price >= cost * 1.05:
                final_status = "🟡 接近停利"
                ai_advice = [
                        "✓ 建議：將停損點無條件上調至「成本價」",
                        "✓ 依據：獲利空間已拉開，確保這筆交易立於不敗",
                        "✓ 狀態：耐心等待達標或轉勢",
                        f"🎯 決策信心：{confidence}%"
                    ]
            elif price < ma20 * 0.95: 
                final_status = "🔴 破線"
                ai_advice = [
                        "✓ 建議：考慮預防性減碼或空手觀望",
                        "✓ 依據：股價明顯跌破月線防守區，短線趨勢轉弱",
                        "✓ 狀態：避開資金閒置風險",
                        f"🎯 決策信心：{100 - confidence}% (偏空防守)"
                    ]
            elif ai_score >= 70: 
                final_status = "🟢 進場"
                ai_advice = [
                        f"✓ 建議：可分批進場佈局，預計投入 {risk_amount:,.0f} 元",
                        "✓ 依據：綜合戰力強勢，各項指標發生共振",
                        f"✓ 狀態：防守線預設為 {atr_stop_price:.1f}",
                        f"🎯 決策信心：{confidence}% (極高勝率)"
                    ]
            else: 
                final_status = "🟡 觀望"
                ai_advice = [
                        "✓ 建議：保持空手，密切盯盤等待",
                        "✓ 依據：條件尚未完全齊備，動能不足",
                        "✓ 狀態：將標的保留在觀察清單中",
                        f"🎯 決策信心：{confidence}%"
                    ]
            
            suggested_shares = min(int(risk_amount / atr), int(cap / price)) if atr > 0 else 0
            
            tags = []
            if is_us_stock: tags.append("🦅美股科技")
            else:
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
                "ai_advice": ai_advice, "confidence": confidence, "pivot_point": pivot_point, "pivot_status": pivot_status,
                "is_us": is_us_stock,
                "score_inst": score_inst, "score_tech": score_tech, "score_vol": score_vol, "score_risk": score_risk 
            })
            
        except Exception as e:
            st.error(f"分析 {code} 發生錯誤: {e}")
            
    save_history(system_history)

    # ===== 持股健康度總覽 =====
    if summary_data:
        health_green = len([d for d in summary_data if "進場" in d['判定'] or "奔跑" in d['判定']])
        health_yellow = len([d for d in summary_data if "觀望" in d['判定'] or "接近" in d['判定']])
        health_red = len([d for d in summary_data if "破" in d['判定'] or "退場" in d['判定']])
        
        st.markdown("### 🌟 持股健康度總覽")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("🟢 優勢/奔跑 (強勢)", f"{health_green} 檔")
        hc2.metric("🟡 觀望/警戒 (震盪)", f"{health_yellow} 檔")
        hc3.metric("🔴 破線/停損 (弱勢)", f"{health_red} 檔")
        st.divider()
            
    if summary_data:
        df_summary = pd.DataFrame(summary_data).sort_values(by="AI分數", ascending=False).reset_index(drop=True)
        st.markdown("### 🎯 盤前決策儀表板")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🏆 最佳標的", f"{df_summary.iloc[0]['名稱']}", f"戰力: {df_summary.iloc[0]['AI分數']}分")
        with c2: st.metric("⚠️ 弱勢警告", f"{df_summary.iloc[-1]['名稱']}", f"戰力: {df_summary.iloc[-1]['AI分數']}分", delta_color="inverse")
        with c3: st.metric("🟢 潛力檔數", f"{len(df_summary[df_summary['AI分數']>=70])} 檔", "可佈局" if len(df_summary[df_summary['AI分數']>=70]) > 0 else "耐心等待", delta_color="normal" if len(df_summary[df_summary['AI分數']>=70]) > 0 else "off")
        st.divider()

    # --- 每日紀律檢核清單 (SOP) ---
    if card_data:
        st.markdown("### ✅ 每日紀律檢核清單 (SOP)")
        with st.expander("展開今日操作任務", expanded=True):
            action_sell = [] 
            action_buy = [] 
            action_watch = [] 
            
            for data in card_data:
                if data['final_status'] == "🔴 破損":
                    action_sell.append(f"🚨 **停損退場**：{data['name']} ({data['code']}) 現價 {data['price']:.2f} 跌破防守點 {data['atr_stop_price']:.1f}，收回資金。")
                elif data['final_status'] == "🔵 停利退場":
                    action_sell.append(f"🛡️ **紀律停利**：{data['name']} ({data['code']}) 現價 {data['price']:.2f} 跌破動態防守線 {data['atr_stop_price']:.1f}，鎖住利潤。")
                elif data['final_status'] == "🟢 達標":
                    action_sell.append(f"🎉 **獲利了結**：{data['name']} ({data['code']}) 達波段目標 {data['take_profit_price']:.1f}，執行分批停利。")
                elif data['final_status'] == "🔥 利潤奔跑":
                    action_watch.append(f"🚀 **獲利續抱**：{data['name']} ({data['code']}) 啟動動態防守，月線 {data['atr_stop_price']:.1f} 不破不賣！")
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
    
    tab_tw, tab_us = st.tabs(["🇹🇼 台股主力陣列 (籌碼監控)", "🇺🇸 美股科技巨頭 (動能監控)"])
    
    with tab_tw:
        tw_cards = [d for d in card_data if not d['is_us']]
        if not tw_cards:
            st.info("目前無台股持股紀錄。")
        for data in tw_cards:
            render_stock_card(data, system_history)

    with tab_us:
        us_cards = [d for d in card_data if d['is_us']]
        if not us_cards:
            st.info("目前無美股持股紀錄。")
        for data in us_cards:
            render_stock_card(data, system_history)

if __name__ == "__main__":
    pass
