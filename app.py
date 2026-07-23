import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime

st.set_page_config(layout="wide", page_title="TaiStock V2.9 全自動紀律決策系統")

# ===== UI 視覺與字體優化模組 =====
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 18px !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; white-space: normal !important; word-break: break-word !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
.ai-advice-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 0. 技術指標輔助函式（V2.9 修正版）---

def calc_kd(h, l, c, period=9):
    """
    正確版 KD 隨機指標：對整段歷史做遞迴平滑，而非只用最後一天套公式。
    K_t = 2/3 * K_(t-1) + 1/3 * RSV_t，初始 K=D=50。
    """
    low_min = l.rolling(period).min()
    high_max = h.rolling(period).max()
    rsv = (c - low_min) / (high_max - low_min + 1e-9) * 100
    k_list, d_list = [], []
    prev_k, prev_d = 50.0, 50.0
    for val in rsv:
        if pd.isna(val):
            k_list.append(np.nan); d_list.append(np.nan)
            continue
        cur_k = 2/3 * prev_k + 1/3 * float(val)
        cur_d = 2/3 * prev_d + 1/3 * cur_k
        k_list.append(cur_k); d_list.append(cur_d)
        prev_k, prev_d = cur_k, cur_d
    return pd.Series(k_list, index=c.index), pd.Series(d_list, index=c.index)

def calc_macd(c, fast=12, slow=26):
    """真正的 EMA 版 MACD DIF（原版誤用 SMA 相減，會失真）。"""
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    return float(ema_fast.iloc[-1] - ema_slow.iloc[-1])

# --- 1. 大盤宏觀環境抓取 ---
@st.cache_data(ttl=1800)
def fetch_macro_data():
    tickers = {'TW': '^TWII', 'US': '^IXIC', 'VIX': '^VIX'}
    macro_status = {}
    for key, symbol in tickers.items():
        try:
            df = yf.download(symbol, period="3mo", progress=False)
            if not df.empty:
                c_series = df['Close'].squeeze()
                if isinstance(c_series, pd.DataFrame): c_series = c_series.iloc[:, 0]
                c = float(c_series.iloc[-1])
                ma20 = float(c_series.rolling(20).mean().iloc[-1])
                macro_status[key] = {'price': c, 'ma20': ma20, 'trend': '🟢 多頭' if c > ma20 else '🔴 空頭'}
        except Exception:
            macro_status[key] = None
    return macro_status

# --- 2. 報價與技術資料抓取 ---
@st.cache_data(ttl=300)
def fetch_stock_data(code):
    try:
        if code.isalpha() or code.endswith('.US'): return yf.download(code.replace('.US', ''), period="6mo", progress=False)
        if code.endswith('.TW') or code.endswith('.TWO'): return yf.download(code, period="6mo", progress=False)
        df_tw = yf.download(f"{code}.TW", period="6mo", progress=False)
        if df_tw is not None and not df_tw.empty and len(df_tw) > 0: return df_tw
        return yf.download(f"{code}.TWO", period="6mo", progress=False)
    except Exception: return pd.DataFrame()

# --- 3. 籌碼資料抓取 ---
@st.cache_data(ttl=3600)
def get_institutional_data(code):
    default_res = {"buy_sell": 0, "days": 0, "trend": "資料不足", "accumulated_shares": 0,
                   "foreign_trend": "無資料", "trust_trend": "無資料", "foreign_days": 0, "trust_days": 0}
    if code.isalpha() or code.endswith('.US'):
        return {"buy_sell": 0, "days": 0, "trend": "美股無籌碼", "accumulated_shares": 0,
                "foreign_trend": "N/A", "trust_trend": "N/A", "foreign_days": 0, "trust_days": 0}
    try:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": code, "start_date": start_date, "end_date": end_date}
        resp = requests.get(url, params=parameter, timeout=5)
        data = resp.json()

        # 【V2.9.2 修正】原版永遠只試 .TW，上櫃股票（如 3324、1595）會抓不到報價、
        # 導致 stock_data 是空的，整段籌碼資料被誤判為「無資料」。改成跟 fetch_stock_data 一樣，
        # 先試 .TW，抓不到再試 .TWO。
        if code.endswith(('.TW', '.TWO')):
            stock_data = yf.download(code, period="1mo", progress=False)
        else:
            stock_data = yf.download(f"{code}.TW", period="1mo", progress=False)
            if stock_data is None or stock_data.empty:
                stock_data = yf.download(f"{code}.TWO", period="1mo", progress=False)
        if data.get("msg") != "success" or not data.get("data") or stock_data is None or stock_data.empty: return default_res

        df_inst = pd.DataFrame(data["data"])
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        daily_net = df_inst.groupby('date')['net_buy'].sum().sort_index(ascending=False)

        f_mask, t_mask = df_inst['name'].str.contains('外資|Foreign', case=False, na=False), df_inst['name'].str.contains('投信|Investment', case=False, na=False)
        df_foreign = df_inst[f_mask].groupby('date')['net_buy'].sum().sort_index(ascending=False)
        df_trust = df_inst[t_mask].groupby('date')['net_buy'].sum().sort_index(ascending=False)

        def calc_trend(series):
            if series.empty: return 0, "無資料"
            days = 0
            for val in series:
                v = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
                if v > 0 and days >= 0: days += 1
                elif v < 0 and days <= 0: days -= 1
                else: break
            return days, f"連{days}買" if days > 0 else (f"連{abs(days)}賣" if days < 0 else "盤整")

        f_days, f_trend = calc_trend(df_foreign)
        t_days, t_trend = calc_trend(df_trust)

        days, accumulated_shares = 0, 0
        for date_key in daily_net.index:
            if date_key in stock_data.index:
                net_buy = float(daily_net[date_key].iloc[0]) if isinstance(daily_net[date_key], pd.Series) else float(daily_net[date_key])
                if net_buy > 0: days += 1; accumulated_shares += net_buy
                elif net_buy <= 0 and days > 0: break
                elif net_buy < 0 and days == 0:
                    for sell_date in daily_net.index:
                        val = float(daily_net[sell_date].iloc[0]) if isinstance(daily_net[sell_date], pd.Series) else float(daily_net[sell_date])
                        if val < 0: days -= 1; accumulated_shares += val
                        else: break
                    break
        trend_str = f"連{days}買" if days > 0 else (f"連{abs(days)}賣" if days < 0 else "盤整")
        return {"days": days, "trend": trend_str, "accumulated_shares": float(accumulated_shares),
                "foreign_trend": f_trend, "trust_trend": t_trend, "foreign_days": f_days, "trust_days": t_days}
    except Exception:
        return default_res

# --- 4. 資料存取（V2.9.1：改用 Google Sheets 當雲端資料庫，取代本機 json 檔）---
import gspread
from google.oauth2.service_account import Credentials

GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
PORTFOLIO_HEADERS = ["code", "name", "cost", "cap", "risk", "status", "break_date"]
HISTORY_HEADERS = ["code", "date", "score", "status", "price"]

DEFAULT_PORTFOLIO = {
    "3035": {"name": "智原", "cost": 300.0, "cap": 20000, "risk": 5.0, "status": "Active"},
    "2317": {"name": "鴻海", "cost": 210.0, "cap": 20000, "risk": 5.0, "status": "Active"},
    "NVDA": {"name": "輝達", "cost": 125.0, "cap": 20000, "risk": 5.0, "status": "Active"}
}

@st.cache_resource
def get_gsheet_client():
    """建立與 Google Sheets 的連線（憑證讀取自 st.secrets['gcp_service_account']）。"""
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_info, scopes=GSHEET_SCOPES)
    return gspread.authorize(creds)

@st.cache_resource
def get_spreadsheet():
    client = get_gsheet_client()
    return client.open_by_key(st.secrets["gsheet"]["sheet_id"])

def get_worksheet(name, headers):
    """取得指定分頁；若試算表裡還沒有這個分頁，就自動建立並寫入表頭。"""
    ss = get_spreadsheet()
    try:
        ws = ss.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=name, rows=200, cols=len(headers))
        ws.append_row(headers)
    return ws

def load_portfolio():
    try:
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS)
        records = ws.get_all_records()
        if not records:
            # 第一次使用、分頁是空的：把預設持股寫進去，讓 Google Sheet 成為資料的起點
            rows = [[code, info["name"], info["cost"], info["cap"], info["risk"], info["status"], ""] for code, info in DEFAULT_PORTFOLIO.items()]
            ws.append_rows(rows)
            return {k: dict(v) for k, v in DEFAULT_PORTFOLIO.items()}
        data = {}
        for row in records:
            code = str(row.get("code", "")).strip()
            if not code: continue
            entry = {
                "name": row.get("name", ""),
                "cost": float(row.get("cost") or 0.0),
                "cap": float(row.get("cap") or 20000.0),
                "risk": float(row.get("risk") or 5.0),
                "status": row.get("status") or "Active",
            }
            break_date = str(row.get("break_date", "")).strip()
            if break_date:
                entry["break_date"] = break_date
            data[code] = entry
        return data
    except Exception as e:
        st.error(f"⚠️ 讀取 Google Sheet 持股資料失敗，暫時使用內建預設值：{e}")
        return {k: dict(v) for k, v in DEFAULT_PORTFOLIO.items()}

def save_portfolio(data):
    try:
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS)
        ws.clear()
        rows = [PORTFOLIO_HEADERS]
        for code, info in data.items():
            rows.append([code, info.get("name", ""), info.get("cost", 0.0), info.get("cap", 20000.0), info.get("risk", 5.0), info.get("status", "Active"), info.get("break_date", "")])
        ws.update(rows)
    except Exception as e:
        st.error(f"⚠️ 寫入 Google Sheet 持股資料失敗：{e}")

def load_history():
    try:
        ws = get_worksheet("history", HISTORY_HEADERS)
        records = ws.get_all_records()
        data = {}
        for row in records:
            code = str(row.get("code", "")).strip()
            date = str(row.get("date", "")).strip()
            if not code or not date: continue
            data.setdefault(code, {})[date] = {
                "score": int(float(row.get("score") or 0)),
                "status": row.get("status", ""),
                "price": float(row.get("price") or 0.0),
            }
        return data
    except Exception as e:
        st.error(f"⚠️ 讀取 Google Sheet 歷史資料失敗：{e}")
        return {}

def save_history(data):
    try:
        ws = get_worksheet("history", HISTORY_HEADERS)
        ws.clear()
        rows = [HISTORY_HEADERS]
        for code, records in data.items():
            for date, rec in records.items():
                score = rec.get("score", 0)
                price = rec.get("price", 0.0)
                # NaN 不是合法的 JSON 值，寫入 Google Sheet 會整批失敗；這裡保險起見再過濾一次
                if score is None or (isinstance(score, float) and score != score): score = 0
                if price is None or (isinstance(price, float) and price != price): price = 0.0
                rows.append([code, date, score, rec.get("status", ""), price])
        ws.update(rows)
    except Exception as e:
        st.error(f"⚠️ 寫入 Google Sheet 歷史資料失敗：{e}")

portfolio, system_history, today_str = load_portfolio(), load_history(), datetime.datetime.now().strftime("%Y-%m-%d")

# --- 5. 側邊欄 UI ---
with st.sidebar:
    st.header("📋 持股與風控設定")
    with st.form("add_stock"):
        new_code = st.text_input("代號 (台股數字 / 美股字母)")
        new_name, new_cost, new_cap, new_risk = st.text_input("名稱 (可留白)"), st.number_input("成本價", value=100.0, step=0.1), st.number_input("分配資金", value=20000, step=5000), st.number_input("單筆風險 (%)", value=5.0, step=0.1)
        if st.form_submit_button("更新設定"):
            if new_code:
                fetch_stock_data.clear(); get_institutional_data.clear()
                existing_break_date = portfolio.get(new_code, {}).get('break_date') if isinstance(portfolio.get(new_code), dict) else None
                portfolio[new_code] = {"name": new_name, "cost": new_cost, "cap": new_cap, "risk": new_risk, "status": "Active"}
                if existing_break_date:
                    portfolio[new_code]['break_date'] = existing_break_date
                save_portfolio(portfolio)
                st.rerun()
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("確認刪除") and del_code in portfolio:
        del portfolio[del_code]
        save_portfolio(portfolio)
        if del_code in system_history: del system_history[del_code]; save_history(system_history)
        st.rerun()

# --- 卡片渲染邏輯 ---
def render_stock_card(data, system_history, portfolio_data):
    with st.container(border=True):
        hist_records = system_history.get(data['code'], {})
        sorted_dates = sorted(hist_records.keys(), reverse=True)
        delta_str = ""
        if len(sorted_dates) > 1:
            yesterday_score = hist_records[sorted_dates[1]]['score']
            diff = data['ai_score'] - yesterday_score
            if diff > 0: delta_str = f" <span style='color: #4ade80;'>(🔺+{diff})</span>"
            elif diff < 0: delta_str = f" <span style='color: #f87171;'>(🔻{diff})</span>"
            else: delta_str = " <span style='color: #94a3b8;'>(➖ 持平)</span>"

        is_broken = data['final_status'] in ["🔴 破損", "🔴 破線", "⚠️ 帳面虧損"]
        broken_label = " <span style='color: red;'>[🚨預警]</span>" if is_broken else ""

        st.markdown(f"#### {data['name']} ({data['code']}){broken_label} - {' '.join(data['tags'][:2])}{delta_str}", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.9em; margin-bottom: 5px; color: #cbd5e1;'>SOP 檢核：{'動能' if data['is_us'] else '籌碼'} {'🟢' if data['step1'] else '⚪'} | 量能 {'🟢' if data['step2'] else '⚪'} | 趨勢 {'🟢' if data['step3'] else '⚪'}</div>", unsafe_allow_html=True)
        _safe_score = data['ai_score']
        if _safe_score is None or (isinstance(_safe_score, float) and _safe_score != _safe_score): _safe_score = 0
        st.progress(max(0, min(100, _safe_score)) / 100)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("現價", f"{data['price']:.2f}")
        col_a.markdown(f"<div style='margin-top: -15px;'><span style='font-size: 0.85em; color: #94a3b8; background-color: #334155; padding: 2px 6px; border-radius: 4px;'>成本 {data['cost']:.2f}</span></div>", unsafe_allow_html=True)
        col_b.metric("多空分水嶺", f"{data['pivot_point']:.2f}", data['pivot_status'], delta_color="normal" if data['pivot_status'] == "🟢 站上" else "inverse")
        col_c.metric("判定", data['final_status'])

        with col_d:
            st.metric("部位", f"{data['shares']}股" if data['final_status'] == "🟢 進場" else "-")
            if data['final_status'] in ["🔵 停利退場", "🔴 破損", "🔴 破線", "⚠️ 帳面虧損"]:
                if st.button("📦 手動歸檔 (已結算)", key=f"close_{data['code']}"):
                    portfolio_data[data['code']]['status'] = "Closed"
                    if 'break_date' in portfolio_data[data['code']]:
                        del portfolio_data[data['code']]['break_date']
                    save_portfolio(portfolio_data)
                    st.rerun()

        st.write("")
        tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs(["⚙️ AI決策與SOP", "📉 技術數據", "🛡️ 風控點位", "📈 決策時間軸"])

        with tab_c1:
            st.markdown(f"<div class='ai-advice-box'><div style='font-size: 1.1em; font-weight: bold; margin-bottom: 8px;'>🤖 AI 執行建議：</div>{''.join([f'<div style=\"margin-bottom: 4px;\">{item}</div>' for item in data['ai_advice']])}</div>", unsafe_allow_html=True)
            st.markdown(f"**🧠 AI 戰力拆解 (總分 {data['ai_score']})**")
            st.code(f"籌碼/長線: +{data['score_inst']:.0f} | 趨勢技術: +{data['score_tech']:.0f} | 量能指標: +{data['score_vol']:.0f} | 風控狀態: +{data['score_risk']:.0f}", language="text")
            # 【V2.9.4 新增】四個子分數各自的滿分不同（40/30/15/15），額外用進度條圖形化呈現，
            # 方便一眼看出每個子項「離滿分還差多少」，不用自己心算百分比。
            for _label, _val, _max in [
                ("籌碼/長線", data['score_inst'], 40),
                ("趨勢技術", data['score_tech'], 30),
                ("量能指標", data['score_vol'], 15),
                ("風控狀態", data['score_risk'], 15),
            ]:
                _safe_val = 0 if (_val is None or (isinstance(_val, float) and _val != _val)) else _val
                st.caption(f"{_label}：{_safe_val:.0f} / {_max}")
                st.progress(max(0.0, min(1.0, _safe_val / _max)))
            if data.get('score_forced_zero'):
                st.warning("⚠️ 已觸發停損防禦機制：現價已跌破防守線，系統強制將總分歸零（不採計上方拆解分數加總），優先保護本金。", icon="⚠️")
            if not data['is_us']:
                st.markdown(f"- **外資動向**: {data['inst']['foreign_trend']} | **投信動向**: {data['inst']['trust_trend']}")
        with tab_c2:
            c_t1, c_t2 = st.columns(2)
            c_t1.write(f"**今日量**: {data['volume']:,.0f} | **5日均量**: {data['vol_ma5']:,.0f}\n**K**: {data['k']:.1f} | **D**: {data['d']:.1f} | **RSI**: {data['rsi']:.1f}")
            c_t2.write(f"**MA20**: {data['ma20']:.2f} | **MA60**: {data['ma60']:.2f}\n**MACD(DIF)**: {data['macd']:.2f} | **季線乖離**: {data['bias']:.2f}%")
        with tab_c3:
            st.write(f"**設定成本**: {data['cost']:.2f}\n**動態防守/停損**: {data['atr_stop_price']:.2f}\n**波段動能目標**: {data['take_profit_price']:.2f}")
        with tab_c4:
            if len(sorted_dates) > 1:
                chart_data = pd.DataFrame([{"Date": d, "Score": hist_records[d]['score']} for d in sorted_dates[:10]]).set_index("Date").sort_index()
                st.write("**📈 近期戰力動能曲線**")
                st.line_chart(chart_data['Score'], height=150)
            st.write("**📝 狀態軌跡**")
            for dt in sorted_dates[:5]: st.write(f"- {dt}: {hist_records[dt]['status']} ({hist_records[dt]['score']}分)")

# --- 6. 主程式執行 ---
st.title("⚡ TaiStock V2.9 全自動決策系統")

macro_data = fetch_macro_data()
st.markdown("### 🌍 雙軌市場環境總覽")
m_col1, m_col2, m_col3 = st.columns(3)

tw_trend = macro_data.get('TW', {})
if tw_trend: m_col1.metric("🇹🇼 台股加權 (大盤方向)", f"{tw_trend['price']:,.0f}", tw_trend['trend'], delta_color="normal" if "多頭" in tw_trend['trend'] else "inverse")
else: m_col1.metric("🇹🇼 台股加權", "連線中...")

us_trend = macro_data.get('US', {})
if us_trend: m_col2.metric("🇺🇸 那斯達克 (科技風向)", f"{us_trend['price']:,.0f}", us_trend['trend'], delta_color="normal" if "多頭" in us_trend['trend'] else "inverse")
else: m_col2.metric("🇺🇸 那斯達克", "連線中...")

vix_trend = macro_data.get('VIX', {})
if vix_trend:
    v_val = vix_trend['price']
    v_status, v_color = ("🚨 極度恐慌", "inverse") if v_val >= 25 else (("⚠️ 波動加劇", "off") if v_val >= 20 else ("🟢 環境穩定", "normal"))
    m_col3.metric("📉 VIX 恐慌指數", f"{v_val:.2f}", v_status, delta_color=v_color)
else: m_col3.metric("📉 VIX 恐慌指數", "連線中...")
st.divider()

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號！")
else:
    summary_data, card_data = [], []

    for code, info in list(portfolio.items()):
        if isinstance(info, dict):
            if info.get('status') == 'Closed': continue
            name, cost, cap, risk_pct = info.get('name', ''), info.get('cost', 0.0), info.get('cap', 20000.0), info.get('risk', 5.0)
        else:
            name, cost, cap, risk_pct = info if len(info) == 4 else (info[0], info[1], 20000.0, 5.0)

        risk_amount = cap * (risk_pct / 100)
        try:
            df = fetch_stock_data(code)
            if df is None or df.empty or len(df) < 60: continue

            c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df.get('Volume', pd.Series(0, index=df.index)).squeeze()
            if isinstance(c, pd.DataFrame): c, h, l, v = c.iloc[:, 0], h.iloc[:, 0], l.iloc[:, 0], v.iloc[:, 0]

            price, volume, vol_ma5 = float(c.iloc[-1]), float(v.iloc[-1]), float(v.rolling(5).mean().iloc[-1])
            pivot_point = (float(h.iloc[-2]) + float(l.iloc[-2]) + float(c.iloc[-2])) / 3 if len(h) >= 2 else price
            pivot_status = "🟢 站上" if price > pivot_point else "🔴 未站上"

            ma10, ma20, ma60 = float(c.rolling(10).mean().iloc[-1]), float(c.rolling(20).mean().iloc[-1]), float(c.rolling(60).mean().iloc[-1])

            # 【V2.9 修正】MACD 改用真正的 EMA 計算（原版用 SMA 相減會失真）
            macd = calc_macd(c)

            # 【V2.9 修正】KD 改用整段歷史遞迴平滑，而非單日套公式（原版 K/D 幾乎恆定在 50 附近）
            k_series, d_series = calc_kd(h, l, c)
            k, d = float(k_series.iloc[-1]), float(d_series.iloc[-1])

            delta = c.diff()
            up, down = delta.clip(lower=0).rolling(14).mean().iloc[-1], -1 * delta.clip(upper=0).rolling(14).mean().iloc[-1]
            rsi = float(100 - (100 / (1 + (np.nan_to_num(up) / (np.nan_to_num(down) + 0.001)))))
            atr = float(sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]) / 14)
            bias = float(((price - ma60) / ma60) * 100)

            # 【V2.9.3 修正】yfinance 偶爾會回傳不完整的資料（例如最後一根K棒缺值），
            # 導致 price/ma/k/d/rsi/atr 等任一數值變成 NaN。NaN 沒被擋下來的話會一路
            # 傳到 st.progress()（讓整個分頁當機）跟 Google Sheet 寫入（NaN 不是合法 JSON，
            # 寫入會直接失敗）。這裡先做一次「健檢」，任何一項是 NaN 就跳過這檔股票，
            # 等下一次重新抓資料時如果正常了，會自動恢復顯示。
            _core_values = [price, volume, vol_ma5, pivot_point, ma10, ma20, ma60, macd, k, d, rsi, atr, bias]
            if any(pd.isna(v) for v in _core_values):
                st.warning(f"⚠️ {name or code} 本次抓到的資料不完整（yfinance 回傳缺值），已跳過這次分析，下次重新整理應會恢復正常。")
                continue

            inst = get_institutional_data(code)
            atr_stop_price = max(cost, ma20) if (cost > 0 and price > cost * 1.10) else (cost - (atr * 2) if cost > 0 else 0)
            take_profit_price = cost * 2.0 if (cost > 0 and price > cost * 1.10) else (cost * 1.10 if cost > 0 else 0)

            is_us_stock = code.isalpha() or code.endswith('.US')
            score_inst = (20 if price > ma60 else 0) + (10 if macd > 0 else 0) + (10 if 0 < bias < 20 else 0) if is_us_stock else min(inst['days'] * 5, 20) + (20 if inst['accumulated_shares'] * price >= 3000000000 else (10 if inst['accumulated_shares'] * price >= 1000000000 else 0))
            score_tech = (10 if k > d else 0) + (10 if rsi > 50 else 0) + (10 if price > ma20 else 0)
            score_vol = min((volume / vol_ma5) * 10, 15) if vol_ma5 > 0 else 0
            score_risk = (10 if price > atr_stop_price else 0) + (5 if price >= take_profit_price or price >= cost * 1.05 else 0) if cost > 0 else 15

            score_forced_zero = bool(cost > 0 and price <= atr_stop_price)
            ai_score = 0 if score_forced_zero else min(int(score_inst + score_tech + score_vol + score_risk), 100)
            is_bull_aligned = (ma10 > ma20 and ma20 > ma60)
            confidence_base = ai_score * 0.8 + (10 if is_bull_aligned else 0) + (5 if price > pivot_point else 0)

            # 【V2.9 修正】多個宏觀警示同時觸發時，訊息會全部保留，不再被後面的條件覆蓋掉
            macro_warnings = []
            if is_us_stock:
                if us_trend and "空頭" in us_trend.get('trend', ''):
                    confidence_base *= 0.85
                    macro_warnings.append("⚠️ 美股大盤跌破月線，系統主動下調部位信心。")
                if vix_trend and vix_trend.get('price', 0) > 25:
                    confidence_base *= 0.70
                    macro_warnings.append("🚨 VIX 恐慌指數過高，系統強制抑制進場訊號！")
            else:
                if tw_trend and "空頭" in tw_trend.get('trend', ''):
                    confidence_base *= 0.85
                    macro_warnings.append("⚠️ 台股大盤跌破月線，逆勢操作風險較高。")

            confidence = min(99, max(10, int(confidence_base)))
            step1_pass = (price > ma60 and macd > 0) if is_us_stock else (inst['days'] >= 3 or inst['accumulated_shares'] * price >= 1000000000)
            step2_pass, step3_pass = (k > d and rsi > 50 and volume > vol_ma5), (price > ma20 and is_bull_aligned)

            ai_advice = []

            if cost > 0 and price <= atr_stop_price:
                final_status = "🔵 停利退場" if price > cost else "🔴 破損"
                ai_advice = [f"✓ 建議：{'立即執行紀律停利' if price > cost else '執行基準停損，絕不凹單'}", f"✓ 依據：股價跌破防守線 ({atr_stop_price:.1f})", "✓ 狀態：收回資金保護本金", f"🎯 決策信心：{confidence}%"]
            elif cost > 0 and price < cost:
                final_status = "⚠️ 帳面虧損"
                ai_advice = ["✓ 建議：注意資金控管，跌破防守線前最後警戒", f"✓ 依據：現價跌破設定成本 ({cost:.2f})", "✓ 狀態：已產生實質帳面虧損，紀律優先", f"🎯 決策信心：0% (防禦狀態)"]
            elif cost > 0 and price >= cost * 1.10:
                final_status = "🔥 利潤奔跑"
                ai_advice = ["✓ 建議：獲利續抱，不預設高點", f"✓ 依據：防守點上調至月線 ({atr_stop_price:.1f})", "✓ 狀態：獲利超過 10%", f"🎯 決策信心：{confidence}% (趨勢保護)"]
            elif cost > 0 and price >= cost * 1.05:
                final_status = "🟡 接近停利"
                ai_advice = ["✓ 建議：將停損點無條件上調至成本價", "✓ 依據：獲利空間已拉開", "✓ 狀態：確保此交易立於不敗", f"🎯 決策信心：{confidence}%"]
            elif price < ma20 * 0.95:
                final_status = "🔴 破線"
                ai_advice = ["✓ 建議：考慮預防性減碼或空手", "✓ 依據：跌破月線防守區", f"🎯 決策信心：{100 - confidence}% (偏空防守)"]
            elif ai_score >= 70:
                final_status = "🟢 進場"
                ai_advice = [f"✓ 建議：可分批進場，防守線 {atr_stop_price:.1f}", "✓ 依據：綜合戰力強勢共振", f"🎯 決策信心：{confidence}%"]
            else:
                final_status = "🟡 觀望"
                ai_advice = ["✓ 建議：保持空手盯盤", "✓ 依據：動能不足", f"🎯 決策信心：{confidence}%"]

            for w in macro_warnings:
                ai_advice.append(f"<span style='color: #fbbf24;'>{w}</span>")
            suggested_shares = min(int(risk_amount / atr), int(cap / price)) if atr > 0 else 0

            if final_status in ["🔴 破損", "🔴 破線", "⚠️ 帳面虧損"]:
                if isinstance(portfolio[code], dict) and 'break_date' not in portfolio[code]:
                    portfolio[code]['break_date'] = today_str
                    save_portfolio(portfolio)
            else:
                if isinstance(portfolio[code], dict) and 'break_date' in portfolio[code]:
                    del portfolio[code]['break_date']
                    save_portfolio(portfolio)

            # 【V2.9 修正】原版用不存在的 inst['t_days'] 判斷標籤，導致「投信作帳」永遠不會出現
            tags = ["🦅美股科技" if is_us_stock else ("🔥投信作帳" if inst.get('trust_days', 0) >= 3 else "🌊外資波段")]
            if is_bull_aligned and price > ma20: tags.append("🚀多頭起漲")
            elif price < ma60 and ma20 < ma60: tags.append("❄️弱勢空頭")
            if len(tags) == 1: tags.append("⏳區間震盪")

            if code not in system_history: system_history[code] = {}
            system_history[code][today_str] = {"score": ai_score, "status": final_status, "price": price}
            if len(system_history[code]) > 10: del system_history[code][sorted(system_history[code].keys())[0]]

            summary_data.append({"代號": code, "名稱": name, "現價": round(price, 2), "成本": round(cost, 2), "AI分數": ai_score, "股性標籤": " | ".join(tags[:2]), "風控點": f"{atr_stop_price:.1f}/{take_profit_price:.1f}" if cost > 0 else "-/-", "判定": final_status})
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume, "vol_ma5": vol_ma5,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi, "atr": atr, "bias": bias, "inst": inst, "tags": tags,
                "cap": cap, "risk_amount": risk_amount, "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "ai_score": ai_score, "final_status": final_status, "shares": suggested_shares, "atr_stop_price": atr_stop_price, "take_profit_price": take_profit_price,
                "ai_advice": ai_advice, "confidence": confidence, "pivot_point": pivot_point, "pivot_status": pivot_status, "is_us": is_us_stock, "score_inst": score_inst, "score_tech": score_tech, "score_vol": score_vol, "score_risk": score_risk, "score_forced_zero": score_forced_zero
            })
        except Exception as e: st.error(f"分析 {code} 發生錯誤: {e}")

    save_history(system_history)

    if summary_data:
        health_green = len([d for d in summary_data if "進場" in d['判定'] or "奔跑" in d['判定']])
        health_yellow = len([d for d in summary_data if "觀望" in d['判定'] or "接近" in d['判定']])
        health_red = len([d for d in summary_data if "破" in d['判定'] or "虧損" in d['判定'] or "退場" in d['判定']])

        st.markdown("### 🌟 持股健康度總覽")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("🟢 優勢/奔跑 (強勢)", f"{health_green} 檔")
        hc2.metric("🟡 觀望/警戒 (震盪)", f"{health_yellow} 檔")
        hc3.metric("🔴 破線/虧損 (弱勢)", f"{health_red} 檔")
        st.divider()

    if summary_data:
        df_summary = pd.DataFrame(summary_data).sort_values(by="AI分數", ascending=False).reset_index(drop=True)
        st.markdown("### 🏆 戰力排行榜 (Top 3 潛力股)")
        top_cols = st.columns(3)
        for i, (idx, row) in enumerate(df_summary.head(3).iterrows()):
            emoji = ["🥇", "🥈", "🥉"][i]
            top_cols[i].metric(f"{emoji} {row['名稱']} ({row['代號']})", f"{row['現價']:.2f}", f"戰力: {row['AI分數']}分", delta_color="normal" if row['AI分數']>=70 else "off")
        st.divider()

    if card_data:
        st.markdown("### ✅ 每日紀律檢核清單 (SOP)")

        overtime_broken = []
        for c, info in portfolio.items():
            if isinstance(info, dict) and info.get('status') != 'Closed':
                b_date_str = info.get('break_date')
                if b_date_str:
                    try:
                        b_date = datetime.datetime.strptime(b_date_str, "%Y-%m-%d")
                        diff_days = (datetime.datetime.now() - b_date).days
                        if diff_days >= 3:
                            overtime_broken.append(f"{info.get('name', c)} (已破線/虧損 {diff_days} 天)")
                    except Exception: pass

        if overtime_broken:
            st.error(f"🚨 **【最高紀律警報】** 以下持股已破線或虧損超過 3 天未處理，請立即執行手動歸檔或停損退場：\n\n" + "、".join(overtime_broken), icon="🚨")

        with st.expander("展開今日操作任務", expanded=True):
            action_sell, action_buy, action_watch = [], [], []
            for data in card_data:
                if data['final_status'] == "🔴 破損": action_sell.append(f"🚨 **停損退場**：{data['name']} 現價 {data['price']:.2f} 跌破防守點 {data['atr_stop_price']:.1f}。")
                elif data['final_status'] == "🔵 停利退場": action_sell.append(f"🛡️ **紀律停利**：{data['name']} 現價 {data['price']:.2f} 跌破動態防守 {data['atr_stop_price']:.1f}。")
                elif data['final_status'] == "⚠️ 帳面虧損": action_sell.append(f"⚠️ **帳面虧損**：{data['name']} 現價 {data['price']:.2f} 已跌破設定成本，請審慎評估。")
                elif data['final_status'] == "🔥 利潤奔跑": action_watch.append(f"🚀 **獲利續抱**：{data['name']} 月線 {data['atr_stop_price']:.1f} 不破不賣！")
                elif data['final_status'] == "🟢 進場": action_buy.append(f"🎯 **進場佈局**：{data['name']} 戰力達 {data['ai_score']} 分，建議部位：{data['shares']} 股。")
                elif data['final_status'] == "🟡 接近停利": action_watch.append(f"⚠️ **防守上調**：{data['name']} 獲利脫離成本，停損設為成本價。")
                elif data['final_status'] == "🔴 破線": action_watch.append(f"📉 **弱勢預警**：{data['name']} 跌破月線防守區。")

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
        if not tw_cards: st.info("目前無台股持股紀錄。")
        for data in tw_cards: render_stock_card(data, system_history, portfolio)

    with tab_us:
        us_cards = [d for d in card_data if d['is_us']]
        if not us_cards: st.info("目前無美股持股紀錄。")
        for data in us_cards: render_stock_card(data, system_history, portfolio)

if __name__ == "__main__":
    pass
