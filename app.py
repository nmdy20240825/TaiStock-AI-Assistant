import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide", page_title="TaiStock V2.11 全自動紀律決策系統")

# ===== UI 視覺與字體優化模組 =====
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 18px !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; white-space: normal !important; word-break: break-word !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
.ai-advice-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 0-0. 簡易密碼保護 ---
def _check_password():
    def _on_submit():
        if st.session_state.get("_pw_input", "") == st.secrets.get("app_password", ""):
            st.session_state["_pw_ok"] = True
            st.session_state["_pw_input"] = ""
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok", False):
        return True

    st.title("🔒 TaiStock 登入")
    if "app_password" not in st.secrets:
        st.error("⚠️ 尚未在 Streamlit Secrets 設定 app_password，暫時無法啟用密碼保護（目前對所有人開放）。")
        return True
    st.text_input("請輸入密碼", type="password", key="_pw_input", on_change=_on_submit)
    if st.session_state.get("_pw_ok") is False:
        st.error("密碼錯誤，請再試一次。")
    return False

if not _check_password():
    st.stop()

# --- 0. 技術指標輔助函式 ---
def calc_kd(h, l, c, period=9):
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
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    return float(ema_fast.iloc[-1] - ema_slow.iloc[-1])

def compute_signal_backtest(history):
    stats = {}
    for code, records in history.items():
        dates_sorted = sorted(records.keys())
        if len(dates_sorted) < 2: continue
        latest_price = records[dates_sorted[-1]].get('price', 0)
        if not latest_price: continue
        for d in dates_sorted[:-1]:
            entry = records[d]
            status = entry.get('status', '')
            entry_price = entry.get('price', 0)
            if not status or not entry_price: continue
            ret_pct = (latest_price - entry_price) / entry_price * 100
            stats.setdefault(status, []).append(ret_pct)
    return stats

# --- 1. 大盤宏觀環境抓取 ---
@st.cache_data(ttl=1800)
def fetch_macro_data():
    tickers = {'TW': '^TWII', 'US': '^IXIC', 'VIX': '^VIX'}
    macro_status = {}
    for key, symbol in tickers.items():
        try:
            df = yf.download(symbol, period="3mo", progress=False)
            df = _trim_trailing_nan_rows(df)
            if df is not None and not df.empty:
                c_series = df['Close'].squeeze()
                if isinstance(c_series, pd.DataFrame): c_series = c_series.iloc[:, 0]
                c = float(c_series.iloc[-1])
                ma20 = float(c_series.rolling(20).mean().iloc[-1])
                _asof = df.index[-1]
                macro_status[key] = {'price': c, 'ma20': ma20, 'trend': '🟢 多頭' if c > ma20 else '🔴 空頭', 'asof': _asof}
        except Exception:
            macro_status[key] = None
    return macro_status

# --- 2. 報價與技術資料抓取 ---
def _trim_trailing_nan_rows(df, max_trim=3, min_keep=60):
    if df is None or df.empty or 'Close' not in df.columns: return df
    close_col = df['Close']
    if isinstance(close_col, pd.DataFrame): close_col = close_col.iloc[:, 0]
    trim = 0
    while trim < max_trim and len(df) - trim > min_keep and pd.isna(close_col.iloc[-1 - trim]):
        trim += 1
    return df.iloc[:-trim] if trim > 0 else df

@st.cache_data(ttl=300)
def fetch_stock_data(code):
    try:
        if code.isalpha() or code.endswith('.US'):
            df = yf.download(code.replace('.US', ''), period="6mo", progress=False)
        elif code.endswith('.TW') or code.endswith('.TWO'):
            df = yf.download(code, period="6mo", progress=False)
        else:
            df_tw = yf.download(f"{code}.TW", period="6mo", progress=False)
            df = df_tw if (df_tw is not None and not df_tw.empty and len(df_tw) > 0) else yf.download(f"{code}.TWO", period="6mo", progress=False)
        return _trim_trailing_nan_rows(df)
    except Exception: return pd.DataFrame()

# --- 3. 籌碼資料抓取 ---
@st.cache_data(ttl=3600)
def get_institutional_data(code):
    default_res = {"buy_sell": 0, "days": 0, "trend": "資料不足", "accumulated_shares": 0, "foreign_trend": "無資料", "trust_trend": "無資料", "foreign_days": 0, "trust_days": 0}
    if code.isalpha() or code.endswith('.US'): return {"buy_sell": 0, "days": 0, "trend": "美股無籌碼", "accumulated_shares": 0, "foreign_trend": "N/A", "trust_trend": "N/A", "foreign_days": 0, "trust_days": 0}
    try:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": code, "start_date": start_date, "end_date": end_date}
        resp = requests.get(url, params=parameter, timeout=5)
        data = resp.json()

        if code.endswith(('.TW', '.TWO')): stock_data = yf.download(code, period="1mo", progress=False)
        else:
            stock_data = yf.download(f"{code}.TW", period="1mo", progress=False)
            if stock_data is None or stock_data.empty: stock_data = yf.download(f"{code}.TWO", period="1mo", progress=False)
        
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
        return {"days": days, "trend": trend_str, "accumulated_shares": float(accumulated_shares), "foreign_trend": f_trend, "trust_trend": t_trend, "foreign_days": f_days, "trust_days": t_days}
    except Exception: return default_res

# --- 4. 資料存取（整合新版 Trade Plan） ---
GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
PORTFOLIO_HEADERS = ["code", "name", "cost", "cap", "risk", "status", "break_date", "qty"]
HISTORY_HEADERS = ["code", "date", "score", "status", "price"]
TRADE_PLAN_HEADERS = [
    "code", "signal_date", "execution_date", "state", "signal_type", "signal_reason",
    "taiwan_data_date", "us_data_date", "entry_price", "breakout_price", "pullback_low",
    "pullback_high", "chase_limit", "invalid_price", "t1_price", "t2_price", "t1_taken",
    "t2_taken", "initial_stop", "previous_trailing_stop", "current_trailing_stop",
    "suggested_shares", "current_shares", "addon_shares_approved", "partial_exit_shares",
    "full_exit_shares", "valid_until", "last_action", "last_evaluated_at"
]

DEFAULT_PORTFOLIO = {
    "3035": {"name": "智原", "cost": 300.0, "cap": 20000, "risk": 5.0, "status": "Active", "qty": 0},
    "2317": {"name": "鴻海", "cost": 210.0, "cap": 20000, "risk": 5.0, "status": "Active", "qty": 0},
    "NVDA": {"name": "輝達", "cost": 125.0, "cap": 20000, "risk": 5.0, "status": "Active", "qty": 0}
}

@st.cache_resource
def get_gsheet_client():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_info, scopes=GSHEET_SCOPES)
    return gspread.authorize(creds)

@st.cache_resource
def get_spreadsheet():
    client = get_gsheet_client()
    return client.open_by_key(st.secrets["gsheet"]["sheet_id"])

def get_worksheet(name, headers):
    ss = get_spreadsheet()
    try: ws = ss.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=name, rows=200, cols=len(headers))
        ws.append_row(headers)
    return ws

def load_portfolio():
    try:
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS)
        records = ws.get_all_records()
        if not records:
            rows = [[code, info["name"], info["cost"], info["cap"], info["risk"], info["status"], "", info.get("qty", 0)] for code, info in DEFAULT_PORTFOLIO.items()]
            ws.append_rows(rows)
            return {k: dict(v) for k, v in DEFAULT_PORTFOLIO.items()}
        data = {}
        for row in records:
            code = str(row.get("code", "")).strip()
            if not code: continue
            data[code] = {
                "name": row.get("name", ""), "cost": float(row.get("cost") or 0.0),
                "cap": float(row.get("cap") or 20000.0), "risk": float(row.get("risk") or 5.0),
                "status": row.get("status") or "Active", "qty": float(row.get("qty") or 0.0),
                "break_date": str(row.get("break_date", "")).strip()
            }
        return data
    except Exception as e:
        st.error(f"⚠️ 讀取 Google Sheet 持股資料失敗：{e}")
        return {k: dict(v) for k, v in DEFAULT_PORTFOLIO.items()}

def save_portfolio(data):
    try:
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS)
        ws.clear()
        rows = [PORTFOLIO_HEADERS]
        for code, info in data.items():
            rows.append([code, info.get("name", ""), info.get("cost", 0.0), info.get("cap", 20000.0), info.get("risk", 5.0), info.get("status", "Active"), info.get("break_date", ""), info.get("qty", 0.0)])
        ws.update(rows)
    except Exception as e: st.error(f"⚠️ 寫入持股資料失敗：{e}")

def load_history():
    try:
        ws = get_worksheet("history", HISTORY_HEADERS)
        records = ws.get_all_records()
        data = {}
        for row in records:
            code, date = str(row.get("code", "")).strip(), str(row.get("date", "")).strip()
            if not code or not date: continue
            data.setdefault(code, {})[date] = {"score": int(float(row.get("score") or 0)), "status": row.get("status", ""), "price": float(row.get("price") or 0.0)}
        return data
    except Exception as e: return {}

def save_history(data):
    try:
        ws = get_worksheet("history", HISTORY_HEADERS)
        ws.clear()
        rows = [HISTORY_HEADERS]
        for code, records in data.items():
            for date, rec in records.items():
                s, p = rec.get("score", 0), rec.get("price", 0.0)
                if pd.isna(s): s = 0
                if pd.isna(p): p = 0.0
                rows.append([code, date, s, rec.get("status", ""), p])
        ws.update(rows)
    except Exception as e: pass

def load_trade_plans():
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        records = ws.get_all_records()
        plans = {}
        for row in records:
            code = str(row.get("code", "")).strip()
            if not code: continue
            plans[code] = {
                "state": row.get("state", "PREPARE"),
                "taiwan_data_date": str(row.get("taiwan_data_date", "")),
                "us_data_date": str(row.get("us_data_date", "")),
                "t1_price": float(row.get("t1_price") or 0.0),
                "t2_price": float(row.get("t2_price") or 0.0),
                "t1_taken": str(row.get("t1_taken", "")).lower() == 'true',
                "t2_taken": str(row.get("t2_taken", "")).lower() == 'true',
                "initial_stop": float(row.get("initial_stop") or 0.0),
                "previous_trailing_stop": float(row.get("current_trailing_stop") or 0.0),
                "current_trailing_stop": float(row.get("current_trailing_stop") or 0.0),
                "valid_until": str(row.get("valid_until", "")),
                "breakout_price": float(row.get("breakout_price") or 0.0),
                "pullback_low": float(row.get("pullback_low") or 0.0),
                "pullback_high": float(row.get("pullback_high") or 0.0),
                "chase_limit": float(row.get("chase_limit") or 0.0),
                "suggested_shares": int(float(row.get("suggested_shares") or 0)),
                "addon_shares_approved": int(float(row.get("addon_shares_approved") or 0)),
                "partial_exit_shares": int(float(row.get("partial_exit_shares") or 0)),
                "full_exit_shares": int(float(row.get("full_exit_shares") or 0)),
                "signal_reason": str(row.get("signal_reason", "")),
            }
        return plans
    except Exception as e:
        st.error(f"⚠️ 讀取 Google Sheet 交易計畫失敗：{e}")
        return {}

def save_trade_plans(plans):
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        ws.clear()
        rows = [TRADE_PLAN_HEADERS]
        for code, p in plans.items():
            rows.append([
                code, datetime.datetime.now().strftime("%Y-%m-%d"), "", p.get("state", ""), "", p.get("signal_reason", ""),
                p.get("taiwan_data_date", ""), p.get("us_data_date", ""), 0.0, p.get("breakout_price", 0.0),
                p.get("pullback_low", 0.0), p.get("pullback_high", 0.0), p.get("chase_limit", 0.0), 0.0,
                p.get("t1_price", 0.0), p.get("t2_price", 0.0), str(p.get("t1_taken", False)), str(p.get("t2_taken", False)),
                p.get("initial_stop", 0.0), p.get("previous_trailing_stop", 0.0), p.get("current_trailing_stop", 0.0),
                p.get("suggested_shares", 0), p.get("current_shares", 0), p.get("addon_shares_approved", 0),
                p.get("partial_exit_shares", 0), p.get("full_exit_shares", 0), p.get("valid_until", ""), "",
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
        ws.update(rows)
    except Exception as e: st.error(f"⚠️ 寫入交易計畫失敗：{e}")

portfolio = load_portfolio()
system_history = load_history()
trade_plans = load_trade_plans()
today_str = datetime.datetime.now().strftime("%Y-%m-%d")

# --- 5. 側邊欄 UI ---
with st.sidebar:
    st.header("📋 持股與風控設定")
    with st.form("add_stock"):
        new_code = st.text_input("代號 (台股數字 / 美股字母)")
        new_name, new_cost, new_cap, new_risk = st.text_input("名稱 (可留白)"), st.number_input("成本價", value=100.0, step=0.1), st.number_input("分配資金", value=20000, step=5000), st.number_input("單筆風險 (%)", value=5.0, step=0.1)
        new_qty = st.number_input("持有股數 (選填，0＝純訊號監控)", value=0, step=1, min_value=0)
        if st.form_submit_button("更新設定"):
            if new_code:
                fetch_stock_data.clear(); get_institutional_data.clear()
                existing_break_date = portfolio.get(new_code, {}).get('break_date') if isinstance(portfolio.get(new_code), dict) else None
                portfolio[new_code] = {"name": new_name, "cost": new_cost, "cap": new_cap, "risk": new_risk, "status": "Active", "qty": new_qty}
                if existing_break_date: portfolio[new_code]['break_date'] = existing_break_date
                save_portfolio(portfolio)
                st.rerun()
    del_code = st.selectbox("刪除持股", [""] + list(portfolio.keys()))
    if st.button("確認刪除") and del_code in portfolio:
        del portfolio[del_code]
        save_portfolio(portfolio)
        if del_code in system_history: del system_history[del_code]; save_history(system_history)
        if del_code in trade_plans: del trade_plans[del_code]; save_trade_plans(trade_plans)
        st.rerun()

    st.divider()
    st.subheader("📤 匯出 / 📥 匯入持股清單 (CSV)")
    _export_rows = [{"code": c, "name": i.get("name", ""), "cost": i.get("cost", 0.0), "cap": i.get("cap", 20000.0), "risk": i.get("risk", 5.0), "qty": i.get("qty", 0.0), "status": i.get("status", "Active")} for c, i in portfolio.items()]
    _csv_bytes = pd.DataFrame(_export_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button("📤 匯出目前持股清單", _csv_bytes, file_name="taistock_持股清單.csv", mime="text/csv")

    _uploaded_csv = st.file_uploader("📥 匯入持股清單 CSV", type=["csv"])
    if _uploaded_csv is not None:
        try:
            _df_import = pd.read_csv(_uploaded_csv)
            if st.button("✅ 確認匯入"):
                for _, _row in _df_import.iterrows():
                    _imp_code = str(_row.get("code", "")).strip()
                    if not _imp_code or _imp_code == "nan": continue
                    portfolio[_imp_code] = {"name": "" if pd.isna(_row.get("name", "")) else str(_row.get("name", "")), "cost": float(_row.get("cost", 0) or 0), "cap": float(_row.get("cap", 20000) or 20000), "risk": float(_row.get("risk", 5.0) or 5.0), "qty": float(_row.get("qty", 0) or 0), "status": "Active" if pd.isna(_row.get("status", "Active")) else str(_row.get("status", "Active"))}
                fetch_stock_data.clear(); get_institutional_data.clear()
                save_portfolio(portfolio); st.rerun()
        except Exception as e: st.error(f"⚠️ CSV 格式讀取失敗：{e}")

    st.divider()
    st.subheader("⏸️ 暫停分析 / ▶️ 恢復分析")
    _active_codes = [c for c, i in portfolio.items() if isinstance(i, dict) and i.get('status', 'Active') == 'Active']
    _paused_codes = [c for c, i in portfolio.items() if isinstance(i, dict) and i.get('status') == 'Paused']
    _pause_target = st.selectbox("選擇要暫停分析的股票", [""] + _active_codes, key="pause_select")
    if st.button("⏸️ 暫停分析") and _pause_target:
        portfolio[_pause_target]['status'] = 'Paused'
        save_portfolio(portfolio); st.rerun()
    _resume_target = st.selectbox("選擇要恢復分析的股票", [""] + _paused_codes, key="resume_select")
    if st.button("▶️ 恢復每日分析") and _resume_target:
        portfolio[_resume_target]['status'] = 'Active'
        save_portfolio(portfolio); st.rerun()

# --- 6. 核心狀態機與業務邏輯 ---
def calculate_trailing_stop(average_cost, atr, ma20, previous_trailing_stop):
    candidate_stop = ma20 - atr
    return max(previous_trailing_stop, candidate_stop, average_cost)

def calculate_exit_plan(close, atr, ma20, current_trailing_stop, h_max, t1_taken, t2_taken, t1_price, t2_price):
    if close < current_trailing_stop: return "FULL_EXIT_NEXT_DAY"
    if t1_price > 0 and close >= t1_price and not t1_taken: return "PARTIAL_EXIT_T1"
    if t2_price > 0 and close >= t2_price and not t2_taken: return "PARTIAL_EXIT_T2"
    return "HOLD"

def calculate_position_size(cap, risk_pct, entry_price, stop_price):
    risk_amount = cap * (risk_pct / 100)
    per_share_risk = max(entry_price - stop_price, 0.1)  # 防禦除0
    risk_based = risk_amount / per_share_risk
    cap_based = cap / entry_price if entry_price > 0 else 0
    return int(np.floor(min(risk_based, cap_based)))

def calculate_addon_shares(curr_qty, curr_price, curr_stop, add_price, add_stop, cap, risk_pct):
    max_risk = cap * (risk_pct / 100)
    remaining_risk = curr_qty * max(curr_price - curr_stop, 0)
    avail_risk = max_risk - remaining_risk
    if avail_risk <= 0: return 0
    per_share_risk = max(add_price - add_stop, 0.1)
    risk_based = avail_risk / per_share_risk
    cap_based = max(0, cap - (curr_qty * curr_price)) / add_price if add_price > 0 else 0
    return int(np.floor(min(risk_based, cap_based)))

def transition_state(plan, new_state, reason):
    plan['state'] = new_state
    plan['signal_reason'] = reason
    return plan

# --- 7. 主程式 UI 渲染與資料遍歷 ---
st.title("⚡ TaiStock V2.11 全自動決策系統 (升級版狀態機)")
st.warning("⚠️ 本系統僅為個人化技術指標整理與紀律提醒工具，所有分數、判定、建議均由你自訂的公式與參數計算而成，不構成任何投資建議。")

macro_data = fetch_macro_data()
st.markdown("### 🌍 雙軌市場環境總覽")
m_col1, m_col2, m_col3 = st.columns(3)

tw_trend = macro_data.get('TW', {})
tw_asof = tw_trend.get('asof', datetime.datetime.now()).strftime("%Y-%m-%d") if tw_trend else today_str
if tw_trend: m_col1.metric("🇹🇼 台股加權", f"{tw_trend['price']:,.0f}", tw_trend['trend'], delta_color="normal" if "多頭" in tw_trend['trend'] else "inverse")
else: m_col1.metric("🇹🇼 台股加權", "連線中...")

us_trend = macro_data.get('US', {})
us_asof = us_trend.get('asof', datetime.datetime.now()).strftime("%Y-%m-%d") if us_trend else today_str
if us_trend: m_col2.metric("🇺🇸 那斯達克", f"{us_trend['price']:,.0f}", us_trend['trend'], delta_color="normal" if "多頭" in us_trend['trend'] else "inverse")
else: m_col2.metric("🇺🇸 那斯達克", "連線中...")

vix_trend = macro_data.get('VIX', {})
if vix_trend:
    v_val = vix_trend['price']
    v_status, v_color = ("🚨 極度恐慌", "inverse") if v_val >= 25 else (("⚠️ 波動加劇", "off") if v_val >= 20 else ("🟢 環境穩定", "normal"))
    m_col3.metric("📉 VIX 恐慌指數", f"{v_val:.2f}", v_status, delta_color=v_color)
else: m_col3.metric("📉 VIX 恐慌指數", "連線中...")

market_regime = "BEARISH" if us_trend and "空頭" in us_trend.get('trend', '') else "BULLISH"
st.divider()

if not portfolio: st.info("👈 請先從左側邊欄新增股票代號！")
else:
    summary_data, card_data, paused_data = [], [], []
    has_plan_updates = False

    for code, info in list(portfolio.items()):
        if isinstance(info, dict) and info.get('status') == 'Closed': continue
        if isinstance(info, dict) and info.get('status') == 'Paused':
            if info.get('qty', 0) > 0:
                try:
                    df_pause = fetch_stock_data(code)
                    if df_pause is not None and not df_pause.empty:
                        paused_data.append({'code': code, 'name': info['name'], 'cost': info['cost'], 'price': float(df_pause['Close'].iloc[-1]), 'qty': info['qty'], 'is_us': code.isalpha() or code.endswith('.US')})
                except: pass
            continue
        
        name, cost, cap, risk_pct = info['name'], info['cost'], info['cap'], info['risk']
        held_qty = info.get('qty', 0)
        risk_amount = cap * (risk_pct / 100)
        
        try:
            df = fetch_stock_data(code)
            if df is None or df.empty or len(df) < 60: continue
            
            c, h, l, v = df['Close'], df['High'], df['Low'], df.get('Volume', pd.Series(0, index=df.index))
            price, volume, vol_ma5 = float(c.iloc[-1]), float(v.iloc[-1]), float(v.rolling(5).mean().iloc[-1])
            h_max = float(h.iloc[-20:].max()) # 前高參考
            
            # 指標計算
            ma10, ma20, ma60 = float(c.rolling(10).mean().iloc[-1]), float(c.rolling(20).mean().iloc[-1]), float(c.rolling(60).mean().iloc[-1])
            macd = calc_macd(c)
            k_series, d_series = calc_kd(h, l, c)
            k, d = float(k_series.iloc[-1]), float(d_series.iloc[-1])
            delta = c.diff()
            up, down = delta.clip(lower=0).rolling(14).mean().iloc[-1], -1 * delta.clip(upper=0).rolling(14).mean().iloc[-1]
            rsi = float(100 - (100 / (1 + (np.nan_to_num(up) / (np.nan_to_num(down) + 0.001)))))
            atr = float(sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-13, 0)]) / 14)
            bias = float(((price - ma60) / ma60) * 100)
            boll_upper = float((c.rolling(20).mean() + 2 * c.rolling(20).std()).iloc[-1])
            pivot_point = (float(h.iloc[-2]) + float(l.iloc[-2]) + float(c.iloc[-2])) / 3 if len(h) >= 2 else price
            pivot_status = "🟢 站上" if price > pivot_point else "🔴 未站上"

            # NaN 防呆 (失敗則保留舊計畫，跳過更新)
            if any(pd.isna(x) for x in [price, ma20, macd, k, rsi, atr]):
                st.warning(f"⚠️ {name} ({code}) 抓取資料含空值，保留原計畫。")
                continue

            inst = get_institutional_data(code)
            is_us_stock = code.isalpha() or code.endswith('.US')
            
            # 分數計算
            score_inst = (20 if price > ma60 else 0) + (10 if macd > 0 else 0) + (10 if 0 < bias < 20 else 0) if is_us_stock else min(inst['days'] * 5, 20) + (20 if inst['accumulated_shares'] * price >= 3000000000 else (10 if inst['accumulated_shares'] * price >= 1000000000 else 0))
            _rsi_bull = 10 if (50 < rsi <= 80) else 0
            score_tech = (10 if k > d else 0) + _rsi_bull + (10 if price > ma20 else 0)
            score_vol = min((volume / vol_ma5) * 10, 15) if vol_ma5 > 0 else 0
            
            ai_score = min(int(score_inst + score_tech + score_vol + 15), 100)
            confidence = min(99, max(10, int(ai_score * 0.8 + (10 if ma10 > ma20 > ma60 else 0))))
            
            step1_pass = (price > ma60 and macd > 0) if is_us_stock else (inst['days'] >= 3 or inst['accumulated_shares'] * price >= 1000000000)
            step2_pass, step3_pass = (k > d and rsi > 50 and volume > vol_ma5), (price > ma20 and ma10 > ma20 > ma60)

            # --- 載入與評估 Trade Plan (狀態機核心) ---
            plan = trade_plans.get(code, {"state": "PREPARE", "taiwan_data_date": "", "us_data_date": ""})
            latest_data_date = tw_asof if not is_us_stock else us_asof
            
            if plan.get("taiwan_data_date") != latest_data_date:
                has_plan_updates = True
                plan["taiwan_data_date"] = latest_data_date
                plan["us_data_date"] = us_asof

                # 檢查過期
                if plan['state'] in ["PREPARE", "BREAKOUT_WAIT", "PULLBACK_WAIT"] and plan.get("valid_until"):
                    if latest_data_date > plan["valid_until"]:
                        transition_state(plan, "EXPIRED", "訊號過期失效")

                # 持倉管理優先
                if held_qty > 0:
                    if not plan.get("initial_stop"): plan["initial_stop"] = cost - 2 * atr
                    if not plan.get("current_trailing_stop"): plan["current_trailing_stop"] = plan["initial_stop"]
                    
                    plan["current_trailing_stop"] = calculate_trailing_stop(cost, atr, ma20, plan["current_trailing_stop"])
                    
                    # 判斷出清或停利
                    exit_res = calculate_exit_plan(price, atr, ma20, plan["current_trailing_stop"], h_max, plan.get("t1_taken"), plan.get("t2_taken"), plan.get("t1_price"), plan.get("t2_price"))
                    
                    if exit_res == "FULL_EXIT_NEXT_DAY":
                        transition_state(plan, "FULL_EXIT_NEXT_DAY", "跌破移動防守線，強制作業")
                        plan["full_exit_shares"] = held_qty
                    elif exit_res == "PARTIAL_EXIT_T1":
                        transition_state(plan, "PARTIAL_EXIT_NEXT_DAY", "達 T1 目標，分批停利")
                        plan["partial_exit_shares"] = max(1, int(held_qty * 0.3))
                        plan["t1_taken"] = True
                    elif exit_res == "PARTIAL_EXIT_T2":
                        transition_state(plan, "PARTIAL_EXIT_NEXT_DAY", "達 T2 目標，分批停利")
                        plan["partial_exit_shares"] = held_qty
                        plan["t2_taken"] = True
                    else:
                        # 加碼判定
                        if step1_pass and step2_pass and step3_pass and market_regime != "BEARISH":
                            add_sh = calculate_addon_shares(held_qty, price, plan["current_trailing_stop"], price, ma20-atr, cap, risk_pct)
                            if add_sh > 0 and price > cost + 0.5 * atr:
                                transition_state(plan, "ADD_NEXT_DAY", "趨勢強勢，核准加碼")
                                plan["addon_shares_approved"] = add_sh
                            else:
                                transition_state(plan, "HOLD", "持有續抱")
                        else:
                            transition_state(plan, "HOLD", "持有續抱")
                
                # 空手進場邏輯
                else:
                    if plan['state'] in ["PREPARE", "EXPIRED", "INVALID", "FULL_EXIT_NEXT_DAY"]:
                        if ai_score >= 70 and step1_pass and step2_pass and market_regime != "BEARISH":
                            b_price = h_max * 1.005
                            c_limit = min(b_price + atr, b_price * 1.03)
                            if price <= c_limit:
                                transition_state(plan, "ENTER_NEXT_DAY", "綜合戰力達標，明日進場")
                                plan["breakout_price"] = b_price
                                plan["suggested_shares"] = calculate_position_size(cap, risk_pct, price, ma20-atr)
                                plan["t1_price"] = price + 2 * atr
                                plan["t2_price"] = price + 4 * atr
                                plan["valid_until"] = (datetime.datetime.strptime(latest_data_date, "%Y-%m-%d") + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
                            else:
                                transition_state(plan, "PULLBACK_WAIT", "現價過高，等待回測")
                                plan["valid_until"] = (datetime.datetime.strptime(latest_data_date, "%Y-%m-%d") + datetime.timedelta(days=5)).strftime("%Y-%m-%d")
                    elif plan['state'] == "BREAKOUT_WAIT" and price > plan.get("breakout_price", 9999):
                        transition_state(plan, "ENTER_NEXT_DAY", "收盤突破前高")
            
            # 美股逆風攔截
            if market_regime == "BEARISH" and plan['state'] in ["PREPARE", "BREAKOUT_WAIT", "PULLBACK_WAIT", "ENTER_NEXT_DAY", "ADD_NEXT_DAY"]:
                transition_state(plan, "SUSPENDED_BY_REGIME", "市場逆風暫停建倉/加碼")

            trade_plans[code] = plan

            # --- UI 狀態對應 ---
            state_to_ui = {
                "ENTER_NEXT_DAY": "🟢 準備進場", "ADD_NEXT_DAY": "📈 準備加碼", "FULL_EXIT_NEXT_DAY": "🔴 全部出清",
                "PARTIAL_EXIT_NEXT_DAY": "🔵 分批停利", "HOLD": "🔥 續抱中", "PREPARE": "🟡 觀望", 
                "BREAKOUT_WAIT": "🟡 等待突破", "PULLBACK_WAIT": "🟡 等待回測", "SUSPENDED_BY_REGIME": "⚠️ 逆風暫停",
                "EXPIRED": "🟡 訊號過期"
            }
            final_status = state_to_ui.get(plan['state'], "🟡 觀望")
            if held_qty > 0 and price < cost and plan['state'] == "HOLD": final_status = "⚠️ 帳面虧損"
            
            ai_advice = [f"✓ 系統狀態：{plan['state']} ({plan.get('signal_reason', '')})"]
            if plan['state'] == "ENTER_NEXT_DAY": ai_advice.append(f"💰 建議建倉：{plan['suggested_shares']} 股")
            elif plan['state'] == "ADD_NEXT_DAY": ai_advice.append(f"📈 核准加碼：{plan['addon_shares_approved']} 股")
            elif plan['state'] == "FULL_EXIT_NEXT_DAY": ai_advice.append(f"🚨 強制出清防守線：{plan['current_trailing_stop']:.1f}")

            if rsi > 80: ai_advice.append(f"<span style='color: #f87171;'>🚨 RSI極度過熱 ({rsi:.1f})，嚴防反轉。</span>")
            elif rsi < 20: ai_advice.append(f"<span style='color: #60a5fa;'>ℹ️ RSI極度過冷 ({rsi:.1f})。</span>")

            tags = ["🦅美股科技" if is_us_stock else ("🔥投信作帳" if inst.get('trust_days', 0) >= 3 else "🌊外資波段")]
            if is_bull_aligned: tags.append("🚀多頭排列")

            if code not in system_history: system_history[code] = {}
            if latest_data_date not in system_history[code]:
                system_history[code][latest_data_date] = {"score": ai_score, "status": final_status, "price": price}
                if len(system_history[code]) > 10: del system_history[code][sorted(system_history[code].keys())[0]]

            summary_data.append({"代號": code, "名稱": name, "現價": round(price, 2), "成本": round(cost, 2), "AI分數": ai_score, "股性標籤": " | ".join(tags[:2]), "風控點": f"{plan.get('current_trailing_stop', 0):.1f}/{plan.get('t1_price', 0):.1f}", "判定": final_status})
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume, "vol_ma5": vol_ma5,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi, "atr": atr, "bias": bias, "inst": inst, "tags": tags,
                "cap": cap, "risk_amount": risk_amount, "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "ai_score": ai_score, "final_status": final_status, "shares_adjusted": plan.get("suggested_shares", 0), 
                "held_qty": held_qty, "addon_shares_approved": plan.get("addon_shares_approved", 0),
                "atr_stop_price": plan.get("current_trailing_stop", 0.0), "take_profit_price": plan.get("t1_price", 0.0),
                "ai_advice": ai_advice, "pivot_point": pivot_point, "pivot_status": pivot_status, "is_us": is_us_stock, 
                "score_inst": score_inst, "score_tech": score_tech, "score_vol": score_vol, "score_risk": 15, "score_forced_zero": False
            })
        except Exception as e: st.error(f"分析 {code} 發生錯誤: {e}")

    if has_plan_updates:
        save_trade_plans(trade_plans)
        save_history(system_history)

    # --- UI 渲染區塊 ---
    if card_data:
        _headline_top = max(card_data, key=lambda x: x['ai_score'])
        st.info(f"🧠 **AI 每日一句**：今天最值得留意的是 **{_headline_top['name']}（{_headline_top['code']}）**，戰力 {_headline_top['ai_score']} 分，狀態「{_headline_top['final_status']}」。")

    if summary_data:
        h_green = len([d for d in summary_data if "進場" in d['判定'] or "續抱" in d['判定'] or "加碼" in d['判定']])
        h_yellow = len([d for d in summary_data if "觀望" in d['判定'] or "等待" in d['判定'] or "暫停" in d['判定'] or "過期" in d['判定']])
        h_red = len([d for d in summary_data if "出清" in d['判定'] or "虧損" in d['判定'] or "停利" in d['判定']])
        st.markdown("### 🌟 持股健康度總覽")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("🟢 強勢 (進場/續抱/加碼)", f"{h_green} 檔")
        hc2.metric("🟡 震盪 (觀望/等待/暫停)", f"{h_yellow} 檔")
        hc3.metric("🔴 弱勢 (出清/停利/虧損)", f"{h_red} 檔")
        st.divider()

    if card_data or paused_data:
        st.markdown("### 💰 資產總覽（依持有股數計算）")
        _valued_cards = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) > 0] + paused_data
        if not _valued_cards: st.info("目前沒有持股填寫「持有股數」。")
        else:
            def _render_asset_group(cards, c_label, c_sym):
                if not cards: return
                st.markdown(f"**{c_label}**")
                tc = sum(d['cost'] * portfolio[d['code']].get('qty', 0) for d in cards)
                tm = sum(d['price'] * portfolio[d['code']].get('qty', 0) for d in cards)
                tp = tm - tc
                tp_pct = (tp / tc * 100) if tc > 0 else 0.0
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric(f"總投入成本 ({c_sym})", f"{tc:,.0f}")
                ac2.metric(f"目前總市值 ({c_sym})", f"{tm:,.0f}")
                ac3.metric(f"總損益 ({c_sym})", f"{tp:,.0f}", f"{tp_pct:+.2f}%", delta_color="normal" if tp >= 0 else "inverse")
            _render_asset_group([d for d in _valued_cards if not d['is_us']], "🇹🇼 台股資產", "TWD")
            _render_asset_group([d for d in _valued_cards if d['is_us']], "🇺🇸 美股資產", "USD")
        st.divider()

    if card_data:
        st.markdown("### ✅ 每日紀律檢核清單 (SOP)")
        with st.expander("展開今日操作任務", expanded=True):
            action_sell, action_buy, action_watch = [], [], []
            for data in card_data:
                if data['final_status'] == "🔴 全部出清": action_sell.append(f"🚨 **強制出清**：{data['name']} 跌破移動防守線 {data['atr_stop_price']:.1f}。")
                elif data['final_status'] == "🔵 分批停利": action_sell.append(f"🛡️ **分批停利**：{data['name']} 達 T1/T2 目標價。")
                elif data['final_status'] == "🟢 準備進場": action_buy.append(f"🎯 **明日進場**：{data['name']} 建議部位 {data['shares_adjusted']} 股。")
                elif data['final_status'] == "📈 準備加碼": action_buy.append(f"📈 **明日加碼**：{data['name']} 核准 {data['addon_shares_approved']} 股。")
                elif data['final_status'] in ["🔥 續抱中", "⚠️ 帳面虧損"]: action_watch.append(f"👀 **持續追蹤**：{data['name']} (防守線: {data['atr_stop_price']:.1f})")

            st.markdown("#### 🟥 優先執行 (出清與停利)")
            if not action_sell: st.write("✅ 今日無急迫出清需求")
            for i, task in enumerate(action_sell): st.checkbox(task, key=f"sell_{i}")

            st.markdown("#### 🟩 佈局與加碼清單")
            if not action_buy: st.write("⏸️ 今日無符合進場或加碼標的")
            for i, task in enumerate(action_buy): st.checkbox(task, key=f"buy_{i}")
        st.divider()

    st.markdown("### 📊 AI 深度解析清單")
    card_data = sorted(card_data, key=lambda x: x['ai_score'], reverse=True)
    _view_filter = st.radio("顯示範圍", ["全部", "只看實際持股 💰", "只看觀察名單 👁️"], horizontal=True, key="view_filter")
    if _view_filter == "只看實際持股 💰": card_data = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) > 0]
    elif _view_filter == "只看觀察名單 👁️": card_data = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) <= 0]

    tab_tw, tab_us = st.tabs(["🇹🇼 台股主力陣列", "🇺🇸 美股科技巨頭"])
    with tab_tw:
        for data in [d for d in card_data if not d['is_us']]: render_stock_card(data, system_history, portfolio)
    with tab_us:
        for data in [d for d in card_data if d['is_us']]: render_stock_card(data, system_history, portfolio)

if __name__ == "__main__":
    pass
