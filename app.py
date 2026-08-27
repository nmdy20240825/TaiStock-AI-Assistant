import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime
import math
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="TaiStock V2.11.x 交易狀態機決策系統")

# ===== UI 視覺與字體優化模組 =====
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 18px !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; white-space: normal !important; word-break: break-word !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
.ai-advice-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 0-0. 簡易密碼保護（V2.10.4 新增）---
# 【重要】這只是「擋掉隨便知道網址就能看」的基本防護，不是正規帳號系統。
# 密碼存在 Streamlit 的 Secrets 裡（st.secrets["app_password"]），不會寫進程式碼或 GitHub。
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
        return True  # 沒設定密碼時不擋，避免自己也被鎖在外面
    st.text_input("請輸入密碼", type="password", key="_pw_input", on_change=_on_submit)
    if st.session_state.get("_pw_ok") is False:
        st.error("密碼錯誤，請再試一次。")
    return False

if not _check_password():
    st.stop()

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

def compute_signal_backtest(history):
    """
    【V2.10 新增】依累積的歷史記錄，回測各判定狀態「後續」的平均報酬與勝率。
    做法：對每一檔股票，把「較早那筆記錄的價格」拿去跟「該股目前累積歷史中最新一筆的價格」比較，
    算出報酬率，再依「較早那筆的判定狀態」分組統計。
    受限於 history 目前每檔股票只保留最近10筆記錄，樣本數會隨使用天數增加而變多，
    不是嚴謹的長期回測，但足夠用來觀察「這套 SOP 過去發出的訊號，後續大致準不準」。
    """
    stats = {}  # 判定狀態 -> 報酬率(%) 清單
    for code, records in history.items():
        dates_sorted = sorted(records.keys())
        if len(dates_sorted) < 2:
            continue
        latest_price = records[dates_sorted[-1]].get('price', 0)
        if not latest_price:
            continue
        for d in dates_sorted[:-1]:
            entry = records[d]
            status = entry.get('status', '')
            entry_price = entry.get('price', 0)
            if not status or not entry_price:
                continue
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
            df = _trim_trailing_nan_rows(df)  # 同樣防範 Yahoo 尾端佔位空列的問題
            if df is not None and not df.empty:
                c_series = df['Close'].squeeze()
                if isinstance(c_series, pd.DataFrame): c_series = c_series.iloc[:, 0]
                c = float(c_series.iloc[-1])
                ma20 = float(c_series.rolling(20).mean().iloc[-1])
                # 【V2.10.8 新增】記錄這筆資料實際對應的交易日期，讓畫面上能顯示「資料日期」，
                # 使用者才能自己判斷這是不是最新資料，而不是完全信任一個數字。
                _asof = df.index[-1]
                macro_status[key] = {'price': c, 'ma20': ma20, 'trend': '🟢 多頭' if c > ma20 else '🔴 空頭', 'asof': _asof}
        except Exception:
            macro_status[key] = None
    return macro_status

# --- 2. 報價與技術資料抓取 ---
def _trim_trailing_nan_rows(df, max_trim=3, min_keep=60):
    """
    【V2.10.1 修正】Yahoo 的台股（TWSE/TPEx）資料源偶爾會在資料尾端多附一筆
    「尚未結算/佔位用」的空列，整列 OHLC 都是 NaN——常發生在週末或跨時區查詢時，
    而且是整個交易所的資料源問題，不是單一個股的問題，所以會一次影響所有台股，
    但不影響美股（美股走的是另一條資料管線）。
    這裡在抓完資料後，先把尾端這種空列去掉，讓後面的技術指標計算不會平白無故拿到 NaN，
    導致整檔股票被 NaN 防呆機制跳過。最多只修剪 3 列，且不會修剪到低於 60 列，避免誤刪正常資料。
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return df
    close_col = df['Close']
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
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
PORTFOLIO_HEADERS = ["code", "name", "cost", "cap", "risk", "status", "break_date", "qty"]
HISTORY_HEADERS = ["code", "date", "score", "status", "price"]

DEFAULT_PORTFOLIO = {
    "3035": {"name": "智原", "cost": 300.0, "cap": 20000, "risk": 5.0, "status": "Active", "qty": 0},
    "2317": {"name": "鴻海", "cost": 210.0, "cap": 20000, "risk": 5.0, "status": "Active", "qty": 0},
    "NVDA": {"name": "輝達", "cost": 125.0, "cap": 20000, "risk": 5.0, "status": "Active", "qty": 0}
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


# ===== V2.11.x Trade Plan / Event-Driven State Machine =====
TAIWAN_CLOSE_UPDATE = "TAIWAN_CLOSE_UPDATE"
US_CLOSE_UPDATE = "US_CLOSE_UPDATE"
VIEW_ONLY = "VIEW_ONLY"

TRADE_PLAN_HEADERS = [
    "code", "signal_type", "state", "signal_reason", "signal_date", "execution_date", "valid_until",
    "last_evaluated_at", "entry_price", "breakout_price", "pullback_low", "pullback_high", "chase_limit",
    "invalid_price", "t1_price", "t2_price", "t1_taken", "t2_taken", "partial_exit_ratio", "remaining_shares",
    "initial_stop", "previous_trailing_stop", "current_trailing_stop", "suggested_shares", "addon_shares",
    "max_risk_amount", "used_risk_amount", "remaining_risk_amount", "taiwan_data_date", "us_data_date", "signal_key",
    "last_action", "last_action_date", "last_known_qty", "plan_version", "origin_state"
]

TRADE_PLAN_LOAD_OK = True

TRADE_STATES = {
    "PREPARE", "BREAKOUT_WAIT", "ENTER_NEXT_DAY", "HOLD", "ADD_NEXT_DAY",
    "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "SUSPENDED_BY_REGIME", "EXPIRED", "INVALID"
}


def _safe_float(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def _bool_value(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _date_str(value):
    if value is None or value == "":
        return ""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


def _next_business_day(date_str):
    try:
        d = pd.Timestamp(date_str)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _add_business_days(date_str, days):
    try:
        d = pd.Timestamp(date_str)
        remaining = int(days)
        while remaining > 0:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5:
                remaining -= 1
        return d.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _normalize_trade_plan_row(row):
    r = {h: row.get(h, "") for h in TRADE_PLAN_HEADERS}
    r["code"] = str(r.get("code", "")).strip()
    r["state"] = str(r.get("state", "PREPARE") or "PREPARE")
    if r["state"] not in TRADE_STATES:
        r["state"] = "PREPARE"
    r["signal_type"] = str(r.get("signal_type", "") or "")
    r["signal_reason"] = str(r.get("signal_reason", "") or "")
    for k in ["entry_price", "breakout_price", "pullback_low", "pullback_high", "chase_limit", "invalid_price",
              "t1_price", "t2_price", "initial_stop", "previous_trailing_stop", "current_trailing_stop",
              "max_risk_amount", "used_risk_amount", "remaining_risk_amount"]:
        r[k] = _safe_float(r.get(k), 0.0)
    for k in ["suggested_shares", "addon_shares", "remaining_shares", "last_known_qty"]:
        r[k] = _safe_int(r.get(k), 0)
    r["partial_exit_ratio"] = _safe_float(r.get("partial_exit_ratio"), 0.30)
    r["t1_taken"] = _bool_value(r.get("t1_taken"))
    r["t2_taken"] = _bool_value(r.get("t2_taken"))
    r["signal_date"] = _date_str(r.get("signal_date"))
    r["execution_date"] = _date_str(r.get("execution_date"))
    r["valid_until"] = _date_str(r.get("valid_until"))
    r["last_evaluated_at"] = _date_str(r.get("last_evaluated_at"))
    r["taiwan_data_date"] = _date_str(r.get("taiwan_data_date"))
    r["us_data_date"] = _date_str(r.get("us_data_date"))
    r["last_action_date"] = _date_str(r.get("last_action_date"))
    r["signal_key"] = str(r.get("signal_key", "") or "")
    r["last_action"] = str(r.get("last_action", "") or "")
    r["plan_version"] = str(r.get("plan_version", "2.11.x" ) or "2.11.x")
    r["origin_state"] = str(r.get("origin_state", "") or "")
    return r


def _trade_plan_defaults(code):
    return _normalize_trade_plan_row({"code": code, "state": "PREPARE", "plan_version": "2.11.x"})


def load_trade_plan():
    global TRADE_PLAN_LOAD_OK
    TRADE_PLAN_LOAD_OK = True
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        records = ws.get_all_records()
        data = {}
        for row in records:
            code = str(row.get("code", "")).strip()
            if not code:
                continue
            data[code] = _normalize_trade_plan_row(row)
        return data
    except Exception as e:
        TRADE_PLAN_LOAD_OK = False
        st.warning(f"⚠️ 讀取 trade_plan 失敗，本次改為 VIEW_ONLY，不修改既有交易計畫：{e}")
        return {}


def save_trade_plan(data):
    if not TRADE_PLAN_LOAD_OK:
        return False
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        rows = [TRADE_PLAN_HEADERS]
        for code, raw in data.items():
            r = _normalize_trade_plan_row(dict(raw, code=code))
            rows.append([r[h] for h in TRADE_PLAN_HEADERS])
        ws.update(rows)
        return True
    except Exception as e:
        st.error(f"⚠️ 寫入 trade_plan 失敗：保留記憶體中的上次成功交易計畫，本次狀態不視為已保存。錯誤：{e}")
        return False


def migrate_trade_plan_sheet():
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        existing = ws.row_values(1)
        if not existing:
            ws.append_row(TRADE_PLAN_HEADERS)
            return
        missing = [h for h in TRADE_PLAN_HEADERS if h not in existing]
        if missing:
            start = len(existing) + 1
            # 新欄位只會追加，不清除、不重排、不覆蓋既有資料。
            ws.update_cell(1, start, missing[0])
            for offset, h in enumerate(missing[1:], start=1):
                ws.update_cell(1, start + offset, h)
    except Exception as e:
        st.warning(f"⚠️ trade_plan Migration 暫時失敗，既有資料不會被清除：{e}")


def detect_update_mode(latest_tw_date, latest_us_date, saved_tw_date, saved_us_date):
    tw_new = bool(latest_tw_date and (not saved_tw_date or latest_tw_date > saved_tw_date))
    us_new = bool(latest_us_date and (not saved_us_date or latest_us_date > saved_us_date))
    if tw_new:
        return TAIWAN_CLOSE_UPDATE
    if us_new:
        return US_CLOSE_UPDATE
    return VIEW_ONLY


def _regime_is_bearish(market_context, is_us_stock):
    if is_us_stock:
        us = market_context.get("US") or {}
        vix = market_context.get("VIX") or {}
        return ("空頭" in str(us.get("trend", ""))) or _safe_float(vix.get("price"), 0) >= 25
    tw = market_context.get("TW") or {}
    return "空頭" in str(tw.get("trend", ""))


def calculate_exit_plan(entry_price, stop_price, risk_reward_ratio=2.0):
    entry = _safe_float(entry_price)
    stop = _safe_float(stop_price)
    if entry <= 0 or stop <= 0 or stop >= entry:
        return {"initial_stop": max(0.0, stop), "t1_price": 0.0, "t2_price": 0.0, "r1": 0.0, "r2": 0.0}
    risk = entry - stop
    r1 = risk
    r2 = risk * _safe_float(risk_reward_ratio, 2.0)
    return {"initial_stop": stop, "t1_price": entry + r1, "t2_price": entry + r2, "r1": r1, "r2": r2}


def calculate_entry_plan(code, indicators, portfolio_info, market_context):
    price = _safe_float(indicators.get("price"))
    atr = _safe_float(indicators.get("atr"))
    pivot = _safe_float(indicators.get("pivot_point"), price)
    ma20 = _safe_float(indicators.get("ma20"), price)
    if price <= 0 or atr <= 0:
        return {}
    breakout = max(price, pivot)
    entry = breakout
    pullback_low = max(0.0, entry - 0.50 * atr)
    pullback_high = entry + 0.25 * atr
    chase_limit = entry + 0.80 * atr
    invalid = max(0.0, entry - 1.00 * atr)
    stop = max(0.0, min(invalid, ma20 - 0.50 * atr))
    if stop >= entry:
        stop = max(0.01, entry - atr)
    exits = calculate_exit_plan(entry, stop, 2.0)
    today = indicators.get("data_date") or datetime.datetime.now().strftime("%Y-%m-%d")
    return {
        "code": code, "entry_price": entry, "breakout_price": breakout, "pullback_low": pullback_low,
        "pullback_high": pullback_high, "chase_limit": chase_limit, "invalid_price": invalid,
        "initial_stop": exits["initial_stop"], "previous_trailing_stop": exits["initial_stop"],
        "current_trailing_stop": exits["initial_stop"], "t1_price": exits["t1_price"], "t2_price": exits["t2_price"],
        "signal_date": today, "execution_date": _next_business_day(today), "valid_until": _add_business_days(today, 3),
        "signal_type": "ENTRY", "partial_exit_ratio": 0.30, "plan_version": "2.11.x",
    }


def calculate_position_size(cap, risk_pct, entry_price, stop_price):
    cap = max(0.0, _safe_float(cap)); risk_pct = max(0.0, _safe_float(risk_pct))
    entry = _safe_float(entry_price); stop = _safe_float(stop_price)
    risk_amount = cap * risk_pct / 100.0
    per_share_risk = max(0.0, entry - stop)
    risk_shares = int(risk_amount / per_share_risk) if per_share_risk > 0 else 0
    capital_shares = int(cap / entry) if entry > 0 else 0
    return {"risk_amount": risk_amount, "per_share_risk": per_share_risk,
            "risk_based_shares": risk_shares, "capital_based_shares": capital_shares,
            "shares_adjusted": min(risk_shares, capital_shares)}


def calculate_trailing_stop(previous_stop, current_price, ma20, atr, cost):
    prev = _safe_float(previous_stop)
    price = _safe_float(current_price)
    ma20 = _safe_float(ma20)
    atr = _safe_float(atr)
    cost = _safe_float(cost)
    candidates = [prev]
    if ma20 > 0 and atr > 0:
        candidates.append(ma20 - 0.50 * atr)
    if cost > 0 and price > cost:
        candidates.append(cost)
    if not candidates:
        return 0.0
    # 嚴格單向：current_stop >= previous_stop。
    return max(candidates)


def calculate_addon_shares(trade_plan, portfolio_info, indicators):
    price = _safe_float(indicators.get("price"))
    stop = _safe_float(trade_plan.get("current_trailing_stop") or trade_plan.get("initial_stop"))
    cap = _safe_float(portfolio_info.get("cap"))
    risk_pct = _safe_float(portfolio_info.get("risk"), 5.0)
    held = _safe_int(portfolio_info.get("qty"))
    max_risk = cap * risk_pct / 100.0
    used_risk = max(0.0, held * max(0.0, price - stop))
    remaining_risk = max(0.0, max_risk - used_risk)
    per_share_risk = max(0.0, price - stop)
    risk_shares = int(remaining_risk / per_share_risk) if per_share_risk > 0 else 0
    remaining_capital = max(0.0, cap - held * price)
    capital_shares = int(remaining_capital / price) if price > 0 else 0
    addon = min(risk_shares, capital_shares)
    return {"addon_shares": max(0, addon), "max_risk_amount": max_risk, "used_risk_amount": used_risk,
            "remaining_risk_amount": remaining_risk, "addon_per_share_risk": per_share_risk}


def is_signal_expired(trade_plan, data_date):
    valid = _date_str(trade_plan.get("valid_until"))
    return bool(valid and data_date > valid)


def is_duplicate_signal(trade_plan, signal_key):
    return bool(signal_key and trade_plan.get("signal_key") == signal_key)


def transition_trade_state(trade_plan, next_state, action, data_date, reason=""):
    current = trade_plan.get("state", "PREPARE")
    trade_plan["origin_state"] = current
    trade_plan["state"] = next_state
    trade_plan["last_action"] = action
    trade_plan["last_action_date"] = data_date
    trade_plan["last_evaluated_at"] = data_date
    if reason:
        trade_plan["signal_reason"] = reason
    return trade_plan


def evaluate_trade_state(trade_plan, indicators, market_context, portfolio_info):
    """嚴格優先序：FULL_EXIT > PARTIAL_EXIT > ADD > ENTRY > HOLD/PREPARE。"""
    plan = _normalize_trade_plan_row(trade_plan)
    price = _safe_float(indicators.get("price"))
    data_date = _date_str(indicators.get("data_date"))
    held = _safe_int(portfolio_info.get("qty"))
    cost = _safe_float(portfolio_info.get("cost"))
    regime_bearish = _regime_is_bearish(market_context, bool(indicators.get("is_us")))
    current_stop = _safe_float(plan.get("current_trailing_stop") or plan.get("initial_stop"))

    if held > 0:
        previous_qty = _safe_int(plan.get("last_known_qty"), held)
        plan["last_known_qty"] = held
        if plan.get("state") in {"ENTER_NEXT_DAY", "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY"} and plan.get("entry_price", 0) <= 0:
            plan["entry_price"] = cost

        # 強制出清永遠第一順位，不能被停利、加碼或續抱覆蓋。
        if current_stop > 0 and price <= current_stop:
            key = f"{plan.get('code')}|FULL_EXIT|{data_date}|{round(current_stop,4)}"
            if not is_duplicate_signal(plan, key):
                plan["signal_key"] = key
                plan["signal_type"] = "FULL_EXIT"
                transition_trade_state(plan, "FULL_EXIT_NEXT_DAY", "FULL_EXIT", data_date, "跌破目前防守線，強制出清優先")
            return plan

        # 已持倉時，只要偵測到實際股數下降，即視為上一筆部分出場已完成。
        if previous_qty > 0 and held < previous_qty:
            if not plan.get("t1_taken"):
                plan["t1_taken"] = True
            elif not plan.get("t2_taken"):
                plan["t2_taken"] = True

        if not plan.get("t1_taken") and _safe_float(plan.get("t1_price")) > 0 and price >= _safe_float(plan.get("t1_price")):
            key = f"{plan.get('code')}|T1|{data_date}|{round(_safe_float(plan.get('t1_price')),4)}"
            if not is_duplicate_signal(plan, key):
                plan["signal_key"] = key
                plan["signal_type"] = "PARTIAL_EXIT"
                transition_trade_state(plan, "PARTIAL_EXIT_NEXT_DAY", "T1_PARTIAL_EXIT", data_date, "到達 T1，隔日分批停利")
            return plan

        if plan.get("t1_taken") and not plan.get("t2_taken") and _safe_float(plan.get("t2_price")) > 0 and price >= _safe_float(plan.get("t2_price")):
            key = f"{plan.get('code')}|T2|{data_date}|{round(_safe_float(plan.get('t2_price')),4)}"
            if not is_duplicate_signal(plan, key):
                plan["signal_key"] = key
                plan["signal_type"] = "PARTIAL_EXIT"
                transition_trade_state(plan, "PARTIAL_EXIT_NEXT_DAY", "T2_PARTIAL_EXIT", data_date, "到達 T2，隔日第二段停利")
            return plan

        if regime_bearish:
            # 逆風不阻止既有持倉的停損/停利管理；只有待執行加碼被暫停。
            if plan.get("state") == "ADD_NEXT_DAY":
                plan["origin_state"] = "ADD_NEXT_DAY"
                plan["state"] = "SUSPENDED_BY_REGIME"
                plan["last_action"] = "SUSPEND_ADD"
                plan["last_action_date"] = data_date
                plan["signal_reason"] = "市場逆風，暫停加碼但保留交易計畫"
                return plan

        # 持倉且沒有更高優先級事件：只上移 trailing stop。
        new_stop = calculate_trailing_stop(current_stop, price, indicators.get("ma20"), indicators.get("atr"), cost)
        if new_stop > current_stop:
            plan["previous_trailing_stop"] = current_stop
            plan["current_trailing_stop"] = new_stop
        plan["remaining_shares"] = held
        if plan.get("state") not in {"FULL_EXIT_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "SUSPENDED_BY_REGIME"}:
            plan["state"] = "HOLD"
        plan["last_evaluated_at"] = data_date
        return plan

    # 空手：先處理既有有效計畫，不因每日 score 小幅波動而重建/覆蓋。
    if plan.get("state") in {"ENTER_NEXT_DAY", "BREAKOUT_WAIT", "SUSPENDED_BY_REGIME"}:
        if is_signal_expired(plan, data_date):
            return transition_trade_state(plan, "EXPIRED", "EXPIRE", data_date, "交易計畫超過有效期限")
        if regime_bearish:
            if plan.get("state") in {"ENTER_NEXT_DAY", "SUSPENDED_BY_REGIME"}:
                plan["origin_state"] = plan.get("state")
                plan["state"] = "SUSPENDED_BY_REGIME"
                plan["last_action"] = "SUSPEND_ENTRY"
                plan["last_action_date"] = data_date
                plan["signal_reason"] = "市場逆風，暫停新倉但保留原交易計畫"
            return plan
        if price > _safe_float(plan.get("chase_limit")) > 0:
            plan["state"] = "BREAKOUT_WAIT"
            plan["signal_type"] = "ENTRY"
            plan["last_evaluated_at"] = data_date
            plan["signal_reason"] = "現價超過追價上限，改等待回測"
            return plan
        if price >= _safe_float(plan.get("entry_price")) > 0:
            key = f"{plan.get('code')}|ENTRY|{data_date}|{round(_safe_float(plan.get('entry_price')),4)}"
            if not is_duplicate_signal(plan, key):
                plan["signal_key"] = key
                plan["signal_type"] = "ENTRY"
                transition_trade_state(plan, "ENTER_NEXT_DAY", "ENTRY", data_date, "突破/進場條件成立，隔日執行")
            return plan
        plan["last_evaluated_at"] = data_date
        return plan

    # 新訊號建立條件：Gate/Score 分離；此處由主迴圈先驗證 gate。
    if indicators.get("entry_gate") and not regime_bearish:
        entry_plan = calculate_entry_plan(plan.get("code", ""), indicators, portfolio_info, market_context)
        if entry_plan:
            plan.update(entry_plan)
            plan["state"] = "ENTER_NEXT_DAY"
            plan["signal_type"] = "ENTRY"
            plan["signal_key"] = f"{plan.get('code')}|ENTRY|{data_date}|{round(_safe_float(plan.get('entry_price')),4)}"
            plan["last_action"] = "CREATE_ENTRY"
            plan["last_action_date"] = data_date
            plan["last_evaluated_at"] = data_date
            plan["signal_reason"] = "Gate 與 Score 同時成立，建立隔日進場計畫"
    return plan


def _trade_state_to_ui(plan, ai_score, held_qty):
    state = plan.get("state", "PREPARE")
    if state == "ENTER_NEXT_DAY": return "🟢 進場"
    if state == "ADD_NEXT_DAY": return "📈 加碼"
    if state == "PARTIAL_EXIT_NEXT_DAY": return "🔵 停利退場"
    if state == "FULL_EXIT_NEXT_DAY": return "🔴 破損" if held_qty > 0 else "🔴 出清"
    if state == "SUSPENDED_BY_REGIME": return "⏸️ 市場逆風"
    if state == "EXPIRED": return "⚪ 訊號過期"
    if state == "INVALID": return "🔴 訊號失效"
    if held_qty > 0 and ai_score >= 70: return "🔥 利潤奔跑"
    if held_qty > 0 and ai_score >= 50: return "🟡 接近停利"
    if ai_score >= 70: return "🟢 進場"
    return "🟡 觀望"


def build_trade_plan_card_fields(plan):
    return {
        "trade_state": plan.get("state", "PREPARE"),
        "execution_date": plan.get("execution_date", ""),
        "valid_until": plan.get("valid_until", ""),
        "entry_price": _safe_float(plan.get("entry_price")),
        "t1_price": _safe_float(plan.get("t1_price")),
        "t2_price": _safe_float(plan.get("t2_price")),
        "current_trailing_stop": _safe_float(plan.get("current_trailing_stop")),
        "signal_reason": plan.get("signal_reason", ""),
        "signal_key": plan.get("signal_key", ""),
    }


def process_taiwan_close_update(portfolio, trade_plan, market_context, latest_tw_date, latest_us_date):
    # 台股新日K：重新計算台股個股指標、籌碼與交易狀態；美股只提供市場背景。
    return


def process_us_close_update(trade_plan, market_context, latest_us_date):
    # 美股新收盤：只更新市場限制，不重新建立台股個股訊號。
    return


def process_view_only(portfolio, trade_plan):
    # 無新資料：正式交易狀態、訊號日期、有效期限、停利與停損均不得被重新計算。
    return

def load_portfolio():
    try:
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS)
        records = ws.get_all_records()
        if not records:
            # 第一次使用、分頁是空的：把預設持股寫進去，讓 Google Sheet 成為資料的起點
            rows = [[code, info["name"], info["cost"], info["cap"], info["risk"], info["status"], "", info.get("qty", 0)] for code, info in DEFAULT_PORTFOLIO.items()]
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
                # 舊的 Google Sheet 分頁可能還沒有 qty 這欄，讀不到就當作 0（代表沒有在追蹤實際股數）
                "qty": float(row.get("qty") or 0.0),
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
            rows.append([code, info.get("name", ""), info.get("cost", 0.0), info.get("cap", 20000.0), info.get("risk", 5.0), info.get("status", "Active"), info.get("break_date", ""), info.get("qty", 0.0)])
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

migrate_trade_plan_sheet()
portfolio, system_history, trade_plan_data, today_str = load_portfolio(), load_history(), load_trade_plan(), datetime.datetime.now().strftime("%Y-%m-%d")

# --- 5. 側邊欄 UI ---
with st.sidebar:
    st.header("📋 持股與風控設定")
    with st.form("add_stock"):
        new_code = st.text_input("代號 (台股數字 / 美股字母)")
        new_name, new_cost, new_cap, new_risk = st.text_input("名稱 (可留白)"), st.number_input("成本價", value=100.0, step=0.1), st.number_input("分配資金", value=20000, step=5000), st.number_input("單筆風險 (%)", value=5.0, step=0.1)
        new_qty = st.number_input("持有股數 (選填，0＝純訊號監控，不計入總損益)", value=0, step=1, min_value=0)
        if st.form_submit_button("更新設定"):
            if new_code:
                fetch_stock_data.clear(); get_institutional_data.clear()
                existing_break_date = portfolio.get(new_code, {}).get('break_date') if isinstance(portfolio.get(new_code), dict) else None
                portfolio[new_code] = {"name": new_name, "cost": new_cost, "cap": new_cap, "risk": new_risk, "status": "Active", "qty": new_qty}
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

    st.divider()
    st.subheader("📤 匯出 / 📥 匯入持股清單 (CSV)")
    _export_rows = [
        {"code": code, "name": info.get("name", ""), "cost": info.get("cost", 0.0), "cap": info.get("cap", 20000.0),
         "risk": info.get("risk", 5.0), "qty": info.get("qty", 0.0), "status": info.get("status", "Active")}
        for code, info in portfolio.items()
    ]
    _csv_bytes = pd.DataFrame(_export_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button("📤 匯出目前持股清單", _csv_bytes, file_name="taistock_持股清單.csv", mime="text/csv")

    _uploaded_csv = st.file_uploader("📥 匯入持股清單 CSV", type=["csv"], help="欄位需包含 code, name, cost, cap, risk；qty、status 選填")
    if _uploaded_csv is not None:
        try:
            _df_import = pd.read_csv(_uploaded_csv)
            st.caption(f"讀到 {len(_df_import)} 筆資料，確認無誤後按下方按鈕匯入（會覆蓋畫面上同代號的既有設定）")
            if st.button("✅ 確認匯入"):
                for _, _row in _df_import.iterrows():
                    _imp_code = str(_row.get("code", "")).strip()
                    if not _imp_code or _imp_code == "nan": continue
                    portfolio[_imp_code] = {
                        "name": "" if pd.isna(_row.get("name", "")) else str(_row.get("name", "")),
                        "cost": float(_row.get("cost", 0) or 0),
                        "cap": float(_row.get("cap", 20000) or 20000),
                        "risk": float(_row.get("risk", 5.0) or 5.0),
                        "qty": float(_row.get("qty", 0) or 0),
                        "status": "Active" if pd.isna(_row.get("status", "Active")) else str(_row.get("status", "Active")),
                    }
                fetch_stock_data.clear(); get_institutional_data.clear()
                save_portfolio(portfolio)
                st.success(f"已匯入 {len(_df_import)} 筆設定")
                st.rerun()
        except Exception as e:
            st.error(f"⚠️ CSV 格式讀取失敗，請確認欄位名稱是否正確：{e}")

    st.divider()
    st.subheader("⏸️ 暫停分析（長期持有）/ ▶️ 恢復分析")
    st.caption("暫停後不會出現在每日分析清單、健康度統計、排行榜、SOP清單裡，但仍會計入資產總覽的損益（如果有填持有股數）。")
    _active_codes = [c for c, i in portfolio.items() if isinstance(i, dict) and i.get('status', 'Active') == 'Active']
    _paused_codes = [c for c, i in portfolio.items() if isinstance(i, dict) and i.get('status') == 'Paused']

    _pause_target = st.selectbox("選擇要暫停分析的股票", [""] + _active_codes, key="pause_select")
    if st.button("⏸️ 暫停分析") and _pause_target:
        portfolio[_pause_target]['status'] = 'Paused'
        save_portfolio(portfolio)
        st.rerun()

    _resume_target = st.selectbox("選擇要恢復分析的股票", [""] + _paused_codes, key="resume_select")
    if st.button("▶️ 恢復每日分析") and _resume_target:
        portfolio[_resume_target]['status'] = 'Active'
        save_portfolio(portfolio)
        st.rerun()

# --- 卡片渲染邏輯 ---
def render_stock_card(data, system_history, portfolio_data, trade_plan_data=None):
    with st.container(border=True):
        hist_records = system_history.get(data['code'], {})
        _tp = (trade_plan_data or {}).get(data['code'], {})
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

        # 【V2.10.3 新增】觀察名單標籤：持有股數=0 代表這是純訊號監控（還沒有實際持股），
        # 用一個藍色標籤直接標在標題上，不用另外開分頁，新手也能一眼分辨「這是我真的有買的」
        # 還是「這只是我在看的」。
        _qty_now = portfolio_data.get(data['code'], {}).get('qty', 0)
        watch_label = " <span style='color: #60a5fa;'>[👁️觀察中]</span>" if _qty_now <= 0 else ""

        st.markdown(f"#### {data['name']} ({data['code']}){broken_label}{watch_label} - {' '.join(data['tags'][:2])}{delta_str}", unsafe_allow_html=True)
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
            # 【V2.10.10 修正】原本只要判定是🟢進場就顯示「建倉建議股數」，沒檢查是否已持有，
            # 導致跟下方「加碼建議」文字互相矛盾（一邊說暫不建議加碼，一邊卻在部位欄顯示一個大數字）。
            # 現在改成：已持有時只顯示核准的加碼股數（沒核准就是「-」），空手時才顯示建倉建議股數。
            if data.get('held_qty', 0) > 0:
                _position_display = f"+{data['addon_shares_approved']}股（加碼）" if data.get('addon_shares_approved', 0) > 0 else "-"
            else:
                _position_display = f"{data['shares_adjusted']}股" if data['final_status'] == "🟢 進場" else "-"
            st.metric("部位", _position_display)
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
            # 【V2.9.5 修正】改用小方塊組成的迷你進度條（而非整條拉滿寬度的 st.progress），
            # 視覺上更接近「一排小方塊」的樣式，且寬度只跟着方塊數走、不會佔滿整個畫面寬度。
            _bar_rows = []
            for _label, _val, _max in [
                ("籌碼/長線", data['score_inst'], 40),
                ("趨勢技術", data['score_tech'], 30),
                ("量能指標", data['score_vol'], 15),
                ("風控狀態", data['score_risk'], 15),
            ]:
                _safe_val = 0 if (_val is None or (isinstance(_val, float) and _val != _val)) else _val
                _ratio = 0.0 if _max == 0 else max(0.0, min(1.0, _safe_val / _max))
                _segments = 10
                _filled = round(_ratio * _segments)
                _bar = "▰" * _filled + "▱" * (_segments - _filled)
                _bar_rows.append(
                    f"<div style='margin-bottom:10px;'>"
                    f"<span style='color:#cbd5e1; font-size:0.85em;'>{_label}</span><br>"
                    f"<span style='letter-spacing:2px; font-size:1.1em; color:#60a5fa;'>{_bar}</span> "
                    f"<span style='color:#94a3b8; font-size:0.85em;'>{_safe_val:.0f} / {_max}</span>"
                    f"</div>"
                )
            st.markdown("".join(_bar_rows), unsafe_allow_html=True)
            if data.get('score_forced_zero'):
                st.warning("⚠️ 已觸發停損防禦機制：現價已跌破防守線，系統強制將總分歸零（不採計上方拆解分數加總），優先保護本金。", icon="⚠️")
            if not data['is_us']:
                st.markdown(f"- **外資動向**: {data['inst']['foreign_trend']} | **投信動向**: {data['inst']['trust_trend']}")
        with tab_c2:
            c_t1, c_t2 = st.columns(2)
            c_t1.write(f"**今日量**: {data['volume']:,.0f} | **5日均量**: {data['vol_ma5']:,.0f}\n**K**: {data['k']:.1f} | **D**: {data['d']:.1f} | **RSI**: {data['rsi']:.1f}")
            c_t2.write(f"**MA20**: {data['ma20']:.2f} | **MA60**: {data['ma60']:.2f}\n**MACD(DIF)**: {data['macd']:.2f} | **季線乖離**: {data['bias']:.2f}%")
            # 【V2.10 新增①／V2.10.12 新增】自動畫K線圖：疊上 MA10/MA20/MA60 跟布林軌道。
            # MA10 是因為系統的「多頭排列」判斷本來就是看 MA10>MA20>MA60，把它畫出來才能親眼核對；
            # 布林軌道（MA20±2倍標準差）則是把「RSI超買超賣」文字提醒的概念視覺化，貼上軌=過熱、貼下軌=過冷，
            # 軌道寬窄變化也能看出最近是盤整還是變動劇烈。台股慣例紅漲綠跌，跟西方常見的紅跌綠漲相反，這裡有特別標明。
            st.markdown("**📉 K線走勢圖（近60日，紅漲綠跌）**")
            try:
                _chart_df = fetch_stock_data(data['code'])
                if _chart_df is not None and not _chart_df.empty and len(_chart_df) >= 20:
                    _cc, _hh, _ll, _oo = _chart_df['Close'].squeeze(), _chart_df['High'].squeeze(), _chart_df['Low'].squeeze(), _chart_df['Open'].squeeze()
                    if isinstance(_cc, pd.DataFrame): _cc, _hh, _ll, _oo = _cc.iloc[:, 0], _hh.iloc[:, 0], _ll.iloc[:, 0], _oo.iloc[:, 0]
                    _ma10_line = _cc.rolling(10).mean()
                    _ma20_line = _cc.rolling(20).mean()
                    _ma60_line = _cc.rolling(60).mean()
                    _boll_std = _cc.rolling(20).std()
                    _boll_upper = _ma20_line + 2 * _boll_std
                    _boll_lower = _ma20_line - 2 * _boll_std
                    _n = min(60, len(_chart_df))
                    _fig = go.Figure(data=[go.Candlestick(
                        x=_chart_df.index[-_n:], open=_oo.iloc[-_n:], high=_hh.iloc[-_n:], low=_ll.iloc[-_n:], close=_cc.iloc[-_n:],
                        increasing_line_color='#f87171', decreasing_line_color='#4ade80', name="K線",
                    )])
                    _fig.add_trace(go.Scatter(x=_chart_df.index[-_n:], y=_boll_upper.iloc[-_n:], line=dict(color='#94a3b8', width=1, dash='dot'), name="布林上軌"))
                    _fig.add_trace(go.Scatter(x=_chart_df.index[-_n:], y=_boll_lower.iloc[-_n:], line=dict(color='#94a3b8', width=1, dash='dot'), name="布林下軌",
                                               fill='tonexty', fillcolor='rgba(148, 163, 184, 0.08)'))
                    _fig.add_trace(go.Scatter(x=_chart_df.index[-_n:], y=_ma10_line.iloc[-_n:], line=dict(color='#c084fc', width=1), name="MA10"))
                    _fig.add_trace(go.Scatter(x=_chart_df.index[-_n:], y=_ma20_line.iloc[-_n:], line=dict(color='#facc15', width=1), name="MA20"))
                    _fig.add_trace(go.Scatter(x=_chart_df.index[-_n:], y=_ma60_line.iloc[-_n:], line=dict(color='#60a5fa', width=1), name="MA60"))
                    _fig.update_layout(
                        height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    )
                    st.plotly_chart(_fig, use_container_width=True, key=f"kchart_{data['code']}")
                else:
                    st.caption("資料不足，暫時無法畫出K線圖。")
            except Exception as _chart_err:
                st.caption(f"K線圖暫時無法載入：{_chart_err}")
        with tab_c3:
            st.write(f"**設定成本**: {data['cost']:.2f}\n**動態防守/停損**: {data['atr_stop_price']:.2f}\n**T1**: {data.get('t1_price', 0):.2f}\n**T2**: {data.get('t2_price', 0):.2f}")
            st.caption(f"交易狀態：{data.get('trade_state', 'PREPARE')} | 隔日執行：{data.get('execution_date', '-') or '-'} | 有效至：{data.get('valid_until', '-') or '-'}")
            if data.get("signal_reason"):
                st.caption(f"訊號原因：{data.get('signal_reason')}")
            # 【V2.10 新增④】波段剩餘空間%：現價距離波段目標價還有多少百分比的路要走，
            # 用「(目標價-現價) ÷ (目標價-成本)」換算成 0~100% 的剩餘空間，不用自己心算。
            _cost, _price, _target = data['cost'], data['price'], data['take_profit_price']
            if _cost > 0 and _target > _cost:
                if _price >= _target:
                    st.caption("🎯 波段剩餘空間：已達成或超越目標價")
                else:
                    _remaining_pct = max(0.0, min(100.0, (_target - _price) / (_target - _cost) * 100))
                    st.caption(f"🎯 波段剩餘空間：還有 {_remaining_pct:.1f}%（距離目標價 {_target - _price:.2f} 元）")

            # 【V2.10.8 新增】風報比 R值 = 報酬空間 ÷ 風險空間，專業交易常見的基本篩選門檻：
            # R < 1 代表賺賠空間比例不划算；1~1.5 普通；≥1.5 才算是有吸引力的賠率結構。
            _r = data.get('risk_reward_ratio')
            if _r is not None:
                _r_icon = "🟢" if _r >= 1.5 else ("🟡" if _r >= 1 else "🔴")
                _r_note = "（賠率結構不錯）" if _r >= 1.5 else ("（普通，可接受）" if _r >= 1 else "（偏低，賺賠不成比例）")
                st.caption(f"{_r_icon} 風報比 R = {_r:.2f} {_r_note}")
            else:
                st.caption("⚪ 風報比：目前無法計算（可能是尚未設定成本，或風險/報酬空間為零）")
        with tab_c4:
            if len(sorted_dates) > 1:
                chart_data = pd.DataFrame([{"Date": d, "Score": hist_records[d]['score']} for d in sorted_dates[:10]]).set_index("Date").sort_index()
                st.write("**📈 近期戰力動能曲線**")
                st.line_chart(chart_data['Score'], height=150)
            st.write("**📝 狀態軌跡**")
            for dt in sorted_dates[:5]: st.write(f"- {dt}: {hist_records[dt]['status']} ({hist_records[dt]['score']}分)")

# --- 6. 主程式執行 ---
st.title("⚡ TaiStock V2.11.x 交易狀態機決策系統")
st.warning("⚠️ 本系統僅為個人化技術指標整理與紀律提醒工具，所有分數、判定、建議均由你自訂的公式與參數計算而成，**不構成任何投資建議**，過去的訊號表現也不保證未來結果。所有操作決策與風險，仍需由你自己判斷並承擔。")

macro_data = fetch_macro_data()
st.markdown("### 🌍 雙軌市場環境總覽")
m_col1, m_col2, m_col3 = st.columns(3)

def _render_macro_asof(col, asof):
    # 【V2.10.8 新增】顯示資料實際對應的交易日期，並在資料超過3天沒更新時跳出警示，
    # 讓使用者自己能判斷「這數字是不是卡住了」，不用只能憑感覺猜。
    if asof is None:
        return
    _asof_ts = pd.Timestamp(asof)
    if _asof_ts.tzinfo is not None:
        _asof_ts = _asof_ts.tz_localize(None)
    _days_old = (pd.Timestamp(datetime.datetime.now()) - _asof_ts).days
    _date_str = _asof_ts.strftime("%Y-%m-%d")
    if _days_old > 3:
        col.caption(f"⚠️ 資料日期：{_date_str}（{_days_old}天前，可能不是最新資料，建議留意）")
    else:
        col.caption(f"資料日期：{_date_str}")

tw_trend = macro_data.get('TW', {})
if tw_trend:
    m_col1.metric("🇹🇼 台股加權 (大盤方向)", f"{tw_trend['price']:,.0f}", tw_trend['trend'], delta_color="normal" if "多頭" in tw_trend['trend'] else "inverse")
    _render_macro_asof(m_col1, tw_trend.get('asof'))
else: m_col1.metric("🇹🇼 台股加權", "連線中...")

us_trend = macro_data.get('US', {})
if us_trend:
    m_col2.metric("🇺🇸 那斯達克 (科技風向)", f"{us_trend['price']:,.0f}", us_trend['trend'], delta_color="normal" if "多頭" in us_trend['trend'] else "inverse")
    _render_macro_asof(m_col2, us_trend.get('asof'))
else: m_col2.metric("🇺🇸 那斯達克", "連線中...")

vix_trend = macro_data.get('VIX', {})
if vix_trend:
    v_val = vix_trend['price']
    v_status, v_color = ("🚨 極度恐慌", "inverse") if v_val >= 25 else (("⚠️ 波動加劇", "off") if v_val >= 20 else ("🟢 環境穩定", "normal"))
    m_col3.metric("📉 VIX 恐慌指數", f"{v_val:.2f}", v_status, delta_color=v_color)
    _render_macro_asof(m_col3, vix_trend.get('asof'))
else: m_col3.metric("📉 VIX 恐慌指數", "連線中...")
st.divider()

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號！")
else:
    summary_data, card_data, paused_data = [], [], []

    # 以宏觀資料日期判定事件模式。沒有新資料時只投影既有 Trade Plan。
    latest_tw_date = _date_str((macro_data.get("TW") or {}).get("asof"))
    latest_us_date = _date_str((macro_data.get("US") or {}).get("asof"))
    saved_tw_dates = [_date_str(v.get("taiwan_data_date")) for v in trade_plan_data.values() if v.get("taiwan_data_date")]
    saved_us_dates = [_date_str(v.get("us_data_date")) for v in trade_plan_data.values() if v.get("us_data_date")]
    saved_tw_date = max(saved_tw_dates) if saved_tw_dates else ""
    saved_us_date = max(saved_us_dates) if saved_us_dates else ""
    execution_mode = detect_update_mode(latest_tw_date, latest_us_date, saved_tw_date, saved_us_date)
    if not TRADE_PLAN_LOAD_OK:
        execution_mode = VIEW_ONLY

    if execution_mode == VIEW_ONLY:
        process_view_only(portfolio, trade_plan_data)
    elif execution_mode == US_CLOSE_UPDATE:
        process_us_close_update(trade_plan_data, macro_data, latest_us_date)
        for _code, _plan in trade_plan_data.items():
            _plan["us_data_date"] = latest_us_date

    st.caption(f"⚙️ 執行模式：**{execution_mode}** | 台股資料：{latest_tw_date or 'N/A'} | 美股資料：{latest_us_date or 'N/A'}")

    for code, info in list(portfolio.items()):
        if isinstance(info, dict):
            _status = info.get('status', 'Active')
            if _status == 'Closed':
                continue
            name = info.get('name', '')
            cost = _safe_float(info.get('cost'))
            cap = _safe_float(info.get('cap'), 20000.0)
            risk_pct = _safe_float(info.get('risk'), 5.0)
            if _status == 'Paused':
                _qty_paused = _safe_int(info.get('qty'))
                if _qty_paused > 0:
                    try:
                        _pdf = fetch_stock_data(code)
                        if _pdf is not None and not _pdf.empty:
                            _pc = _pdf['Close'].squeeze()
                            if isinstance(_pc, pd.DataFrame): _pc = _pc.iloc[:, 0]
                            _pprice = float(_pc.iloc[-1])
                            if not pd.isna(_pprice):
                                paused_data.append({'code': code, 'name': name, 'cost': cost, 'price': _pprice, 'qty': _qty_paused,
                                                     'is_us': code.isalpha() or code.endswith('.US')})
                    except Exception:
                        pass
                continue
        else:
            name, cost, cap, risk_pct = info if len(info) == 4 else (info[0], info[1], 20000.0, 5.0)

        risk_amount = cap * (risk_pct / 100.0)
        try:
            df = fetch_stock_data(code)
            if df is None or df.empty or len(df) < 60:
                continue
            c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df.get('Volume', pd.Series(0, index=df.index)).squeeze()
            if isinstance(c, pd.DataFrame): c, h, l, v = c.iloc[:, 0], h.iloc[:, 0], l.iloc[:, 0], v.iloc[:, 0]
            price = float(c.iloc[-1]); volume = float(v.iloc[-1]); vol_ma5 = float(v.rolling(5).mean().iloc[-1])
            data_date = _date_str(df.index[-1])
            pivot_point = (float(h.iloc[-2]) + float(l.iloc[-2]) + float(c.iloc[-2])) / 3 if len(h) >= 2 else price
            pivot_status = "🟢 站上" if price > pivot_point else "🔴 未站上"
            ma10, ma20, ma60 = float(c.rolling(10).mean().iloc[-1]), float(c.rolling(20).mean().iloc[-1]), float(c.rolling(60).mean().iloc[-1])
            macd = calc_macd(c)
            k_series, d_series = calc_kd(h, l, c); k, d = float(k_series.iloc[-1]), float(d_series.iloc[-1])
            delta = c.diff(); up = delta.clip(lower=0).rolling(14).mean().iloc[-1]; down = -1 * delta.clip(upper=0).rolling(14).mean().iloc[-1]
            rsi = float(100 - (100 / (1 + (np.nan_to_num(up) / (np.nan_to_num(down) + 0.001)))))
            true_ranges = []
            for i in range(max(1, len(c)-14), len(c)):
                true_ranges.append(max(float(h.iloc[i]-l.iloc[i]), abs(float(h.iloc[i]-c.iloc[i-1])), abs(float(l.iloc[i]-c.iloc[i-1]))))
            atr = float(np.mean(true_ranges)) if true_ranges else 0.0
            bias = float(((price - ma60) / ma60) * 100) if ma60 else 0.0
            boll_upper = float((c.rolling(20).mean() + 2 * c.rolling(20).std()).iloc[-1])
            core_named = {"現價": price, "成交量": volume, "5日均量": vol_ma5, "多空分水嶺": pivot_point, "MA10": ma10, "MA20": ma20, "MA60": ma60,
                          "MACD": macd, "K": k, "D": d, "RSI": rsi, "ATR": atr, "季線乖離": bias, "布林上軌": boll_upper}
            bad_fields = [k_name for k_name, val in core_named.items() if pd.isna(val)]
            if bad_fields:
                st.warning(f"⚠️ {name or code} 本次資料不完整（缺值：{'、'.join(bad_fields)}），已跳過；既有交易計畫不變。")
                continue

            is_us_stock = code.isalpha() or code.endswith('.US')
            inst = get_institutional_data(code)
            score_inst = ((20 if price > ma60 else 0) + (10 if macd > 0 else 0) + (10 if 0 < bias < 20 else 0)) if is_us_stock else min(inst['days'] * 5, 20) + (20 if inst['accumulated_shares'] * price >= 3000000000 else (10 if inst['accumulated_shares'] * price >= 1000000000 else 0))
            rsi_bull_point = 10 if (50 < rsi <= 80) else 0
            score_tech = (10 if k > d else 0) + rsi_bull_point + (10 if price > ma20 else 0)
            score_vol = min((volume / vol_ma5) * 10, 15) if vol_ma5 > 0 else 0
            # P0：風控分數只作評分，不再反向覆蓋持久化交易計畫。
            old_plan = _normalize_trade_plan_row(trade_plan_data.get(code, _trade_plan_defaults(code)))
            old_stop = _safe_float(old_plan.get('current_trailing_stop') or old_plan.get('initial_stop'))
            fallback_stop = max(0.0, cost - 2 * atr) if cost > 0 else 0.0
            if old_stop <= 0 and cost > 0:
                old_stop = fallback_stop
            score_risk = (10 if price > old_stop else 0) + (5 if cost > 0 and price >= cost * 1.05 else 0) if cost > 0 else 15
            score_forced_zero = bool(cost > 0 and old_stop > 0 and price <= old_stop)
            ai_score = 0 if score_forced_zero else min(int(score_inst + score_tech + score_vol + score_risk), 100)
            is_bull_aligned = ma10 > ma20 and ma20 > ma60
            confidence_base = ai_score * 0.8 + (10 if is_bull_aligned else 0) + (5 if price > pivot_point else 0)
            macro_warnings = []
            if is_us_stock:
                if macro_data.get('US') and '空頭' in macro_data['US'].get('trend', ''):
                    confidence_base *= 0.85; macro_warnings.append("⚠️ 美股大盤跌破月線，系統主動下調部位信心。")
                if macro_data.get('VIX') and macro_data['VIX'].get('price', 0) > 25:
                    confidence_base *= 0.70; macro_warnings.append("🚨 VIX 恐慌指數過高，系統強制抑制進場訊號！")
            else:
                if macro_data.get('TW') and '空頭' in macro_data['TW'].get('trend', ''):
                    confidence_base *= 0.85; macro_warnings.append("⚠️ 台股大盤跌破月線，逆勢操作風險較高。")
            confidence = min(99, max(10, int(confidence_base)))
            step1_pass = (price > ma60 and macd > 0) if is_us_stock else (inst['days'] >= 3 or inst['accumulated_shares'] * price >= 1000000000)
            step2_pass = k > d and rsi > 50 and volume > vol_ma5
            step3_pass = price > ma20 and is_bull_aligned
            entry_gate = bool(ai_score >= 70 and confidence >= 70 and step1_pass and step2_pass and step3_pass)
            held_qty = _safe_int(info.get('qty', 0)) if isinstance(info, dict) else 0
            indicators = {"price": price, "atr": atr, "pivot_point": pivot_point, "ma20": ma20, "data_date": data_date,
                          "is_us": is_us_stock, "entry_gate": entry_gate}

            # VIEW_ONLY 嚴格唯讀；只有新台股資料才更新個股交易狀態。美股更新只處理市場限制。
            should_update_stock = execution_mode == TAIWAN_CLOSE_UPDATE or (not old_plan.get('taiwan_data_date') and bool(data_date))
            if should_update_stock:
                new_plan = evaluate_trade_state(old_plan, indicators, macro_data, info if isinstance(info, dict) else {"cost": cost, "cap": cap, "risk": risk_pct, "qty": held_qty})
                if latest_us_date:
                    new_plan['us_data_date'] = latest_us_date
                new_plan['taiwan_data_date'] = data_date
                trade_plan_data[code] = _normalize_trade_plan_row(new_plan)
            else:
                new_plan = old_plan

            # 若沒有已建立計畫但符合 Gate，只有在 TAIWAN_CLOSE_UPDATE 才能建立新訊號。
            if execution_mode == TAIWAN_CLOSE_UPDATE and new_plan.get('state') == 'PREPARE' and held_qty == 0 and entry_gate and not _regime_is_bearish(macro_data, is_us_stock):
                created = calculate_entry_plan(code, indicators, info if isinstance(info, dict) else {"cost": cost, "cap": cap, "risk": risk_pct}, macro_data)
                if created:
                    new_plan.update(created)
                    new_plan['state'] = 'ENTER_NEXT_DAY'
                    new_plan['signal_type'] = 'ENTRY'
                    new_plan['signal_key'] = f"{code}|ENTRY|{data_date}|{round(_safe_float(created['entry_price']),4)}"
                    new_plan['last_action'] = 'CREATE_ENTRY'; new_plan['last_action_date'] = data_date; new_plan['last_evaluated_at'] = data_date
                    new_plan['signal_reason'] = 'Gate 與 Score 同時成立，建立隔日進場計畫'
                    trade_plan_data[code] = _normalize_trade_plan_row(new_plan)

            # P0：建立交易計畫後固定 Entry/T1/T2；Trailing Stop 只上移。
            plan_entry = _safe_float(new_plan.get('entry_price'))
            plan_stop = _safe_float(new_plan.get('current_trailing_stop') or new_plan.get('initial_stop'))
            if plan_entry <= 0 and cost > 0 and held_qty > 0:
                plan_entry = cost
                plan_stop = max(0.01, cost - 2 * atr)
                new_plan['entry_price'] = plan_entry; new_plan['initial_stop'] = plan_stop; new_plan['current_trailing_stop'] = plan_stop
                ex = calculate_exit_plan(plan_entry, plan_stop, 2.0)
                new_plan['t1_price'] = ex['t1_price']; new_plan['t2_price'] = ex['t2_price']
            position_info = calculate_position_size(cap, risk_pct, plan_entry if plan_entry > 0 else price, plan_stop if plan_stop > 0 else max(0.01, price-atr))
            suggested_shares_adjusted = position_info['shares_adjusted']
            if new_plan.get('suggested_shares', 0) <= 0 and plan_entry > 0:
                new_plan['suggested_shares'] = suggested_shares_adjusted
            addon_calc = calculate_addon_shares(new_plan, info if isinstance(info, dict) else {"cap": cap, "risk": risk_pct, "qty": held_qty}, indicators)
            addon_shares_approved = addon_calc['addon_shares'] if (execution_mode == TAIWAN_CLOSE_UPDATE and held_qty > 0 and entry_gate and not _regime_is_bearish(macro_data, is_us_stock) and price > cost + 0.5 * atr and new_plan.get('state') in {'HOLD', 'SUSPENDED_BY_REGIME'}) else 0
            if addon_shares_approved > 0 and new_plan.get('state') == 'HOLD':
                key = f"{code}|ADD|{data_date}|{addon_shares_approved}"
                if not is_duplicate_signal(new_plan, key):
                    new_plan['signal_key'] = key; new_plan['signal_type'] = 'ADD'; new_plan['addon_shares'] = addon_shares_approved
                    new_plan['state'] = 'ADD_NEXT_DAY'; new_plan['execution_date'] = _next_business_day(data_date)
                    new_plan['last_action'] = 'ADD'; new_plan['last_action_date'] = data_date; new_plan['signal_reason'] = '三燈、信心與風險額度成立，隔日加碼'
            elif new_plan.get('state') == 'SUSPENDED_BY_REGIME' and not _regime_is_bearish(macro_data, is_us_stock) and held_qty > 0:
                # 恢復時重新驗證風險，不直接沿用舊加碼股數。
                if entry_gate and price > cost + 0.5 * atr and addon_calc['addon_shares'] > 0:
                    new_plan['state'] = 'ADD_NEXT_DAY'; new_plan['addon_shares'] = addon_calc['addon_shares']; new_plan['execution_date'] = _next_business_day(data_date)
                    new_plan['signal_type'] = 'ADD'; new_plan['last_action'] = 'RESUME_ADD'; new_plan['last_action_date'] = data_date

            final_status = _trade_state_to_ui(new_plan, ai_score, held_qty)
            # 顯示建議與提醒
            ai_advice = []
            if new_plan.get('state') == 'FULL_EXIT_NEXT_DAY':
                ai_advice.append(f"🚨 強制出清：現價 {price:.2f} ≤ 防守線 {plan_stop:.2f}，出場優先於停利、加碼與續抱。")
            elif new_plan.get('state') == 'PARTIAL_EXIT_NEXT_DAY':
                ai_advice.append(f"🎯 分批停利：下一交易日執行，T1={_safe_float(new_plan.get('t1_price')):.2f} / T2={_safe_float(new_plan.get('t2_price')):.2f}")
            elif new_plan.get('state') == 'ENTER_NEXT_DAY':
                ai_advice.append(f"✓ 隔日進場計畫：建議執行區間 {_safe_float(new_plan.get('entry_price')):.2f}，追價上限 {_safe_float(new_plan.get('chase_limit')):.2f}")
            elif new_plan.get('state') == 'ADD_NEXT_DAY':
                ai_advice.append(f"📈 隔日加碼：核准 {addon_shares_approved or _safe_int(new_plan.get('addon_shares'))} 股；加碼後總風險不得超過設定上限。")
            elif new_plan.get('state') == 'SUSPENDED_BY_REGIME':
                ai_advice.append("⏸️ 市場逆風：暫停新倉/加碼，保留既有交易計畫；既有停損/停利仍有效。")
            elif held_qty > 0:
                ai_advice.append(f"✓ 持倉管理：Trailing Stop {plan_stop:.2f} 只會上移，不會下移。")
            else:
                ai_advice.append("✓ 保持觀察：正式交易計畫只在新完成日K資料到達時建立或更新。")
            ai_advice.extend([f"🎯 決策信心：{confidence}%"] + [f"<span style='color:#fbbf24;'>{w}</span>" for w in macro_warnings])
            boll_touch = price >= boll_upper
            if rsi > 80:
                ai_advice.append(f"<span style='color:#f87171;'>🚨 RSI 極度過熱：{rsi:.1f}；{\"同時觸及布林上軌。\" if boll_touch else \"請避免追高。\"}</span>")
            elif rsi > 70:
                ai_advice.append(f"<span style='color:#fbbf24;'>⚠️ RSI 短線過熱：{rsi:.1f}，留意分批停利。</span>")
            elif rsi < 30:
                ai_advice.append(f"<span style='color:#60a5fa;'>ℹ️ RSI 偏冷：{rsi:.1f}，不建議僅因超賣貿然殺低。</span>")
            if vol_ma5 > 0 and vol_ma5 < 200000:
                ai_advice.append(f"<span style='color:#fbbf24;'>⚠️ 流動性偏低：5日均量約 {vol_ma5:,.0f} 股。</span>")

            tags = ["🦅美股科技" if is_us_stock else ("🔥投信作帳" if inst.get('trust_days', 0) >= 3 else "🌊外資波段")]
            if is_bull_aligned and price > ma20: tags.append("🚀多頭起漲")
            elif price < ma60 and ma20 < ma60: tags.append("❄️弱勢空頭")
            if len(tags) == 1: tags.append("⏳區間震盪")

            if execution_mode != VIEW_ONLY:
                if code not in system_history: system_history[code] = {}
                system_history[code][data_date] = {"score": ai_score, "status": final_status, "price": price}
                while len(system_history[code]) > 10:
                    del system_history[code][sorted(system_history[code].keys())[0]]

            t1 = _safe_float(new_plan.get('t1_price'))
            t2 = _safe_float(new_plan.get('t2_price'))
            summary_data.append({"代號": code, "名稱": name, "現價": round(price,2), "成本": round(cost,2), "AI分數": ai_score,
                                 "交易狀態": new_plan.get('state','PREPARE'), "股性標籤": " | ".join(tags[:2]),
                                 "風控點": f"{plan_stop:.1f}/{t1:.1f}/{t2:.1f}" if cost > 0 or plan_entry > 0 else "-/-/-",
                                 "隔日執行": new_plan.get('execution_date','') or "-", "判定": final_status})
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume, "vol_ma5": vol_ma5,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi, "atr": atr, "bias": bias, "inst": inst, "tags": tags,
                "cap": cap, "risk_amount": risk_amount, "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "ai_score": ai_score, "final_status": final_status, "shares": position_info['risk_based_shares'], "shares_adjusted": suggested_shares_adjusted,
                "position_label": "100%（可分批布局）" if confidence >= 80 else ("60%（可小量試單）" if confidence >= 60 else ("20%（僅觀察）" if confidence >= 40 else "0%（不建議進場）")),
                "held_qty": held_qty, "addon_shares_approved": addon_shares_approved,
                "atr_stop_price": plan_stop, "take_profit_price": t1 or (price + 3*atr), "t1_price": t1, "t2_price": t2,
                "ai_advice": ai_advice, "confidence": confidence, "pivot_point": pivot_point, "pivot_status": pivot_status, "is_us": is_us_stock,
                "score_inst": score_inst, "score_tech": score_tech, "score_vol": score_vol, "score_risk": score_risk, "score_forced_zero": score_forced_zero,
                "risk_reward_ratio": ((t1-plan_entry)/(plan_entry-plan_stop)) if (t1 > plan_entry > plan_stop > 0) else None,
                **build_trade_plan_card_fields(new_plan)
            })
        except Exception as e:
            st.warning(f"⚠️ 分析 {code} 發生暫時性錯誤：{e}；本次跳過，不清空既有 Trade Plan。")

    # 只有非 VIEW_ONLY 才保存本次正式狀態；寫入失敗也不清除記憶體中的既有計畫。
    if execution_mode != VIEW_ONLY:
        save_history(system_history)
        save_trade_plan(trade_plan_data)

    # 【V2.10 新增②】AI 每日一句：從今天戰力最高的持股，自動拼一句話當作頭條，
    # 不用先看完整份排行榜跟卡片才知道「今天最值得注意的是哪一檔」。
    if card_data:
        _headline_top = max(card_data, key=lambda x: x['ai_score'])
        if _headline_top['ai_score'] > 0:
            _sub_scores = {"籌碼/長線動能": _headline_top['score_inst'], "趨勢技術": _headline_top['score_tech'], "量能表現": _headline_top['score_vol'], "風控狀態": _headline_top['score_risk']}
            _top_sub_label = max(_sub_scores, key=_sub_scores.get)
            _tag_str = "、".join(_headline_top['tags'][:2])
            st.info(f"🧠 **AI 每日一句**：今天最值得留意的是 **{_headline_top['name']}（{_headline_top['code']}）**，戰力 {_headline_top['ai_score']} 分，判定「{_headline_top['final_status']}」。優勢主要來自「{_top_sub_label}」，標籤：{_tag_str}。")
        else:
            st.info("🧠 **AI 每日一句**：今天所有持股都沒有出現戰力突出的標的，建議耐心觀望，不用勉強找機會。")

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

    # 【V2.10.2 修正】資產總覽依幣別（台幣／美金）分開計算，原本會把美股(美金)跟台股(台幣)
    # 直接加總，數字沒有意義；現在拆成兩組，各自算總投入成本、總市值、總損益。
    # 【V2.10.11 新增】暫停分析中的長期持有（paused_data）也會併入計算，避免歸檔/暫停後
    # 這筆部位的損益悄悄從總覽消失，讓數字不再反映真實資產狀況。
    if card_data or paused_data:
        st.markdown("### 💰 資產總覽（依持有股數計算）")
        _valued_cards = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) > 0] + paused_data
        if not _valued_cards:
            st.info("目前沒有任何持股填寫「持有股數」，所以無法計算實際總損益。到側邊欄的「持有股數」欄位填入你實際持有的股數（留 0 代表純訊號監控），這裡就會自動算出總投入成本、總市值與總損益。")
        else:
            def _render_asset_group(cards, currency_label, currency_symbol):
                if not cards:
                    return
                st.markdown(f"**{currency_label}**")
                _tc = sum(d['cost'] * portfolio[d['code']].get('qty', 0) for d in cards)
                _tm = sum(d['price'] * portfolio[d['code']].get('qty', 0) for d in cards)
                _tp = _tm - _tc
                _tp_pct = (_tp / _tc * 100) if _tc > 0 else 0.0
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric(f"總投入成本 ({currency_symbol})", f"{_tc:,.0f}")
                ac2.metric(f"目前總市值 ({currency_symbol})", f"{_tm:,.0f}")
                ac3.metric(f"總損益 ({currency_symbol})", f"{_tp:,.0f}", f"{_tp_pct:+.2f}%", delta_color="normal" if _tp >= 0 else "inverse")
                with st.expander(f"展開 {currency_label} 各檔損益明細"):
                    _rows = []
                    for d in cards:
                        _qty = portfolio[d['code']].get('qty', 0)
                        _pl = (d['price'] - d['cost']) * _qty
                        _pl_pct = ((d['price'] - d['cost']) / d['cost'] * 100) if d['cost'] > 0 else 0.0
                        _is_paused = portfolio.get(d['code'], {}).get('status') == 'Paused'
                        _display_name = ("⏸️ " if _is_paused else "") + d['name']
                        _rows.append({"代號": d['code'], "名稱": _display_name, "股數": _qty, "成本": round(d['cost'], 2),
                                      "現價": round(d['price'], 2), "損益": round(_pl, 0), "損益%": round(_pl_pct, 2)})
                    st.dataframe(pd.DataFrame(_rows).sort_values("損益", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)

            _render_asset_group([d for d in _valued_cards if not d['is_us']], "🇹🇼 台股資產（新台幣 TWD）", "TWD")
            _render_asset_group([d for d in _valued_cards if d['is_us']], "🇺🇸 美股資產（美金 USD）", "USD")
            if paused_data:
                st.caption("⏸️ 標記的股票是「暫停分析／長期持有」狀態，明細表裡看得到但不會出現在每日分析清單、健康度統計、排行榜裡。")
            st.caption("⚠️ 兩組數字幣別不同，不會加總在一起顯示；如果你想看合併後的台幣總資產，需要自己乘上當下匯率換算，系統目前沒有自動抓匯率。")
        st.divider()

    # 【V2.10.5 新增】新手風險檢查：把「單筆風險%」串起來看整體，並檢查標籤集中度。
    # 這兩個檢查依賴的是你自己在側邊欄設定的分配資金/風險%，以及系統標籤，
    # 分母是「所有 Active 持股規劃的分配資金加總」，不是你真正的總資產，算是概估值。
    if card_data:
        st.markdown("### 🛡️ 新手風險檢查")
        _total_cap_plan = sum(d['cap'] for d in card_data)
        _total_risk_plan = sum(d['risk_amount'] for d in card_data)
        if _total_cap_plan > 0:
            _risk_exposure_pct = _total_risk_plan / _total_cap_plan * 100
            if _risk_exposure_pct >= 20:
                st.error(f"🚨 整體風險曝露：{_risk_exposure_pct:.1f}%（把你所有持股設定的「分配資金 × 單筆風險%」加總，除以總分配資金）。這個比例偏高，代表如果所有持股同時觸發停損，虧損金額會佔你規劃資金相當大的比例，建議重新檢視各檔的單筆風險%設定。")
            elif _risk_exposure_pct >= 10:
                st.warning(f"⚠️ 整體風險曝露：{_risk_exposure_pct:.1f}%，中等偏高，建議留意不要再繼續加碼提高單筆風險%。")
            else:
                st.success(f"✅ 整體風險曝露：{_risk_exposure_pct:.1f}%，屬於相對保守的範圍。")
            st.caption("這個數字是用你側邊欄設定的「分配資金」與「單筆風險%」概算出來的整體風險預算比例，不是你真實總資產的風險占比，僅供參考。")

        _tag_counter = {}
        for d in card_data:
            for t in d['tags'][:1]:  # 只計第一個標籤（籌碼/動能屬性），第二個標籤是趨勢狀態，不適合拿來看集中度
                _tag_counter[t] = _tag_counter.get(t, 0) + 1
        if len(card_data) >= 3:
            _dominant_tag, _dominant_count = max(_tag_counter.items(), key=lambda x: x[1])
            _dominant_ratio = _dominant_count / len(card_data) * 100
            if _dominant_ratio >= 60:
                st.warning(f"⚠️ 標籤集中度偏高：你追蹤的 {len(card_data)} 檔股票裡，有 {_dominant_count} 檔（{_dominant_ratio:.0f}%）都屬於「{_dominant_tag}」這個屬性，這些股票的漲跌行為可能高度連動，不算真正分散。（此為依系統標籤概估，非正式產業分類）")
        st.divider()

    if summary_data:
        df_summary = pd.DataFrame(summary_data).sort_values(by="AI分數", ascending=False).reset_index(drop=True)
        st.markdown("### 🏆 戰力排行榜 (Top 3 潛力股)")
        top_cols = st.columns(3)
        for i, (idx, row) in enumerate(df_summary.head(3).iterrows()):
            emoji = ["🥇", "🥈", "🥉"][i]
            top_cols[i].metric(f"{emoji} {row['名稱']} ({row['代號']})", f"{row['現價']:.2f}", f"戰力: {row['AI分數']}分", delta_color="normal" if row['AI分數']>=70 else "off")
        st.divider()

    # 【V2.10.9 新增】AI 等待清單：找出目前判定為🟡觀望、但分數已經接近70分進場門檻的股票，
    # 只顯示「還差幾分」這種能從現有資料算出來的具體事實，不編造「預估幾天內達標」這類無法可靠預測的內容。
    if card_data:
        _waiting = sorted(
            [d for d in card_data if d['final_status'] == "🟡 觀望" and d['ai_score'] >= 50],
            key=lambda x: x['ai_score'], reverse=True
        )
        st.markdown("### ⏳ AI 等待清單（快接近進場門檻）")
        if _waiting:
            for d in _waiting[:5]:
                _gap = 70 - d['ai_score']
                st.write(f"**{d['name']} ({d['code']})** — 目前戰力 {d['ai_score']} 分，距離進場門檻（70分）還差 **{_gap} 分**")
            st.caption("這裡只列出目前判定為🟡觀望、分數已≥50的股票，依分數高到低排序，最多顯示5檔。純粹反映「現在」的分數差距，不代表之後一定會達標，也不預測需要幾天。")
        else:
            st.info("目前沒有任何股票落在「🟡觀望且分數≥50」的區間，等待清單暫時是空的。")
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
                elif data['final_status'] == "🟢 進場":
                    if data.get('held_qty', 0) > 0:
                        if data.get('addon_shares_approved', 0) > 0:
                            action_watch.append(f"📈 **可考慮加碼**：{data['name']} 已持有中，資金額度內約可加碼 {data['addon_shares_approved']} 股。")
                        # 已持有但未核准加碼時，這檔股票的「🟢進場」狀態對你來說不是新機會，不重複顯示在佈局清單
                    else:
                        action_buy.append(f"🎯 **進場佈局**：{data['name']} 戰力達 {data['ai_score']} 分，建議部位：{data['shares_adjusted']} 股（倉位比例 {data['position_label']}）。")
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

    # 【V2.10.3 新增】顯示範圍篩選：清單一多，可以只看實際持股或只看觀察名單，
    # 不用兩種混在一起逐一分辨標籤。
    _view_filter = st.radio("顯示範圍", ["全部", "只看實際持股 💰", "只看觀察名單 👁️"], horizontal=True, key="view_filter")
    if _view_filter == "只看實際持股 💰":
        card_data = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) > 0]
    elif _view_filter == "只看觀察名單 👁️":
        card_data = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) <= 0]

    tab_tw, tab_us = st.tabs(["🇹🇼 台股主力陣列 (籌碼監控)", "🇺🇸 美股科技巨頭 (動能監控)"])

    with tab_tw:
        tw_cards = [d for d in card_data if not d['is_us']]
        if not tw_cards: st.info("目前無符合篩選條件的台股。")
        for data in tw_cards: render_stock_card(data, system_history, portfolio, trade_plan_data)

    with tab_us:
        us_cards = [d for d in card_data if d['is_us']]
        if not us_cards: st.info("目前無符合篩選條件的美股。")
        for data in us_cards: render_stock_card(data, system_history, portfolio, trade_plan_data)

    st.divider()
    st.markdown("### 📈 訊號準確度回測（依累積歷史記錄統計）")
    _bt_stats = compute_signal_backtest(system_history)
    if not _bt_stats:
        st.info("目前累積的歷史記錄還太少（至少要有同一檔股票連續兩天以上的記錄才能比較），先讓系統多跑幾天，這裡的統計會隨時間慢慢累積。")
    else:
        _bt_rows = []
        for _status, _rets in _bt_stats.items():
            _win_rate = sum(1 for r in _rets if r > 0) / len(_rets) * 100
            _avg_ret = sum(_rets) / len(_rets)
            _bt_rows.append({"判定狀態": _status, "樣本數": len(_rets), "後續平均報酬%": round(_avg_ret, 2), "上漲勝率%": round(_win_rate, 1)})
        _df_bt = pd.DataFrame(_bt_rows).sort_values("後續平均報酬%", ascending=False).reset_index(drop=True)
        st.dataframe(_df_bt, use_container_width=True, hide_index=True)
        st.caption("「後續平均報酬」＝拿每筆歷史記錄當天的價格，對照同一檔股票目前歷史中最新一筆的價格計算漲跌幅，再依「當時的判定狀態」分組平均。樣本數會隨使用天數增加而增加；目前每檔股票最多保留最近10筆記錄，天數越久統計越有參考價值。")

if __name__ == "__main__":
    pass
