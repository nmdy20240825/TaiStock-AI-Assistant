import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime
import plotly.graph_objects as go

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

def calc_atr_series(h, l, c, period=14):
    """
    【V2.11.2 新增】完整 ATR 序列（不只是最後一個數值），供移動停利棘輪計算使用。
    公式：真實波幅 TR = max(當日高-當日低, |當日高-前日收|, |當日低-前日收|)，
    再取 period 天滾動平均。
    """
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_structural_target(high_series, current_price, atr, lookback=60, min_gap_atr_multiple=1.0):
    """
    【V2.11.2 新增】結構化目標價 T1/T2，取代原本「成本×固定倍數」的僵化公式。
    T1：抓過去 lookback 個交易日（不含今日）的最高價當壓力位；
        如果前高離現價太近（< min_gap_atr_multiple 倍ATR，代表沒有參考價值，
        或現價已經站上前高），改用「現價+2倍ATR」外推。
    T2：T1 再往上延伸 2 倍 ATR。
    回傳 (t1, t2, branch)，branch 是 "resistance"（用前高）或 "atr_fallback"（用外推）。
    """
    if atr is None or atr <= 0 or pd.isna(atr):
        return current_price, current_price, "atr_fallback"
    window = high_series.iloc[-(lookback + 1):-1] if len(high_series) > lookback else high_series.iloc[:-1]
    recent_high = float(window.max()) if len(window) > 0 else current_price
    min_gap = min_gap_atr_multiple * atr
    if recent_high > current_price + min_gap:
        t1 = recent_high
        t2 = recent_high + 2 * atr
        branch = "resistance"
    else:
        t1 = current_price + 2 * atr
        t2 = current_price + 4 * atr
        branch = "atr_fallback"
    return t1, t2, branch

def calc_trailing_stop(close_series, ma20_series, atr_series, cost, lookback=60, profit_trigger_pct=10.0):
    """
    【V2.11.2 新增】無狀態版「移動停利只能上移不能下移」。
    不依賴任何持久化的「昨天防守線」欄位（系統本來就沒存這個），而是把過去 lookback 天內
    「獲利率已經超過 profit_trigger_pct%」的每一天，都算一次候選防守線 max(MA20(t)-ATR(t), 成本)，
    取這些候選值的最大值，等同於重建一次「防守線只會往上走」的完整歷程。
    回傳 (stop_price, method)：method 是 "ratchet"（找到候選、用棘輪結果）或
    "fallback"（視窗內找不到任何獲利超過門檻的日子，呼叫端應改用原本的固定公式）。
    """
    if cost is None or cost <= 0:
        return cost, "fallback"
    n = min(lookback, len(close_series), len(ma20_series), len(atr_series))
    if n == 0:
        return cost, "fallback"
    closes, ma20s, atrs = close_series.iloc[-n:], ma20_series.iloc[-n:], atr_series.iloc[-n:]
    candidates = []
    for cc, ma20v, atrv in zip(closes, ma20s, atrs):
        if pd.isna(cc) or pd.isna(ma20v) or pd.isna(atrv):
            continue
        profit_pct = (cc - cost) / cost * 100
        if profit_pct > profit_trigger_pct:
            candidates.append(max(ma20v - atrv, cost))
    if not candidates:
        return None, "fallback"
    return max(candidates), "ratchet"

# --- 0-1. V2.11.x 交易計畫 / 事件驅動狀態機：共用型別轉換與日期工具 ---
def _safe_float(value, default=0.0):
    """任何輸入安全轉 float；None、空字串、NaN、無法轉換一律回傳 default。"""
    try:
        if value is None or value == "":
            return default
        f = float(value)
        if f != f:  # NaN 自身不等於自身
            return default
        return f
    except Exception:
        return default

def _safe_int(value, default=0):
    """任何輸入安全轉 int（先轉 float 再取整，容忍 Google Sheet 存成字串的數字）。"""
    try:
        if value is None or value == "":
            return default
        f = float(value)
        if f != f:
            return default
        return int(f)
    except Exception:
        return default

def _bool_value(value):
    """Google Sheet 存回來的布林值常常是字串 'True'/'FALSE'/'1'，統一轉成 Python bool。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

def _date_str(value):
    """任何日期輸入（Timestamp、字串、None）統一轉成 'YYYY-MM-DD' 字串，方便直接用字串比較大小。"""
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
    """回傳輸入日期之後的下一個交易日（僅排除週末，未接台股行事曆，不排除國定假日）。"""
    try:
        d = pd.Timestamp(date_str)
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    except Exception:
        return ""

def _add_business_days(date_str, days):
    """回傳輸入日期往後推 N 個交易日的日期（僅排除週末，未接台股行事曆，不排除國定假日）。
    用於計算訊號有效期限 valid_until（規格書 7.6：PREPARE/BREAKOUT_WAIT=3個交易日，PULLBACK_WAIT=5個交易日）。"""
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

def round_to_tick(price, is_us_stock=False):
    """
    【V2.11.x 新增】依台股實際最小跳動單位（tick size）取整，避免系統建議的突破價/追價上限
    是一個實務上不可能成交的價格（例如 87.503 元）。美股不受台股tick制度限制，只取到分。
    台股現行制度（依價格級距）：
      <10元：0.01　10~50元：0.05　50~100元：0.1　100~500元：0.5　500~1000元：1　>=1000元：5
    """
    p = _safe_float(price, 0.0)
    if p <= 0:
        return 0.0
    if is_us_stock:
        return round(p, 2)
    if p < 10:
        tick = 0.01
    elif p < 50:
        tick = 0.05
    elif p < 100:
        tick = 0.1
    elif p < 500:
        tick = 0.5
    elif p < 1000:
        tick = 1.0
    else:
        tick = 5.0
    return round(round(p / tick) * tick, 2)

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

# ===== V2.11.x 交易計畫 / 事件驅動狀態機（規格書 TaiStock_V2_11 反應速度與交易流程修正報告）=====
TAIWAN_CLOSE_UPDATE = "TAIWAN_CLOSE_UPDATE"
US_CLOSE_UPDATE = "US_CLOSE_UPDATE"
VIEW_ONLY = "VIEW_ONLY"

# 規格書 12.2 TRADE_PLAN_HEADERS 為基礎，依第二階段確認的設計追加：
#   pullback_taken、full_exit_shares、partial_exit_shares、addon_shares_suggested
# 並將原本語意含糊的 addon_shares 拆成 addon_shares_suggested / addon_shares_approved。
TRADE_PLAN_HEADERS = [
    # 識別與狀態
    "code", "signal_type", "state", "origin_state", "signal_reason", "signal_key",
    # 時間錨點
    "signal_date", "execution_date", "valid_until", "last_evaluated_at",
    "taiwan_data_date", "us_data_date", "last_action", "last_action_date",
    # 進場相關
    "entry_price", "breakout_price", "pullback_low", "pullback_high",
    "chase_limit", "invalid_price", "pullback_taken",
    # 停利相關
    "t1_price", "t2_price", "t1_taken", "t2_taken", "partial_exit_ratio", "partial_exit_shares",
    # 出清相關
    "initial_stop", "previous_trailing_stop", "current_trailing_stop", "full_exit_shares",
    # 股數與風險
    "suggested_shares", "addon_shares_suggested", "addon_shares_approved", "remaining_shares",
    "max_risk_amount", "used_risk_amount", "remaining_risk_amount", "last_known_qty",
    # 版本
    "plan_version",
]

# 規格書第六節定義的11種狀態；ChatGPT 草稿版少了 PULLBACK_WAIT，這裡補齊。
TRADE_STATES = {
    "PREPARE", "BREAKOUT_WAIT", "PULLBACK_WAIT", "ENTER_NEXT_DAY", "HOLD",
    "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY",
    "SUSPENDED_BY_REGIME", "INVALID", "EXPIRED",
}

# 規格書第六節「狀態轉移」表格的合法轉移清單。transition_state() 會用這張表擋掉不合法的跳轉。
ALLOWED_TRANSITIONS = {
    # 【首次導入 bootstrap】既有持股在 trade_plan 分頁第一次建立時預設是 PREPARE，
    # 但實際上使用者可能早已持有股數 > 0，evaluate_trade_state() 會直接依現況判給 HOLD／
    # ADD_NEXT_DAY／PARTIAL_EXIT_NEXT_DAY／FULL_EXIT_NEXT_DAY，因此這幾種轉移也要開放。
    "PREPARE": {"BREAKOUT_WAIT", "PULLBACK_WAIT", "ENTER_NEXT_DAY", "INVALID", "EXPIRED",
                "HOLD", "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "PREPARE"},
    "BREAKOUT_WAIT": {"ENTER_NEXT_DAY", "PULLBACK_WAIT", "INVALID", "EXPIRED", "BREAKOUT_WAIT"},
    "PULLBACK_WAIT": {"ENTER_NEXT_DAY", "INVALID", "EXPIRED", "PULLBACK_WAIT"},
    "ENTER_NEXT_DAY": {"HOLD", "SUSPENDED_BY_REGIME", "ENTER_NEXT_DAY"},
    # 【修正：股數變 0 的重置路徑】HOLD/ADD_NEXT_DAY/PARTIAL_EXIT_NEXT_DAY 都可能因為使用者在系統之外
    # 手動賣出全部持股，導致 held_qty 直接變 0，這種情況也要能重置回 PREPARE 重新追蹤訊號，
    # 否則狀態會卡死在「理論上已經沒有部位、卻永遠回不去空手訊號流程」的中間態。
    "HOLD": {"ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "PREPARE", "HOLD"},
    "ADD_NEXT_DAY": {"SUSPENDED_BY_REGIME", "HOLD", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "PREPARE", "ADD_NEXT_DAY"},
    # 逆風解除後恢復到暫停前的原狀態；出清判斷仍優先於恢復。
    "SUSPENDED_BY_REGIME": {"ENTER_NEXT_DAY", "ADD_NEXT_DAY", "HOLD", "FULL_EXIT_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "BREAKOUT_WAIT", "PULLBACK_WAIT", "PREPARE", "SUSPENDED_BY_REGIME"},
    "PARTIAL_EXIT_NEXT_DAY": {"HOLD", "FULL_EXIT_NEXT_DAY", "PREPARE", "PARTIAL_EXIT_NEXT_DAY"},
    "FULL_EXIT_NEXT_DAY": {"PREPARE", "FULL_EXIT_NEXT_DAY"},   # 出清後歸零，重新開始追蹤新訊號
    "INVALID": {"PREPARE", "INVALID"},
    "EXPIRED": {"PREPARE", "EXPIRED"},
}

# 若 load_trade_plan() 失敗，強制整個執行流程降級為 VIEW_ONLY，不允許本次任何寫入或狀態推進。
TRADE_PLAN_LOAD_OK = True

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

# --- 4-1. trade_plan 分頁讀寫（V2.11.x 新增）---
def _normalize_trade_plan_row(row):
    """
    把任何一筆 trade_plan 資料（不論來自 Google Sheet 讀回的舊資料、還是記憶體中剛算出的新計畫）
    統一補齊成含有 TRADE_PLAN_HEADERS 全部欄位、型別正確的 dict。缺欄位一律給預設值，
    不合法的 state 一律重置為 PREPARE，確保任何髒資料都不會讓系統崩潰（規格書向後相容原則）。
    """
    r = {h: row.get(h, "") for h in TRADE_PLAN_HEADERS}
    r["code"] = str(r.get("code", "")).strip()
    r["state"] = str(r.get("state", "") or "PREPARE")
    if r["state"] not in TRADE_STATES:
        r["state"] = "PREPARE"
    r["origin_state"] = str(r.get("origin_state", "") or "")
    r["signal_type"] = str(r.get("signal_type", "") or "")
    r["signal_reason"] = str(r.get("signal_reason", "") or "")
    r["signal_key"] = str(r.get("signal_key", "") or "")
    r["last_action"] = str(r.get("last_action", "") or "")
    r["plan_version"] = str(r.get("plan_version", "") or "2.11.x")
    for k in ["entry_price", "breakout_price", "pullback_low", "pullback_high", "chase_limit",
              "invalid_price", "t1_price", "t2_price", "initial_stop", "previous_trailing_stop",
              "current_trailing_stop", "max_risk_amount", "used_risk_amount", "remaining_risk_amount"]:
        r[k] = _safe_float(r.get(k), 0.0)
    r["partial_exit_ratio"] = _safe_float(r.get("partial_exit_ratio"), 0.30)
    for k in ["suggested_shares", "addon_shares_suggested", "addon_shares_approved",
              "remaining_shares", "partial_exit_shares", "full_exit_shares", "last_known_qty"]:
        r[k] = _safe_int(r.get(k), 0)
    r["t1_taken"] = _bool_value(r.get("t1_taken"))
    r["t2_taken"] = _bool_value(r.get("t2_taken"))
    r["pullback_taken"] = _bool_value(r.get("pullback_taken"))
    for k in ["signal_date", "execution_date", "valid_until", "last_evaluated_at",
              "taiwan_data_date", "us_data_date", "last_action_date"]:
        r[k] = _date_str(r.get(k))
    return r

def _trade_plan_defaults(code):
    """全新股票代號、trade_plan 分頁裡還沒有任何紀錄時使用的初始空白計畫。"""
    return _normalize_trade_plan_row({"code": code, "state": "PREPARE", "plan_version": "2.11.x"})

def load_trade_plan():
    """
    讀取 trade_plan 分頁。若分頁不存在，get_worksheet() 會自動建立空白分頁（向後相容既有機制，不需要改動）。
    若讀取過程發生任何例外（額度限制、網路逾時、認證失敗……），一律回傳空 dict，並把
    TRADE_PLAN_LOAD_OK 設為 False，讓主程式強制整個流程降級為 VIEW_ONLY——絕不能因為這裡失敗
    就假裝「大家都是空手、都沒有計畫」去跑正式決策，那樣反而會誤刪或誤判既有訊號。
    """
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
        st.warning(f"⚠️ 讀取 Google Sheet 交易計畫（trade_plan）失敗，本次強制改為 VIEW_ONLY 唯讀模式，不會修改任何既有交易計畫：{e}")
        return {}

def save_trade_plan(data):
    """
    整表覆寫 trade_plan 分頁。若 TRADE_PLAN_LOAD_OK 是 False（代表這次執行一開始讀取就失敗），
    直接拒絕寫入並回傳 False——避免拿一份「可能基於不完整讀取算出來」的資料去覆蓋 Google Sheet
    上原本可能還完好的資料。寫入本身若失敗，也只回傳 False、印出錯誤，不拋例外中斷整個頁面。
    """
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
        st.error(f"⚠️ 寫入 Google Sheet 交易計畫（trade_plan）失敗，本次狀態變更不視為已保存，既有已保存的計畫不受影響：{e}")
        return False

def migrate_trade_plan_sheet():
    """
    一次性欄位遷移：比對 Google Sheet 第一列表頭與程式內建的 TRADE_PLAN_HEADERS，
    缺的欄位只會「追加」到最後，不清除、不重排、不覆蓋既有資料列——未來若再擴充欄位，
    只要沿用這個函式並更新 TRADE_PLAN_HEADERS 常數即可，不需要手動改 Google Sheet。
    """
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        existing = ws.row_values(1)
        if not existing:
            ws.append_row(TRADE_PLAN_HEADERS)
            return
        missing = [h for h in TRADE_PLAN_HEADERS if h not in existing]
        if missing:
            start_col = len(existing) + 1
            for offset, h in enumerate(missing):
                ws.update_cell(1, start_col + offset, h)
    except Exception as e:
        st.warning(f"⚠️ trade_plan 欄位遷移暫時失敗，既有資料不會被清除，下次啟動會再嘗試：{e}")

# --- 4-2. 執行模式判斷（規格書 5.3）---
def detect_update_mode(latest_tw_date, latest_us_date, saved_tw_date, saved_us_date):
    """
    比對「這次抓到的大盤資料日期」跟「trade_plan 中已保存的資料日期」，決定本次執行模式。
    台股資料更新的優先權高於美股：只要台股有新資料，一律視為 TAIWAN_CLOSE_UPDATE
    （因為台股更新時本來就會一併確認美股/VIX狀態，不需要再獨立跑一次美股模式）。
    """
    tw_new = bool(latest_tw_date and (not saved_tw_date or latest_tw_date > saved_tw_date))
    us_new = bool(latest_us_date and (not saved_us_date or latest_us_date > saved_us_date))
    if tw_new:
        return TAIWAN_CLOSE_UPDATE
    if us_new:
        return US_CLOSE_UPDATE
    return VIEW_ONLY

def derive_market_regime(macro_data):
    """
    把 fetch_macro_data() 回傳的台股/美股/VIX資料，轉成規格書十一節定義的三段式市場燈號文字，
    純粹給畫面顯示與人工判讀用；實際攔截新倉/加碼的判斷是用 _regime_is_bearish()（依個股是台股或美股分開判定）。
    """
    tw = macro_data.get("TW") or {}
    us = macro_data.get("US") or {}
    vix = macro_data.get("VIX") or {}
    tw_bear = "空頭" in str(tw.get("trend", ""))
    us_bear = "空頭" in str(us.get("trend", ""))
    vix_high = _safe_float(vix.get("price"), 0) >= 25
    if us_bear or vix_high or tw_bear:
        return "🔴 逆風"
    return "🟢 順風"

def _regime_is_bearish(market_context, is_us_stock):
    """
    逆風判定：台股個股看加權指數是否跌破月線；美股個股看那斯達克跌破月線 或 VIX>=25。
    這個函式只回傳 True/False，供 evaluate_trade_state() 決定是否攔截新倉／加碼——
    規格書明確要求「逆風只限制新倉與加碼，不得刪除個股既有出場計畫」，所以出清/停利判斷完全不使用這個函式。
    """
    if is_us_stock:
        us = market_context.get("US") or {}
        vix = market_context.get("VIX") or {}
        return ("空頭" in str(us.get("trend", ""))) or _safe_float(vix.get("price"), 0) >= 25
    tw = market_context.get("TW") or {}
    return "空頭" in str(tw.get("trend", ""))

# --- 4-3. 重複訊號與有效期限判斷（規格書 7.6、十八節）---
def is_duplicate_signal(plan, signal_key):
    """同一個 signal_key 已經是目前計畫記錄的 signal_key，代表這個訊號今天已經處理過一次，不重複建立。"""
    return bool(signal_key) and plan.get("signal_key") == signal_key

def is_signal_expired(plan, data_date):
    """valid_until 存在、且目前資料日期已經超過它，代表這個等待中的訊號過期了。"""
    valid_until = plan.get("valid_until", "")
    if not valid_until or not data_date:
        return False
    return data_date > valid_until

def is_signal_invalid(plan, price):
    """現價跌破 invalid_price（規格書用語：訊號條件被破壞），代表原本等待的突破/回測邏輯已經不成立。"""
    invalid_price = _safe_float(plan.get("invalid_price"), 0.0)
    if invalid_price <= 0:
        return False
    return price < invalid_price

# --- 4-4. 狀態轉移（規格書第六節狀態轉移表）---
def transition_state(plan, next_state, extra_fields, data_date, reason=""):
    """
    唯一允許改變 plan['state'] 的地方。會先查 ALLOWED_TRANSITIONS 確認這是合法轉移，
    不合法就直接忽略、維持原狀態（寧可卡住讓使用者發現，也不要跳到不該去的狀態），
    合法的話才更新 state、origin_state、時間戳記與傳入的其餘欄位。
    """
    current = plan.get("state", "PREPARE")
    allowed = ALLOWED_TRANSITIONS.get(current, {current})
    if next_state not in allowed:
        plan["signal_reason"] = f"（忽略不合法的狀態轉移 {current}→{next_state}，原狀態維持）{plan.get('signal_reason','')}"
        return plan
    plan["origin_state"] = current
    plan["state"] = next_state
    plan.update(extra_fields)
    plan["last_action_date"] = data_date
    plan["last_evaluated_at"] = data_date
    if reason:
        plan["signal_reason"] = reason
    return plan

# --- 4-5. 核心計算函式（規格書第七、八、九、十、十五節）---
def calculate_position_size(cap, risk_pct, entry_price, stop_price, available_cash):
    """
    建議股數（規格書 15.1）：用「實際停損距離」而非單純 ATR，risk_amount/atr 的舊 bug 已在
    taistock_v2_9.py 修正過，這裡抽成獨立函式，公式與既有版本完全一致，數值不會改變。
    """
    risk_amount = _safe_float(cap) * _safe_float(risk_pct) / 100.0
    per_share_risk = max(_safe_float(entry_price) - _safe_float(stop_price), 0.0)
    if per_share_risk <= 0:
        return 0
    risk_based_shares = risk_amount / per_share_risk
    cash_limit = min(_safe_float(cap), _safe_float(available_cash)) if available_cash else _safe_float(cap)
    capital_based_shares = cash_limit / _safe_float(entry_price) if _safe_float(entry_price) > 0 else 0
    return int(np.floor(min(risk_based_shares, capital_based_shares)))

def calculate_trailing_stop_stateful(previous_stop, current_price, ma20, atr, cost):
    """
    【V2.11.x 核心修正】有狀態版移動防守線，取代舊版「每次重新掃描過去60天重建棘輪」的無狀態算法。
    直接讀取上一次已保存的 previous_stop 當基準，新防守線只會是三者取最大值，天然滿足「只能上移」：
      initial_stop（只在 previous_stop 還不存在時，即首次建倉時使用）＝ cost − 2×ATR
      candidate_stop ＝ MA20 − ATR
      new_stop = max(previous_stop 或 initial_stop, candidate_stop, cost)
    """
    cost_f = _safe_float(cost)
    if cost_f <= 0:
        return 0.0
    initial_stop = cost_f - 2 * _safe_float(atr)
    base = _safe_float(previous_stop) if _safe_float(previous_stop) > 0 else initial_stop
    candidate_stop = _safe_float(ma20) - _safe_float(atr)
    return max(base, candidate_stop, cost_f)

def calculate_exit_plan(price, average_cost, atr, ma20, previous_trailing_stop, previous_high,
                         t1_taken, t2_taken, current_shares, is_us_stock=False, partial_exit_ratio=0.30):
    """
    出清／分批停利計畫（規格書 9、10節）。回傳 dict 一定含 current_trailing_stop 與 t1_price/t2_price，
    並在符合條件時附上 next_state 建議（呼叫端 evaluate_trade_state 會再用 transition_state 實際套用，
    確保優先權判斷：全部出清 > T2 > T1 > 續抱，全部在這個函式內部就決定好，呼叫端不需要再重排順序）。
    """
    current_trailing_stop = calculate_trailing_stop_stateful(previous_trailing_stop, price, ma20, atr, average_cost)
    if _safe_float(previous_high) > 0 and _safe_float(atr) > 0:
        t1_price = round_to_tick(previous_high, is_us_stock)
        t2_price = round_to_tick(previous_high + 2 * atr, is_us_stock)
    elif _safe_float(atr) > 0:
        t1_price = round_to_tick(price + 2 * atr, is_us_stock)
        t2_price = round_to_tick(price + 4 * atr, is_us_stock)
    else:
        t1_price, t2_price = 0.0, 0.0

    result = {"current_trailing_stop": current_trailing_stop, "t1_price": t1_price, "t2_price": t2_price,
              "next_state": None, "signal_type": "", "signal_reason": "", "partial_exit_shares": 0, "full_exit_shares": 0}

    # 最高優先權：收盤跌破移動防守線，不論停利分數高低，一律強制全部出清（規格書10.3、10.4）
    if current_trailing_stop > 0 and price <= current_trailing_stop:
        result.update({"next_state": "FULL_EXIT_NEXT_DAY", "signal_type": "FULL_EXIT",
                        "signal_reason": "收盤跌破移動防守線，強制全部出清",
                        "full_exit_shares": current_shares})
        return result

    if not t1_taken and t1_price > 0 and price >= t1_price:
        result.update({"next_state": "PARTIAL_EXIT_NEXT_DAY", "signal_type": "T1_PARTIAL_EXIT",
                        "signal_reason": f"到達第一目標 T1（{t1_price:.2f}），隔日分批停利",
                        "partial_exit_shares": max(1, int(current_shares * _safe_float(partial_exit_ratio, 0.30)))})
        return result

    if t1_taken and not t2_taken and t2_price > 0 and price >= t2_price:
        result.update({"next_state": "PARTIAL_EXIT_NEXT_DAY", "signal_type": "T2_PARTIAL_EXIT",
                        "signal_reason": f"到達第二目標 T2（{t2_price:.2f}），隔日第二段停利",
                        "partial_exit_shares": current_shares})
        return result

    result.update({"next_state": "HOLD", "signal_reason": "持有續抱，移動防守線持續追蹤"})
    return result

def calculate_entry_plan(code, indicators, portfolio_info, market_context):
    """
    空手進場計畫（規格書 7節）。indicators 需含：price, atr, previous_high, ma20, decision_score,
    trend_gate, chip_gate, volume_gate, r1, market_regime, is_us_stock, data_date。
    entry_gate 沒通過或 decision_score < 70 時回傳 None（不建立計畫，維持 PREPARE）。
    """
    price = _safe_float(indicators.get("price"))
    atr = _safe_float(indicators.get("atr"))
    previous_high = _safe_float(indicators.get("previous_high"))
    decision_score = _safe_float(indicators.get("decision_score"))
    is_us_stock = bool(indicators.get("is_us_stock"))
    data_date = indicators.get("data_date", "")

    entry_gate_pass = bool(
        indicators.get("trend_gate") and indicators.get("chip_gate") and indicators.get("volume_gate")
        and (indicators.get("r1") is not None and indicators.get("r1") >= 1.5)
        and indicators.get("market_regime") != "BEARISH"
    )
    if not entry_gate_pass or decision_score < 70 or atr <= 0:
        return None

    breakout_price = round_to_tick(previous_high * 1.005, is_us_stock) if previous_high > 0 else round_to_tick(price, is_us_stock)
    chase_limit = round_to_tick(min(breakout_price + atr, breakout_price * 1.03), is_us_stock)
    pullback_low = round_to_tick(previous_high - 0.5 * atr, is_us_stock)
    pullback_high = round_to_tick(previous_high + 0.2 * atr, is_us_stock)
    # invalid_price（失效價）規格書未給明確公式，此處採用「前高－1倍ATR」作為初版基準，
    # 屬於【需要人工確認的參數】，請依實際回測結果調整。
    invalid_price = round_to_tick(max(previous_high - 1.0 * atr, 0), is_us_stock)

    if price > chase_limit > 0:
        state = "PULLBACK_WAIT"
        valid_until = _add_business_days(data_date, 5)
        reason = "現價已超過追價上限，改為等待回測區間"
    elif price >= breakout_price > 0:
        state = "ENTER_NEXT_DAY"
        valid_until = _add_business_days(data_date, 3)
        reason = "突破確認且未超過追價上限，隔日執行進場"
    else:
        state = "BREAKOUT_WAIT"
        valid_until = _add_business_days(data_date, 3)
        reason = "Gate 與 Score 同時成立，等待價格突破"

    return {
        "signal_type": "ENTRY", "entry_price": breakout_price, "breakout_price": breakout_price,
        "pullback_low": pullback_low, "pullback_high": pullback_high, "chase_limit": chase_limit,
        "invalid_price": invalid_price, "state": state, "signal_date": data_date,
        "execution_date": _next_business_day(data_date) if state == "ENTER_NEXT_DAY" else "",
        "valid_until": valid_until, "signal_reason": reason,
        "suggested_shares": calculate_position_size(
            portfolio_info.get("cap", 20000.0), portfolio_info.get("risk", 5.0),
            breakout_price, breakout_price - 2 * atr, portfolio_info.get("available_cash", portfolio_info.get("cap", 20000.0))
        ),
        "initial_stop": round_to_tick(breakout_price - 2 * atr, is_us_stock),
    }

def calculate_addon_shares(current_shares, current_price, current_stop, add_price, add_stop,
                            allocated_capital, risk_percent, available_cash, suggested_shares_cap=None):
    """
    加碼股數（規格書8節）。三重上限取最小值：加碼後總風險不超過 max_risk、資金餘額、以及可選的
    「原建倉股數上限」（既有 taistock_v2_9.py 已驗證過的保守設計，這裡保留但改為可選參數，
    未提供時不套用這個額外上限，行為與規格書8.2原始公式完全一致）。
    """
    max_risk = _safe_float(allocated_capital) * _safe_float(risk_percent) / 100.0
    remaining_risk = _safe_float(current_shares) * max(_safe_float(current_price) - _safe_float(current_stop), 0.0)
    available_add_risk = max_risk - remaining_risk
    add_per_share_risk = max(_safe_float(add_price) - _safe_float(add_stop), 0.0)
    if available_add_risk <= 0 or add_per_share_risk <= 0:
        return 0
    risk_based_add_shares = available_add_risk / add_per_share_risk
    capital_based_add_shares = _safe_float(available_cash) / _safe_float(add_price) if _safe_float(add_price) > 0 else 0
    candidates = [risk_based_add_shares, capital_based_add_shares]
    if suggested_shares_cap is not None:
        candidates.append(suggested_shares_cap)
    return int(np.floor(min(candidates))) if candidates else 0

# --- 4-6. 狀態機主體（規格書十四節主程式正確執行順序：持倉優先於新倉、出清優先於停利/加碼/續抱）---
def evaluate_trade_state(trade_plan, indicators, market_context, portfolio_info):
    """
    唯一會呼叫 transition_state() 推進正式狀態的地方。只在 TAIWAN_CLOSE_UPDATE（或首次建立計畫）
    時被呼叫；US_CLOSE_UPDATE/VIEW_ONLY 一律不呼叫這個函式（見 process_us_close_update / process_view_only）。
    """
    plan = _normalize_trade_plan_row(trade_plan)
    code = plan.get("code", indicators.get("code", ""))
    price = _safe_float(indicators.get("price"))
    data_date = indicators.get("data_date", "")
    held_qty = _safe_int(portfolio_info.get("qty"))
    cost = _safe_float(portfolio_info.get("cost"))
    is_us_stock = bool(indicators.get("is_us_stock"))
    regime_bearish = _regime_is_bearish(market_context, is_us_stock)

    # ===== 持倉分支：全部出清 > T1/T2分批停利 > 逆風暫停加碼 > 加碼 > 續抱 =====
    if held_qty > 0:
        previous_qty = _safe_int(plan.get("last_known_qty"), held_qty)
        plan["last_known_qty"] = held_qty
        if plan.get("entry_price", 0) <= 0:
            plan["entry_price"] = cost

        # 用「股數比上次少」推斷使用者已經照系統建議執行了上一筆分批出場，標記 t1_taken/t2_taken，
        # 避免系統對已經賣掉的部位重複發出同一批停利訊號。
        if previous_qty > 0 and held_qty < previous_qty:
            if not plan.get("t1_taken"):
                plan["t1_taken"] = True
            elif not plan.get("t2_taken"):
                plan["t2_taken"] = True

        exit_plan = calculate_exit_plan(
            price, cost, indicators.get("atr"), indicators.get("ma20"),
            plan.get("current_trailing_stop") or plan.get("initial_stop"),
            indicators.get("previous_high"), plan.get("t1_taken"), plan.get("t2_taken"),
            held_qty, is_us_stock, plan.get("partial_exit_ratio", 0.30),
        )
        plan["t1_price"] = exit_plan["t1_price"] if plan.get("t1_price", 0) <= 0 else plan["t1_price"]
        plan["t2_price"] = exit_plan["t2_price"] if plan.get("t2_price", 0) <= 0 else plan["t2_price"]

        if exit_plan["next_state"] == "FULL_EXIT_NEXT_DAY":
            key = f"{code}|FULL_EXIT|{data_date}|{round(exit_plan['current_trailing_stop'], 2)}"
            if is_duplicate_signal(plan, key):
                return plan
            plan["signal_key"] = key
            return transition_state(plan, "FULL_EXIT_NEXT_DAY",
                                     {"current_trailing_stop": exit_plan["current_trailing_stop"],
                                      "full_exit_shares": exit_plan["full_exit_shares"],
                                      "signal_type": exit_plan["signal_type"]},
                                     data_date, exit_plan["signal_reason"])

        if exit_plan["next_state"] == "PARTIAL_EXIT_NEXT_DAY":
            key = f"{code}|{exit_plan['signal_type']}|{data_date}"
            if is_duplicate_signal(plan, key):
                return plan
            plan["signal_key"] = key
            return transition_state(plan, "PARTIAL_EXIT_NEXT_DAY",
                                     {"current_trailing_stop": exit_plan["current_trailing_stop"],
                                      "partial_exit_shares": exit_plan["partial_exit_shares"],
                                      "signal_type": exit_plan["signal_type"]},
                                     data_date, exit_plan["signal_reason"])

        # 逆風時：只暫停「即將要執行的加碼」，已經是 HOLD 續抱的部位完全不受影響
        if regime_bearish and plan.get("state") == "ADD_NEXT_DAY":
            return transition_state(plan, "SUSPENDED_BY_REGIME",
                                     {"current_trailing_stop": exit_plan["current_trailing_stop"]},
                                     data_date, "市場逆風，暫停加碼但保留交易計畫")

        # 逆風解除：若上次是因為加碼被暫停，恢復回 ADD_NEXT_DAY 讓使用者重新看到加碼建議
        # （實際加碼股數會在下面重新計算，不會沿用暫停當下的舊數字）。
        if not regime_bearish and plan.get("state") == "SUSPENDED_BY_REGIME" and plan.get("origin_state") == "ADD_NEXT_DAY":
            plan = transition_state(plan, "HOLD", {"current_trailing_stop": exit_plan["current_trailing_stop"]},
                                     data_date, "市場逆風解除，重新評估加碼條件")

        if not regime_bearish:
            addon_shares = calculate_addon_shares(
                held_qty, price, exit_plan["current_trailing_stop"], price, exit_plan["current_trailing_stop"],
                portfolio_info.get("cap", 20000.0), portfolio_info.get("risk", 5.0),
                max(0.0, _safe_float(portfolio_info.get("cap", 20000.0)) - held_qty * price),
                suggested_shares_cap=int(plan.get("suggested_shares", 0) * 0.5) if plan.get("suggested_shares", 0) > 0 else None,
            )
            if addon_shares > 0:
                key = f"{code}|ADD|{data_date}|{addon_shares}"
                if not is_duplicate_signal(plan, key):
                    plan["signal_key"] = key
                    return transition_state(plan, "ADD_NEXT_DAY",
                                             {"current_trailing_stop": exit_plan["current_trailing_stop"],
                                              "addon_shares_suggested": addon_shares,
                                              "addon_shares_approved": addon_shares, "signal_type": "ADD"},
                                             data_date, f"SOP條件成立，資金/風險額度內約可加碼 {addon_shares} 股")

        return transition_state(plan, "HOLD",
                                 {"current_trailing_stop": exit_plan["current_trailing_stop"],
                                  "remaining_shares": held_qty, "addon_shares_approved": 0},
                                 data_date, exit_plan["signal_reason"])

    # ===== 空手分支：先處理既有計畫（過期/失效/推進），逆風時不建立新計畫，但不刪除既有計畫 =====
    plan["last_known_qty"] = 0
    active_wait_states = {"PREPARE", "BREAKOUT_WAIT", "PULLBACK_WAIT", "ENTER_NEXT_DAY", "SUSPENDED_BY_REGIME"}

    # 【重置防卡死機制】任何「非等待中」的舊狀態（HOLD/ADD_NEXT_DAY/PARTIAL_EXIT_NEXT_DAY/
    # FULL_EXIT_NEXT_DAY/INVALID/EXPIRED）在偵測到目前是空手時，代表這筆交易已經結束
    # （不論是照系統計畫出清、還是使用者在系統外手動賣出），一律重置回 PREPARE 並清空舊的
    # 進場/停利/防守欄位，才能重新開始追蹤全新訊號，避免卡在轉移表允許範圍外的死狀態。
    if plan.get("state") not in active_wait_states:
        plan = transition_state(
            plan, "PREPARE",
            {"entry_price": 0.0, "breakout_price": 0.0, "chase_limit": 0.0, "pullback_low": 0.0,
             "pullback_high": 0.0, "invalid_price": 0.0, "signal_key": "", "t1_taken": False, "t2_taken": False,
             "t1_price": 0.0, "t2_price": 0.0, "current_trailing_stop": 0.0, "initial_stop": 0.0,
             "addon_shares_approved": 0, "addon_shares_suggested": 0, "partial_exit_shares": 0, "full_exit_shares": 0,
             "execution_date": "", "valid_until": ""},
            data_date, f"部位已全部出清（原狀態 {plan.get('state')}），重置為 PREPARE 重新追蹤新訊號")

    if plan.get("state") in active_wait_states and plan.get("entry_price", 0) > 0:
        if plan.get("state") != "PREPARE" and is_signal_expired(plan, data_date):
            return transition_state(plan, "EXPIRED", {}, data_date, "交易計畫超過有效期限")
        if is_signal_invalid(plan, price):
            return transition_state(plan, "INVALID", {}, data_date, f"現價跌破失效價 {plan.get('invalid_price'):.2f}，訊號條件已被破壞")

        if regime_bearish:
            if plan.get("state") in {"ENTER_NEXT_DAY"}:
                return transition_state(plan, "SUSPENDED_BY_REGIME", {}, data_date, "市場逆風，暫停新倉但保留原交易計畫")
            return plan  # BREAKOUT_WAIT / PULLBACK_WAIT 本來就還沒到下單階段，逆風時單純不推進，不強制暫停

        if plan.get("state") == "SUSPENDED_BY_REGIME":
            origin = plan.get("origin_state") or "BREAKOUT_WAIT"
            return transition_state(plan, origin if origin in active_wait_states else "BREAKOUT_WAIT", {}, data_date, "市場逆風解除，恢復原交易計畫")

        chase_limit = _safe_float(plan.get("chase_limit"))
        entry_price = _safe_float(plan.get("entry_price"))
        if chase_limit > 0 and price > chase_limit and plan.get("state") != "PULLBACK_WAIT":
            return transition_state(plan, "PULLBACK_WAIT", {}, data_date, "現價超過追價上限，改為等待回測區間")

        if plan.get("state") in {"BREAKOUT_WAIT", "PULLBACK_WAIT"} and entry_price > 0 and price >= entry_price:
            key = f"{code}|ENTRY|{data_date}|{round(entry_price, 2)}"
            if is_duplicate_signal(plan, key):
                return plan
            plan["signal_key"] = key
            return transition_state(plan, "ENTER_NEXT_DAY",
                                     {"execution_date": _next_business_day(data_date)},
                                     data_date, "突破/回測進場條件成立，隔日執行")
        return plan

    # ===== 沒有既有計畫（PREPARE 且尚未有 entry_price）：檢查是否符合建立新計畫的條件 =====
    if regime_bearish:
        return plan  # 逆風時不建立新倉訊號，但完全不動既有（空的）計畫

    entry_result = calculate_entry_plan(code, indicators, portfolio_info, market_context)
    if entry_result:
        key = f"{code}|ENTRY_CREATE|{data_date}|{round(entry_result['entry_price'], 2)}"
        if is_duplicate_signal(plan, key):
            return plan
        plan["signal_key"] = key
        return transition_state(plan, entry_result["state"], entry_result, data_date, entry_result["signal_reason"])

    return plan

# --- 4-7. 三種執行模式的入口函式（規格書五節；主迴圈只呼叫這三個函式，不再內嵌分流判斷）---
def process_taiwan_close_update(old_plan, indicators, market_context, portfolio_info):
    """台股收盤更新：完整跑一次狀態機，個股訊號正式推進。"""
    return evaluate_trade_state(old_plan, indicators, market_context, portfolio_info)

def process_us_close_update(old_plan, market_context, us_data_date, is_us_stock):
    """
    美股收盤更新：白名單方式，只允許修改「市場允許度」相關欄位（state 在 ADD_NEXT_DAY/ENTER_NEXT_DAY
    與 SUSPENDED_BY_REGIME 之間切換），絕不觸碰 t1_price/t2_price/current_trailing_stop/entry_price
    等個股價格與技術欄位——這是規格書「美股資料不得覆蓋台股個股的價格、技術指標、Score或歷史訊號」的硬性邊界。
    """
    plan = dict(old_plan)
    regime_bearish = _regime_is_bearish(market_context, is_us_stock)
    if regime_bearish:
        if plan.get("state") == "ADD_NEXT_DAY":
            plan["origin_state"] = "ADD_NEXT_DAY"
            plan["state"] = "SUSPENDED_BY_REGIME"
            plan["last_action"] = "SUSPEND_ADD"
            plan["signal_reason"] = "市場逆風（美股收盤更新），暫停加碼但保留交易計畫"
        elif plan.get("state") == "ENTER_NEXT_DAY":
            plan["origin_state"] = "ENTER_NEXT_DAY"
            plan["state"] = "SUSPENDED_BY_REGIME"
            plan["last_action"] = "SUSPEND_ENTRY"
            plan["signal_reason"] = "市場逆風（美股收盤更新），暫停新倉但保留原交易計畫"
    else:
        if plan.get("state") == "SUSPENDED_BY_REGIME" and plan.get("origin_state") in ("ENTER_NEXT_DAY", "ADD_NEXT_DAY"):
            plan["state"] = plan["origin_state"]
            plan["last_action"] = "RESUME_FROM_REGIME"
            plan["signal_reason"] = "市場逆風解除（美股收盤更新），恢復原交易計畫"
    plan["us_data_date"] = us_data_date
    return plan

def process_view_only(old_plan):
    """
    無新資料：嚴格唯讀，原封不動回傳既有計畫的複本，不重建、不重設有效期限、
    不修改T1/T2、不修改防守線、不重複產生加碼/停利建議。
    """
    return dict(old_plan)

portfolio, system_history, trade_plan_data, today_str = load_portfolio(), load_history(), load_trade_plan(), datetime.datetime.now().strftime("%Y-%m-%d")
migrate_trade_plan_sheet()

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
        if del_code in trade_plan_data: del trade_plan_data[del_code]; save_trade_plan(trade_plan_data)
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
        tab_c1, tab_c2, tab_c3, tab_c4, tab_c5 = st.tabs(["⚙️ AI決策與SOP", "📉 技術數據", "🛡️ 風控點位", "📈 決策時間軸", "🗓️ 交易計畫"])

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
            _branch_note = "（依前高壓力位）" if data.get('target_branch') == "resistance" else "（前高不明顯，改用ATR外推）"
            st.write(f"**設定成本**: {data['cost']:.2f}\n**動態防守/停損**: {data['atr_stop_price']:.2f}\n**第一目標 T1**: {data['t1']:.2f} {_branch_note}\n**第二目標 T2**: {data['t2']:.2f}")
            # 【V2.11.2 正式導入】波段剩餘空間%：現價距離「第一目標T1」還有多少百分比的路要走，
            # 用「(T1-現價) ÷ (T1-成本)」換算成 0~100% 的剩餘空間，不用自己心算。
            _cost, _price, _target = data['cost'], data['price'], data['t1']
            if _cost > 0 and _target > _cost:
                if _price >= _target:
                    st.caption("🎯 波段剩餘空間（對T1）：已達成或超越第一目標")
                else:
                    _remaining_pct = max(0.0, min(100.0, (_target - _price) / (_target - _cost) * 100))
                    st.caption(f"🎯 波段剩餘空間（對T1）：還有 {_remaining_pct:.1f}%（距離T1 {_target - _price:.2f} 元）")

            # 【V2.11.2 正式導入】雙階段風報比：R1對第一目標T1（較近、較保守），R2對第二目標T2（較遠、較樂觀）。
            # R < 1 代表賺賠空間比例不划算；1~1.5 普通；≥1.5 才算是有吸引力的賠率結構。
            def _render_r(label, r_val, reached_label):
                if r_val is None:
                    st.caption(f"⚪ {label}：{reached_label}")
                else:
                    _icon = "🟢" if r_val >= 1.5 else ("🟡" if r_val >= 1 else "🔴")
                    _note = "（賠率結構不錯）" if r_val >= 1.5 else ("（普通，可接受）" if r_val >= 1 else "（偏低，賺賠不成比例）")
                    st.caption(f"{_icon} {label} = {r_val:.2f} {_note}")

            _r1, _r2 = data.get('r1'), data.get('r2')
            if data['cost'] <= 0:
                st.caption("⚪ 風報比：目前無法計算（尚未設定成本）")
            elif data['price'] <= data['atr_stop_price']:
                st.caption("⚪ 風報比：現價已跌破防守線，風控上應視為已觸發停損，R值不再具參考意義")
            else:
                _render_r("風報比 R1（對T1）", _r1, "已達成T1")
                _render_r("風報比 R2（對T2）", _r2, "已達成T2")
        with tab_c4:
            if len(sorted_dates) > 1:
                chart_data = pd.DataFrame([{"Date": d, "Score": hist_records[d]['score']} for d in sorted_dates[:10]]).set_index("Date").sort_index()
                st.write("**📈 近期戰力動能曲線**")
                st.line_chart(chart_data['Score'], height=150)
            st.write("**📝 狀態軌跡**")
            for dt in sorted_dates[:5]: st.write(f"- {dt}: {hist_records[dt]['status']} ({hist_records[dt]['score']}分)")

        with tab_c5:
            # 【V2.11.x 新增】交易計畫狀態機顯示：與上方「判定」(final_status) 是兩套獨立系統並行顯示，
            # final_status 是「當下即時分類」，這裡顯示的是「持久保存、事件驅動」的正式交易計畫。
            _plan_state = data.get('plan_state', 'PREPARE')
            _state_label_map = {
                "PREPARE": "⚪ 準備中（尚未符合條件）", "BREAKOUT_WAIT": "🟡 等待突破",
                "PULLBACK_WAIT": "🟡 等待回測", "ENTER_NEXT_DAY": "🟢 下一交易日可進場",
                "HOLD": "🔵 持有續抱", "ADD_NEXT_DAY": "🟢 下一交易日可加碼",
                "PARTIAL_EXIT_NEXT_DAY": "🟠 下一交易日分批出場", "FULL_EXIT_NEXT_DAY": "🔴 下一交易日全部出清",
                "SUSPENDED_BY_REGIME": "⏸️ 市場逆風，暫停新倉/加碼", "INVALID": "🔴 訊號失效", "EXPIRED": "⚪ 訊號已過期",
            }
            st.markdown(f"**交易計畫狀態**：{_state_label_map.get(_plan_state, _plan_state)}")
            if data.get('plan_signal_reason'):
                st.caption(f"📝 {data['plan_signal_reason']}")

            _pc1, _pc2 = st.columns(2)
            with _pc1:
                st.write(f"**訊號日期資料**\n台股資料日：{data.get('plan_taiwan_data_date') or '—'}\n美股資料日：{data.get('plan_us_data_date') or '—'}\n建議執行日：{data.get('plan_execution_date') or '—'}\n有效期限：{data.get('plan_valid_until') or '—'}")
            with _pc2:
                st.write(f"**目前執行模式**：{_mode_display.get(execution_mode, execution_mode)}\n**下一交易日**：{_next_business_day(data.get('plan_taiwan_data_date') or today_str) or '—'}")

            if data.get('held_qty', 0) <= 0:
                st.markdown("**空手訊號資訊**")
                st.write(f"突破價：{data.get('plan_breakout_price', 0):.2f}　｜　追價上限：{data.get('plan_chase_limit', 0):.2f}")
                st.write(f"回測區間：{data.get('plan_pullback_low', 0):.2f} ～ {data.get('plan_pullback_high', 0):.2f}")
                st.write(f"失效價：{data.get('plan_invalid_price', 0):.2f}　｜　建議進場股數：{data.get('plan_suggested_shares', 0)} 股")
                if _plan_state == "SUSPENDED_BY_REGIME":
                    st.warning("⏸️ 市場目前處於逆風狀態，新倉暫停，但交易計畫本身未被刪除，逆風解除後會自動恢復。")
            else:
                st.markdown("**持倉計畫資訊**")
                st.write(f"持有股數：{data.get('held_qty', 0)} 股　｜　平均成本：{data.get('cost', 0):.2f}")
                st.write(f"T1：{data.get('plan_t1_price', 0):.2f}（{'✅已執行' if data.get('plan_t1_taken') else '⬜未執行'}）　｜　T2：{data.get('plan_t2_price', 0):.2f}（{'✅已執行' if data.get('plan_t2_taken') else '⬜未執行'}）")
                st.write(f"初始防守線：{data.get('atr_stop_price', 0):.2f}　｜　今日移動防守線（計畫值）：{data.get('plan_current_trailing_stop', 0):.2f}")
                if _plan_state == "ADD_NEXT_DAY":
                    st.success(f"📈 建議加碼股數：{data.get('plan_addon_shares_approved', 0)} 股（下一交易日執行）")
                if _plan_state == "PARTIAL_EXIT_NEXT_DAY":
                    st.warning(f"🟠 建議分批停利股數：{data.get('plan_partial_exit_shares', 0)} 股（{data.get('plan_signal_type','')}，下一交易日執行）")
                if _plan_state == "FULL_EXIT_NEXT_DAY":
                    st.error(f"🔴 建議全部出清股數：{data.get('plan_full_exit_shares', 0)} 股（下一交易日執行）\n⚠️ 系統不保證一定能以防守觸發價成交，實際成交價可能因跳空而偏離，請留意跳空風險。")

# --- 6. 主程式執行 ---
st.title("⚡ TaiStock V2.9 全自動決策系統")
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

# --- 6-0. V2.11.x 執行模式判斷（規格書 5.2、5.3）---
# 用「trade_plan 中各股已保存的 taiwan_data_date / us_data_date 最大值」當作「上次正式更新到哪一天」的基準。
_saved_tw_dates = [p.get("taiwan_data_date", "") for p in trade_plan_data.values() if p.get("taiwan_data_date")]
_saved_us_dates = [p.get("us_data_date", "") for p in trade_plan_data.values() if p.get("us_data_date")]
saved_tw_date = max(_saved_tw_dates) if _saved_tw_dates else ""
saved_us_date = max(_saved_us_dates) if _saved_us_dates else ""
latest_tw_date = _date_str((macro_data.get("TW") or {}).get("asof"))
latest_us_date = _date_str((macro_data.get("US") or {}).get("asof"))

execution_mode = detect_update_mode(latest_tw_date, latest_us_date, saved_tw_date, saved_us_date)
if not TRADE_PLAN_LOAD_OK:
    execution_mode = VIEW_ONLY  # trade_plan 讀取失敗，強制唯讀，本次不允許任何狀態推進或寫入

market_regime_label = derive_market_regime(macro_data)
_mode_display = {"TAIWAN_CLOSE_UPDATE": "🇹🇼 台股收盤更新", "US_CLOSE_UPDATE": "🇺🇸 美股收盤更新", "VIEW_ONLY": "👁️ 唯讀檢視（無新資料）"}
st.caption(f"⚙️ 執行模式：**{_mode_display.get(execution_mode, execution_mode)}** ｜台股資料日期：{latest_tw_date or 'N/A'}｜美股資料日期：{latest_us_date or 'N/A'}｜市場燈號：{market_regime_label}｜上次已保存：台{saved_tw_date or 'N/A'} / 美{saved_us_date or 'N/A'}")
st.divider()

if not portfolio:
    st.info("👈 請先從左側邊欄新增股票代號！")
else:
    summary_data, card_data, paused_data = [], [], []

    for code, info in list(portfolio.items()):
        if isinstance(info, dict):
            _status = info.get('status', 'Active')
            if _status == 'Closed': continue
            name, cost, cap, risk_pct = info.get('name', ''), info.get('cost', 0.0), info.get('cap', 20000.0), info.get('risk', 5.0)
            if _status == 'Paused':
                # 【V2.10.11 新增】暫停分析（長期持有）：跳過完整的技術指標/AI分數計算，
                # 不出現在每日分析清單、健康度統計、排行榜、SOP清單裡，也不消耗額外的籌碼資料API額度，
                # 但如果有填持有股數，仍然抓一次現價，讓「資產總覽」的總損益能繼續反映這筆部位，不會悄悄消失。
                _qty_paused = info.get('qty', 0)
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

        risk_amount = cap * (risk_pct / 100)
        try:
            df = fetch_stock_data(code)
            if df is None or df.empty or len(df) < 60: continue

            c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df.get('Volume', pd.Series(0, index=df.index)).squeeze()
            if isinstance(c, pd.DataFrame): c, h, l, v = c.iloc[:, 0], h.iloc[:, 0], l.iloc[:, 0], v.iloc[:, 0]

            price, volume, vol_ma5 = float(c.iloc[-1]), float(v.iloc[-1]), float(v.rolling(5).mean().iloc[-1])
            # 【V2.11.2 新增】未完成K棒提醒（輕量版）：如果抓到的最後一筆資料日期是「今天」，
            # 代表這根K棒可能還在交易時段中持續變動（尤其影響量能、KD、RSI），收盤後數字才會定案。
            _last_bar_date = pd.Timestamp(df.index[-1])
            if _last_bar_date.tzinfo is not None:
                _last_bar_date = _last_bar_date.tz_localize(None)
            is_today_bar = _last_bar_date.date() == datetime.datetime.now().date()
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
            # 【V2.11.2 修正】原本 range(-13,0) 只加總13天卻除以14，跟新增的 calc_atr_series()（14期）
            # 對不齊，微幅低估ATR。改成 range(-14,0) 真正取14天，兩處ATR計算基準一致。
            atr = float(sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-14, 0)]) / 14)
            bias = float(((price - ma60) / ma60) * 100)
            # 【V2.11 修正②】布林上軌：用於跟 RSI 超買訊號交叉確認，不進計分公式，純粹是文字警示用。
            boll_upper = float((c.rolling(20).mean() + 2 * c.rolling(20).std()).iloc[-1])

            # 【V2.9.3／V2.10.1 修正】yfinance 偶爾會回傳不完整的資料（例如最後一根K棒缺值），
            # 導致 price/ma/k/d/rsi/atr 等任一數值變成 NaN。NaN 沒被擋下來的話會一路
            # 傳到 st.progress()（讓整個分頁當機）跟 Google Sheet 寫入（NaN 不是合法 JSON，
            # 寫入會直接失敗）。這裡先做一次「健檢」，任何一項是 NaN 就跳過這檔股票，
            # 並且把是哪個欄位出問題列出來，方便下次追查是資料源哪裡不完整。
            _core_named = {"現價": price, "成交量": volume, "5日均量": vol_ma5, "多空分水嶺": pivot_point,
                           "MA10": ma10, "MA20": ma20, "MA60": ma60, "MACD": macd, "K": k, "D": d,
                           "RSI": rsi, "ATR": atr, "季線乖離": bias, "布林上軌": boll_upper}
            _bad_fields = [k_name for k_name, v in _core_named.items() if pd.isna(v)]
            if _bad_fields:
                st.warning(f"⚠️ {name or code} 本次抓到的資料不完整（缺值欄位：{'、'.join(_bad_fields)}），已跳過這次分析，下次重新整理應會恢復正常。")
                continue

            inst = get_institutional_data(code)
            # 【V2.11.2 正式導入】把原本「獲利>10%用固定門檻切換」的停損/目標價，
            # 換成結構化版本：移動停利用「無狀態棘輪」重建只能上移的歷程；目標價改用
            # 前高（結構壓力位）或 ATR 外推，取代僵化的「成本×固定倍數」。
            # atr_stop_price／take_profit_price 這兩個變數名稱保留不變，
            # 讓後面所有既有的分數/顯示邏輯不用跟著大改；take_profit_price = T1（較近的第一目標）。
            _ma20_series = c.rolling(20).mean()
            _atr_series = calc_atr_series(h, l, c, period=14)

            if cost > 0 and price > cost * 1.10:
                _ratchet_stop, _stop_method = calc_trailing_stop(c, _ma20_series, _atr_series, cost)
                atr_stop_price = _ratchet_stop if _ratchet_stop is not None else max(cost, ma20)
                t1, t2, _target_branch = calc_structural_target(h, price, atr)
            elif cost > 0:
                atr_stop_price = cost - (atr * 2)
                t1, t2 = cost * 1.10, cost * 1.10  # 未達10%獲利時，維持原本「先看5%/10%門檻」的既有邏輯，不套用結構目標
                _target_branch = "atr_fallback"
            else:
                atr_stop_price = 0
                t1, t2 = 0, 0
                _target_branch = "atr_fallback"

            take_profit_price = t1

            # 【V2.10.8 新增／V2.11.2 修正】風報比改成雙階段 R1（對T1）/R2（對T2）。
            # 現價已跌破防守線時 R1=R2=None（風控上應視為已觸發停損，R值不再有意義）；
            # 已達成目標時也回傳 None，改由呼叫端顯示「已達成」文字，不顯示奇怪的負值。
            _risk_dist = price - atr_stop_price
            if cost > 0 and _risk_dist > 0:
                r1 = (t1 - price) / _risk_dist if price < t1 else None
                r2 = (t2 - price) / _risk_dist if price < t2 else None
            else:
                r1, r2 = None, None
            risk_reward_ratio = r1  # 保留舊變數名，供既有「🟢進場但風報比<1」警示邏輯使用（對照T1）

            is_us_stock = code.isalpha() or code.endswith('.US')
            score_inst = (20 if price > ma60 else 0) + (10 if macd > 0 else 0) + (10 if 0 < bias < 20 else 0) if is_us_stock else min(inst['days'] * 5, 20) + (20 if inst['accumulated_shares'] * price >= 3000000000 else (10 if inst['accumulated_shares'] * price >= 1000000000 else 0))
            # 【V2.10.7 修正】RSI>50 原本無條件+10分，但 RSI 極度過熱時（>80）已經是短線反轉風險區，
            # 不該再算作「健康的偏多確認」，所以把這個加分的上限收窄到 50~80 之間；
            # RSI 對趨勢分數的貢獻在 >80 時歸零，避免系統在超買區還持續給高分。
            _rsi_bull_point = 10 if (rsi > 50 and rsi <= 80) else 0
            score_tech = (10 if k > d else 0) + _rsi_bull_point + (10 if price > ma20 else 0)
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
                if risk_reward_ratio is not None and risk_reward_ratio < 1:
                    ai_advice.append(f"<span style='color: #f87171;'>⚠️ 風報比偏低（R1={risk_reward_ratio:.2f}，對第一目標T1）：報酬空間比風險空間還小，就算分數達標，賠率結構也不划算，建議謹慎評估。</span>")
            else:
                final_status = "🟡 觀望"
                ai_advice = ["✓ 建議：保持空手盯盤", "✓ 依據：動能不足", f"🎯 決策信心：{confidence}%"]

            for w in macro_warnings:
                ai_advice.append(f"<span style='color: #fbbf24;'>{w}</span>")

            # 【V2.11.2 新增】未完成K棒提醒：今天的資料如果還在交易時段中，量能/KD/RSI等數字
            # 可能隨盤中交易持續變動，收盤後才是最終定案的數值，避免把盤中的暫時訊號當成正式決策。
            if is_today_bar:
                ai_advice.append("<span style='color: #60a5fa;'>ℹ️ 今天的K棒可能還在交易時段中，量能/KD/RSI等數值會隨盤中交易變動，收盤後才是最終數字，僅供參考。</span>")

            # 【V2.10.7 新增／V2.11 修正②】RSI 超買超賣警示：用台股較適合的 70/30 門檻（而非美股常用的80/20），
            # 分「短線過熱/過冷」與「極度過熱/過冷」兩級，純粹是提醒性質，不影響上面已經算好的判定與分數。
            # V2.11 加入布林上軌交叉確認：RSI過熱同時股價又觸及/突破布林上軌，代表兩個獨立訊號一起示警，
            # 用更強烈的文字標示，但仍然只是提醒，不改動任何分數或判定。
            _boll_touch = price >= boll_upper
            if rsi > 80:
                if _boll_touch:
                    ai_advice.append("<span style='color: #f87171;'>🚨 雙重過熱確認：RSI已達{:.1f}（>80）同時股價已觸及/突破布林上軌，短線反轉風險更高，強烈不建議此時追高。</span>".format(rsi))
                else:
                    ai_advice.append("<span style='color: #fbbf24;'>⚠️ RSI已達極度過熱（{:.1f}，>80），短線反轉機率較高，不適合追高，若已持有可考慮分批獲利了結。</span>".format(rsi))
            elif rsi > 70:
                if _boll_touch:
                    ai_advice.append("<span style='color: #f87171;'>🚨 雙重過熱確認：RSI偏向過熱（{:.1f}，>70）同時股價已觸及/突破布林上軌，兩個訊號一起示警，若已持有建議留意分批獲利了結。</span>".format(rsi))
                else:
                    ai_advice.append("<span style='color: #fbbf24;'>⚠️ RSI偏向短線過熱（{:.1f}，>70），若已持有可留意分批獲利了結，避免此時追價。</span>".format(rsi))
            elif rsi < 20:
                ai_advice.append("<span style='color: #60a5fa;'>ℹ️ RSI已達極度過冷（{:.1f}，<20），短線反彈機率較高，但不建議貿然殺低出場。</span>".format(rsi))
            elif rsi < 30:
                ai_advice.append("<span style='color: #60a5fa;'>ℹ️ RSI偏向短線過冷（{:.1f}，<30），可開始留意是否有反彈買點，仍需搭配其他指標確認。</span>".format(rsi))

            # 【V2.10.5 新增】低流動性警示：5日均量過低代表買賣價差可能較大，
            # 新手照系統建議股數直接下市價單，容易買貴或賣便宜。門檻是粗略經驗值，
            # 不是嚴謹的流動性模型，僅供提醒留意，不同股本大小的股票基準本來就不同。
            _low_liquidity_threshold = 200000
            if vol_ma5 > 0 and vol_ma5 < _low_liquidity_threshold:
                ai_advice.append(f"<span style='color: #fbbf24;'>⚠️ 流動性偏低：5日均量僅約 {vol_ma5:,.0f} 股，買賣價差可能較大，建議用限價單，避免市價單成交價偏離太多。</span>")

            # 【V2.11.2 正式導入／重要bug修正】原本用「風險金額÷ATR」算股數，但實際停損距離是2倍ATR
            # （或利潤奔跑階段的棘輪防守距離），兩者對不齊，導致真正停損時賠的錢跟設定的風險金額兜不起來。
            # 改成用「現價−實際防守價」當每股風險，股數才會跟你真正會賠多少錢一致。
            _per_share_risk = max(price - atr_stop_price, 0)
            suggested_shares = min(int(risk_amount / _per_share_risk), int(cap / price)) if _per_share_risk > 0 else 0

            # 【V2.10 新增】AI 倉位建議：依「決策信心」分級，把原本單純用 ATR 算出來的建議股數，
            # 再乘上一個信心對應的倉位比例，讓建議部位更貼近「信心越低、部位越小」的實際下單邏輯。
            if confidence >= 80: position_pct, position_label = 1.0, "100%（可分批布局）"
            elif confidence >= 60: position_pct, position_label = 0.6, "60%（可小量試單）"
            elif confidence >= 40: position_pct, position_label = 0.2, "20%（僅觀察，避免重倉）"
            else: position_pct, position_label = 0.0, "0%（不建議進場）"
            suggested_shares_adjusted = int(suggested_shares * position_pct)
            _held_qty = info.get('qty', 0) if isinstance(info, dict) else 0

            if final_status == "🟢 進場" and _held_qty == 0:
                # 這個比例只在「真的是進場訊號」且「手上還沒有這檔股票」時才有意義：
                # 它是假設從零開始建倉的建議倉位。只要你已經持有（哪怕只有1股），
                # 就改用下面的「加碼建議」邏輯，避免兩種建議同時出現造成混淆。
                ai_advice.append(f"💰 建議倉位比例：{position_label}")

            # 【V2.10.6 新增／V2.11 修正④】加碼建議：專門給「手上已經有庫存」的人看，跟上面「建議倉位比例」
            # （假設從零建倉）是互斥的兩件事——已持有時只會顯示這一段。設計上刻意做得保守，核心原則是：
            # 絕不建議在虧損/警示狀態下加碼攤平（這是新手最常見的致命錯誤），
            # 只有在「本來就賺錢、而且籌碼/量能/趨勢三燈同時確認、決策信心也夠高」時才會給加碼空間，
            # 而且加碼股數會被你自己設定的「分配資金」上限鎖住，不會讓你越加越重倉。
            # V2.11 加入價格間距限制：現價要比成本價至少拉開0.5倍ATR，才核准加碼，避免在成本附近小區間
            # 盤整、趨勢還沒真正走出來的時候就被建議加碼。系統沒有交易日誌記錄「上次加碼價位」，
            # 這裡用「距離成本價」當替代基準，精神一致但不是逐筆追蹤每次加碼的間距。
            addon_shares_approved = 0
            if _held_qty > 0:
                if final_status in ["🔴 破損", "🔴 破線", "⚠️ 帳面虧損", "🔵 停利退場"]:
                    ai_advice.append("<span style='color: #f87171;'>❌ 不建議加碼：目前處於警示/停損停利狀態，加碼等於攤平虧損部位，違反紀律。</span>")
                elif final_status not in ["🟢 進場", "🔥 利潤奔跑"]:
                    ai_advice.append("⏸️ 暫不建議加碼：目前訊號不夠明確（觀望或接近停利階段），等待更清楚的多頭訊號再考慮。")
                elif not (step1_pass and step2_pass and step3_pass):
                    ai_advice.append("⏸️ 暫不建議加碼：SOP 三燈還沒有同時亮起（籌碼/量能/趨勢未同步確認）。")
                elif confidence < 80:
                    ai_advice.append(f"⏸️ 暫不建議加碼：決策信心僅 {confidence}%，還沒到高信心加碼的門檻（80%以上）。")
                elif atr > 0 and price < cost + 0.5 * atr:
                    ai_advice.append(f"⏸️ 暫不建議加碼：現價距離成本還沒拉開足夠空間（門檻約 {cost + 0.5 * atr:.2f}），可能還在整理區間，避免提早加碼。")
                else:
                    _current_value = _held_qty * price
                    _remaining_room = max(0.0, cap - _current_value)
                    # 【V2.11.2 正式導入】加碼後總風險上限檢查：現有持倉風險 + 加碼部位風險，
                    # 不能超過這檔股票原本設定的風險金額（分配資金×單筆風險%），避免加碼把整體風險越墊越高。
                    _current_position_risk = _held_qty * _per_share_risk
                    _remaining_risk_budget = max(risk_amount - _current_position_risk, 0.0)
                    _risk_based_addon_cap = int(_remaining_risk_budget / _per_share_risk) if _per_share_risk > 0 else 0
                    _addon_shares = int(min(_remaining_room / price, suggested_shares_adjusted * 0.5, _risk_based_addon_cap)) if price > 0 else 0
                    if _remaining_risk_budget <= 0:
                        ai_advice.append("⏸️ 不建議加碼：目前持倉的風險已達（或超過）這檔股票原始設定的風險預算上限，加碼會讓總風險超出你原本能接受的範圍。")
                    elif _remaining_room <= 0:
                        ai_advice.append("⏸️ 不建議加碼：目前持有市值已達到你設定的分配資金上限，加碼會超出原本的資金規劃。")
                    elif _addon_shares > 0:
                        addon_shares_approved = _addon_shares
                        ai_advice.append(f"📈 可考慮加碼：SOP三燈全亮、決策信心{confidence}%、現價已與成本拉開足夠空間，資金額度內約可加碼 {_addon_shares} 股（同時受分配資金上限、原始建倉股數一半、加碼後總風險上限三重限制，避免單押過重）。")
                    else:
                        ai_advice.append("⏸️ 資金或風險額度所剩不多，加碼股數不足1股，暫不建議加碼。")

            # ===== V2.11.x 交易計畫狀態機：與上方既有 final_status 邏輯並行運作，不修改既有變數 =====
            # 只把「持久化的交易計畫」疊加上去，既有的 ai_score/final_status/atr_stop_price/t1/t2/
            # suggested_shares_adjusted/addon_shares_approved 全部原封不動，UI 既有分頁行為不受影響。
            _plan_data_date = _date_str(_last_bar_date)
            _plan_previous_high_window = h.iloc[-61:-1] if len(h) > 60 else h.iloc[:-1]
            _plan_previous_high = float(_plan_previous_high_window.max()) if len(_plan_previous_high_window) > 0 else price

            _old_plan = _normalize_trade_plan_row(trade_plan_data.get(code, _trade_plan_defaults(code)))
            _plan_indicators = {
                "code": code, "price": price, "atr": atr, "ma20": ma20, "previous_high": _plan_previous_high,
                "decision_score": ai_score, "trend_gate": step3_pass, "chip_gate": step1_pass, "volume_gate": step2_pass,
                "r1": r1, "market_regime": "BEARISH" if _regime_is_bearish(macro_data, is_us_stock) else "NORMAL",
                "is_us_stock": is_us_stock, "data_date": _plan_data_date,
            }
            _plan_portfolio_info = {"cost": cost, "cap": cap, "risk": risk_pct, "qty": _held_qty,
                                     "available_cash": max(0.0, cap - _held_qty * price)}

            if execution_mode == TAIWAN_CLOSE_UPDATE or (not _old_plan.get("taiwan_data_date") and _plan_data_date):
                # 台股有新日K，或這檔股票從未被 evaluate 過（第一次遷移／新增持股時的一次性 bootstrap）
                _new_plan = process_taiwan_close_update(_old_plan, _plan_indicators, macro_data, _plan_portfolio_info)
                _new_plan["taiwan_data_date"] = _plan_data_date
                if latest_us_date:
                    _new_plan["us_data_date"] = latest_us_date
            elif execution_mode == US_CLOSE_UPDATE:
                _new_plan = process_us_close_update(_old_plan, macro_data, latest_us_date, is_us_stock)
            else:
                _new_plan = process_view_only(_old_plan)  # VIEW_ONLY：嚴格唯讀，原封不動

            trade_plan_data[code] = _normalize_trade_plan_row(dict(_new_plan, code=code))

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

            summary_data.append({"代號": code, "名稱": name, "現價": round(price, 2), "成本": round(cost, 2), "AI分數": ai_score, "股性標籤": " | ".join(tags[:2]), "風控點": f"{atr_stop_price:.1f}/{take_profit_price:.1f}" if cost > 0 else "-/-", "判定": final_status, "交易計畫狀態": trade_plan_data[code]["state"]})
            card_data.append({
                "code": code, "name": name, "cost": cost, "price": price, "volume": volume, "vol_ma5": vol_ma5,
                "ma10": ma10, "ma20": ma20, "ma60": ma60, "macd": macd, "k": k, "d": d, "rsi": rsi, "atr": atr, "bias": bias, "inst": inst, "tags": tags,
                "cap": cap, "risk_amount": risk_amount, "step1": step1_pass, "step2": step2_pass, "step3": step3_pass,
                "ai_score": ai_score, "final_status": final_status, "shares": suggested_shares, "shares_adjusted": suggested_shares_adjusted, "position_label": position_label,
                "held_qty": _held_qty, "addon_shares_approved": addon_shares_approved,
                "atr_stop_price": atr_stop_price, "take_profit_price": take_profit_price,
                "ai_advice": ai_advice, "confidence": confidence, "pivot_point": pivot_point, "pivot_status": pivot_status, "is_us": is_us_stock, "score_inst": score_inst, "score_tech": score_tech, "score_vol": score_vol, "score_risk": score_risk, "score_forced_zero": score_forced_zero, "risk_reward_ratio": risk_reward_ratio,
                "t1": t1, "t2": t2, "r1": r1, "r2": r2, "target_branch": _target_branch, "is_today_bar": is_today_bar,
                # ===== V2.11.x 交易計畫（trade_plan）欄位，統一用 plan_ 前綴，跟既有欄位分開，互不覆蓋 =====
                "plan_state": trade_plan_data[code]["state"], "plan_origin_state": trade_plan_data[code]["origin_state"],
                "plan_signal_type": trade_plan_data[code]["signal_type"], "plan_signal_reason": trade_plan_data[code]["signal_reason"],
                "plan_entry_price": trade_plan_data[code]["entry_price"], "plan_breakout_price": trade_plan_data[code]["breakout_price"],
                "plan_pullback_low": trade_plan_data[code]["pullback_low"], "plan_pullback_high": trade_plan_data[code]["pullback_high"],
                "plan_chase_limit": trade_plan_data[code]["chase_limit"], "plan_invalid_price": trade_plan_data[code]["invalid_price"],
                "plan_t1_price": trade_plan_data[code]["t1_price"], "plan_t2_price": trade_plan_data[code]["t2_price"],
                "plan_t1_taken": trade_plan_data[code]["t1_taken"], "plan_t2_taken": trade_plan_data[code]["t2_taken"],
                "plan_current_trailing_stop": trade_plan_data[code]["current_trailing_stop"],
                "plan_suggested_shares": trade_plan_data[code]["suggested_shares"],
                "plan_addon_shares_approved": trade_plan_data[code]["addon_shares_approved"],
                "plan_partial_exit_shares": trade_plan_data[code]["partial_exit_shares"],
                "plan_full_exit_shares": trade_plan_data[code]["full_exit_shares"],
                "plan_execution_date": trade_plan_data[code]["execution_date"], "plan_valid_until": trade_plan_data[code]["valid_until"],
                "plan_taiwan_data_date": trade_plan_data[code]["taiwan_data_date"], "plan_us_data_date": trade_plan_data[code]["us_data_date"],
            })
        except Exception as e: st.error(f"分析 {code} 發生錯誤: {e}")

    save_history(system_history)
    # 【V2.11.x 新增】trade_plan 只在真的有新資料時才寫回（VIEW_ONLY 模式嚴格唯讀，不做任何寫入，
    # 避免同一天重複開啟頁面時，把「唯讀複本」誤當作正式結果覆蓋回 Google Sheet）。
    if execution_mode != VIEW_ONLY:
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

        # 【V2.11.2 新增】組合最大損失：用實際持有股數 × (現價−防守線) 加總，換算成具體金額，
        # 補充上面「整體風險曝露%」的百分比視角，直接回答「如果現在全部觸及防守線，最多會賠多少錢」。
        # 僅計入 Active（分析中）且持有股數>0的持股；暫停分析（Paused）的股票沒有跑防守線計算，不計入。
        _held_cards = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) > 0]
        if _held_cards:
            def _max_loss_group(cards):
                return sum(portfolio[d['code']].get('qty', 0) * max(d['price'] - d['atr_stop_price'], 0) for d in cards)
            _max_loss_tw = _max_loss_group([d for d in _held_cards if not d['is_us']])
            _max_loss_us = _max_loss_group([d for d in _held_cards if d['is_us']])
            _loss_cols = st.columns(2)
            _loss_cols[0].metric("🇹🇼 組合最大損失 (TWD)", f"-{_max_loss_tw:,.0f}")
            _loss_cols[1].metric("🇺🇸 組合最大損失 (USD)", f"-{_max_loss_us:,.0f}")
            st.caption("把「持有股數 × (現價−目前防守線)」加總算出來的具體金額——如果所有持股同時觸及防守線出場，大約會賠多少錢。只計入分析中(Active)且有填股數的持股，暫停分析(Paused)的股票沒有跑防守線計算，不計入此金額。")

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

                # ===== V2.11.x 交易計畫狀態機驅動的任務（獨立於上面的 final_status 判斷，兩套並列顯示）=====
                _ps = data.get('plan_state', 'PREPARE')
                if _ps == "FULL_EXIT_NEXT_DAY":
                    action_sell.append(f"🔴 **【交易計畫】全部出清**：{data['name']} {data.get('plan_signal_reason','')}，建議出清 {data.get('plan_full_exit_shares',0)} 股（{data.get('plan_execution_date','下一交易日')} 執行）。")
                elif _ps == "PARTIAL_EXIT_NEXT_DAY":
                    action_sell.append(f"🟠 **【交易計畫】分批停利（{data.get('plan_signal_type','')}）**：{data['name']} 建議出脫 {data.get('plan_partial_exit_shares',0)} 股（{data.get('plan_execution_date','下一交易日')} 執行）。")
                elif _ps == "ADD_NEXT_DAY":
                    action_watch.append(f"📈 **【交易計畫】下一交易日可加碼**：{data['name']} 核准加碼 {data.get('plan_addon_shares_approved',0)} 股。")
                elif _ps == "ENTER_NEXT_DAY" and data.get('held_qty', 0) <= 0:
                    action_buy.append(f"🎯 **【交易計畫】下一交易日可進場**：{data['name']} 突破價 {data.get('plan_breakout_price',0):.2f}，建議股數 {data.get('plan_suggested_shares',0)} 股。")
                elif _ps == "SUSPENDED_BY_REGIME":
                    action_watch.append(f"⏸️ **【交易計畫】市場逆風暫停**：{data['name']} 原訂計畫已保留，等待逆風解除後自動恢復。")

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
        for data in tw_cards: render_stock_card(data, system_history, portfolio)

    with tab_us:
        us_cards = [d for d in card_data if d['is_us']]
        if not us_cards: st.info("目前無符合篩選條件的美股。")
        for data in us_cards: render_stock_card(data, system_history, portfolio)

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
