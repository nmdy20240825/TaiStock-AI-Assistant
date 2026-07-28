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

portfolio, system_history, today_str = load_portfolio(), load_history(), datetime.datetime.now().strftime("%Y-%m-%d")

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
            st.metric("部位", f"{data['shares_adjusted']}股" if data['final_status'] == "🟢 進場" else "-")
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
            # 【V2.10 新增①】自動畫K線圖：直接用已經抓過（有快取）的價量資料畫蠟燭圖疊 MA20/MA60，
            # 不用另外跳去看 TradingView。台股慣例紅漲綠跌，跟西方常見的紅跌綠漲相反，這裡有特別標明。
            st.markdown("**📉 K線走勢圖（近60日，紅漲綠跌）**")
            try:
                _chart_df = fetch_stock_data(data['code'])
                if _chart_df is not None and not _chart_df.empty and len(_chart_df) >= 20:
                    _cc, _hh, _ll, _oo = _chart_df['Close'].squeeze(), _chart_df['High'].squeeze(), _chart_df['Low'].squeeze(), _chart_df['Open'].squeeze()
                    if isinstance(_cc, pd.DataFrame): _cc, _hh, _ll, _oo = _cc.iloc[:, 0], _hh.iloc[:, 0], _ll.iloc[:, 0], _oo.iloc[:, 0]
                    _ma20_line = _cc.rolling(20).mean()
                    _ma60_line = _cc.rolling(60).mean()
                    _n = min(60, len(_chart_df))
                    _fig = go.Figure(data=[go.Candlestick(
                        x=_chart_df.index[-_n:], open=_oo.iloc[-_n:], high=_hh.iloc[-_n:], low=_ll.iloc[-_n:], close=_cc.iloc[-_n:],
                        increasing_line_color='#f87171', decreasing_line_color='#4ade80', name="K線",
                    )])
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
            st.write(f"**設定成本**: {data['cost']:.2f}\n**動態防守/停損**: {data['atr_stop_price']:.2f}\n**波段動能目標**: {data['take_profit_price']:.2f}")
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

            # 【V2.9.3／V2.10.1 修正】yfinance 偶爾會回傳不完整的資料（例如最後一根K棒缺值），
            # 導致 price/ma/k/d/rsi/atr 等任一數值變成 NaN。NaN 沒被擋下來的話會一路
            # 傳到 st.progress()（讓整個分頁當機）跟 Google Sheet 寫入（NaN 不是合法 JSON，
            # 寫入會直接失敗）。這裡先做一次「健檢」，任何一項是 NaN 就跳過這檔股票，
            # 並且把是哪個欄位出問題列出來，方便下次追查是資料源哪裡不完整。
            _core_named = {"現價": price, "成交量": volume, "5日均量": vol_ma5, "多空分水嶺": pivot_point,
                           "MA10": ma10, "MA20": ma20, "MA60": ma60, "MACD": macd, "K": k, "D": d,
                           "RSI": rsi, "ATR": atr, "季線乖離": bias}
            _bad_fields = [k_name for k_name, v in _core_named.items() if pd.isna(v)]
            if _bad_fields:
                st.warning(f"⚠️ {name or code} 本次抓到的資料不完整（缺值欄位：{'、'.join(_bad_fields)}），已跳過這次分析，下次重新整理應會恢復正常。")
                continue

            inst = get_institutional_data(code)
            atr_stop_price = max(cost, ma20) if (cost > 0 and price > cost * 1.10) else (cost - (atr * 2) if cost > 0 else 0)
            take_profit_price = cost * 2.0 if (cost > 0 and price > cost * 1.10) else (cost * 1.10 if cost > 0 else 0)

            # 【V2.10.8 新增】風報比 R值：報酬空間 ÷ 風險空間。這是專業交易最基本、但先前系統沒算的一環——
            # 就算 SOP 三燈全亮、AI分數再高，如果賺賠空間比例本身不划算（R<1），這筆交易的賠率結構就是差的。
            _risk_dist = price - atr_stop_price
            _reward_dist = take_profit_price - price
            risk_reward_ratio = (_reward_dist / _risk_dist) if (cost > 0 and _risk_dist > 0 and _reward_dist > 0) else None

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
                    ai_advice.append(f"<span style='color: #f87171;'>⚠️ 風報比偏低（R={risk_reward_ratio:.2f}）：報酬空間比風險空間還小，就算分數達標，賠率結構也不划算，建議謹慎評估。</span>")
            else:
                final_status = "🟡 觀望"
                ai_advice = ["✓ 建議：保持空手盯盤", "✓ 依據：動能不足", f"🎯 決策信心：{confidence}%"]

            for w in macro_warnings:
                ai_advice.append(f"<span style='color: #fbbf24;'>{w}</span>")

            # 【V2.10.7 新增】RSI 超買超賣警示：用台股較適合的 70/30 門檻（而非美股常用的80/20），
            # 分「短線過熱/過冷」與「極度過熱/過冷」兩級，純粹是提醒性質，不影響上面已經算好的判定與分數。
            if rsi > 80:
                ai_advice.append("<span style='color: #fbbf24;'>⚠️ RSI已達極度過熱（{:.1f}，>80），短線反轉機率較高，不適合追高，若已持有可考慮分批獲利了結。</span>".format(rsi))
            elif rsi > 70:
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

            suggested_shares = min(int(risk_amount / atr), int(cap / price)) if atr > 0 else 0

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

            # 【V2.10.6 新增】加碼建議：專門給「手上已經有庫存」的人看，跟上面「建議倉位比例」
            # （假設從零建倉）是互斥的兩件事——已持有時只會顯示這一段。設計上刻意做得保守，核心原則是：
            # 絕不建議在虧損/警示狀態下加碼攤平（這是新手最常見的致命錯誤），
            # 只有在「本來就賺錢、而且籌碼/量能/趨勢三燈同時確認、決策信心也夠高」時才會給加碼空間，
            # 而且加碼股數會被你自己設定的「分配資金」上限鎖住，不會讓你越加越重倉。
            if _held_qty > 0:
                if final_status in ["🔴 破損", "🔴 破線", "⚠️ 帳面虧損", "🔵 停利退場"]:
                    ai_advice.append("<span style='color: #f87171;'>❌ 不建議加碼：目前處於警示/停損停利狀態，加碼等於攤平虧損部位，違反紀律。</span>")
                elif final_status not in ["🟢 進場", "🔥 利潤奔跑"]:
                    ai_advice.append("⏸️ 暫不建議加碼：目前訊號不夠明確（觀望或接近停利階段），等待更清楚的多頭訊號再考慮。")
                elif not (step1_pass and step2_pass and step3_pass):
                    ai_advice.append("⏸️ 暫不建議加碼：SOP 三燈還沒有同時亮起（籌碼/量能/趨勢未同步確認）。")
                elif confidence < 80:
                    ai_advice.append(f"⏸️ 暫不建議加碼：決策信心僅 {confidence}%，還沒到高信心加碼的門檻（80%以上）。")
                else:
                    _current_value = _held_qty * price
                    _remaining_room = max(0.0, cap - _current_value)
                    _addon_shares = int(min(_remaining_room / price, suggested_shares_adjusted * 0.5)) if price > 0 else 0
                    if _remaining_room <= 0:
                        ai_advice.append("⏸️ 不建議加碼：目前持有市值已達到你設定的分配資金上限，加碼會超出原本的資金規劃。")
                    elif _addon_shares > 0:
                        ai_advice.append(f"📈 可考慮加碼：SOP三燈全亮、決策信心{confidence}%，資金額度內約可加碼 {_addon_shares} 股（不超過分配資金上限，且僅為原始建倉股數的一半，避免單押過重）。")
                    else:
                        ai_advice.append("⏸️ 資金額度所剩不多，加碼股數不足1股，暫不建議加碼。")

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
                "ai_score": ai_score, "final_status": final_status, "shares": suggested_shares, "shares_adjusted": suggested_shares_adjusted, "position_label": position_label,
                "atr_stop_price": atr_stop_price, "take_profit_price": take_profit_price,
                "ai_advice": ai_advice, "confidence": confidence, "pivot_point": pivot_point, "pivot_status": pivot_status, "is_us": is_us_stock, "score_inst": score_inst, "score_tech": score_tech, "score_vol": score_vol, "score_risk": score_risk, "score_forced_zero": score_forced_zero, "risk_reward_ratio": risk_reward_ratio
            })
        except Exception as e: st.error(f"分析 {code} 發生錯誤: {e}")

    save_history(system_history)

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
    if card_data:
        st.markdown("### 💰 資產總覽（依持有股數計算）")
        _valued_cards = [d for d in card_data if portfolio.get(d['code'], {}).get('qty', 0) > 0]
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
                        _rows.append({"代號": d['code'], "名稱": d['name'], "股數": _qty, "成本": round(d['cost'], 2),
                                      "現價": round(d['price'], 2), "損益": round(_pl, 0), "損益%": round(_pl_pct, 2)})
                    st.dataframe(pd.DataFrame(_rows).sort_values("損益", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)

            _render_asset_group([d for d in _valued_cards if not d['is_us']], "🇹🇼 台股資產（新台幣 TWD）", "TWD")
            _render_asset_group([d for d in _valued_cards if d['is_us']], "🇺🇸 美股資產（美金 USD）", "USD")
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
                elif data['final_status'] == "🟢 進場": action_buy.append(f"🎯 **進場佈局**：{data['name']} 戰力達 {data['ai_score']} 分，建議部位：{data['shares_adjusted']} 股（倉位比例 {data['position_label']}）。")
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
