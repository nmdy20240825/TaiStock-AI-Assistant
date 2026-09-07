import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import datetime
from zoneinfo import ZoneInfo
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Dict, Any, List, Tuple
import plotly.graph_objects as go

# 【新增】單一版本常數：畫面上所有顯示版本號的地方都從這裡讀取，避免像過去那樣
# 標題寫死成舊版本號、卻在程式碼各處的異動註解裡另外散落著不同的版本標記，
# 導致「畫面顯示的版本」「程式碼註解裡的版本」「操作說明書裡的版本」三邊互相矛盾。
# 之後每次做重大功能異動，記得同步更新這個常數（以及對應更新操作說明書的版本標示）。
APP_VERSION = "V2.11.45"
APP_TITLE = f"TaiStock {APP_VERSION} 波段紀律決策系統"

st.set_page_config(layout="wide", page_title=APP_TITLE)

# ===== UI 視覺與字體優化模組 =====
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 18px !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; white-space: normal !important; word-break: break-word !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
.ai-advice-box { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ===== V2.11.24新增：PWA安裝支援 =====
# 【重要】這幾行只負責讓瀏覽器/PWABuilder找得到 manifest.json 跟圖示，本身不會產生.apk檔案——
# 產生真正能安裝的 Android 套件，還是要透過 PWABuilder.com 這個外部工具完成，見操作說明書。
# 需要搭配 .streamlit/config.toml 裡的 enableStaticServing=true，以及 static/ 資料夾內的
# manifest.json + 圖示檔案（見說明書第XX節PWA安裝章節的完整檔案清單與放置位置）。
st.markdown("""
<link rel="manifest" href="app/static/manifest.json">
<link rel="apple-touch-icon" href="app/static/icon-192.png">
<meta name="theme-color" content="#0f172a">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="TaiStock">
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

    【V2.11.22】原本唯一的呼叫端是「進場專用R1預檢」（用 current_price=突破價=前高×1.005 呼叫
    這個函式），但那個呼叫方式有結構性瑕疵：current_price 本身就是從同一個 high_series 算出來的
    前高推導出來的，導致 recent_high 必然小於 current_price，這個函式必然回傳 atr_fallback，
    使得那個R1預檢永遠算出精確1.0，永遠無法通過entry_gate_pass的1.5門檻，等於「偵測新突破」功能
    完全失效。這道關卡已經拿掉（詳見 calculate_entry_plan 的 docstring），這個函式目前沒有任何
    呼叫端在使用，先保留函式定義本身（沒有壞，只是暫時沒人呼叫），未來若要在其他情境下（例如
    current_price 是「現價」而非「突破價」，兩者不是同一組資料）重新使用，這個函式本身可以直接
    沿用，不需要改動。
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

# --- 0-0-1. MACD 動能變化與背離分析模組（新增）---
# 依據「柱狀體主導、背離為轉折預警、快慢線交叉僅作次要確認」的架構，
# 提供日線／週線通用的 MACD 訊號分析器，輸出結構化 MACDSignalResult。

OSCStatus = Literal["正值", "負值", "收腳中", "翻紅第1根", "翻黑", "資料不足"]
DivergenceType = Literal["無", "頂背離", "底背離", "低檔雙背離", "資料不足"]
SignalAction = Literal["觀望", "預警關注", "分批試單", "核心進場", "減碼50%", "出場", "資料不足"]

def calc_macd_full_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    計算完整的 DIF／DEA／OSC 序列（而非只取最後一個數值），供背離偵測與「連續遞增/遞減」
    這類需要比對多根歷史值的判斷使用。公式與既有 calc_macd() 的 EMA 版一致，只是回傳整段序列。
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    osc = dif - dea
    return dif, dea, osc

def calculate_momentum_acceleration_score(rsi_series, osc_series, volume_series, lookback=3):
    """
    【V2.11.10新增，AI Score動能加速度，觀察用】±10分的子分數，不併入現有 ai_score 總分、
    不影響任何進場/加碼門檻（decision_score≥70、confidence≥80完全不會用到這個數字）。

    目的：現有 ai_score 只看「現在是不是處於多頭區間」（K>D、RSI 50~80、price>MA20 這種靜態門檻），
    分不出「正在轉強」跟「正在轉弱」——例如 RSI 55→62→69 跟 75→70→64 只要還在50以上，
    現有分數完全看不出差別。這個子分數改看最近幾天的「變化方向」（斜率），來補上這個缺口：
      RSI 斜率：最近lookback天的RSI變化量，每2點記1分，上限±4分
      MACD柱狀體(OSC)斜率：用近20天OSC標準差正規化後的變化量，上限±4分
      成交量斜率：5日均量最近lookback天的變化百分比，每20%記1分，上限±2分
    回傳 (分數, 說明文字)。任何一項資料不足時該項不計分，不會讓整體分數異常。
    """
    score = 0.0
    detail_parts = []
    try:
        if rsi_series is not None and len(rsi_series) > lookback:
            _r0, _r1 = float(rsi_series.iloc[-1]), float(rsi_series.iloc[-1 - lookback])
            if not (pd.isna(_r0) or pd.isna(_r1)):
                rsi_slope = _r0 - _r1
                pts = max(-4.0, min(4.0, rsi_slope / 2.0))
                score += pts
                detail_parts.append(f"RSI{lookback}日變化{rsi_slope:+.1f}（{pts:+.1f}分）")
    except Exception:
        pass
    try:
        if osc_series is not None and len(osc_series) > max(lookback, 20):
            _o0, _o1 = float(osc_series.iloc[-1]), float(osc_series.iloc[-1 - lookback])
            if not (pd.isna(_o0) or pd.isna(_o1)):
                osc_slope = _o0 - _o1
                _osc_std = float(osc_series.iloc[-20:].std())
                _osc_ref = _osc_std if _osc_std > 0 else (abs(_o1) or 1.0)
                norm_slope = osc_slope / _osc_ref
                pts = max(-4.0, min(4.0, norm_slope * 2.0))
                score += pts
                detail_parts.append(f"OSC{lookback}日變化{osc_slope:+.3f}（{pts:+.1f}分）")
    except Exception:
        pass
    try:
        if volume_series is not None and len(volume_series) > lookback + 5:
            _vma = volume_series.rolling(5).mean()
            _v0, _v1 = float(_vma.iloc[-1]), float(_vma.iloc[-1 - lookback])
            if not (pd.isna(_v0) or pd.isna(_v1)) and _v1 > 0:
                vol_change_pct = (_v0 - _v1) / _v1 * 100.0
                pts = max(-2.0, min(2.0, vol_change_pct / 20.0))
                score += pts
                detail_parts.append(f"量能5日均量變化{vol_change_pct:+.1f}%（{pts:+.1f}分）")
    except Exception:
        pass
    return round(max(-10.0, min(10.0, score)), 1), "；".join(detail_parts) if detail_parts else "資料不足，無法計算動能加速度"

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    把日線 OHLCV 轉成週線（週五收盤為一週結尾，符合台股與美股慣例）。
    資料不足或缺少必要欄位時回傳空 DataFrame，呼叫端需自行檢查長度。
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        needed = ["Open", "High", "Low", "Close"]
        if any(col not in df.columns for col in needed):
            return pd.DataFrame()
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        weekly = df.resample("W-FRI").agg(agg)
        weekly = weekly.dropna(subset=["Close"])
        return weekly
    except Exception:
        return pd.DataFrame()

@dataclass
class MACDSignalResult:
    """MACD 動能與背離分析的結構化輸出，支援 dict / DataFrame / JSON 三種取用方式。"""
    stock_id: str
    stock_name: str
    timeframe: str                          # "日線" 或 "週線"
    dif: Optional[float]
    dea: Optional[float]
    osc: Optional[float]
    osc_status: OSCStatus
    divergence_type: DivergenceType
    signal_action: SignalAction
    risk_management: Optional[float]        # 關鍵支撐停損價（跌破前低參考價）
    detail: str = ""                        # 人類可讀的判斷依據說明
    error: Optional[str] = None             # 資料不足或計算失敗時的錯誤訊息

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

class MACDStrategyAnalyzer:
    """
    MACD 動能變化與背離分析器。接收股票代號、名稱與 OHLCV DataFrame（日線或週線皆可，
    呼叫端自行決定要不要先用 resample_to_weekly() 轉換），回傳 MACDSignalResult。

    設計原則（對應角色定義的三層權重）：
      1. 柱狀體（OSC）狀態是主導訊號（收腳/翻紅/翻黑）。
      2. 背離型態是反轉與風險預警訊號，優先權高於柱狀體單純翻紅（頂背離時即使OSC還是正值，
         也要示警減碼；底背離時即使OSC還沒翻紅，也可以先分批試單）。
      3. DIF/DEA黃金交叉只做為趨勢確立的次要確認，不單獨驅動任何 signal_action。
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 divergence_lookback: int = 20, min_bars: int = 35, kd_period: int = 9,
                 min_new_extreme_pct: float = 0.5, min_osc_change_pct: float = 10.0,
                 min_bar_gap: int = 3) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.divergence_lookback = divergence_lookback
        self.min_bars = min_bars          # 至少需要這麼多根K棒才進行分析（EMA26+訊號9需要暖身期）
        self.kd_period = kd_period
        # 【V2.11.10新增，P2背離降噪】原本只要「現價比前高多1分錢」「OSC隨便遞減一點點」就會判定背離，
        # 對強勢噴出股（例如創意3443）很容易誤判——因為那種股票本來就是OSC自然收斂但價格仍持續創高。
        # 新增兩道最小幅度濾網：創高/破低要有意義的幅度，OSC的變化也要有意義的幅度，才算真背離。
        self.min_new_extreme_pct = min_new_extreme_pct  # 創新高/新低，至少要比前波高/低點多這個百分比（預設0.5%）
        self.min_osc_change_pct = min_osc_change_pct     # OSC變化幅度，至少要是參考OSC值的這個百分比（預設10%）
        # 【V2.11.14新增】最小bar間距：拿來當比較基準的前波高/低點，至少要距離今天這根K棒這麼多根，
        # 避免抓到「昨天、前天剛形成、根本還在雜訊範圍內」的極短期高低點當作背離比較基準。
        self.min_bar_gap = min_bar_gap

    def _empty_result(self, stock_id: str, stock_name: str, timeframe: str, error: str) -> MACDSignalResult:
        return MACDSignalResult(
            stock_id=stock_id, stock_name=stock_name, timeframe=timeframe,
            dif=None, dea=None, osc=None, osc_status="資料不足", divergence_type="資料不足",
            signal_action="資料不足", risk_management=None, detail=error, error=error,
        )

    def _detect_osc_status(self, osc: pd.Series) -> Tuple[OSCStatus, str]:
        """
        依據角色定義：
          翻紅第1根：OSC_{t-1}<0 且 OSC_t>0
          翻黑     ：OSC_{t-1}>0 且 OSC_t<0
          收腳中   ：OSC_t<0 且 |OSC_t| < |OSC_{t-1}|（負值柱狀體縮短）
          正值/負值：其餘一般狀態
        """
        cur, prev = float(osc.iloc[-1]), float(osc.iloc[-2])
        if pd.isna(cur) or pd.isna(prev):
            return "資料不足", "OSC 最新兩筆數值含 NaN，無法判斷狀態"
        if prev < 0 and cur > 0:
            return "翻紅第1根", f"OSC 由負轉正（{prev:.3f} → {cur:.3f}），多頭取回主導權"
        if prev > 0 and cur < 0:
            return "翻黑", f"OSC 由正轉負（{prev:.3f} → {cur:.3f}），多頭動能結束"
        if cur < 0 and abs(cur) < abs(prev):
            return "收腳中", f"負值柱狀體縮短（|{prev:.3f}| → |{cur:.3f}|），空頭動能衰退"
        if cur > 0:
            return "正值", f"OSC 維持正值（{cur:.3f}）"
        return "負值", f"OSC 維持負值（{cur:.3f}），尚未見收腳"

    def _detect_divergence(self, close: pd.Series, low: pd.Series, high: pd.Series,
                            osc: pd.Series, k: Optional[pd.Series]) -> Tuple[DivergenceType, str]:
        """
        簡化版背離偵測：在 lookback 視窗內找出「前一個波段低點/高點」，
        跟「當下這一根」比較價格與OSC的相對高低，判斷是否符合底背離/頂背離定義。
        這是一個實用近似算法（非嚴謹的zigzag轉折點演算法），足以捕捉角色定義的背離型態，
        但對雜訊敏感度會比專業轉折點演算法高，建議之後可以再疊加最小波段幅度過濾雜訊。
        """
        n = min(self.divergence_lookback, len(close) - 1)
        if n < 5:
            return "無", "資料長度不足以偵測背離（lookback視窗過短）"

        window_close = close.iloc[-(n + 1):-1]
        window_low = low.iloc[-(n + 1):-1]
        window_high = high.iloc[-(n + 1):-1]
        window_osc = osc.iloc[-(n + 1):-1]

        cur_close, cur_osc = float(close.iloc[-1]), float(osc.iloc[-1])
        if pd.isna(cur_close) or pd.isna(cur_osc) or window_close.empty:
            return "無", "最新資料含 NaN，跳過背離判斷"

        # ---- 底背離：現價創新低，但 OSC 在對應前波低點時的位置比現在的 OSC 還低 ----
        prev_low_idx = window_close.idxmin()
        prev_low_close = float(window_close.loc[prev_low_idx])
        prev_low_osc = float(window_osc.loc[prev_low_idx]) if not pd.isna(window_osc.loc[prev_low_idx]) else None

        # 【V2.11.10降噪】創新低要有意義的幅度（預設至少低於前低0.5%），避免現價只比前低低一點點雜訊
        # 就被判定「創新低」；OSC的變化也要有意義的幅度（預設至少是前低OSC值的10%），
        # 避免OSC只是隨機微幅波動就被當成「未破前低」的背離證據。
        _min_low_gap = prev_low_close * (self.min_new_extreme_pct / 100.0)
        _osc_ref = abs(prev_low_osc) if prev_low_osc is not None and prev_low_osc != 0 else None
        _min_osc_gap = (_osc_ref * self.min_osc_change_pct / 100.0) if _osc_ref else 0.0
        # 【V2.11.14新增】前低點至少要距離今天這根K棒 min_bar_gap 根，太近視為雜訊、不採用。
        _low_bars_back = len(close) - 1 - close.index.get_loc(prev_low_idx)
        _low_gap_ok = _low_bars_back >= self.min_bar_gap

        if (_low_gap_ok and cur_close < prev_low_close - _min_low_gap and prev_low_osc is not None
                and cur_osc > prev_low_osc + _min_osc_gap):
            note = f"價格創新低（{cur_close:.2f} < 前波低點{prev_low_close:.2f}，已超過最小幅度門檻，前波低點距今{_low_bars_back}根K棒），但OSC未破前低（{cur_osc:.3f} > {prev_low_osc:.3f}，差距已超過雜訊門檻）"
            if k is not None and len(k) >= 2 and not pd.isna(k.iloc[-1]) and float(k.iloc[-1]) < 20:
                return "低檔雙背離", note + f"；KD同步低檔區（K={float(k.iloc[-1]):.1f} < 20）"
            return "底背離", note

        # ---- 頂背離：現價創新高，且 OSC 連續3根遞減（含當日）----
        prev_high_idx = window_high.idxmax()
        prev_high_close = float(window_close.loc[prev_high_idx]) if prev_high_idx in window_close.index else float(window_high.loc[prev_high_idx])
        if len(osc) >= 3:
            o0, o1, o2 = float(osc.iloc[-1]), float(osc.iloc[-2]), float(osc.iloc[-3])
            osc_declining_3 = (not any(pd.isna(x) for x in (o0, o1, o2))) and (o0 < o1 < o2)
        else:
            osc_declining_3 = False
            o0 = o1 = o2 = None

        # 【V2.11.10降噪】同樣道理：創新高要有意義的幅度，OSC從o2到o0的總遞減量也要有意義的幅度
        # （預設至少是o2本身的10%），避免強勢噴出股「價格持續創高、OSC只是自然小幅收斂」被誤判頂背離
        # ——這正是我們實際遇過的創意(3443)案例。
        _min_high_gap = prev_high_close * (self.min_new_extreme_pct / 100.0)
        _osc_decline_significant = False
        if osc_declining_3 and o2 != 0:
            _decline_pct = (o2 - o0) / abs(o2) * 100.0
            _osc_decline_significant = _decline_pct >= self.min_osc_change_pct
        # 【V2.11.14新增】前高點同樣要求最小bar間距。
        _high_bars_back = len(close) - 1 - close.index.get_loc(prev_high_idx)
        _high_gap_ok = _high_bars_back >= self.min_bar_gap

        if _high_gap_ok and cur_close > prev_high_close + _min_high_gap and osc_declining_3 and _osc_decline_significant:
            return "頂背離", f"價格創新高（{cur_close:.2f} > 前波高點{prev_high_close:.2f}，已超過最小幅度門檻，前波高點距今{_high_bars_back}根K棒），且OSC柱狀體連續3根遞減（{o2:.3f}→{o1:.3f}→{o0:.3f}，遞減幅度{_decline_pct:.1f}%已超過雜訊門檻），多頭力竭"

        return "無", "目前價格與OSC走勢一致，或幅度/間距未超過最小雜訊門檻，未偵測到顯著背離"

    def _compute_risk_management(self, low: pd.Series) -> Optional[float]:
        """關鍵支撐停損價：lookback 視窗內（不含當日）的最低價，作為「跌破前低」的風控參考。"""
        n = min(self.divergence_lookback, len(low) - 1)
        if n < 3:
            return None
        window = low.iloc[-(n + 1):-1]
        if window.empty or window.isna().all():
            return None
        return float(window.min())

    def _decide_signal_action(self, osc_status: OSCStatus, divergence_type: DivergenceType) -> Tuple[SignalAction, str]:
        """
        優先序（由高到低）：出場(翻黑) > 減碼50%(頂背離) > 分批試單(底背離/低檔雙背離)
        > 核心進場(翻紅第1根) > 預警關注(收腳中) > 觀望。
        黃金交叉不在這裡出現——它只作為趨勢確認的次要訊號，不驅動 action。
        """
        if osc_status == "翻黑":
            return "出場", "柱狀體翻黑，出場風險升高，建議檢查停損與趨勢是否仍然成立"
        if divergence_type == "頂背離":
            return "減碼50%", "偵測到頂背離，獲利保護警示，建議評估分批減碼（非強制）"
        if divergence_type in ("底背離", "低檔雙背離"):
            extra = "（疊加KD低檔背離，信心水準較高）" if divergence_type == "低檔雙背離" else ""
            return "分批試單", f"偵測到{divergence_type}，屬於左側觀察訊號{extra}，不代表反轉已確認，建議分批小量試單、待柱狀體翻紅再補齊部位"
        if osc_status == "翻紅第1根":
            return "核心進場", "柱狀體由負轉正第1根，轉強候選，建議進一步確認價格與量能是否同步配合"
        if osc_status == "收腳中":
            return "預警關注", "負值柱狀體收腳，空頭動能衰退，納入觀察但不宜重押"
        return "觀望", "無明確訊號，維持空手觀望"

    def analyze(self, stock_id: str, stock_name: str, ohlcv: pd.DataFrame, timeframe: str = "日線") -> MACDSignalResult:
        """
        主分析入口。ohlcv 需含 Open/High/Low/Close（Volume 可選），index 為日期。
        資料長度不足、缺少必要欄位、或關鍵欄位是 NaN 時，一律回傳「資料不足」的安全結果，
        絕不拋出例外中斷呼叫端（既有系統的個股迴圈仍要能繼續跑其他股票）。
        """
        try:
            if ohlcv is None or ohlcv.empty:
                return self._empty_result(stock_id, stock_name, timeframe, "OHLCV 資料為空")
            for col in ("Open", "High", "Low", "Close"):
                if col not in ohlcv.columns:
                    return self._empty_result(stock_id, stock_name, timeframe, f"缺少必要欄位：{col}")

            close = ohlcv["Close"].squeeze()
            high = ohlcv["High"].squeeze()
            low = ohlcv["Low"].squeeze()
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame): low = low.iloc[:, 0]

            if len(close) < self.min_bars:
                return self._empty_result(stock_id, stock_name, timeframe,
                                           f"資料長度不足（{len(close)}根 < 最少需求{self.min_bars}根），暖身期不足，不予計算")

            dif, dea, osc = calc_macd_full_series(close, self.fast, self.slow, self.signal)
            if pd.isna(dif.iloc[-1]) or pd.isna(dea.iloc[-1]) or pd.isna(osc.iloc[-1]) or pd.isna(osc.iloc[-2]):
                return self._empty_result(stock_id, stock_name, timeframe, "最新 MACD 數值為 NaN，暫時跳過本次分析")

            osc_status, osc_note = self._detect_osc_status(osc)

            k_series = None
            try:
                k_series, _ = calc_kd(high, low, close, period=self.kd_period)
            except Exception:
                k_series = None

            divergence_type, div_note = self._detect_divergence(close, low, high, osc, k_series)
            risk_stop = self._compute_risk_management(low)
            signal_action, action_note = self._decide_signal_action(osc_status, divergence_type)

            cross_note = ""
            if len(dif) >= 2 and not pd.isna(dif.iloc[-1]) and not pd.isna(dea.iloc[-1]) and not pd.isna(dif.iloc[-2]) and not pd.isna(dea.iloc[-2]):
                golden_cross = dif.iloc[-2] <= dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]
                death_cross = dif.iloc[-2] >= dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]
                if golden_cross:
                    cross_note = "；DIF今日向上突破DEA（黃金交叉，趨勢確立次要確認）"
                elif death_cross:
                    cross_note = "；DIF今日向下跌破DEA（死亡交叉，次要確認）"

            detail = f"{osc_note}；{div_note}{cross_note}。判定：{action_note}"

            return MACDSignalResult(
                stock_id=stock_id, stock_name=stock_name, timeframe=timeframe,
                dif=round(float(dif.iloc[-1]), 4), dea=round(float(dea.iloc[-1]), 4), osc=round(float(osc.iloc[-1]), 4),
                osc_status=osc_status, divergence_type=divergence_type, signal_action=signal_action,
                risk_management=round(risk_stop, 2) if risk_stop is not None else None,
                detail=detail, error=None,
            )
        except Exception as e:
            return self._empty_result(stock_id, stock_name, timeframe, f"MACD分析發生未預期錯誤：{e}")

def build_macd_report(macd_results: List[MACDSignalResult]) -> pd.DataFrame:
    """把一批 MACDSignalResult 轉成報表用的 DataFrame，供 UI 表格顯示或 CSV/JSON 匯出使用。"""
    if not macd_results:
        return pd.DataFrame()
    rows = [r.to_dict() for r in macd_results]
    return pd.DataFrame(rows)

# 全域單一實例，供主迴圈重複呼叫；參數皆為預設值，若要調整（例如背離lookback天數）改這裡即可。
macd_analyzer = MACDStrategyAnalyzer()

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

# 【V2.11.9新增，P1-4】台股／美股靜態假日表——用來補強 _next_business_day/_add_business_days
# 原本「只排除週末」的限制。這是手動整理的靜態清單（資料來源：台灣證券交易所、NYSE官方休市公告），
# 不是即時行事曆服務，需要每年手動更新一次（通常年底前補上下一年度資料）。若忘記更新，
# 效果會退回「只排除週末」，不會出錯、只是遇到連假時 valid_until/execution_date 可能提早1~2天。
TW_MARKET_HOLIDAYS = {
    # 2026（資料來源：臺灣證券交易所 https://www.twse.com.tw/zh/trading/holiday.html）
    "2026-01-01", "2026-02-12", "2026-02-13", "2026-02-16", "2026-02-17",
    "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-27", "2026-04-03",
    "2026-04-06", "2026-05-01", "2026-06-19", "2026-09-25", "2026-09-28",
    "2026-10-09", "2026-10-26", "2026-12-25",
    # 2027（依行政院人事行政總處行事曆與證交所慣例推算，正式公告前可能微調）
    "2027-01-01", "2027-02-02", "2027-02-03", "2027-02-04", "2027-02-05",
    "2027-02-08", "2027-02-09", "2027-02-10", "2027-03-01", "2027-04-05",
    "2027-04-06", "2027-04-30", "2027-06-09", "2027-09-15", "2027-09-28",
    "2027-10-11", "2027-10-25", "2027-12-24", "2027-12-31",
}
US_MARKET_HOLIDAYS = {
    # 2026（資料來源：NYSE https://www.nyse.com/markets/hours-calendars）
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

def _is_market_session_open(is_us_stock, now=None):
    """
    【V2.11.33新增】判斷「現在」這個當下，對應市場是不是還在正式交易時段中，用來修正
    is_today_bar 原本「只比日期、不比時間」的真實bug（收盤幾個小時後系統還誤判「可能還在
    交易時段中」，因為原本只看抓到的資料日期是不是今天，完全沒管現在幾點）。

    台股：09:00~13:30（Asia/Taipei）；美股：09:30~16:00（America/New_York，zoneinfo會自動
    處理日光節約時間EST/EDT轉換，不需要手動判斷該用UTC-4還是UTC-5）。週末一律視為休市。

    這裡刻意不比對 US_MARKET_HOLIDAYS／TW_MARKET_HOLIDAYS（國定假日休市）——盤中時段判斷
    只是給「這根K棒的數字還會不會變動」這個提醒用的參考資訊，不是硬性關卡，國定假日當天
    抓到的資料本來就會是前一個交易日的舊資料（日期對不上今天），is_today_bar 的日期比對
    那一半自然就會是False，不需要在這裡重複判斷假日。

    now：可傳入已經算好的 timezone-aware datetime 重複使用，避免呼叫端重複呼叫 now()；
    不傳的話這裡自己算一次。
    """
    tz = ZoneInfo("America/New_York") if is_us_stock else ZoneInfo("Asia/Taipei")
    if now is None:
        now = datetime.datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    if now.weekday() >= 5:  # 週六(5)/週日(6)
        return False
    if is_us_stock:
        open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    else:
        open_t = now.replace(hour=9, minute=0, second=0, microsecond=0)
        close_t = now.replace(hour=13, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t

def _next_business_day(date_str, is_us_stock=False):
    """回傳輸入日期之後的下一個交易日（排除週末，並依 is_us_stock 對照台股或美股的靜態假日表；
    假日表只到2027年，超出範圍時自動退回「只排除週末」，不會拋錯）。"""
    try:
        holidays = US_MARKET_HOLIDAYS if is_us_stock else TW_MARKET_HOLIDAYS
        d = pd.Timestamp(date_str)
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in holidays:
            d += pd.Timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    except Exception:
        return ""

def _add_business_days(date_str, days, is_us_stock=False):
    """回傳輸入日期往後推 N 個交易日的日期（排除週末與靜態假日表，同上限制）。
    用於計算訊號有效期限 valid_until（規格書 7.6：PREPARE/BREAKOUT_WAIT=3個交易日，PULLBACK_WAIT=5個交易日）。"""
    try:
        holidays = US_MARKET_HOLIDAYS if is_us_stock else TW_MARKET_HOLIDAYS
        d = pd.Timestamp(date_str)
        remaining = int(days)
        while remaining > 0:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in holidays:
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
    """
    【V2.11.42修正】明確加上 auto_adjust=True。原本沒有指定這個參數，yfinance 回傳的股價
    是否已經還原股利/股票分割，完全取決於當下安裝的 yfinance 版本預設值（不同版本的預設值
    不一樣，行為不確定）。如果剛好是「不還原」的版本，台股在除息日附近，Close/High/Low
    會出現一個假的跳空缺口——這本身不影響「今天」這一天的真實成交價（今天的價格本來就
    沒有調整前後的差異），但會扭曲用「過去N天」計算的指標（MA20/60、ATR、前高、波段低點），
    在除息日之後的一段時間內（約MA60的暖身期長度）可能讓這些指標失真，進而影響進出場判斷。
    明確指定 auto_adjust=True，確保不管安裝的yfinance版本是哪一個，行為都一致、正確地還原
    股利/分割，不再依賴不確定的預設值。
    """
    try:
        if code.isalpha() or code.endswith('.US'):
            df = yf.download(code.replace('.US', ''), period="6mo", progress=False, auto_adjust=True)
        elif code.endswith('.TW') or code.endswith('.TWO'):
            df = yf.download(code, period="6mo", progress=False, auto_adjust=True)
        else:
            df_tw = yf.download(f"{code}.TW", period="6mo", progress=False, auto_adjust=True)
            df = df_tw if (df_tw is not None and not df_tw.empty and len(df_tw) > 0) else yf.download(f"{code}.TWO", period="6mo", progress=False, auto_adjust=True)
        return _trim_trailing_nan_rows(df)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_stock_data_extended(code):
    """
    【MACD模組新增】專供「週線」分析使用的較長期資料抓取。既有 fetch_stock_data() 只抓6個月
    （約26根週K），對週線 MACD（EMA26+訊號9）而言暖身期完全不夠，週線分析會永遠卡在「資料不足」。
    這裡另開一個獨立、較長快取時間（30分鐘）的抓取函式，抓 2 年資料，跟 fetch_stock_data()
    完全分開快取、互不影響，不會改變既有日線指標/K線圖/任何既有功能的行為或抓取頻率。

    【V2.11.42修正】同 fetch_stock_data()，明確加上 auto_adjust=True，理由見該函式docstring。
    這裡跨度2年，涵蓋除息事件的機率比6個月的日線資料更高，這個修正的影響也更明顯。
    """
    try:
        if code.isalpha() or code.endswith('.US'):
            df = yf.download(code.replace('.US', ''), period="2y", progress=False, auto_adjust=True)
        elif code.endswith('.TW') or code.endswith('.TWO'):
            df = yf.download(code, period="2y", progress=False, auto_adjust=True)
        else:
            df_tw = yf.download(f"{code}.TW", period="2y", progress=False, auto_adjust=True)
            df = df_tw if (df_tw is not None and not df_tw.empty and len(df_tw) > 0) else yf.download(f"{code}.TWO", period="2y", progress=False, auto_adjust=True)
        return _trim_trailing_nan_rows(df)
    except Exception:
        return pd.DataFrame()

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
# 【V2.11.39新增，Decision Log】純append-only的人工決策紀錄，記錄「系統建議 vs 你實際執行了什麼」
# ——系統自己的建議/理由已經存在 trade_plan 分頁（state／signal_reason），這裡只補系統不可能
# 知道的那一半：你有沒有真的照做、實際成交價位、偏離的原因。
DECISION_LOG_HEADERS = ["log_date", "code", "system_suggestion", "action_taken", "actual_price", "reason", "logged_at"]

# 【V2.11.41新增，Trade Plan Snapshot】每次某檔股票的資料日期真的往前推進到新的一天時，把「前一天
# 最終確定版」的計畫內容凍結存一份，不管之後同一天內盤中重新評估幾次、覆寫幾次，都不會動到這份
# 已經凍結的舊快照——解決「盤中打開App，發現昨晚看到的計畫已經被當天盤中還沒收盤的資料覆寫掉」
# 這個真實操作痛點。純append-only，只新增不覆寫。
TRADE_PLAN_SNAPSHOT_HEADERS = ["code", "snapshot_date", "state", "signal_reason", "entry_price", "breakout_price",
                               "chase_limit", "invalid_price", "t1_price", "t2_price", "current_trailing_stop",
                               "current_trailing_stop_source", "suggested_shares", "addon_shares_approved",
                               "partial_exit_shares", "full_exit_shares", "saved_at"]

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
    # 回測品質（V2.11.17新增，Retest Engine）
    "retest_min_price", "retest_quality",
    # 突破品質（V2.11.17新增，Breakout Quality Engine；V2.11.28補上四個子項，見說明書V2.11.28修復說明）
    "breakout_quality_score", "breakout_quality_grade",
    "bq_volume", "bq_macd", "bq_breakout_margin", "bq_decision_score",
    # 停利相關
    "t1_price", "t2_price", "t1_taken", "t2_taken", "partial_exit_ratio", "partial_exit_shares",
    # 出清相關
    "initial_stop", "previous_trailing_stop", "current_trailing_stop", "current_trailing_stop_source", "full_exit_shares",
    # 股數與風險
    "suggested_shares", "addon_shares_suggested", "addon_shares_approved", "remaining_shares",
    "max_risk_amount", "used_risk_amount", "remaining_risk_amount", "last_known_qty",
    # 人工覆核（V2.11.12新增，簡化版）
    "review_state", "review_at",
    # 版本
    "plan_version",
]

# 規格書第六節定義的11種狀態；ChatGPT 草稿版少了 PULLBACK_WAIT，這裡補齊。
# V2.11.10新增 BREAKOUT_FAILED（Breakout Engine）：突破後隔日站不穩（收盤跌破突破價且量能萎縮
# 或單日跌幅過大），跟「INVALID」（價格已經跌破更寬的失效價）是兩種不同嚴重程度的失敗，
# BREAKOUT_FAILED 抓得比較早、比較貼近「這次突破品質不夠」的判斷，不用等真的崩到失效價才反應。
TRADE_STATES = {
    "PREPARE", "BREAKOUT_WAIT", "PULLBACK_WAIT", "ENTER_NEXT_DAY", "HOLD",
    "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY",
    "SUSPENDED_BY_REGIME", "INVALID", "EXPIRED", "BREAKOUT_FAILED",
}

# 規格書第六節「狀態轉移」表格的合法轉移清單。transition_state() 會用這張表擋掉不合法的跳轉。
ALLOWED_TRANSITIONS = {
    # 【首次導入 bootstrap】既有持股在 trade_plan 分頁第一次建立時預設是 PREPARE，
    # 但實際上使用者可能早已持有股數 > 0，evaluate_trade_state() 會直接依現況判給 HOLD／
    # ADD_NEXT_DAY／PARTIAL_EXIT_NEXT_DAY／FULL_EXIT_NEXT_DAY，因此這幾種轉移也要開放。
    "PREPARE": {"BREAKOUT_WAIT", "PULLBACK_WAIT", "ENTER_NEXT_DAY", "INVALID", "EXPIRED",
                "HOLD", "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "PREPARE"},
    "BREAKOUT_WAIT": {"ENTER_NEXT_DAY", "PULLBACK_WAIT", "INVALID", "EXPIRED", "BREAKOUT_FAILED", "BREAKOUT_WAIT"},
    "PULLBACK_WAIT": {"ENTER_NEXT_DAY", "INVALID", "EXPIRED", "BREAKOUT_FAILED", "PULLBACK_WAIT"},
    # 【修正】原本漏了 INVALID／EXPIRED：訊號確認「下一交易日可進場」後，如果執行前價格突然
    # 跌破失效價、或超過有效期限使用者都還沒來得及執行，理論上應該要能判定失效／過期，
    # 原本的轉移表卻擋住這條路，導致這兩種情況發生時轉移一直被拒絕（見上方 transition_state 修正說明）。
    # 【V2.11.14修正，真實bug】同樣道理也漏了 PULLBACK_WAIT：已經確認「下一交易日可進場」後，
    # 如果隔日真的開太高、追價超過追價上限，理應改判「追價過高、改等回測」，但轉移表原本沒開放
    # 這條路，導致這個轉移一律被拒絕，狀態會卡在ENTER_NEXT_DAY不動，等於沒有真正擋下追高風險。
    "ENTER_NEXT_DAY": {"HOLD", "SUSPENDED_BY_REGIME", "INVALID", "EXPIRED", "BREAKOUT_FAILED", "PULLBACK_WAIT", "ENTER_NEXT_DAY"},
    # 【修正：股數變 0 的重置路徑】HOLD/ADD_NEXT_DAY/PARTIAL_EXIT_NEXT_DAY 都可能因為使用者在系統之外
    # 手動賣出全部持股，導致 held_qty 直接變 0，這種情況也要能重置回 PREPARE 重新追蹤訊號，
    # 否則狀態會卡死在「理論上已經沒有部位、卻永遠回不去空手訊號流程」的中間態。
    "HOLD": {"ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "PREPARE", "HOLD"},
    "ADD_NEXT_DAY": {"SUSPENDED_BY_REGIME", "HOLD", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "PREPARE", "ADD_NEXT_DAY"},
    # 逆風解除後恢復到暫停前的原狀態；出清判斷仍優先於恢復。
    "SUSPENDED_BY_REGIME": {"ENTER_NEXT_DAY", "ADD_NEXT_DAY", "HOLD", "FULL_EXIT_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "BREAKOUT_WAIT", "PULLBACK_WAIT", "PREPARE", "SUSPENDED_BY_REGIME"},
    "PARTIAL_EXIT_NEXT_DAY": {"HOLD", "FULL_EXIT_NEXT_DAY", "PREPARE", "PARTIAL_EXIT_NEXT_DAY"},
    # 【V2.11.32修正，真實P0 bug】原本只開放到PREPARE，假設「出清訊號一定會被執行」（執行後qty
    # 變0，才會走到下面的PREPARE歸零重置路徑）。但這個假設不成立：使用者可能沒有在建議當天執行
    # （忘記、猶豫、或不同意），股數仍然>0；如果後續價格回到防守線之上，evaluate_trade_state()
    # 的「持有中」分支其實已經正確判斷該回到HOLD／ADD_NEXT_DAY／PARTIAL_EXIT_NEXT_DAY（依當下條件
    # 而定，不是固定只會是HOLD——測試時發現如果復原當下同時符合加碼資格，系統會嘗試轉去
    # ADD_NEXT_DAY，同樣會被舊版轉移表擋下），卻被這裡擋下來，導致「交易計畫」分頁永遠卡在
    # 「全部出清」，即使現價早已回到防守線之上、跟「AI決策與SOP」分頁（完全獨立、每次都重新計算，
    # 沒有這個卡死問題）顯示的建議互相矛盾，使用者會看到兩個分頁給出完全相反的訊號卻不知道原因。
    # 開放這三條路，讓「出清訊號沒有被執行、價格後來又回到防守線之上」時，能正確回到對應的持有中
    # 狀態繼續追蹤，不會卡死；如果之後價格又真的再度跌破防守線，會產生一個全新的FULL_EXIT_NEXT_DAY
    # 訊號，這是正常、即時反映當下價格的行為，不是「忽略風險」。
    "FULL_EXIT_NEXT_DAY": {"PREPARE", "FULL_EXIT_NEXT_DAY", "HOLD", "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY"},
    "INVALID": {"PREPARE", "INVALID"},
    "EXPIRED": {"PREPARE", "EXPIRED"},
    "BREAKOUT_FAILED": {"PREPARE", "BREAKOUT_FAILED"},   # 突破失敗後歸零，重新開始追蹤新訊號
}

# 若 load_trade_plan() 失敗，強制整個執行流程降級為 VIEW_ONLY，不允許本次任何寫入或狀態推進。
TRADE_PLAN_LOAD_OK = True

# 【V2.11.15新增】若 load_portfolio() 失敗，強制整個執行流程降級為安全模式：不顯示任何分析、
# 不允許任何持股/交易計畫的建立或覆寫。理由跟 TRADE_PLAN_LOAD_OK 完全一樣——絕不能讓「讀取失敗」
# 悄悄變成「假裝用內建示範持股去跑正式分析」，那樣系統會誤以為自己知道使用者的真實部位。
PORTFOLIO_LOAD_OK = True

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

def get_worksheet(name, headers, bootstrap_rows=None):
    """
    取得指定分頁；若試算表裡還沒有這個分頁，就自動建立並寫入表頭。

    bootstrap_rows（V2.11.20新增）：只有在這個分頁『真的不存在、這次執行才剛被建立』時才會
    一併寫入的預設資料列（list of list）。這個時機是唯一『毫無疑義的第一次使用』——分頁不存在，
    絕對沒有任何既有資料可能被誤蓋。刻意不在「分頁已存在、但讀到空結果」這種情況下寫入
    bootstrap_rows，因為那種情況無法區分「真的是空的」跟「暫時性讀取異常」，見 load_portfolio()
    的說明。
    """
    ss = get_spreadsheet()
    try:
        ws = ss.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=name, rows=200, cols=len(headers))
        ws.append_row(headers)
        if bootstrap_rows:
            ws.append_rows(bootstrap_rows)
    return ws

def load_portfolio():
    """
    讀取 portfolio 分頁。

    【V2.11.15修正，P0】原本這裡不論「分頁真的是空的（第一次使用）」還是「讀取過程發生例外
    （額度限制、網路逾時、認證失敗……）」，最後都會回傳同一份 DEFAULT_PORTFOLIO（程式碼內建的
    示範持股），導致系統在「Google Sheets 讀取失敗」時完全沒有察覺自己拿到的是假資料。
    V2.11.15 修正了「讀取過程發生例外」這一半：改回傳空 dict、把 PORTFOLIO_LOAD_OK 設為 False，
    強制安全模式。

    【V2.11.20修正，真實bug，P0】V2.11.15 沒有修到的另一半：原本只要 `ws.get_all_records()`
    回傳空結果（不論是分頁『真的第一次使用』、還是分頁『明明已經有~20幾檔真實持股，這次讀取卻剛好
    讀到空結果』），一律當成「第一次使用」寫入3檔內建示範持股。這造成一次真實的資料損毀事故：
    使用者原本有20幾檔持股，某次讀取回傳空結果被誤判成「第一次使用」，寫入3檔示範持股後，
    緊接著任何一次存檔動作（新增股票/刪除/加碼小工具/CSV匯入，都會呼叫 save_portfolio()）
    就把這3檔示範持股整批覆寫回 Google Sheet，永久蓋掉原本的真實持股（`trade_plan`／`history`
    這兩個分頁沒有這個自動bootstrap機制，因此沒有被波及，仍保有完整資料，這也是為什麼只有
    `portfolio` 分頁受損的原因）。

    修正後不再用「records是不是空的」判斷要不要bootstrap，改用「這個分頁是不是這次執行才剛被
    建立的全新分頁」判斷（唯一不會誤判的依據——分頁不存在就是不存在，沒有模糊地帶）：
      1. 分頁『這次執行才剛被建立』（原本完全不存在）：這是唯一毫無疑義的「第一次使用」，
         由 get_worksheet() 的 bootstrap_rows 機制在建立分頁的當下就一併寫入預設持股。
      2. 分頁『已經存在』，但這次讀到空結果：不再自動假設「一定是真的清空了」，因為這個假設
         之前造成過真實的資料損毀。改成跟「讀取例外」同等級的保守處理：回傳空 dict，並把
         PORTFOLIO_LOAD_OK 設為 False，強制安全模式（不顯示分析、禁止任何持股寫入）——
         如果這次真的只是暫時性讀取異常，系統完全沒有寫入任何東西，下次重新整理／重新執行
         就會自動恢復正常；如果持股真的被清空了，安全模式會提示你確認，而不是靜悄悄地
         幫你「補」上3檔示範持股、然後在你毫不知情的情況下讓下一次存檔把這3檔永久寫回去。
    """
    global PORTFOLIO_LOAD_OK
    PORTFOLIO_LOAD_OK = True
    try:
        _bootstrap_rows = [[code, info["name"], info["cost"], info["cap"], info["risk"], info["status"], "", info.get("qty", 0)] for code, info in DEFAULT_PORTFOLIO.items()]
        _existed_before = True
        try:
            get_spreadsheet().worksheet("portfolio")
        except gspread.WorksheetNotFound:
            _existed_before = False
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS, bootstrap_rows=_bootstrap_rows)
        records = ws.get_all_records()
        if not records:
            if not _existed_before:
                # 分頁是這次執行才剛被建立的：bootstrap_rows 已經在 get_worksheet() 裡寫入，
                # 這裡直接回傳同一份資料即可（理論上 get_all_records() 現在應該讀得到剛寫入的
                # 3列，走到這個分支代表寫入後立刻讀取失敗，保留防禦性判斷，仍然回傳預設持股，
                # 不會有資料損毀風險，因為這個分頁本來就是全新的，沒有任何既有資料可能被誤蓋）
                return {k: dict(v) for k, v in DEFAULT_PORTFOLIO.items()}
            # 分頁『原本就存在』，這次卻讀到空結果：無法區分「真的清空了」還是「暫時性讀取異常」，
            # 一律當成需要人工確認的情況處理，不自動寫入任何東西，也不假裝這是正常的空持股狀態
            PORTFOLIO_LOAD_OK = False
            st.error("⚠️ 持股資料讀取結果異常（分頁存在，但讀不到任何資料），本次強制改為安全模式（僅檢視，不顯示任何分析，且已停用「儲存持股」以免覆蓋你在 Google Sheet 上的既有真實資料）。若這只是暫時性讀取異常，重新整理頁面即可恢復；若持股真的被清空，請先確認 Google Sheet 上的實際內容再繼續操作。")
            return {}
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
        PORTFOLIO_LOAD_OK = False
        st.error(f"⚠️ 讀取 Google Sheet 持股資料失敗，本次強制改為安全模式（僅檢視，不顯示任何分析，且已停用「儲存持股」以免覆蓋你在 Google Sheet 上的既有真實資料），避免誤用內建示範持股去計算你的真實部位：{e}")
        return {}

def save_portfolio(data):
    """
    【V2.11.15修正，P0】若 PORTFOLIO_LOAD_OK 是 False（代表這次執行一開始讀取 portfolio 就失敗），
    直接拒絕寫入並回傳 False——理由跟 save_trade_plan() 的 TRADE_PLAN_LOAD_OK 守門完全一樣：
    避免拿一份「基於讀取失敗、可能只是空 dict 或不完整內容」的資料去覆蓋 Google Sheet 上
    原本可能還完好的真實持股資料。這一道守門保護了側邊欄所有會呼叫 save_portfolio() 的入口
    （新增股票、刪除持股、加減碼小工具、CSV匯入、暫停/恢復分析、手動歸檔），不需要每個呼叫點各自檢查。
    """
    if not PORTFOLIO_LOAD_OK:
        st.error("⚠️ 持股資料本次讀取失敗，安全模式下已停用「儲存持股」以免覆蓋 Google Sheet 上的既有資料，請重新整理頁面確認連線/權限後再試一次。")
        return False
    try:
        ws = get_worksheet("portfolio", PORTFOLIO_HEADERS)
        ws.clear()
        rows = [PORTFOLIO_HEADERS]
        for code, info in data.items():
            rows.append([code, info.get("name", ""), info.get("cost", 0.0), info.get("cap", 20000.0), info.get("risk", 5.0), info.get("status", "Active"), info.get("break_date", ""), info.get("qty", 0.0)])
        ws.update(rows)
        return True
    except Exception as e:
        st.error(f"⚠️ 寫入 Google Sheet 持股資料失敗：{e}")
        return False

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

def load_decision_log():
    """
    讀取決策日誌分頁，回傳 list of dict，新到舊排序（最新的紀錄看起來比較方便）。
    這是純append-only的人工紀錄，不需要像 portfolio/trade_plan 那樣做正規化/schema比對，
    讀取失敗時安靜回傳空列表即可（這個分頁純粹是給使用者自己回顧用，不是任何決策邏輯的輸入，
    讀取失敗不影響任何交易判斷，不需要比照 portfolio/trade_plan 的安全模式機制）。
    """
    try:
        ws = get_worksheet("decision_log", DECISION_LOG_HEADERS)
        records = ws.get_all_records()
        return list(reversed(records))
    except Exception as e:
        st.error(f"⚠️ 讀取決策日誌失敗：{e}")
        return []

def append_decision_log(log_date, code, system_suggestion, action_taken, actual_price, reason):
    """
    新增一筆決策日誌記錄。刻意用 append_row()（只新增一行）而不是 clear()+update()（整批清空
    重寫）——這是純append-only的紀錄，每次只新增一行，用append天生就不會有「清空成功、寫入
    失敗」的資料損毀窗口期（這正是V2.11.15/20修過的那類真實bug的成因，這裡從設計上直接避開，
    不需要再套用同一套安全模式機制）。
    """
    try:
        ws = get_worksheet("decision_log", DECISION_LOG_HEADERS)
        ws.append_row([log_date, str(code), system_suggestion, action_taken,
                        actual_price if actual_price else "", reason or "",
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")])
        return True
    except Exception as e:
        st.error(f"⚠️ 寫入決策日誌失敗：{e}")
        return False

def load_trade_plan_snapshots():
    """
    讀取所有交易計畫快照，回傳 dict：{code: 該股票最新的一筆快照(dict)}，只保留每檔股票「最新」
    的一筆——使用者要看的是「上一個已經確定的版本」，不需要看到完整歷史，只需要最近一筆。
    純讀取失敗時安靜回傳空dict即可，這個分頁只是給使用者比對用，不是任何決策邏輯的輸入。
    """
    try:
        ws = get_worksheet("trade_plan_snapshot", TRADE_PLAN_SNAPSHOT_HEADERS)
        records = ws.get_all_records()
        latest = {}
        for row in records:
            code = str(row.get("code", "")).strip()
            if not code: continue
            # records 是照試算表由上到下的順序（也就是append的先後順序），後面出現的會覆蓋前面的，
            # 天然就會留下「最新一筆」，不需要額外排序
            latest[code] = row
        return latest
    except Exception as e:
        st.error(f"⚠️ 讀取交易計畫快照失敗：{e}")
        return {}

def append_trade_plan_snapshot(code, plan):
    """
    把「即將被覆寫掉的前一天最終確定版」凍結存一份。呼叫時機是：偵測到這檔股票的資料日期真的
    要往前推進到新的一天（不是同一天內的盤中重新評估）——這樣存下來的，正好就是「前一天收盤後
    最後一次確定的計畫內容」，不管當天之後盤中重新評估、覆寫幾次，這份快照都不會被動到。
    刻意用 append_row()（純append-only），理由跟 append_decision_log() 一樣：避免「清空重寫」
    類型的資料損毀風險，這裡從設計上直接避開。
    """
    try:
        ws = get_worksheet("trade_plan_snapshot", TRADE_PLAN_SNAPSHOT_HEADERS)
        ws.append_row([
            str(code), plan.get("taiwan_data_date", ""), plan.get("state", ""), plan.get("signal_reason", ""),
            plan.get("entry_price", 0), plan.get("breakout_price", 0), plan.get("chase_limit", 0),
            plan.get("invalid_price", 0), plan.get("t1_price", 0), plan.get("t2_price", 0),
            plan.get("current_trailing_stop", 0), plan.get("current_trailing_stop_source", ""),
            plan.get("suggested_shares", 0), plan.get("addon_shares_approved", 0),
            plan.get("partial_exit_shares", 0), plan.get("full_exit_shares", 0),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ])
        return True
    except Exception as e:
        st.error(f"⚠️ 寫入交易計畫快照失敗：{e}")
        return False

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
    r["current_trailing_stop_source"] = str(r.get("current_trailing_stop_source", "") or "")
    r["review_state"] = str(r.get("review_state", "") or "PENDING")
    if r["review_state"] not in ("PENDING", "ACKNOWLEDGED"):
        r["review_state"] = "PENDING"
    r["plan_version"] = str(r.get("plan_version", "") or "2.11.x")
    for k in ["entry_price", "breakout_price", "pullback_low", "pullback_high", "chase_limit",
              "invalid_price", "t1_price", "t2_price", "initial_stop", "previous_trailing_stop",
              "current_trailing_stop", "max_risk_amount", "used_risk_amount", "remaining_risk_amount",
              "retest_min_price", "breakout_quality_score",
              "bq_volume", "bq_macd", "bq_breakout_margin", "bq_decision_score"]:
        r[k] = _safe_float(r.get(k), 0.0)
    r["retest_quality"] = str(r.get("retest_quality", "") or "")
    r["breakout_quality_grade"] = str(r.get("breakout_quality_grade", "") or "")
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

    【V2.11.12修正】原本這裡沒有 ws.clear()，只靠 ws.update(rows) 覆寫——gspread的update()
    只會覆寫「這次傳入的資料範圍」，範圍以外的舊儲存格不會被清掉。這代表如果你刪除一檔股票，
    trade_plan_data 少了一筆，Google Sheet上那一列舊資料卻還留在原地；下次 load_trade_plan()
    讀取整張表時，這筆已刪除股票的舊交易計畫會被誤讀回來、混進當次分析結果（孤兒列復活）。
    補上 clear() 後，每次都是「先清空、再完整寫入目前的data」，不會再有殘留孤兒列。
    """
    if not TRADE_PLAN_LOAD_OK:
        return False
    try:
        ws = get_worksheet("trade_plan", TRADE_PLAN_HEADERS)
        ws.clear()
        rows = [TRADE_PLAN_HEADERS]
        for code, raw in data.items():
            r = _normalize_trade_plan_row(dict(raw, code=code))
            rows.append([r[h] for h in TRADE_PLAN_HEADERS])
        ws.update(rows)
        return True
    except Exception as e:
        # 【V2.11.12修正】原本這裡宣稱「既有已保存的計畫不受影響」，但清空後若中途寫入失敗
        # （例如網路中斷），Google Sheet 可能已經被清空、新資料卻還沒寫完，不是真正意義上的
        # 「不受影響」。改成如實描述風險，不做過度保證。
        st.error(f"⚠️ 寫入 Google Sheet 交易計畫（trade_plan）失敗，這次的狀態變更可能沒有完整保存（含已清空但尚未寫入新資料的可能），建議重新整理頁面確認資料是否正常：{e}")
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

REGIME_BEARISH_THRESHOLD = 40.0  # 【V2.11.16新增，需要人工確認的參數】分數低於此視為「逆風」，
# 攔截新倉與加碼；對應五級分類中的「逆風」與「極端逆風」兩級，數值跟舊版二元條件的臨界點相近
# （見 calculate_regime_score 內的映射說明），是初版經驗值，建議依實際使用效果調整。

BREAKOUT_QUALITY_GATE_THRESHOLD = 50.0  # 【V2.11.34新增，需要人工確認的參數】突破品質分數
# （calculate_breakout_quality_score，V2.11.30/31重新設計後）低於此門檻時，暫緩確認進場，
# 留在BREAKOUT_WAIT/PULLBACK_WAIT繼續觀察，不會強制進場也不會被砍掉追蹤。50分對應五級評等
# 的C/D級分界，是刻意保守的起始值（只擋最明顯偏弱的訊號），依據V2.11.30/31用45~55筆小樣本
# 驗證出量能甜蜜點/突破強度子項有正相關（相關係數約0.17~0.26，屬中等偏弱訊號，不是強力保證），
# 屬於初版經驗值，建議之後用更大樣本的回測結果重新校準這個門檻。

REGIME_NEUTRAL_ZONE_UPPER = 60.0  # 【V2.11.38新增，需要人工確認的參數】市場燈號分數在
# REGIME_BEARISH_THRESHOLD（40分）到這個數字之間，視為「中性」——還沒到逆風攔截的地步，
# 但也稱不上明確偏多。用V2.11.37的跨時段一致性檢查發現：回測期間市場燈號平均59分的那一段
# Expectancy是負的（-1.93%），平均73分的那一段是強烈正值（+7.82%），兩者差距懸殊——代表
# 目前系統可能只在「明確偏多」的環境下才有優勢，中性環境下容易虧錢，但entry_gate原本只有
# 「低於40分才攔截」這一道關卡，40~100分之間完全一視同仁放行，沒有區分「還可以」跟「真的
# 很好」。60分是初版經驗值，取跨時段檢查裡兩段平均分數（59分/73分）中間值附近，抓一個大致
# 的分界，之後應該用更多樣本重新校準。
BREAKOUT_QUALITY_GATE_THRESHOLD_NEUTRAL = 65.0  # 【V2.11.38新增，需要人工確認的參數】市場燈號
# 分數落在「中性區」（REGIME_BEARISH_THRESHOLD~REGIME_NEUTRAL_ZONE_UPPER之間）時，改用這個
# 較高的品質門檻（原本50分提高到65分，對應五級評等的B/C分界），只放行更扎實的訊號，不是完全
# 不能進場——呼應V2.11.16設計Market Regime Score五級分類時「中性應該降低部位」的構想，但沒有
# 直接動部位大小計算（風險較高），改用「拉高品質要求」這個更小範圍的方式間接做到類似效果。

def _breakout_quality_gate_threshold_for_regime(regime_score):
    """
    【V2.11.38新增】依市場燈號分數決定這次要用哪個突破品質門檻，不再是「逆風才擋、其他所有
    情況一視同仁」的二元處理。regime_score < REGIME_BEARISH_THRESHOLD 的情況理論上已經在更早
    的 _regime_is_bearish() 那一關就被攔截、不會走到品質門檻這一步，這裡列出來是防禦性判斷。
    """
    if regime_score is None:
        return BREAKOUT_QUALITY_GATE_THRESHOLD  # 資料缺失時用基礎門檻，不额外收緊也不放寬
    if regime_score >= REGIME_NEUTRAL_ZONE_UPPER:
        return BREAKOUT_QUALITY_GATE_THRESHOLD
    return BREAKOUT_QUALITY_GATE_THRESHOLD_NEUTRAL

def calculate_regime_score(macro_data):
    """
    【V2.11.16新增，Market Regime Score】把「順風/逆風」從二元開關改成 0~100 分的連續分數，
    取代舊版「任一條件成立就整體翻黑」的二元判斷——原本只要台股跌破月線、或那斯達克跌破月線、
    或VIX≥25其中一項成立，就直接整個判定「🔴逆風」，沒有中間地帶（例如VIX=26跟VIX=45的風險
    程度天差地遠，舊版一視同仁）。

    子分數（各自 0~100，越高越偏多／環境越穩定）：
      - TW分數：台股加權指數距離月線（MA20）的百分比，±10% 線性映射到 0~100（剛好在月線上＝50分）
      - US分數：那斯達克距離月線的百分比，映射方式同上
      - VIX分數：VIX恐慌指數，12以下＝100分（環境穩定），35以上＝0分（極度恐慌），中間線性內插

    組合規則（刻意採用 min()「取最差」而非加權平均，維持舊版「任一項變差就該保守」的一貫精神，
    只是從「單一因子一踩線就整體翻黑」改成連續分數，不再是稍微風吹草動就整個熄燈）：
      - TW股適用分數（tw_regime）：就是 tw_score 本身，跟舊版一致，只看台股加權，不看美股/VIX
      - US股適用分數（us_regime）：min(us_score, vix_score)，跟舊版一致，那斯達克或VIX任一轉差
        都會壓低分數
      - 總覽分數（overview，頁面最上方顯示用）：min(tw_score, us_score, vix_score)，三個市場只要
        有一個明顯轉差，總覽分數就會被拉低，維持「保守示警優先」的一貫精神，不是三者平均掉

    五級分類（0~100）：80~100 🟢🟢順風／60~79 🟢偏多／40~59 🟡中性／20~39 🟠逆風／0~19 🔴極端逆風

    資料缺失（macro_data 抓取失敗）時，對應子分數給 50 分（中性），不會因為資料暫時缺失就誤判成
    極端值去攔截或放行交易。

    【需要人工確認的參數】±10%（指數距月線的映射範圍）、VIX 12~35（映射範圍）這兩組區間，以及
    REGIME_BEARISH_THRESHOLD＝40（逆風門檻）都是初版經驗值，跟 invalid_price 等既有參數一樣，
    建議依實際使用效果調整。
    """
    def _index_score(info):
        if not info:
            return 50.0
        price = _safe_float(info.get("price"))
        ma20 = _safe_float(info.get("ma20"))
        if price <= 0 or ma20 <= 0:
            return 50.0
        distance_pct = max(-10.0, min(10.0, (price / ma20 - 1.0) * 100.0))
        return (distance_pct + 10.0) / 20.0 * 100.0

    def _vix_score(info):
        if not info:
            return 50.0
        v = _safe_float(info.get("price"))
        if v <= 0:
            return 50.0
        v = max(12.0, min(35.0, v))
        return (35.0 - v) / (35.0 - 12.0) * 100.0

    tw_score = _index_score(macro_data.get("TW"))
    us_score = _index_score(macro_data.get("US"))
    vix_score = _vix_score(macro_data.get("VIX"))
    return {
        "tw": tw_score, "us": us_score, "vix": vix_score,
        "tw_regime": tw_score,
        "us_regime": min(us_score, vix_score),
        "overview": min(tw_score, us_score, vix_score),
    }

def regime_tier_label(score):
    """分數轉五級文字標籤，純顯示用；實際攔截邏輯只依賴數字本身（REGIME_BEARISH_THRESHOLD），不依賴這個文字。"""
    if score >= 80: return "🟢🟢 順風"
    if score >= 60: return "🟢 偏多"
    if score >= 40: return "🟡 中性"
    if score >= 20: return "🟠 逆風"
    return "🔴 極端逆風"

def derive_market_regime(macro_data):
    """
    【V2.11.16改為分數制】把 calculate_regime_score() 算出的「總覽分數」轉成顯示文字，
    取代舊版純二元的🟢順風/🔴逆風判斷，回傳例如「🟡 中性 52/100」這樣附帶分數與各子項細節的字串。
    實際攔截新倉/加碼的判斷是用 _regime_is_bearish()（依個股是台股或美股分開判定，使用各自對應的
    子分數，不是這裡的總覽分數）。
    """
    scores = calculate_regime_score(macro_data)
    overview = scores["overview"]
    return f"{regime_tier_label(overview)} {overview:.0f}/100（台{scores['tw']:.0f}／美{scores['us']:.0f}／VIX{scores['vix']:.0f}）"

def _regime_is_bearish(market_context, is_us_stock):
    """
    逆風判定：【V2.11.16改為分數制】改呼叫 calculate_regime_score()，用同一份分數判斷，不再各自
    維護一套二元條件。台股個股用 tw_regime 分數（只看台股加權，跟舊版一致）；美股個股用 us_regime
    分數（那斯達克與VIX取最差，跟舊版一致）。判斷門檻是 REGIME_BEARISH_THRESHOLD（目前40分，對應
    五級分類的「逆風」與「極端逆風」兩級），跟舊版二元條件（台股跌破月線／那斯達克跌破月線或VIX≥25）
    的臨界點相近，只是從「一踩線就整個翻黑」改成分數連續變化，體感上會略為放寬邊緣情況（例如台股
    剛好在月線下方0.5%這種極輕微破線，新制度下可能還沒觸發攔截，需要真的明顯轉弱才會擋新倉/加碼）。

    這個函式只回傳 True/False，供 evaluate_trade_state() 決定是否攔截新倉／加碼——規格書明確要求
    「逆風只限制新倉與加碼，不得刪除個股既有出場計畫」，所以出清/停利判斷完全不使用這個函式，
    這一點跟改版前完全沒有變化。
    """
    scores = calculate_regime_score(market_context)
    regime_score = scores["us_regime"] if is_us_stock else scores["tw_regime"]
    return regime_score < REGIME_BEARISH_THRESHOLD

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

def classify_next_day_execution(plan, price):
    """
    【V2.11.14新增，Next-Day Execution Gate】把「隔日開盤價落在哪個區間該怎麼做」整理成一個
    好懂的執行燈號，純粹是顯示層面的分類，不建立新狀態、不影響狀態機本身的轉移邏輯——
    刻意這樣設計是因為前幾輪多次因為「新增狀態卻漏掉合法轉移表分支」出過真實bug（包含這次
    順便修掉的 ENTER_NEXT_DAY→PULLBACK_WAIT 遺漏），這裡只做分類、不做決策，降低風險。

    回傳 (代號, 說明文字)：
      EXECUTE：條件正常，可依計畫執行
      EXECUTE_WITH_LIMIT：價格略低於突破價，可視情況小量執行或觀察是否站穩
      WAIT_RETEST：開太高、追價過高，不追，等回測
      CANCEL：已跌破失效條件或已判定突破失敗，不執行
      NO_TRADE：目前沒有待執行的進場計畫
    """
    state = plan.get("state")
    if state == "ENTER_NEXT_DAY":
        breakout_price = _safe_float(plan.get("breakout_price"))
        chase_limit = _safe_float(plan.get("chase_limit"))
        invalid_price = _safe_float(plan.get("invalid_price"))
        if invalid_price > 0 and price is not None and _safe_float(price) < invalid_price:
            return "CANCEL", "🔴 已跌破失效條件，不執行"
        if chase_limit > 0 and price is not None and _safe_float(price) > chase_limit:
            return "WAIT_RETEST", "🟠 開盤已超過追價上限，不追價，改等回測"
        if breakout_price > 0 and price is not None and _safe_float(price) < breakout_price:
            return "EXECUTE_WITH_LIMIT", "🟡 價格略低於突破價，建議先觀察是否站穩再執行，或酌量減碼執行"
        return "EXECUTE", "🟢 條件正常，可依計畫執行"
    if state == "BREAKOUT_FAILED":
        return "CANCEL", "🔴 已判定突破失敗，不執行"
    return "NO_TRADE", "⚪ 目前沒有待執行的進場計畫"

def is_breakout_failed(plan, price, volume, vol_ma5, atr, prev_close):
    """
    【V2.11.10新增，Breakout Engine「T+1 Execution Gate」】比 is_signal_invalid 抓得更早、更貼近
    「這次突破的品質不夠、可能是假突破」的判斷，不用等真的崩到更寬的失效價才反應。

    判定條件（採用「收盤跌破突破價」+「量能萎縮 或 單日跌幅過大」的組合，避免對正常的
    突破後拉回測試（throwback）反應過度——如果只看「收盤跌破突破價」，會連很多正常的
    小幅拉回都判定失敗；但如果只看「量縮才算」，又會漏掉「帶量重跌」這種更危險的假突破，
    所以量縮跟單日跌幅過大兩個條件是「或」的關係，任一成立就判定突破失敗）：
      收盤價 < 突破價
      且 (當日成交量 < 5日均量  或  單日跌幅 ≥ 1.5×ATR)
    """
    breakout_price = _safe_float(plan.get("breakout_price"))
    if breakout_price <= 0 or price is None or _safe_float(price) >= breakout_price:
        return False
    vol_ma5_f = _safe_float(vol_ma5)
    volume_shrunk = vol_ma5_f > 0 and _safe_float(volume) < vol_ma5_f
    atr_f = _safe_float(atr)
    prev_close_f = _safe_float(prev_close)
    sharp_drop = prev_close_f > 0 and atr_f > 0 and (prev_close_f - _safe_float(price)) >= 1.5 * atr_f
    return volume_shrunk or sharp_drop

def classify_retest_quality(retest_min_price, pullback_low, pullback_high, invalid_price):
    """
    【V2.11.17新增，Retest Engine（P1-2）】依 PULLBACK_WAIT 期間追蹤到的最低價，判斷這次「回測不破
    再突破」是不是真的經過回測，還是價格從沒真的拉回等待區間就直接反彈站上（V型急拉，較不可靠）。

    純粹是分類/顯示用，不影響任何狀態轉移或進場判斷本身——理由跟 classify_next_day_execution()
    （V2.11.14）一樣：前幾輪多次因為「新增判斷邏輯卻漏掉合法轉移分支」出過真實bug，這裡刻意只做
    事後分類、不做決策，降低風險。

    回傳 (中文說明, 分類代碼)：
      RETESTED_HELD：期間最低價曾經拉回進等待區間（≤pullback_high）、且沒有跌破失效價，屬於
                     真正意義上的「回測不破再突破」
      NO_RETEST：期間最低價從未進入等待區間，代表這次是直接站上、沒有經過真實回測（V型），
                 訊號可靠度相對存疑，建議額外留意量能是否同步認證
      INVALID_TOUCHED：期間最低價曾經跌破失效價（理論上這種情況通常會先被 is_signal_invalid()
                       攔下、走不到這裡，這是防禦性判斷，避免萬一資料跳動漏判）
      （retest_min_price 沒有記錄到有效資料時，回傳空字串，不做任何提示）
    """
    m = _safe_float(retest_min_price)
    hi = _safe_float(pullback_high)
    inv = _safe_float(invalid_price)
    if m <= 0:
        return "", ""
    if inv > 0 and m <= inv:
        return "回測期間曾跌破失效價，訊號品質存疑", "INVALID_TOUCHED"
    if hi > 0 and m <= hi:
        return f"回測期間最低來到 {m:.2f}，已實際拉回等待區間內確認支撐，屬真實回測後再突破", "RETESTED_HELD"
    return "回測期間股價從未實際拉回等待區間，屬直接反彈站上（V型），未經真實回測確認", "NO_RETEST"

# --- 4-4. 狀態轉移（規格書第六節狀態轉移表）---
def transition_state(plan, next_state, extra_fields, data_date, reason=""):
    """
    唯一允許改變 plan['state'] 的地方。會先查 ALLOWED_TRANSITIONS 確認這是合法轉移，
    不合法就直接忽略、維持原狀態（寧可卡住讓使用者發現，也不要跳到不該去的狀態），
    合法的話才更新 state、origin_state、時間戳記與傳入的其餘欄位。

    【修正】拒絕轉移時的提示文字改成「直接覆蓋」而不是「接在舊文字前面」。舊寫法會導致同一個
    被反覆拒絕的轉移（例如訊號還在等待執行、盤中股價短暫觸及失效價又拉回）每被拒絕一次就多疊加
    一段文字，搭配「今天K棒未收斂時強制重新評估」的機制，一天內可能被拒絕好幾次，疊出一大段
    重複的灰色文字。現在只保留最新一次的拒絕原因，不會再無限增長。
    """
    current = plan.get("state", "PREPARE")
    allowed = ALLOWED_TRANSITIONS.get(current, {current})
    if next_state not in allowed:
        plan["signal_reason"] = f"（忽略不合法的狀態轉移 {current}→{next_state}，原狀態維持）"
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
def calculate_daily_score(price, cost, ma10, ma20, ma60, macd, bias, k, d, rsi, volume, vol_ma5,
                           atr_stop_price, take_profit_price, pivot_point, inst, is_us_stock,
                           tw_bearish=False, us_bearish=False, vix_high=False):
    """
    【V2.11.21新增，從main loop抽出】決策分數（ai_score）／決策信心（confidence）／SOP三燈
    （step1/2/3_pass）的唯一權威計算來源。

    這段公式原本直接寫在主迴圈裡，只有UI一個呼叫點。這次為了建立回測系統（P2），需要在回測的
    逐日重演迴圈裡呼叫「跟UI完全同一套」的評分公式——如果讓我在回測那邊另外抄一份公式，
    未來只要UI這邊改了分數公式、忘記同步改回測那份，兩邊就會產生你在這次對話裡已經看過好幾次的
    那種「兩套獨立維護、算出不同答案」的裂縫。所以直接把公式抽成這個獨立函式，UI跟回測都呼叫
    同一份，公式改一次、兩邊同時生效，不會再有漏改的風險。

    這是**純函式重構，不是行為變更**：抽出來的公式跟抽出前逐行完全一致，UI呼叫這個函式後，
    算出來的 ai_score/confidence/step1_pass/step2_pass/step3_pass 應該跟改版前的數字分毫不差。

    tw_bearish/us_bearish/vix_high：呼叫端自行判斷好的三個布林值（大盤跌破月線/那斯達克跌破月線/
    VIX過高），取代原本直接讀 tw_trend/us_trend/vix_trend 字典的寫法，讓這個函式不依賴任何
    外部字典結構，方便回測端用歷史資料自己組出這三個布林值。

    回傳 dict：ai_score, confidence, step1_pass, step2_pass, step3_pass, macro_warnings,
    is_bull_aligned, score_inst, score_tech, score_vol, score_risk。
    """
    score_inst = (20 if price > ma60 else 0) + (10 if macd > 0 else 0) + (10 if 0 < bias < 20 else 0) if is_us_stock else min(inst['days'] * 5, 20) + (20 if inst['accumulated_shares'] * price >= 3000000000 else (10 if inst['accumulated_shares'] * price >= 1000000000 else 0))
    _rsi_bull_point = 10 if (rsi > 50 and rsi <= 80) else 0
    score_tech = (10 if k > d else 0) + _rsi_bull_point + (10 if price > ma20 else 0)
    score_vol = min((volume / vol_ma5) * 10, 15) if vol_ma5 > 0 else 0
    score_risk = (10 if price > atr_stop_price else 0) + (5 if price >= take_profit_price or price >= cost * 1.05 else 0) if cost > 0 else 15

    score_forced_zero = bool(cost > 0 and price <= atr_stop_price)
    ai_score = 0 if score_forced_zero else min(int(score_inst + score_tech + score_vol + score_risk), 100)
    is_bull_aligned = (ma10 > ma20 and ma20 > ma60)
    confidence_base = ai_score * 0.8 + (10 if is_bull_aligned else 0) + (5 if price > pivot_point else 0)

    macro_warnings = []
    if is_us_stock:
        if us_bearish:
            confidence_base *= 0.85
            macro_warnings.append("⚠️ 美股大盤跌破月線，系統主動下調部位信心。")
        if vix_high:
            confidence_base *= 0.70
            macro_warnings.append("🚨 VIX 恐慌指數過高，系統強制抑制進場訊號！")
    else:
        if tw_bearish:
            confidence_base *= 0.85
            macro_warnings.append("⚠️ 台股大盤跌破月線，逆勢操作風險較高。")

    confidence = min(99, max(10, int(confidence_base)))
    step1_pass = (price > ma60 and macd > 0) if is_us_stock else (inst['days'] >= 3 or inst['accumulated_shares'] * price >= 1000000000)
    step2_pass, step3_pass = (k > d and rsi > 50 and volume > vol_ma5), (price > ma20 and is_bull_aligned)

    return {
        "ai_score": ai_score, "confidence": confidence, "step1_pass": step1_pass,
        "step2_pass": step2_pass, "step3_pass": step3_pass, "macro_warnings": macro_warnings,
        "is_bull_aligned": is_bull_aligned, "score_inst": score_inst, "score_tech": score_tech,
        "score_vol": score_vol, "score_risk": score_risk,
    }

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

def calculate_target_plan(price, cost, atr, previous_high, is_us_stock=False, profit_trigger_pct=10.0, min_gap_atr_multiple=1.0,
                           entry_price=None, initial_stop=None, t1_r_multiple=1.5, t2_r_multiple=2.5):
    """
    【單一目標價權威來源，V2.11.8新增，V2.11.18改為R倍數+前高混合】T1/T2 計算的唯一入口。

    【V2.11.18】原本T1/T2純粹用「前高（結構）或現價+N×ATR（外推）」決定，完全不管你當初這筆交易
    實際承擔了多少風險——兩檔ATR差很多的股票，即使風報比天差地遠，算出來的T1/T2距離可能很接近，
    因為公式只吃ATR，不吃你的風險距離。這次改成「R倍數 + 前高」混合，兩者取較大值：

      R（風險距離）＝entry_price − initial_stop，這是進場當下就鎖定、之後不會再變動的固定值
      （不像ATR會隨時間變動，R代表「這筆交易當初決定承擔的風險」，是更穩定的度量單位）。
      entry_price/initial_stop 缺失或不合法（例如很舊的資料、或用CSV匯入的部位沒有經過正式的
      進場計畫流程）時，退回用 2×ATR 當R的替代值（等同於 initial_stop 原本的定義公式，數值上
      相近，只是精神上從「用當下浮動ATR」退化成近似值）。

      structural_t1 ＝前高（若前高離現價 > min_gap_atr_multiple×ATR，代表前高還有參考價值）；
                      否則現價已經追平/超過前高，改用「現價+2×ATR」外推（沿用V2.11.8既有邏輯不變）
      structural_t2 ＝structural_t1 + 2×ATR（沿用既有的固定差距）

      r_floor_t1 ＝cost + t1_r_multiple×R（預設1.5R，對齊 entry_gate 原本就要求 R1≥1.5 的門檻——
                   第一目標至少要值回你當初願意承擔的風險等級的1.5倍，不然只是隨便挑一個近的價位，
                   不是真正「值得」的第一目標）
      r_floor_t2 ＝cost + t2_r_multiple×R（預設2.5R）

      T1 ＝max(structural_t1, r_floor_t1)：前高夠遠、本身就值超過1.5R，就用前高（有真實壓力位
           支撐，比純數學算出來的R倍數更貼近市場現實）；前高太近、不值1.5R，就用R floor頂上去
           （這種情況下，前高不是一個有意義的第一目標，用R倍數確保至少有基本的風報比才算數）
      T2 ＝max(structural_t2, r_floor_t2)：邏輯相同

    其餘規則不變：
      - cost<=0（空手）：T1=T2=0
      - atr 無效：T1=T2=cost×1.10（保底值，這個分支不會用到R，因為沒有ATR也就沒有structural候選）
      - 獲利尚未超過 profit_trigger_pct（預設10%）：T1=T2=cost×1.10（未達門檻不做任何分析，R倍數
        邏輯同樣不會啟用）

    回傳 (t1, t2, branch)，branch 標示T1是哪個候選方法算出來的（resistance／atr_fallback／
    r_floor_t1，新增最後一種），T2一律沿用同一個branch文字（不額外區分T2自己的來源，維持原本
    函式「一個branch代表整體算法路徑」的慣例，即使極少數情況下T2實際上是被R floor頂上去、但T1
    卻是前高，這種混合情形branch文字只反映T1的來源，屬於已知的顯示簡化）。

    【需要人工確認的參數】t1_r_multiple（1.5）／t2_r_multiple（2.5）是本次新增的參數，跟
    min_gap_atr_multiple 等既有參數一樣屬於初版經驗值，建議依實際使用效果調整。
    """
    if cost is None or cost <= 0:
        return 0.0, 0.0, "no_position"
    if atr is None or atr <= 0 or pd.isna(atr):
        return round_to_tick(cost * 1.10, is_us_stock), round_to_tick(cost * 1.10, is_us_stock), "atr_unavailable"
    if price <= cost * (1 + profit_trigger_pct / 100.0):
        t1 = t2 = cost * 1.10
        return round_to_tick(t1, is_us_stock), round_to_tick(t2, is_us_stock), "profit_gate"

    _entry_f = _safe_float(entry_price)
    _init_stop_f = _safe_float(initial_stop)
    r = (_entry_f - _init_stop_f) if (_entry_f > 0 and _init_stop_f > 0 and _entry_f > _init_stop_f) else 2 * atr

    ph = _safe_float(previous_high)
    min_gap = min_gap_atr_multiple * atr
    if ph > price + min_gap:
        structural_t1, structural_t2, structural_branch = ph, ph + 2 * atr, "resistance"
    else:
        structural_t1, structural_t2, structural_branch = price + 2 * atr, price + 4 * atr, "atr_fallback"

    r_floor_t1 = cost + t1_r_multiple * r
    r_floor_t2 = cost + t2_r_multiple * r

    if r_floor_t1 > structural_t1:
        t1, branch = r_floor_t1, "r_floor_t1"
    else:
        t1, branch = structural_t1, structural_branch
    t2 = max(structural_t2, r_floor_t2)
    return round_to_tick(t1, is_us_stock), round_to_tick(t2, is_us_stock), branch

def calculate_stop_plan(price, cost, atr, ma20, previous_stop=None, profit_trigger_pct=30.0, swing_low=None, trend_confirmed=False, profit_protection_atr_multiple=1.0):
    """
    【單一防守線權威來源，V2.11.9新增，V2.11.10擴充Trend Runner多方法，V2.11.17改為三層防守，
    V2.11.18把Level1→2切換條件改為「趨勢是否形成」，V2.11.36收緊Level3的ATR倍數】

    【V2.11.18】依你的指示，把 Level1→2 的切換條件從「獲利%」改成「趨勢是否形成」，Level2→3
    則維持用「獲利%」不變——這兩段各自代表不同的問題，適合用不同的判斷依據：
    「什麼時候該開始跟結構」是趨勢問題，「什麼時候該重兵保護已經到手的獲利」是金錢問題。

      Level 1「Initial Risk Stop」：trend_confirmed 為 False（趨勢尚未確認形成）
        防守線 = max(previous_stop, cost−2×ATR)，趨勢還沒走出來前用最保守的固定緩衝，
        不做任何結構分析（此時價格結構還不夠明確，用結構支撐容易被雜訊洗）

      Level 2「Structural Stop」：trend_confirmed 為 True，但獲利還沒到 profit_trigger_pct（V2.11.19
        起預設30%，依你的指示從10%調高——原本10%太早就切進最緊的貼身防守，容易在正常回檔time
        把趨勢股洗出場，犧牲「讓利潤奔跑」的精神。調高後結構防守的持有時間拉長，趨勢股有更多
        空間走完整段行情，才會進入Level 3的緊縮保護）
        趨勢已經確認形成（例如站上MA20、均線多頭排列——沿用你系統既有SOP三燈的「趨勢燈」定義，
        不是另外發明一套新邏輯），但獲利還不多，還沒到「重兵保護」的程度：
        candidate = max(MA20−ATR, 波段低點(swing_low，若有提供))
        new_stop = max(previous_stop 或 cost−2×ATR起點, candidate)
        （只用結構性方法，不用「現價−X×ATR」這種貼近現價的緊縮方法——趨勢剛確認、獲利還不多，
        不需要為了保護還沒賺到的利潤而把防守線收得太緊，避免正常回檔就被洗出去）

      Level 3「Profit Protection Stop」：獲利 ≥ profit_trigger_pct（V2.11.19起預設30%，不論
        trend_confirmed 當下是True還是False——已經到手的獲利要優先保護，不因為趨勢燈臨時熄滅
        就放鬆防守）
        candidate = max(MA20−ATR, 波段低點(swing_low，若有提供), 現價−profit_protection_atr_multiple×ATR)
        new_stop = max(previous_stop 或 cost−2×ATR起點, candidate)
        （沿用V2.11.10「Trend Runner」三方法取最大值，加入「現價−X×ATR」這種貼近現價、
        鎖住最多獲利的候選方法）

    【V2.11.36修正，依MFE/MAE回測證據調整】profit_protection_atr_multiple 從原本固定寫死的1.5
    改成可傳入的參數，預設值同時從1.5調緊為1.0。調整依據：V2.11.35新增的MFE/MAE追蹤，用48筆
    真實回測資料顯示「贏的交易平均最高曾浮盈到35.2%，實際出場只拿到21.1%，回吐了14.1個百分點
    （約40%的峰值獲利被吐回去）」，而輸的交易那邊MAE（-9.8%）跟實際平均虧損（-8.3%）差距不大，
    代表初始停損（Level1的2×ATR）執行得還算確實——這代表問題比較可能出在Level3「已經賺錢之後
    要不要收緊防守線」這一段，不是「一開始的停損」，所以這次只收緊Level3的倍數，Level1的
    2×ATR跟Level2都沒有變動。收緊到1.0×ATR是否真的能縮小回吐缺口、同時不會讓太多原本能延續的
    趨勢股提早出場，需要重新跑一輪回測比較前後的Expectancy/回吐缺口才能確認，屬於
    【需要人工確認的參數】，這次的1.0只是根據現有證據的一次調整嘗試，不是最終定案。

    【V2.11.19重要澄清】這裡的 profit_trigger_pct 跟 calculate_target_plan() 裡同名的
    profit_trigger_pct 是兩個完全獨立的參數（各自函式自己的預設值，不是共用同一個全域常數）：
    這裡的門檻只決定「防守線什麼時候收緊」，calculate_target_plan() 的門檻只決定「T1/T2什麼時候
    開始真正計算」——依你的指示，這次只調高這裡（防守線），T1/T2的10%門檻維持不變。調整其中一個
    不會影響另一個。

    trend_confirmed 建議直接傳入 SOP三燈裡的「趨勢燈」（indicators["trend_gate"]，定義為
    price>MA20 且 MA10>MA20>MA60），這是系統既有、每天都會算的判斷，不是新發明的獨立邏輯，
    也不會因此新增額外的判斷子系統或濾網。trend_confirmed 缺失（None）時視為 False（Level 1，
    最保守），跟其他判斷邏輯「風控類參數缺失時預設保守」的慣例一致。

    三層之間共用同一條鐵律：**只能上移，不能下移**——不論在哪一層，new_stop 永遠是
    max(previous_stop, 這一層的候選值)。這條鐵律也保護了 trend_confirmed 在Level1/2之間來回
    切換（例如均線交叉反覆發生）時不會讓防守線跟著忽上忽下：即使某天 trend_confirmed 從True
    翻回False，防守線本身只會沿用先前已經墊高的 previous_stop，不會真的退回 Level 1 的水準，
    只有「這一天用哪一組候選公式」這個標籤會變動，實際數值不受影響。

    previous_stop：呼叫端傳入「上一次已經算出、且已經持久化保存的防守線」（沒有就傳 None 或 0，
    會用 cost−2×ATR 當起點）。UI跟狀態機都必須傳入「同一個來源」（trade_plan.current_trailing_stop）
    的 previous_stop，才能確保兩邊算出同一個答案。

    回傳值 (new_stop, source)，source 標示這次的防守線是哪個候選方法算出來的
    （initial_stop／locked_previous／ratchet_ma20_atr／ratchet_price_atr／ratchet_swing_low／
    no_position），讓UI可以告訴使用者「防守線為什麼在這裡」——Level2跟Level3共用
    ratchet_ma20_atr／ratchet_swing_low 這兩個來源標籤（公式完全一樣），只有 ratchet_price_atr
    這個來源只會在Level3出現（Level2的候選集合裡沒有這個方法）。
    """
    cost_f = _safe_float(cost)
    if cost_f <= 0:
        return 0.0, "no_position"
    flat_stop = cost_f - 2 * _safe_float(atr)
    prev = _safe_float(previous_stop)
    price_f = _safe_float(price) if price is not None else 0.0
    profit_pct = (price_f / cost_f - 1.0) * 100.0 if price is not None else -999.0

    # Level 3 優先判斷：獲利已超過 profit_trigger_pct 時，不論 trend_confirmed 為何都優先重兵保護
    if price is not None and profit_pct >= profit_trigger_pct:
        base = prev if prev > 0 else flat_stop
        _candidates = {
            "ratchet_ma20_atr": _safe_float(ma20) - _safe_float(atr),
            "ratchet_price_atr": _safe_float(price) - profit_protection_atr_multiple * _safe_float(atr),
        }
        if swing_low is not None and _safe_float(swing_low) > 0:
            _candidates["ratchet_swing_low"] = _safe_float(swing_low)
        best_source, best_value = max(_candidates.items(), key=lambda kv: kv[1])
        if base >= best_value:
            return base, "locked_previous"
        return best_value, best_source

    # Level 1：Initial Risk Stop —— 趨勢尚未確認形成（trend_confirmed 缺失一律視為 False，取保守值）
    if price is None or not trend_confirmed:
        if prev > 0 and prev >= flat_stop:
            return prev, "locked_previous"
        return flat_stop, "initial_stop"

    # Level 2：Structural Stop —— 趨勢已確認形成，但獲利還沒到 profit_trigger_pct
    base = prev if prev > 0 else flat_stop
    _candidates = {"ratchet_ma20_atr": _safe_float(ma20) - _safe_float(atr)}
    if swing_low is not None and _safe_float(swing_low) > 0:
        _candidates["ratchet_swing_low"] = _safe_float(swing_low)
    best_source, best_value = max(_candidates.items(), key=lambda kv: kv[1])
    if base >= best_value:
        return base, "locked_previous"
    return best_value, best_source

def calculate_trailing_stop_stateful(previous_stop, current_price, ma20, atr, cost):
    """
    【V2.11.9】改為 calculate_stop_plan() 的薄包裝，保留舊名稱／參數順序以維持既有呼叫端不用改，
    實際計算邏輯已經全部收斂進 calculate_stop_plan()，不再各自維護一份公式。
    """
    return calculate_stop_plan(current_price, cost, atr, ma20, previous_stop)[0]

def calculate_exit_plan(price, average_cost, atr, ma20, previous_trailing_stop, previous_high,
                         t1_taken, t2_taken, current_shares, is_us_stock=False, partial_exit_ratio=0.30,
                         macd_osc_status=None, swing_low=None, trend_confirmed=False, entry_price=None, initial_stop=None):
    """
    出清／分批停利計畫（規格書 9、10節）。回傳 dict 一定含 current_trailing_stop 與 t1_price/t2_price，
    並在符合條件時附上 next_state 建議（呼叫端 evaluate_trade_state 會再用 transition_state 實際套用，
    確保優先權判斷：全部出清 > T2 > T1 > 續抱，全部在這個函式內部就決定好，呼叫端不需要再重排順序）。

    【MACD深度整合】macd_osc_status 是「日線」MACD柱狀體狀態（翻紅第1根/正值/負值/收腳中/翻黑/None）。
    只用「已確認翻黑」（柱狀體由正轉負，不是還在醞釀的頂背離）觸發分批停利，且觸發條件是「價格到達
    T1/T2 結構化目標」或「MACD翻黑」兩者任一成立即可——用兩套獨立的訊號來源互相補位：結構化目標抓
    「漲多少該獲利了結」，MACD翻黑抓「動能真的轉弱了，不用等價格真的碰到目標價才反應」。刻意不用
    「頂背離」這種還在醞釀階段的訊號來觸發賣出，避免對正常回檔反應過度、犧牲「讓利潤奔跑」的精神；
    頂背離的用途是攔阻「新進場」與「加碼」，不是拿來提前出場。

    swing_low（V2.11.10新增，Trend Runner）：近期波段低點，傳給 calculate_stop_plan() 當第三個
    防守線候選方法（跟MA20−ATR、現價−1.5×ATR取最大值），沒有提供時該候選值不計入，行為退回
    V2.11.9版本（只有MA20−ATR與現價−1.5×ATR兩種候選）。

    trend_confirmed（V2.11.18新增）：直接轉傳給 calculate_stop_plan()，決定Level1→2的切換
    （見該函式docstring）。entry_price/initial_stop（V2.11.18新增）：直接轉傳給
    calculate_target_plan()，用來計算R倍數目標（見該函式docstring）。
    """
    current_trailing_stop, _stop_source = calculate_stop_plan(price, average_cost, atr, ma20, previous_trailing_stop, swing_low=swing_low, trend_confirmed=trend_confirmed)
    t1_price, t2_price, _target_branch = calculate_target_plan(price, average_cost, atr, previous_high, is_us_stock, entry_price=entry_price, initial_stop=initial_stop)

    result = {"current_trailing_stop": current_trailing_stop, "stop_source": _stop_source, "t1_price": t1_price, "t2_price": t2_price,
              "next_state": None, "signal_type": "", "signal_reason": "", "partial_exit_shares": 0, "full_exit_shares": 0}

    # 最高優先權：收盤跌破移動防守線，不論停利分數高低，一律強制全部出清（規格書10.3、10.4）
    if current_trailing_stop > 0 and price <= current_trailing_stop:
        result.update({"next_state": "FULL_EXIT_NEXT_DAY", "signal_type": "FULL_EXIT",
                        "signal_reason": "收盤跌破移動防守線，強制全部出清",
                        "full_exit_shares": current_shares})
        return result

    macd_confirmed_bearish = macd_osc_status == "翻黑"
    is_profitable = _safe_float(average_cost) > 0 and price > average_cost

    if not t1_taken and ((t1_price > 0 and price >= t1_price) or (macd_confirmed_bearish and is_profitable)):
        if t1_price > 0 and price >= t1_price:
            reason = f"到達第一目標 T1（{t1_price:.2f}），隔日分批停利"
            signal_type = "T1_PARTIAL_EXIT"
        else:
            reason = "MACD日線柱狀體翻黑（動能確認轉弱），尚未到達T1但提前分批停利，保留獲利"
            signal_type = "MACD_REVERSAL_T1"
        result.update({"next_state": "PARTIAL_EXIT_NEXT_DAY", "signal_type": signal_type,
                        "signal_reason": reason,
                        "partial_exit_shares": max(1, int(current_shares * _safe_float(partial_exit_ratio, 0.30)))})
        return result

    if t1_taken and not t2_taken and ((t2_price > 0 and price >= t2_price) or (macd_confirmed_bearish and is_profitable)):
        if t2_price > 0 and price >= t2_price:
            reason = f"到達第二目標 T2（{t2_price:.2f}），隔日第二段停利"
            signal_type = "T2_PARTIAL_EXIT"
        else:
            reason = "MACD日線柱狀體翻黑（動能確認轉弱），尚未到達T2但提前出清剩餘部位，保留獲利"
            signal_type = "MACD_REVERSAL_T2"
        result.update({"next_state": "PARTIAL_EXIT_NEXT_DAY", "signal_type": signal_type,
                        "signal_reason": reason,
                        "partial_exit_shares": current_shares})
        return result

    result.update({"next_state": "HOLD", "signal_reason": "持有續抱，移動防守線持續追蹤"})
    return result

def calculate_breakout_quality_score(volume, vol_ma5, macd_osc_atr_ratio, breakout_margin_atr, decision_score):
    """
    【V2.11.17新增，V2.11.30重新設計，Breakout Quality Engine（P1-1）】把 is_breakout_failed() 之外
    「這次突破強不強」的判斷，從 calculate_entry_plan() 原本的純布林關卡（過關/不過關）升級成
    0~100 的連續分數，純粹是「這次通過關卡的訊號有多強」的參考指標，**不是新的關卡**——entry_gate
    有沒有通過、要不要建立這筆計畫，完全不受這個分數影響，這裡只是替已經通過關卡的訊號多附上一個
    強度分數。

    【V2.11.30重新設計的原因】跑了兩輪真實回測（45~55筆交易）後，用逐項拆解發現舊版四個子項裡
    有兩個是「死欄位」、一個是雜訊、一個方向是反的，總分完全沒有應有的判別力：
      - 舊版「風報比子項」（20分）：V2.11.22已經把進場R1預檢整個拿掉，這個子項的輸入自此永遠是
        None，55筆交易裡固定給10分，沒有任何一次例外——這20分對總分零貢獻
      - 舊版「MACD子項」（25分）：entry_gate本身就已經要求MACD必須是「正值」或「翻紅第1根」才會
        放行進場，代表能出現在計畫裡的每一筆交易，這個子項本來就已經被entry_gate篩過一次、永遠
        滿分25——這25分本質上是在重複測試entry_gate已經測過的同一件事，對總分同樣零貢獻
      - 舊版「量能子項」（35分，佔比最高）：假設「量能倍數越高分數越高」，但回測顯示相關係數是
        負的（-0.305）——量能分數最高的10筆交易勝率只有20%，最低的10筆反而50%，方向整個是反的。
        合理的推測（非確定因果）：極端爆量可能代表「衝刺竭盡／出貨」而非「健康承接」，原本的
        線性假設本身就有問題

    這次改成三個真正有機會互相區分的子項（合計100分）：
      - 量能甜蜜點（30分）：不再是「越大越好」，改成鐘形曲線——1.0倍以下0分，1.5~1.8倍區間最健康
        給滿分30分，超過2.5倍開始隨爆量程度扣分（最極端只會腰斬到15分，不會直接歸零，避免對真正
        強勢股一竿子打死，畢竟這只是參考分數不是關卡）
      - MACD動能強度（25分，取代原本永遠滿分的死欄位）：改用柱狀體OSC的實際數值（不是「正值/
        翻紅」這種entry_gate已經篩過一次的分類），【V2.11.31修正】用ATR正規化後才給分（不是直接
        用OSC原始數值——原始數值天生跟股價水位成正比，股價幾千元的股票OSC本來就比股價幾十元的
        股票大，不正規化的話分數高低反映的是股價水位而不是動能強弱，是混淆因子）——「剛轉正、
        動能還很弱」跟「動能已經很強」這兩種都能通過entry_gate的突破訊號，在這裡才真正被區分開來
      - 突破強度（25分，取代原本永遠固定10分的死欄位r1子項）：現價收在突破價之上多遠（用ATR
        正規化），收越高代表這次突破的確認力道越強，不是勉強擦線過關
      - 決策分數強度（20分，沿用不變）：decision_score 70（gate最低門檻）給10分，90以上給滿分
        20分，中間線性內插

    五級不是這裡的重點，先只回傳連續分數＋一個簡單的A/B/C/D字母評等（A≥80／B≥65／C≥50／D<50）。
    回傳 (score, grade, detail)，detail 是子分數明細 dict，方便顯示「這個分數是怎麼組成的」。

    【需要人工確認的參數】量能甜蜜點的1.5~1.8倍區間、2.5倍衰減起點、突破強度的1.0倍ATR滿分門檻、
    MACD動能強度的0.4倍ATR滿分門檻（V2.11.31新增，OSC正規化後通常小於一整根ATR，這個門檻抓得比
    突破強度更保守），全部都是初版經驗值，這次回測顯示的方向性只有45~55筆的小樣本，之後有更多
    回測資料建議再校準一次，不要當成已經驗證過的最終版本。
    """
    def _lerp_score(value, lo, hi, max_pts, min_pts=0.0):
        if value is None:
            return max_pts / 2.0
        v = _safe_float(value)
        if v <= lo:
            return min_pts
        if v >= hi:
            return max_pts
        return min_pts + (v - lo) / (hi - lo) * (max_pts - min_pts)

    # ---- 量能甜蜜點（30分）----
    vol_ma5_f = _safe_float(vol_ma5) if vol_ma5 is not None else None
    if vol_ma5_f is not None and vol_ma5_f > 0:
        vol_ratio = _safe_float(volume) / vol_ma5_f
        if vol_ratio <= 1.0:
            vol_score = 0.0
        elif vol_ratio <= 1.5:
            vol_score = _lerp_score(vol_ratio, 1.0, 1.5, 30.0)
        elif vol_ratio <= 1.8:
            vol_score = 30.0
        elif vol_ratio <= 2.5:
            vol_score = 30.0 - _lerp_score(vol_ratio, 1.8, 2.5, 10.0)  # 30分緩降到20分
        else:
            # 2.5倍以上持續扣分，但設下限15分，避免對真正的強勢股一竿子打死
            _over = min(_safe_float(vol_ratio) - 2.5, 2.5)  # 最多再看2.5倍的超額部分（即5倍封頂）
            vol_score = max(15.0, 20.0 - _over / 2.5 * 5.0)
    else:
        vol_score = 15.0  # 資料缺失給甜蜜點區間中段偏低的中性值，不是滿分

    # ---- MACD動能強度（25分）：改用OSC實際數值，不是entry_gate已經篩過的分類 ----
    # 【V2.11.31修正】原本直接用OSC的原始數值（價格單位）去對應0~1.0這個寫死的區間，但OSC的絕對
    # 大小天生跟股價水位成正比——股價幾千元的股票（例如台光電、聯發科）OSC原始數值本來就會比
    # 股價幾十元的股票大上好幾倍，分數高低反映的是「這檔股票股價多高」而不是「這次動能有多強」，
    # 是一個混淆因子。改成跟 breakout_margin 同樣的做法：用ATR正規化，讓不同股價水位的股票可以
    # 放在同一把尺上比較（OSC本身通常小於一整根ATR，所以這裡的滿分門檻抓得比breakout_margin更
    # 保守，見下方【需要人工確認的參數】）。
    macd_score = _lerp_score(macd_osc_atr_ratio, 0.0, 0.4, 25.0, min_pts=5.0)

    # ---- 突破強度（25分）：現價收在突破價之上多少ATR ----
    breakout_score = _lerp_score(breakout_margin_atr, 0.0, 1.0, 25.0)

    decision_score_score = _lerp_score(decision_score, 70.0, 90.0, 20.0, min_pts=10.0)

    total = vol_score + macd_score + breakout_score + decision_score_score
    total = max(0.0, min(100.0, total))
    grade = "A" if total >= 80 else ("B" if total >= 65 else ("C" if total >= 50 else "D"))
    detail = {"volume": round(vol_score, 1), "macd": round(macd_score, 1), "breakout_margin": round(breakout_score, 1), "decision_score": round(decision_score_score, 1)}
    return round(total, 1), grade, detail

def calculate_entry_plan(code, indicators, portfolio_info, market_context):
    """
    空手進場計畫（規格書 7節）。indicators 需含：price, atr, previous_high, ma20, decision_score,
    trend_gate, chip_gate, volume_gate, r1, market_regime, is_us_stock, data_date。
    entry_gate 沒通過或 decision_score < 70 時回傳 None（不建立計畫，維持 PREPARE）。

    【MACD深度整合】indicators 可另外帶 macd_osc_status／macd_divergence_type（日線）。
    只要日線出現「頂背離」（價格創新高但動能已經在衰竭，假突破的典型特徵）或「翻黑」，
    就直接擋下這次建立新進場計畫——這是本次整合最核心的目的：避開假突破。
    缺這兩個欄位（資料不足/尚未計算）時視為中性，不影響進場判斷。

    【V2.11.10新增，Breakout Engine】在既有關卡之上，再加兩道「真突破」確認：
      量價確認：當日成交量 ≥ 5日均量×1.5，突破沒有放量的話容易是誘多，不核准建立計畫
      MACD確認：柱狀體要是「正值」或「翻紅第1根」才算真正確認，不只是「不逆風」這種消極不擋
    這兩項資料不足（volume/vol_ma5缺失、MACD資料不足）時視為中性通過，不會誤擋新股或資料不全的情況。

    【V2.11.22移除，真實P0 bug修復】原本這裡還有一道「r1（風報比）>=1.5」的進場前預檢，這道關卡
    在數學上永遠不可能通過，等於整個「偵測全新突破」功能自V2.11.9這道關卡加入後就完全失效——
    原因是：突破價定義為「前高×1.005」，而這道預檢拿「同一個前高」去找目標價，前高必然小於
    「前高×1.005」，導致目標價永遠掉進備援公式（突破價+2×ATR），跟停損距離（突破價−2×ATR算出
    的2×ATR風險）相除，永遠精確等於1.0，永遠小於1.5門檻，entry_gate_pass永遠是False，
    calculate_entry_plan()永遠回傳None，沒有任何一筆新交易計畫能被建立。

    跟你討論後決定拿掉這道關卡（而不是修補回溯窗口）：因為「剛突破的當下」本來就無法预先知道
    後續會漲到哪，硬要在進場前算出一個「進場後才會知道」的報酬數字，本來就是在編造精確度；
    值不值得，交給進場後持續運作的T1/T2（V2.11.18已改為R倍數+前高混合）跟分批停利機制動態決定，
    比在進場前用一個結構性有瑕疵的公式一次性判死刑更合理。
    """
    price = _safe_float(indicators.get("price"))
    atr = _safe_float(indicators.get("atr"))
    previous_high = _safe_float(indicators.get("previous_high"))
    decision_score = _safe_float(indicators.get("decision_score"))
    is_us_stock = bool(indicators.get("is_us_stock"))
    data_date = indicators.get("data_date", "")
    macd_osc_status = indicators.get("macd_osc_status")
    macd_divergence_type = indicators.get("macd_divergence_type")
    macd_blocks_entry = macd_divergence_type == "頂背離" or macd_osc_status == "翻黑"

    _volume = indicators.get("volume")
    _vol_ma5 = indicators.get("vol_ma5")
    volume_confirms = (_safe_float(_volume) >= _safe_float(_vol_ma5) * 1.5) if (_vol_ma5 is not None and _safe_float(_vol_ma5) > 0) else True
    macd_confirms = (macd_osc_status in ("正值", "翻紅第1根")) if macd_osc_status is not None else True

    entry_gate_pass = bool(
        indicators.get("trend_gate") and indicators.get("chip_gate") and indicators.get("volume_gate")
        and indicators.get("market_regime") != "BEARISH"
        and not macd_blocks_entry
        and volume_confirms
        and macd_confirms
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

    # 【V2.11.30】breakout_margin_atr：現價收在突破價之上多少倍ATR，取代已移除的r1當突破強度指標；
    # price可能還沒真的越過breakout_price（例如剛建立BREAKOUT_WAIT、還在等待階段），此時margin
    # 會是負值，calculate_breakout_quality_score內部的_lerp_score會把它壓到0分（尚無確認力道）。
    _breakout_margin_atr = (price - breakout_price) / atr if atr > 0 else None
    # 【V2.11.31修正】macd_osc_value 是價格單位的原始數值，天生跟股價水位成正比，直接拿去打分數
    # 會混淆「股價高低」跟「動能強弱」，這裡先用ATR正規化成跟breakout_margin同一把尺，才丟進去。
    _macd_osc_value = indicators.get("macd_osc_value")
    _macd_osc_atr_ratio = (abs(_macd_osc_value) / atr) if (_macd_osc_value is not None and atr > 0) else None
    _bq_score, _bq_grade, _bq_detail = calculate_breakout_quality_score(
        indicators.get("volume"), _vol_ma5, _macd_osc_atr_ratio, _breakout_margin_atr, decision_score)

    if price > chase_limit > 0:
        state = "PULLBACK_WAIT"
        valid_until = _add_business_days(data_date, 5, is_us_stock)
        reason = "現價已超過追價上限，改為等待回測區間"
    elif price >= breakout_price > 0:
        # 【V2.11.34新增，Breakout Quality Gate；V2.11.38改為依市場燈號分兩級門檻】品質分數未達
        # 門檻時，先不直接確認進場，改成BREAKOUT_WAIT繼續追蹤觀察——這裡是「這次剛好第一次偵測到
        # 時，價格已經越過突破價」的情境，這時候的品質分數是有意義的（breakout_margin_atr是真實
        # 數字，不是等待階段的0），可以直接拿來判斷。之後如果分數改善（例如量能持續放大），會在
        # evaluate_trade_state()的WAIT→ENTER_NEXT_DAY重新確認流程裡用當天最新資料重算一次品質，
        # 不是延用這裡凍結的舊分數（那個分數只在「這一天首次建立計畫」時有意義，之後每天狀況都在
        # 變，應該重新判斷）。
        # 【V2.11.38】門檻本身不再是單一固定值：市場燈號處於中性區間時，改用較高的門檻
        # （BREAKOUT_QUALITY_GATE_THRESHOLD_NEUTRAL），只放行更扎實的訊號；明確偏多以上維持
        # 原本基礎門檻。理由見該常數的說明（V2.11.37跨時段一致性檢查發現中性市場的期望值是負的）。
        _regime_scores_here = calculate_regime_score(market_context)
        _regime_score_here = _regime_scores_here["us_regime"] if is_us_stock else _regime_scores_here["tw_regime"]
        _quality_threshold_here = _breakout_quality_gate_threshold_for_regime(_regime_score_here)
        if _bq_score < _quality_threshold_here:
            state = "BREAKOUT_WAIT"
            valid_until = _add_business_days(data_date, 3, is_us_stock)
            reason = f"突破確認但品質分數{_bq_score:.0f}分未達門檻{_quality_threshold_here:.0f}分（市場燈號{_regime_score_here:.0f}分），先追蹤觀察，不強制進場"
        else:
            state = "ENTER_NEXT_DAY"
            valid_until = _add_business_days(data_date, 3, is_us_stock)
            reason = "突破確認且未超過追價上限，隔日執行進場"
    else:
        state = "BREAKOUT_WAIT"
        valid_until = _add_business_days(data_date, 3, is_us_stock)
        reason = "Gate 與 Score 同時成立，等待價格突破"

    return {
        "signal_type": "ENTRY", "entry_price": breakout_price, "breakout_price": breakout_price,
        "pullback_low": pullback_low, "pullback_high": pullback_high, "chase_limit": chase_limit,
        "invalid_price": invalid_price, "state": state, "signal_date": data_date,
        "execution_date": _next_business_day(data_date, is_us_stock) if state == "ENTER_NEXT_DAY" else "",
        "valid_until": valid_until, "signal_reason": reason,
        "suggested_shares": calculate_position_size(
            portfolio_info.get("cap", 20000.0), portfolio_info.get("risk", 5.0),
            breakout_price, breakout_price - 2 * atr, portfolio_info.get("available_cash", portfolio_info.get("cap", 20000.0))
        ),
        "initial_stop": round_to_tick(breakout_price - 2 * atr, is_us_stock),
        "breakout_quality_score": _bq_score, "breakout_quality_grade": _bq_grade,
        # 【V2.11.28新增，V2.11.30重新設計】拆開突破品質分數的子項，供事後分析各子項的判別力。
        # V2.11.30把 bq_r1 改名成 bq_breakout_margin（原本的r1子項已經是死欄位，改用突破強度取代，
        # 見 calculate_breakout_quality_score 的docstring），bq_macd 名稱沿用但底層算法已經改用
        # OSC實際數值，不再是entry_gate已經篩過的分類。
        "bq_volume": _bq_detail.get("volume", 0), "bq_macd": _bq_detail.get("macd", 0),
        "bq_breakout_margin": _bq_detail.get("breakout_margin", 0), "bq_decision_score": _bq_detail.get("decision_score", 0),
    }

def calculate_addon_shares(current_shares, current_price, current_stop, add_price, add_stop,
                            allocated_capital, risk_percent, available_cash, suggested_shares_cap=None,
                            confidence_multiplier=1.0):
    """
    加碼股數（規格書8節）。三重上限取最小值：加碼後總風險不超過 max_risk、資金餘額、以及可選的
    「原建倉股數上限」（既有 taistock_v2_9.py 已驗證過的保守設計，這裡保留但改為可選參數，
    未提供時不套用這個額外上限，行為與規格書8.2原始公式完全一致）。

    confidence_multiplier（V2.11.9新增，P0-3統一）：0~1之間的縮減比例，對應「決策信心」分級
    （100%/60%/20%/0%）。這個機制原本只存在UI端（乘在suggested_shares_adjusted裡），狀態機完全沒有，
    導致「要不要加碼」統一了、但「加碼多少股」UI跟狀態機還是兩個數字。現在UI跟狀態機都呼叫這同一個
    函式、都傳入同一個confidence_multiplier來源，兩邊算出來的加碼股數會永遠一致。
    suggested_shares_cap 現在統一約定傳「未經信心縮減的原始建倉股數的一半」，信心縮減只在這裡
    最後統一乘一次，不會被同時套用兩次。
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
    raw_shares = min(candidates) if candidates else 0
    return int(np.floor(raw_shares * _safe_float(confidence_multiplier, 1.0)))

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
            macd_osc_status=indicators.get("macd_osc_status"),
            swing_low=indicators.get("swing_low"),
            trend_confirmed=bool(indicators.get("trend_gate")),
            entry_price=plan.get("entry_price"), initial_stop=plan.get("initial_stop"),
        )
        # 【V2.11.8 修正】原本只在「第一次設定」時寫入 t1_price/t2_price，之後永遠凍結不再更新——
        # 即使已經統一成同一套公式（calculate_target_plan），只要時間拉長、前高或股價變化，
        # 畫面上凍結的舊數字還是會跟即時算出來的新數字兜不起來。改成每次評估都直接採用
        # calculate_exit_plan 剛算出來的最新值，確保「🗓️交易計畫」顯示的T1/T2永遠等於當下
        # 用同一套公式即時算出的結果，跟「🛡️風控點位」分頁不會再對不上。
        plan["t1_price"] = exit_plan["t1_price"]
        plan["t2_price"] = exit_plan["t2_price"]

        if exit_plan["next_state"] == "FULL_EXIT_NEXT_DAY":
            key = f"{code}|FULL_EXIT|{data_date}|{round(exit_plan['current_trailing_stop'], 2)}"
            if is_duplicate_signal(plan, key):
                return plan
            plan["signal_key"] = key
            plan["review_state"] = "PENDING"  # 【V2.11.12新增】任何新訊號（signal_key改變）一律重置為尚未確認
            return transition_state(plan, "FULL_EXIT_NEXT_DAY",
                                     {"current_trailing_stop": exit_plan["current_trailing_stop"], "current_trailing_stop_source": exit_plan["stop_source"],
                                      "full_exit_shares": exit_plan["full_exit_shares"],
                                      "signal_type": exit_plan["signal_type"]},
                                     data_date, exit_plan["signal_reason"])

        if exit_plan["next_state"] == "PARTIAL_EXIT_NEXT_DAY":
            key = f"{code}|{exit_plan['signal_type']}|{data_date}"
            if is_duplicate_signal(plan, key):
                return plan
            plan["signal_key"] = key
            plan["review_state"] = "PENDING"  # 【V2.11.12新增】任何新訊號（signal_key改變）一律重置為尚未確認
            return transition_state(plan, "PARTIAL_EXIT_NEXT_DAY",
                                     {"current_trailing_stop": exit_plan["current_trailing_stop"], "current_trailing_stop_source": exit_plan["stop_source"],
                                      "partial_exit_shares": exit_plan["partial_exit_shares"],
                                      "signal_type": exit_plan["signal_type"]},
                                     data_date, exit_plan["signal_reason"])

        # 逆風時：只暫停「即將要執行的加碼」，已經是 HOLD 續抱的部位完全不受影響
        if regime_bearish and plan.get("state") == "ADD_NEXT_DAY":
            return transition_state(plan, "SUSPENDED_BY_REGIME",
                                     {"current_trailing_stop": exit_plan["current_trailing_stop"], "current_trailing_stop_source": exit_plan["stop_source"]},
                                     data_date, "市場逆風，暫停加碼但保留交易計畫")

        # 逆風解除：若上次是因為加碼被暫停，恢復回 ADD_NEXT_DAY 讓使用者重新看到加碼建議
        # （實際加碼股數會在下面重新計算，不會沿用暫停當下的舊數字）。
        if not regime_bearish and plan.get("state") == "SUSPENDED_BY_REGIME" and plan.get("origin_state") == "ADD_NEXT_DAY":
            plan = transition_state(plan, "HOLD", {"current_trailing_stop": exit_plan["current_trailing_stop"], "current_trailing_stop_source": exit_plan["stop_source"]},
                                     data_date, "市場逆風解除，重新評估加碼條件")

        if not regime_bearish and portfolio_info.get("addon_quality_gate", True):
            addon_shares = calculate_addon_shares(
                held_qty, price, exit_plan["current_trailing_stop"], price, exit_plan["current_trailing_stop"],
                portfolio_info.get("cap", 20000.0), portfolio_info.get("risk", 5.0),
                max(0.0, _safe_float(portfolio_info.get("cap", 20000.0)) - held_qty * price),
                suggested_shares_cap=int(plan.get("suggested_shares", 0) * 0.5) if plan.get("suggested_shares", 0) > 0 else None,
                confidence_multiplier=portfolio_info.get("confidence_multiplier", 1.0),
            )
            if addon_shares > 0:
                key = f"{code}|ADD|{data_date}|{addon_shares}"
                if not is_duplicate_signal(plan, key):
                    plan["signal_key"] = key
                    plan["review_state"] = "PENDING"  # 【V2.11.12新增】任何新訊號（signal_key改變）一律重置為尚未確認
                    return transition_state(plan, "ADD_NEXT_DAY",
                                             {"current_trailing_stop": exit_plan["current_trailing_stop"], "current_trailing_stop_source": exit_plan["stop_source"],
                                              "addon_shares_suggested": addon_shares,
                                              "addon_shares_approved": addon_shares, "signal_type": "ADD"},
                                             data_date, f"SOP三燈/信心/價格間距等品質關卡與資金風險上限均已通過，約可加碼 {addon_shares} 股")

        # 【V2.11.32新增】從 FULL_EXIT_NEXT_DAY 恢復到 HOLD 時，原本的 exit_plan["signal_reason"]
        # 是空字串（見 calculate_exit_plan 的result初始化），而 transition_state() 只在 reason
        # 非空時才會覆蓋 plan["signal_reason"]——這代表如果不特別處理，畫面上會繼續顯示上一個
        # 「收盤跌破移動防守線，強制全部出清」的舊文字，即使狀態已經改回HOLD，非常容易誤導使用者。
        # 這裡明確給一個誠實反映「怎麼回事」的新理由文字。
        _hold_reason = exit_plan["signal_reason"]
        if plan.get("state") == "FULL_EXIT_NEXT_DAY" and not _hold_reason:
            _hold_reason = "先前的出清訊號未被執行，現價已回到防守線之上，重新評估為持有中（若之後再度跌破，會產生新的出清訊號）"
        return transition_state(plan, "HOLD",
                                 {"current_trailing_stop": exit_plan["current_trailing_stop"], "current_trailing_stop_source": exit_plan["stop_source"],
                                  "remaining_shares": held_qty, "addon_shares_approved": 0},
                                 data_date, _hold_reason)

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
             "execution_date": "", "valid_until": "", "retest_min_price": 0.0, "retest_quality": "",
             "breakout_quality_score": 0.0, "breakout_quality_grade": "",
             "bq_volume": 0.0, "bq_macd": 0.0, "bq_breakout_margin": 0.0, "bq_decision_score": 0.0},
            data_date, f"部位已全部出清（原狀態 {plan.get('state')}），重置為 PREPARE 重新追蹤新訊號")

    if plan.get("state") in active_wait_states and plan.get("entry_price", 0) > 0:
        if plan.get("state") != "PREPARE" and is_signal_expired(plan, data_date):
            return transition_state(plan, "EXPIRED", {}, data_date, "交易計畫超過有效期限")
        # 【V2.11.10新增，Breakout Engine】比失效價更早的一道防線：突破後隔日站不穩就先判定失敗，
        # 不用等真的崩到更寬的失效價才反應。只在還沒真正持有時檢查（這裡本來就是空手分支）。
        if is_breakout_failed(plan, price, indicators.get("volume"), indicators.get("vol_ma5"),
                               indicators.get("atr"), indicators.get("prev_close")):
            return transition_state(plan, "BREAKOUT_FAILED", {}, data_date,
                                     f"收盤跌破突破價 {plan.get('breakout_price'):.2f} 且量能萎縮或單日跌幅過大，判定突破失敗")
        if is_signal_invalid(plan, price):
            return transition_state(plan, "INVALID", {}, data_date, f"現價跌破失效價 {plan.get('invalid_price'):.2f}，訊號條件已被破壞")

        # 【修正】建議股數（suggested_shares）原本只在「建立這個計畫的當下」算一次就凍結，之後
        # 使用者調整資金／風險% 都不會反映在畫面上，容易誤以為設定沒生效。這裡讓它在計畫還處於
        # 等待狀態（尚未實際執行）時，每次評估都用「當下」的資金/風險%/ATR 重新計算，不會凍結。
        # entry_price 是已經鎖定的突破價，不會因為重算股數而跟著變動，只有股數本身會即時跟上設定。
        _recalc_stop = plan.get("entry_price", 0) - 2 * _safe_float(indicators.get("atr"))
        plan["suggested_shares"] = calculate_position_size(
            portfolio_info.get("cap", 20000.0), portfolio_info.get("risk", 5.0),
            plan.get("entry_price", 0), _recalc_stop,
            portfolio_info.get("available_cash", portfolio_info.get("cap", 20000.0)),
        )

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

        # 【V2.11.17新增，Retest Engine】追蹤PULLBACK_WAIT期間價格是否真的回測進等待區間，純粹記錄用，
        # 不新增狀態、不影響任何轉移判斷——只在下面真的轉去ENTER_NEXT_DAY那一刻，用這份記錄附上
        # 「這次突破前有沒有經過真實回測」的品質提示，不是「來回擺盪誤觸發」也不是「原地空等」。
        if plan.get("state") == "PULLBACK_WAIT" and price > 0:
            _prev_min = _safe_float(plan.get("retest_min_price"))
            plan["retest_min_price"] = price if _prev_min <= 0 else min(_prev_min, price)

        if plan.get("state") in {"BREAKOUT_WAIT", "PULLBACK_WAIT"} and entry_price > 0 and price >= entry_price:
            # 【V2.11.34新增，Breakout Quality Gate】用「今天」的最新資料重新算一次品質分數，不是
            # 沿用建立BREAKOUT_WAIT/PULLBACK_WAIT當時凍結的舊分數——那是計畫剛建立那天的快照，
            # 量能/MACD/決策分數每天都在變，確認進場前應該看「現在」的品質，不是好幾天前的舊資料。
            _atr_now = _safe_float(indicators.get("atr"))
            _reconfirm_margin_atr = (price - entry_price) / _atr_now if _atr_now > 0 else None
            _reconfirm_macd_value = indicators.get("macd_osc_value")
            _reconfirm_macd_ratio = (abs(_reconfirm_macd_value) / _atr_now) if (_reconfirm_macd_value is not None and _atr_now > 0) else None
            _reconfirm_score, _reconfirm_grade, _reconfirm_detail = calculate_breakout_quality_score(
                indicators.get("volume"), indicators.get("vol_ma5"), _reconfirm_macd_ratio, _reconfirm_margin_atr, indicators.get("decision_score"))
            # 【V2.11.38新增】門檻依當下市場燈號分數分兩級，理由見
            # _breakout_quality_gate_threshold_for_regime() 的說明（V2.11.37跨時段一致性檢查
            # 發現中性市場的期望值是負的）。
            _reconfirm_regime_scores = calculate_regime_score(market_context)
            _reconfirm_regime_score = _reconfirm_regime_scores["us_regime"] if is_us_stock else _reconfirm_regime_scores["tw_regime"]
            _reconfirm_threshold = _breakout_quality_gate_threshold_for_regime(_reconfirm_regime_score)
            if _reconfirm_score < _reconfirm_threshold:
                # 品質還沒到，留在原狀態（BREAKOUT_WAIT／PULLBACK_WAIT）繼續觀察，不強制進場、
                # 也不砍掉這筆追蹤。既有的 is_signal_invalid（跌破失效價）／valid_until（有效期限
                # 到期）機制完全不受影響、照常運作，不會因為這道品質關卡而讓計畫無限期卡死。
                return plan

            key = f"{code}|ENTRY|{data_date}|{round(entry_price, 2)}"
            if is_duplicate_signal(plan, key):
                return plan
            plan["signal_key"] = key
            plan["review_state"] = "PENDING"  # 【V2.11.12新增】任何新訊號（signal_key改變）一律重置為尚未確認
            _retest_note, _retest_quality = ("", "")
            if plan.get("state") == "PULLBACK_WAIT":
                _retest_note, _retest_quality = classify_retest_quality(
                    plan.get("retest_min_price"), plan.get("pullback_low"), plan.get("pullback_high"), plan.get("invalid_price"))
            _reason = "突破/回測進場條件成立，隔日執行"
            if _retest_note:
                _reason += f"（{_retest_note}）"
            return transition_state(plan, "ENTER_NEXT_DAY",
                                     {"execution_date": _next_business_day(data_date, is_us_stock), "retest_quality": _retest_quality,
                                      # 用重新確認當下的分數覆蓋掉建立計畫當天的舊快照，反映真正進場那一刻的品質；
                                      # 子項明細一併更新，避免「總分是新的、拆解明細卻是舊的」互相對不上
                                      "breakout_quality_score": _reconfirm_score, "breakout_quality_grade": _reconfirm_grade,
                                      "bq_volume": _reconfirm_detail.get("volume", 0), "bq_macd": _reconfirm_detail.get("macd", 0),
                                      "bq_breakout_margin": _reconfirm_detail.get("breakout_margin", 0), "bq_decision_score": _reconfirm_detail.get("decision_score", 0)},
                                     data_date, _reason)
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
        plan["review_state"] = "PENDING"  # 【V2.11.12新增】任何新訊號（signal_key改變）一律重置為尚未確認
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

# ===== V2.11.22 新增：策略回測引擎（P2）=====
# 目的：把歷史資料逐日餵給「跟即時系統完全相同」的決策函式（calculate_daily_score／
# calculate_entry_plan／evaluate_trade_state／calculate_exit_plan），重演系統過去每一天會做出
# 的判斷，統計出勝率、Profit Factor、Expectancy 等真正有意義的數字，取代「猜參數」。

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_backtest_indicator_frame(code, years=2):
    """
    抓取單一股票的歷史OHLCV，算出回測需要的完整「因果」（只看過去、不看未來）指標序列。
    ticker判斷邏輯沿用 fetch_stock_data_extended()，確保跟即時系統抓到同一個標的。
    所有指標都用 rolling/ewm（天生只看過去）向量化算好整段序列一次，等同於「每天用當天以前的
    資料重新算一次」，但快非常多——這是安全的，因為 pandas 的 rolling().mean() 在第 t 列只會
    用到第 t 列以前(含)的資料，不會偷看未來。

    回傳 (df, raw, error)：df 是處理過的因果指標框（小寫欄位），raw 是原始 yfinance 格式
    （大寫Open/High/Low/Close，供 MACDStrategyAnalyzer 直接使用），error 是失敗原因文字。

    【V2.11.42修正】明確加上 auto_adjust=True，理由跟 fetch_stock_data() 相同（見該函式
    docstring），但這裡影響更大：回測跨度長達2年，遠比即時系統的6個月／週線的2年更容易涵蓋到
    除息事件，如果沒有正確還原股利，回測算出來的MA/ATR/前高/波段低點在除息日附近會被扭曲，
    可能產生不該出現的假訊號，直接影響回測結果的可信度——這是我們一路驗證回測系統時，
    之前沒有檢查過的一個資料正確性缺口。
    """
    try:
        is_us = code.isalpha() or code.endswith('.US')
        period = f"{years}y" if years != 2 else "2y"
        if is_us:
            raw = yf.download(code.replace('.US', ''), period=period, progress=False, auto_adjust=True)
        elif code.endswith('.TW') or code.endswith('.TWO'):
            raw = yf.download(code, period=period, progress=False, auto_adjust=True)
        else:
            raw = yf.download(f"{code}.TW", period=period, progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                raw = yf.download(f"{code}.TWO", period=period, progress=False, auto_adjust=True)
        raw = _trim_trailing_nan_rows(raw)
        if raw is None or raw.empty or len(raw) < 90:
            return None, None, f"歷史資料不足（僅{0 if raw is None else len(raw)}筆，至少需要90筆才能起算）"

        c, h, l, o = raw['Close'].squeeze(), raw['High'].squeeze(), raw['Low'].squeeze(), raw['Open'].squeeze()
        v = raw.get('Volume', pd.Series(0, index=raw.index)).squeeze()
        if isinstance(c, pd.DataFrame): c, h, l, o, v = c.iloc[:, 0], h.iloc[:, 0], l.iloc[:, 0], o.iloc[:, 0], v.iloc[:, 0]

        df = pd.DataFrame(index=raw.index)
        df['open'], df['high'], df['low'], df['close'], df['volume'] = o, h, l, c, v
        df['ma10'] = c.rolling(10).mean()
        df['ma20'] = c.rolling(20).mean()
        df['ma60'] = c.rolling(60).mean()
        df['atr'] = calc_atr_series(h, l, c, period=14)
        df['vol_ma5'] = v.rolling(5).mean()
        k_series, d_series = calc_kd(h, l, c)
        df['k'], df['d'] = k_series, d_series
        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -1 * delta.clip(upper=0).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (up / (down + 0.001))))
        dif, _dea, _osc = calc_macd_full_series(c)
        df['macd'] = dif
        df['bias'] = (c - df['ma60']) / df['ma60'] * 100
        # previous_high／swing_low：跟即時系統 h.iloc[-61:-1]／l.iloc[-21:-1]（過去N天不含今日）
        # 邏輯完全一致，改用 rolling+shift 向量化寫法
        df['previous_high'] = h.rolling(60).max().shift(1)
        df['swing_low'] = l.rolling(20).min().shift(1)
        df['pivot_point'] = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
        df['is_us_stock'] = is_us
        return df, raw, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_institutional_history(code, years=2):
    """
    【回測專用】TW股法人籌碼歷史重建。即時系統的 get_institutional_data() 寫死只抓「最近30天」，
    沒辦法回答「兩年前某一天法人連買幾天」。這裡改成一次抓整個回測區間的原始法人買賣超資料，
    用跟 get_institutional_data() 裡 calc_trend() 完全同樣的規則（net_buy>0且streak>=0則
    streak+=1；net_buy<0且streak<=0則streak-=1；規則被打破就從新的一天重新起算），改寫成正向
    掃描（由舊到新），一次算出每一天的 days／accumulated_shares，數學上等效於「如果那天呼叫
    get_institutional_data()會得到的答案」，只是一次性抓取＋本地計算，避免對FinMind API發出
    數千次請求。美股不使用籌碼資料（回傳空表），跟 get_institutional_data() 對美股的行為一致。
    """
    is_us = code.isalpha() or code.endswith('.US')
    if is_us:
        return pd.DataFrame(columns=['days', 'accumulated_shares']), None
    try:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=int(years * 365.25) + 30)).strftime("%Y-%m-%d")
        _plain_code = code.replace('.TW', '').replace('.TWO', '')
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": _plain_code, "start_date": start_date, "end_date": end_date}
        resp = requests.get(url, params=parameter, timeout=15)
        data = resp.json()
        if data.get("msg") != "success" or not data.get("data"):
            return pd.DataFrame(columns=['days', 'accumulated_shares']), "FinMind無資料或請求失敗"

        df_inst = pd.DataFrame(data["data"])
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        daily_net = df_inst.groupby('date')['net_buy'].sum().sort_index(ascending=True)

        days, accumulated_shares = 0, 0.0
        rows = []
        for date_key, net_buy in daily_net.items():
            net_buy = float(net_buy)
            if net_buy > 0 and days >= 0:
                days += 1; accumulated_shares += net_buy
            elif net_buy < 0 and days <= 0:
                days -= 1; accumulated_shares += net_buy
            else:
                days = 1 if net_buy > 0 else (-1 if net_buy < 0 else 0)
                accumulated_shares = net_buy
            rows.append({'date': date_key, 'days': days, 'accumulated_shares': accumulated_shares})
        out = pd.DataFrame(rows).set_index('date')
        out.index = pd.to_datetime(out.index)
        return out, None
    except Exception as e:
        return pd.DataFrame(columns=['days', 'accumulated_shares']), str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_regime_history(years=2):
    """
    【回測專用】抓取 ^TWII／^IXIC／^VIX 的歷史 price／MA20，回傳格式對齊 fetch_macro_data()
    （只是整段歷史區間，不是只有「現在」）。刻意不在這裡就先算出 regime score——讓
    evaluate_trade_state() 內部照樣呼叫 _regime_is_bearish()→calculate_regime_score()，
    用跟即時系統完全相同的呼叫鏈，這裡只準備原始輸入，不重複計算分數邏輯本身。
    """
    try:
        period = f"{years}y" if years != 2 else "2y"
        tickers = {'TW': '^TWII', 'US': '^IXIC', 'VIX': '^VIX'}
        raw = {}
        for key, symbol in tickers.items():
            _df = yf.download(symbol, period=period, progress=False)
            _df = _trim_trailing_nan_rows(_df)
            if _df is None or _df.empty:
                return None, f"{key}({symbol}) 歷史資料抓取失敗"
            _c = _df['Close'].squeeze()
            if isinstance(_c, pd.DataFrame): _c = _c.iloc[:, 0]
            raw[key] = pd.DataFrame({'price': _c, 'ma20': _c.rolling(20).mean()})
        all_dates = raw['TW'].index.union(raw['US'].index).union(raw['VIX'].index)
        for key in raw:
            raw[key] = raw[key].reindex(all_dates).ffill()
        merged = pd.DataFrame(index=all_dates)
        for key in raw:
            merged[f'{key}_price'] = raw[key]['price']
            merged[f'{key}_ma20'] = raw[key]['ma20']
        return merged.sort_index(), None
    except Exception as e:
        return None, str(e)

def _simulate_stock_backtest(code, ind_df, raw_df, inst_df, regime_df, cap=20000.0, risk_pct=5.0):
    """
    對單一股票逐日重演狀態機。核心原則：
      - 每一天只看得到「當天以前」的資料（ind_df每一列本身就是因果的，rolling/ewm天生只看過去）
      - 呼叫跟即時系統完全同一套函式（calculate_daily_score／calculate_entry_plan／
        evaluate_trade_state／calculate_exit_plan），不另外寫一份簡化邏輯
      - 訊號在「今天收盤」產生，用「隔天開盤價」模擬實際成交，對齊你系統「盤後決策、隔日執行」
        的定位
      - qty／cost 是這個函式自己管理的模擬帳本，不會動到真實的 Google Sheet

    回傳 (trades, error)：trades 是完整交易記錄的 list of dict；error 是中途失敗原因（若有）。
    """
    try:
        plan = _trade_plan_defaults(code)
        qty, cost = 0, 0.0
        is_us_stock = bool(ind_df['is_us_stock'].iloc[0]) if len(ind_df) else False
        trades = []
        open_trade = None

        dates = ind_df.index
        warmup = 60
        n = len(dates)
        for i in range(warmup, n - 1):
            row = ind_df.iloc[i]
            if pd.isna(row[['ma10', 'ma20', 'ma60', 'atr', 'k', 'd', 'rsi']]).any():
                continue
            price = float(row['close'])
            today = dates[i]
            next_open = float(ind_df.iloc[i + 1]['open'])
            if pd.isna(next_open) or next_open <= 0:
                continue
            today_str = today.strftime('%Y-%m-%d')

            # 【V2.11.35新增，MFE/MAE】如果現在有持有中的部位，用「今天」的High/Low更新這筆交易
            # 持有期間曾經到過的最高/最低價——這裡用的是進場後第一次輪到「今天」被當作當下日期處理
            # 的那一天開始（也就是實際成交隔天起），不含成交前的等待期間（那時候還沒有部位）。
            if open_trade is not None:
                _day_high = _safe_float(row.get('high'))
                _day_low = _safe_float(row.get('low'))
                if _day_high > 0:
                    open_trade['mfe_price'] = max(open_trade['mfe_price'], _day_high)
                if _day_low > 0:
                    open_trade['mae_price'] = min(open_trade['mae_price'], _day_low)

            if not is_us_stock and today in inst_df.index:
                inst = {'days': int(inst_df.loc[today, 'days']), 'accumulated_shares': float(inst_df.loc[today, 'accumulated_shares'])}
            else:
                inst = {'days': 0, 'accumulated_shares': 0.0}  # 跟 get_institutional_data() 失敗時的預設值一致

            if regime_df is not None and today in regime_df.index:
                rr = regime_df.loc[today]
                market_context = {
                    'TW': {'price': float(rr['TW_price']), 'ma20': float(rr['TW_ma20'])},
                    'US': {'price': float(rr['US_price']), 'ma20': float(rr['US_ma20'])},
                    'VIX': {'price': float(rr['VIX_price']), 'ma20': float(rr['VIX_ma20'])},
                }
            else:
                market_context = {'TW': {}, 'US': {}, 'VIX': {}}

            # MACD日線訊號：用跟即時系統同一個 MACDStrategyAnalyzer，餵入「截至今天」的原始OHLCV
            _slice = raw_df.iloc[:i + 1]
            _macd_result = macd_analyzer.analyze(code, code, _slice, "日線") if len(_slice) >= macd_analyzer.min_bars else None
            macd_osc_status = _macd_result.osc_status if _macd_result else None
            macd_divergence_type = _macd_result.divergence_type if _macd_result else None
            macd_osc_value = _macd_result.osc if _macd_result else None

            atr, ma20, ma60, ma10 = float(row['atr']), float(row['ma20']), float(row['ma60']), float(row['ma10'])
            previous_high = float(row['previous_high']) if not pd.isna(row['previous_high']) else price
            swing_low = float(row['swing_low']) if not pd.isna(row['swing_low']) else 0.0

            t1, t2, _branch = calculate_target_plan(price, cost, atr, previous_high, is_us_stock,
                                                      entry_price=plan.get('entry_price'), initial_stop=plan.get('initial_stop'))
            trend_confirmed = price > ma20 and (ma10 > ma20 > ma60)
            prev_stop_for_calc = plan.get('current_trailing_stop') or plan.get('initial_stop')
            atr_stop_price, _stop_source = calculate_stop_plan(price, cost, atr, ma20, prev_stop_for_calc, swing_low=swing_low, trend_confirmed=trend_confirmed)

            _score = calculate_daily_score(
                price, cost, ma10, ma20, ma60, float(row['macd']), float(row['bias']),
                float(row['k']), float(row['d']), float(row['rsi']), float(row['volume']), float(row['vol_ma5']),
                atr_stop_price, t1, float(row['pivot_point']) if not pd.isna(row['pivot_point']) else price, inst, is_us_stock,
                tw_bearish=bool(market_context['TW'].get('price', 1) < market_context['TW'].get('ma20', 0)) if market_context['TW'] else False,
                us_bearish=bool(market_context['US'].get('price', 1) < market_context['US'].get('ma20', 0)) if market_context['US'] else False,
                vix_high=bool(market_context.get('VIX', {}).get('price', 0) >= 25),
            )
            decision_score = _score['ai_score']

            # 【V2.11.22移除】原本這裡算的「進場用R1」在數學上永遠精確等於1.0（見
            # calculate_entry_plan docstring 的完整推導），已跟即時系統同步拿掉這道預檢，
            # 這裡也同步不再計算，直接傳 None，跟即時系統維持同一套邏輯。
            _entry_r1 = None

            indicators = {
                "code": code, "price": price, "atr": atr, "ma20": ma20, "previous_high": previous_high,
                "decision_score": decision_score, "trend_gate": _score['step3_pass'], "chip_gate": _score['step1_pass'],
                "volume_gate": _score['step2_pass'], "r1": _entry_r1, "is_us_stock": is_us_stock, "data_date": today_str,
                "macd_osc_status": macd_osc_status, "macd_divergence_type": macd_divergence_type,
                "macd_osc_value": macd_osc_value,
                "swing_low": swing_low, "volume": float(row['volume']), "vol_ma5": float(row['vol_ma5']),
                "prev_close": float(ind_df.iloc[i - 1]['close']) if i > 0 else price,
            }
            portfolio_info = {"cost": cost, "cap": cap, "risk": risk_pct, "qty": qty,
                               "available_cash": max(0.0, cap - qty * price), "addon_quality_gate": True, "confidence_multiplier": 1.0}

            new_plan = evaluate_trade_state(plan, indicators, market_context, portfolio_info)
            state = new_plan.get('state')

            if state == 'ENTER_NEXT_DAY':
                fill_qty = int(new_plan.get('suggested_shares', 0))
                if fill_qty > 0:
                    qty, cost = fill_qty, next_open
                    # 【V2.11.26新增】記錄進場當下的市場燈號分數，供事後分析「虧損交易是不是集中在
                    # 大盤本身就轉弱的時段」——用進場決策當天（i）的 market_context 算分數，不是
                    # 隔天實際成交那天，因為決策本身是「今天收盤」做的，這才是當時系統看到的市場環境。
                    _entry_regime = calculate_regime_score(market_context)
                    open_trade = {'code': code, 'entry_date': dates[i + 1], 'entry_price': next_open,
                                   'breakout_quality_score': new_plan.get('breakout_quality_score', 0),
                                   # 【V2.11.28新增，V2.11.30改名】子項分數，供事後分析哪個子項在拖累整體判別力
                                   'bq_volume': new_plan.get('bq_volume', 0), 'bq_macd': new_plan.get('bq_macd', 0),
                                   'bq_breakout_margin': new_plan.get('bq_breakout_margin', 0), 'bq_decision_score': new_plan.get('bq_decision_score', 0),
                                   'retest_quality': new_plan.get('retest_quality', ''), 'adds': 0, 'partial_exits': [], 'is_us_stock': is_us_stock,
                                   'entry_regime_tw': _entry_regime['tw_regime'], 'entry_regime_us': _entry_regime['us_regime'],
                                   'entry_regime_overview': _entry_regime['overview'],
                                   # 【V2.11.35新增，MFE/MAE】用進場成交價當起點，之後每天用當天的
                                   # High/Low更新，追蹤持有期間「最高曾經浮盈到哪」跟「最深曾經浮虧到哪」。
                                   'mfe_price': next_open, 'mae_price': next_open}
                    new_plan = dict(new_plan); new_plan['state'] = 'HOLD'
                else:
                    new_plan = dict(new_plan); new_plan['state'] = 'PREPARE'
            elif state == 'ADD_NEXT_DAY':
                add_qty = int(new_plan.get('addon_shares_approved', 0))
                if add_qty > 0 and qty > 0:
                    new_cost = (cost * qty + next_open * add_qty) / (qty + add_qty)
                    qty, cost = qty + add_qty, new_cost
                    if open_trade:
                        open_trade['adds'] += 1
                        # 【V2.11.27新增】記錄每一次加碼當下的市場燈號分數，跟進場時同一個算法
                        # （用決策當天i的market_context），供分析「加碼是不是發生在市場正在轉弱
                        # 的時段」——這是比「進場當下分數」更精確的問題：V2.11.26驗證過進場分數
                        # 跟輸贏基本無關，因為進場後市場才轉壞、進場分數量不到；加碼卻是「持有期間」
                        # 才發生的決策，理論上應該要能反映當下市場環境轉差與否。
                        _add_regime = calculate_regime_score(market_context)
                        _add_regime_score = _add_regime['us_regime'] if is_us_stock else _add_regime['tw_regime']
                        open_trade.setdefault('add_regime_scores', []).append(round(_add_regime_score, 1))
                new_plan = dict(new_plan); new_plan['state'] = 'HOLD'
            elif state == 'PARTIAL_EXIT_NEXT_DAY':
                exit_qty = min(int(new_plan.get('partial_exit_shares', 0)), qty)
                if exit_qty > 0:
                    if open_trade: open_trade['partial_exits'].append({'date': dates[i + 1], 'price': next_open, 'qty': exit_qty})
                    qty -= exit_qty
                new_plan = dict(new_plan)
                new_plan['state'] = 'HOLD' if qty > 0 else 'PREPARE'
            elif state == 'FULL_EXIT_NEXT_DAY':
                if open_trade and qty > 0:
                    open_trade['exit_date'], open_trade['exit_price'], open_trade['exit_qty'] = dates[i + 1], next_open, qty
                    open_trade['exit_reason'] = new_plan.get('signal_reason', '')
                    trades.append(open_trade)
                open_trade = None
                qty, cost = 0, 0.0
                new_plan = dict(new_plan); new_plan['state'] = 'PREPARE'

            plan = new_plan

        # 回測結束時仍持有部位：以最後一天收盤價強制平倉結算，不然這筆交易無法統計進報表
        if open_trade and qty > 0:
            open_trade['exit_date'], open_trade['exit_price'], open_trade['exit_qty'] = dates[-1], float(ind_df.iloc[-1]['close']), qty
            open_trade['exit_reason'] = '回測結束強制平倉（尚未實際出場）'
            trades.append(open_trade)

        return trades, None
    except Exception as e:
        return [], str(e)

def _aggregate_backtest_trades(trade):
    """把一筆 open_trade 記錄（含分批加碼/分批出場）換算成單筆交易的績效指標。"""
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    holding_days = (pd.Timestamp(trade['exit_date']) - pd.Timestamp(trade['entry_date'])).days
    pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
    t1_hit = len(trade.get('partial_exits', [])) >= 1
    t2_hit = len(trade.get('partial_exits', [])) >= 2
    _add_scores = trade.get('add_regime_scores', [])
    # 【V2.11.35新增，MFE/MAE】把持有期間追蹤到的最高/最低價，換算成相對進場價的百分比。
    # MFE（Maximum Favorable Excursion）：這筆交易期間「曾經浮盈到最多」是多少%，用來檢驗
    # 「停利設定是不是抓在對的位置」（例如：如果很多交易MFE遠高於實際出場的獲利%，代表利潤
    # 沒有被有效鎖住，回吐了很多）。
    # MAE（Maximum Adverse Excursion）：這筆交易期間「曾經浮虧到最深」是多少%（負值），用來
    # 檢驗「停損設定是不是抓在對的位置」（例如：如果最終獲利的交易，MAE也顯示曾經大幅虧損過
    # 又拉回來，代表停損可能設太緊、容易把之後會成功的交易洗掉；反過來說，如果最終虧損的交易
    # MAE都很淺，代表停損抓得早、防守有發揮作用）。
    _mfe_price = trade.get('mfe_price', entry_price)
    _mae_price = trade.get('mae_price', entry_price)
    mfe_pct = (_mfe_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
    mae_pct = (_mae_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
    return {
        'code': trade['code'], 'entry_date': trade['entry_date'], 'exit_date': trade['exit_date'],
        'entry_price': entry_price, 'exit_price': exit_price, 'pnl_pct': pnl_pct,
        'mfe_pct': round(mfe_pct, 2), 'mae_pct': round(mae_pct, 2),
        'holding_days': holding_days, 'adds': trade.get('adds', 0), 'partial_exits': len(trade.get('partial_exits', [])),
        't1_hit': t1_hit, 't2_hit': t2_hit, 'breakout_quality_score': trade.get('breakout_quality_score', 0),
        # 【V2.11.34修正】retest_quality 原本在回測裡一直是空字串（寫死，忘記接上實際算出來的值），
        # 這次修正為讀取 evaluate_trade_state() 真正算出的分類（RETESTED_HELD／NO_RETEST／
        # INVALID_TOUCHED／空字串代表這筆交易不是走PULLBACK_WAIT路徑，沒有回測品質可判斷）。
        'retest_quality': trade.get('retest_quality', ''),
        # 【V2.11.28新增，V2.11.30改名】突破品質分數的子項，供分析哪個子項在拖累整體判別力。
        'bq_volume': trade.get('bq_volume', 0), 'bq_macd': trade.get('bq_macd', 0),
        'bq_breakout_margin': trade.get('bq_breakout_margin', 0), 'bq_decision_score': trade.get('bq_decision_score', 0),
        # 【V2.11.26新增】進場當下的市場燈號分數，供分析「虧損是不是集中在大盤本身轉弱的時段」。
        # 依股票是台股/美股，取對應的那個子分數（跟 _regime_is_bearish() 的判斷依據一致）。
        'entry_regime_score': round(trade.get('entry_regime_us' if trade.get('is_us_stock') else 'entry_regime_tw', 0), 1),
        'entry_regime_overview': round(trade.get('entry_regime_overview', 0), 1),
        # 【V2.11.27新增】這筆交易期間每一次加碼當下的市場燈號分數，取「最低分那次」跟「平均」——
        # 最低分最能回答「這筆交易有沒有在市場明顯轉弱時還繼續加碼」這個問題；沒有加碼過的交易
        # （adds=0）這兩欄是 None，不是0分，避免跟「加碼時剛好遇到0分」混淆。
        'add_regime_score_min': round(min(_add_scores), 1) if _add_scores else None,
        'add_regime_score_avg': round(sum(_add_scores) / len(_add_scores), 1) if _add_scores else None,
        'exit_reason': trade.get('exit_reason', ''), 'win': pnl_pct > 0,
    }

def calculate_backtest_metrics(trades_df):
    """
    P2回測指標（規格書三十五節12個問題的具體實作）：勝率／平均獲利／平均虧損／Profit Factor／
    Expectancy／平均持有天數／T1命中率／T2命中率／停損率（虧損出場佔比）。
    最大回撤（基於逐筆交易累積報酬%的簡化權益曲線，不是真實資金回撤，見UI端的說明文字）跟
    假突破率（P2-9，需要對照 BREAKOUT_FAILED 狀態，這次先不含在這份指標裡，等下一輪擴充）。

    【V2.11.26新增】avg_regime_score_win／avg_regime_score_loss：贏的交易 vs 輸的交易，進場當下
    平均的市場燈號分數各是多少。用來直接檢驗「虧損是不是集中在大盤本身轉弱的時段」這個假設——
    如果輸的交易平均進場分數明顯低於贏的交易，代表逆風攔截的門檻可能設得不夠嚴；如果兩者分數
    差不多，代表虧損不是市場環境造成的，是個股/訊號本身的問題。

    【V2.11.35新增，MFE/MAE總覽】用來檢驗「停損/停利設定到底抓得準不準」，這是30年資深操盤手
    校準風控參數最基本的工具，比單純調整倍數參數更有根據：
      - avg_mfe_win／avg_win_pct 的差距（「獲利回吐」缺口）：贏的交易平均「曾經浮盈到最多」是
        多少%，跟「實際出場拿到」多少%之間的差距。如果缺口很大，代表移動防守線可能太鬆，
        利潤在拉回時被吐回去太多，沒有確實鎖住；缺口很小，代表停利機制抓得緊，獲利保護得好。
      - avg_mae_loss：輸的交易平均「曾經浮虧到最深」是多少%（負值）。理論上這個數字應該
        接近初始停損的理論距離（約-2×ATR換算成的百分比）；如果實際MAE比理論停損距離深很多，
        代表停損可能沒有確實執行、或有跳空造成的滑價；如果MAE遠比理論停損距離淺，代表停損
        可能設太緊，還沒真正跌到理論停損價，其他機制（例如MACD翻黑）就先出場了。
      - avg_mae_win：贏的交易在最終獲利之前，平均「曾經浮虧到最深」是多少%。如果這個數字
        明顯偏深（例如超過-5%），代表這些最終賺錢的交易一路上經歷過不小的帳面虧損才翻正——
        如果停損設更緊，這些交易可能會提早被洗出場、變成虧損，這是判斷「停損是不是設太緊」
        的直接證據（比單純猜測「10天內停損的都是0%勝率」更精確）。
    """
    if trades_df is None or trades_df.empty:
        return None
    wins = trades_df[trades_df['win']]
    losses = trades_df[~trades_df['win']]
    win_rate = len(wins) / len(trades_df) * 100
    avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0.0
    gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    equity_curve = (1 + trades_df.sort_values('exit_date')['pnl_pct'] / 100).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    _has_mfe_mae = 'mfe_pct' in trades_df.columns and 'mae_pct' in trades_df.columns
    return {
        'total_trades': len(trades_df), 'win_rate': win_rate, 'avg_win_pct': avg_win, 'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor, 'expectancy_pct': expectancy, 'avg_holding_days': trades_df['holding_days'].mean(),
        't1_hit_rate': trades_df['t1_hit'].mean() * 100, 't2_hit_rate': trades_df['t2_hit'].mean() * 100,
        'stop_rate': (1 - win_rate / 100) * 100, 'max_drawdown_pct': max_drawdown, 'equity_curve': equity_curve,
        'avg_regime_score_win': wins['entry_regime_score'].mean() if len(wins) > 0 and 'entry_regime_score' in trades_df.columns else None,
        'avg_regime_score_loss': losses['entry_regime_score'].mean() if len(losses) > 0 and 'entry_regime_score' in trades_df.columns else None,
        'avg_mfe_win': wins['mfe_pct'].mean() if len(wins) > 0 and _has_mfe_mae else None,
        'avg_mae_win': wins['mae_pct'].mean() if len(wins) > 0 and _has_mfe_mae else None,
        'avg_mae_loss': losses['mae_pct'].mean() if len(losses) > 0 and _has_mfe_mae else None,
    }

def calculate_walk_forward_split(trades_df, n_periods=2):
    """
    【V2.11.37新增】把交易明細依「進場日期」切成N個時間段，各自完整算一次
    calculate_backtest_metrics()，用來檢驗策略表現是不是集中在某一段特定期間（過度擬合的
    警訊），還是在不同時間段都維持一致的方向。

    【重要澄清】這不是嚴謹學術定義的「walk-forward optimization」（那需要自動在每一段用不同
    參數重新尋優、再用下一段驗證，工程量大很多）。這裡做的是更務實的「跨時段一致性檢查」：
    用同一組固定參數（你已經手動決定好的那些），檢查策略在不同時間段的表現是不是穩定——如果
    表現高度集中在其中一段、另一段明顯轉差甚至由正轉負，代表你目前看到的整體正期望值可能
    只是某幾筆特定交易撐起來的，不是策略本身穩定的優勢；如果各段方向一致，才比較有信心這個
    優勢不是單一時段的偶然。

    依「進場日期」的時間範圍等分成n_periods段（預設2段，前半/後半），不是依交易筆數等分——
    因為進場日期在時間軸上的分布不一定均勻，依日期切才能真正反映「不同時間段」的意思。

    回傳每個時間段的 list of dict：{period_label, start_date, end_date, trade_count, metrics}，
    metrics 是該段的完整 calculate_backtest_metrics() 結果（該段沒有任何交易時為 None）。
    """
    if trades_df is None or trades_df.empty:
        return []
    df = trades_df.copy()
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df = df.sort_values('entry_date').reset_index(drop=True)

    min_date, max_date = df['entry_date'].min(), df['entry_date'].max()
    total_days = (max_date - min_date).days
    if total_days <= 0:
        return [{'period_label': '全部（時間範圍過短無法切分）', 'start_date': min_date.strftime('%Y-%m-%d'),
                  'end_date': max_date.strftime('%Y-%m-%d'), 'trade_count': len(df), 'metrics': calculate_backtest_metrics(df)}]

    boundaries = [min_date + pd.Timedelta(days=total_days * i / n_periods) for i in range(n_periods + 1)]

    segments = []
    for idx in range(n_periods):
        start, end = boundaries[idx], boundaries[idx + 1]
        if idx == n_periods - 1:
            mask = (df['entry_date'] >= start) & (df['entry_date'] <= end)
        else:
            mask = (df['entry_date'] >= start) & (df['entry_date'] < end)
        segment_df = df[mask]
        metrics = calculate_backtest_metrics(segment_df) if not segment_df.empty else None
        segments.append({
            'period_label': f"第{idx + 1}段（{'較早' if idx == 0 else '較晚'}）" if n_periods == 2 else f"第{idx + 1}段",
            'start_date': start.strftime('%Y-%m-%d'), 'end_date': end.strftime('%Y-%m-%d'),
            'trade_count': len(segment_df), 'metrics': metrics,
        })
    return segments

def calculate_backtest_by_code(trades_df):
    """
    【V2.11.25新增】依股票代碼分組的績效拆解——回答「哪幾檔貢獻最多獲利/虧損」。

    「總報酬貢獻%」是把該股票所有交易的 pnl_pct 加總，這是一個粗略估計，不是真正依資金配置
    加權算出的貢獻度（因為回測目前還沒實作P2-2/P2-3的真實資金分配/排擠效應，見
    calculate_backtest_metrics 的docstring），但足夠用來快速定位「這個股票池裡，哪幾檔是主要
    賺錢引擎、哪幾檔是主要拖累」，供你決定要不要調整股票清單。

    回傳依「總報酬貢獻%」由高到低排序的 DataFrame，欄位：code／交易筆數／勝率%／平均報酬%／
    總報酬貢獻%／最大單筆獲利%／最大單筆虧損%／平均持有天數。
    """
    if trades_df is None or trades_df.empty:
        return None
    rows = []
    for code, g in trades_df.groupby('code'):
        wins = g[g['win']]
        rows.append({
            'code': code,
            '交易筆數': len(g),
            '勝率%': round(len(wins) / len(g) * 100, 1),
            '平均報酬%': round(g['pnl_pct'].mean(), 2),
            '總報酬貢獻%': round(g['pnl_pct'].sum(), 2),
            '最大單筆獲利%': round(g['pnl_pct'].max(), 2),
            '最大單筆虧損%': round(g['pnl_pct'].min(), 2),
            '平均持有天數': round(g['holding_days'].mean(), 1),
        })
    out = pd.DataFrame(rows).sort_values('總報酬貢獻%', ascending=False).reset_index(drop=True)
    return out

def run_backtest(codes, years=2, progress_callback=None):
    """
    回測總入口。逐檔抓資料、逐檔重演狀態機，任何單一股票失敗都不會中斷整批回測（跟即時系統
    main loop的個股try/except防護原則一致）。回傳 (trades_df, metrics, errors)。
    """
    regime_df, regime_err = _fetch_regime_history(years)
    all_trades = []
    errors = {}
    for idx, code in enumerate(codes):
        if progress_callback:
            progress_callback(idx, len(codes), code)
        ind_df, raw_df, err = _fetch_backtest_indicator_frame(code, years)
        if err:
            errors[code] = f"股價資料：{err}"; continue
        inst_df, inst_err = _fetch_institutional_history(code, years)
        if inst_err and not (code.isalpha() or code.endswith('.US')):
            errors[code] = f"籌碼資料：{inst_err}（仍會繼續回測，籌碼燈這段期間會偏保守判定不過關）"
        trades, sim_err = _simulate_stock_backtest(code, ind_df, raw_df, inst_df, regime_df)
        if sim_err:
            errors.setdefault(code, ""); errors[code] += f" 模擬失敗：{sim_err}"
            continue
        for t in trades:
            all_trades.append(_aggregate_backtest_trades(t))
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    metrics = calculate_backtest_metrics(trades_df) if not trades_df.empty else None
    if regime_err:
        errors['__regime__'] = f"市場燈號歷史資料：{regime_err}（仍會繼續回測，市場燈號這段期間會視為中性）"
    return trades_df, metrics, errors

portfolio, system_history, trade_plan_data, today_str = load_portfolio(), load_history(), load_trade_plan(), datetime.datetime.now().strftime("%Y-%m-%d")
trade_plan_snapshots = load_trade_plan_snapshots()  # 【V2.11.41新增】每檔股票「前一個確定版」的凍結快照
migrate_trade_plan_sheet()

# --- 5. 側邊欄 UI ---
with st.sidebar:
    st.header("📋 持股與風控設定")
    if not PORTFOLIO_LOAD_OK:
        st.error("🛡️ 安全模式中：持股資料本次讀取失敗，下方所有會修改持股的操作（新增/刪除/加減碼/CSV匯入/暫停恢復/歸檔）都已停用，避免覆蓋 Google Sheet 上的既有資料。重新整理頁面重試即可恢復正常。")
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
    with st.expander("🧮 加減碼成本計算小工具"):
        # 【新增】這個小工具跟上面「新增股票」表單是兩條互相獨立的路徑：
        # 如果你已經自己算好新成本，不需要用這個工具，直接在上面表單填入新的成本價與股數、
        # 按「更新設定」即可，一樣會生效——兩種方式擇一使用都可以，這裡只是幫你省去手動心算的步驟。
        st.caption("幫你算加碼／減碼後的新成本與股數，算完可以直接套用，也可以只看數字自己去上面表單填。")
        _calc_targets = [c for c, info in portfolio.items() if isinstance(info, dict) and _safe_float(info.get('qty', 0)) > 0]
        if not _calc_targets:
            st.info("目前沒有持股數>0的股票可以計算。")
        else:
            _calc_code = st.selectbox("選擇股票", _calc_targets, key="calc_code")
            _calc_info = portfolio.get(_calc_code, {}) if isinstance(portfolio.get(_calc_code), dict) else {}
            _calc_old_qty = _safe_float(_calc_info.get('qty', 0))
            _calc_old_cost = _safe_float(_calc_info.get('cost', 0))
            st.write(f"目前：持有 {_calc_old_qty:.0f} 股，成本 {_calc_old_cost:.2f}")
            _calc_action = st.radio("動作", ["加碼（買進更多）", "減碼／出場（賣出部分或全部）"], key="calc_action")

            if _calc_action == "加碼（買進更多）":
                _calc_trade_qty = st.number_input("加碼股數", min_value=0.0, value=0.0, step=1.0, key="calc_add_qty")
                _calc_trade_price = st.number_input("加碼價格", min_value=0.0, value=_calc_old_cost, step=0.1, key="calc_add_price")
                if _calc_trade_qty > 0:
                    _calc_new_qty = _calc_old_qty + _calc_trade_qty
                    _calc_new_cost = (_calc_old_qty * _calc_old_cost + _calc_trade_qty * _calc_trade_price) / _calc_new_qty if _calc_new_qty > 0 else 0.0
                    st.success(f"加碼後：持有 {_calc_new_qty:.0f} 股，新加權平均成本 {_calc_new_cost:.2f}")
                    if st.button("✅ 套用到成本欄位", key="calc_apply_add"):
                        portfolio[_calc_code]['qty'] = _calc_new_qty
                        portfolio[_calc_code]['cost'] = round(_calc_new_cost, 2)
                        save_portfolio(portfolio)
                        st.rerun()
            else:
                _calc_trade_qty = st.number_input("減碼／出場股數", min_value=0.0, max_value=_calc_old_qty, value=0.0, step=1.0, key="calc_reduce_qty")
                _calc_trade_price = st.number_input("賣出價格（選填，只用來顯示這筆的損益，不影響剩餘部位成本）", min_value=0.0, value=_calc_old_cost, step=0.1, key="calc_reduce_price")
                if _calc_trade_qty > 0:
                    _calc_new_qty = _calc_old_qty - _calc_trade_qty
                    # 減碼／部分出場不會改變「剩餘部位」的加權平均成本，只有股數變少——這跟加碼不同，
                    # 加碼是混入新一批不同價位的股票才需要重新算加權平均，減碼只是把同一批成本的股票賣掉一部分。
                    _calc_realized_pnl = (_calc_trade_price - _calc_old_cost) * _calc_trade_qty
                    st.success(f"減碼後：剩餘 {_calc_new_qty:.0f} 股，成本維持 {_calc_old_cost:.2f}（減碼不影響剩餘部位的平均成本）")
                    st.caption(f"這筆賣出的損益：{'+' if _calc_realized_pnl >= 0 else ''}{_calc_realized_pnl:,.0f} 元")
                    if st.button("✅ 套用到股數欄位", key="calc_apply_reduce"):
                        portfolio[_calc_code]['qty'] = _calc_new_qty
                        if _calc_new_qty <= 0:
                            portfolio[_calc_code]['cost'] = 0.0
                        save_portfolio(portfolio)
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
# 【V2.11.12新增，V2.11.17更新標籤】防守線來源的中文標籤，供UI顯示「這條防守線為什麼在這個位置」，
# 不再只丟一個數字。V2.11.17三層化後：initial_stop＝Level1，ratchet_ma20_atr／ratchet_swing_low
# 在Level2、Level3都可能出現（公式相同），ratchet_price_atr只會在Level3出現。
STOP_SOURCE_LABELS = {
    "no_position": "無持股",
    "initial_stop": "Level1 成本−2×ATR（尚未獲利，初始防守）",
    "locked_previous": "沿用前次防守線（本次候選都沒有更高）",
    "ratchet_ma20_atr": "MA20−ATR（結構防守）",
    "ratchet_price_atr": "現價−1.0×ATR（Level3 獲利保護，V2.11.36收緊）",
    "ratchet_swing_low": "近20日波段低點（結構防守）",
}

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
        # 【V2.11.14新增，三燈強度顯示】原本三燈只回答「有沒有過」，看不出「勉強過關」還是「過得很輕鬆」，
        # 這裡把驅動每個燈號判斷的原始數值印出來，讓你自己判斷強弱，不改動燈號本身的PASS/FAIL邏輯。
        if data['is_us']:
            _light1_detail = f"價格{'>' if data['price']>data['ma60'] else '≤'}MA60（{data['price']:.1f} vs {data['ma60']:.1f}）、MACD {data['macd']:+.2f}"
        else:
            _light1_detail = f"連{data['inst'].get('days',0)}買、累積買超{data['inst'].get('accumulated_shares',0)*data['price']/1e8:.1f}億"
        _light2_detail = f"K{data['k']:.0f}/D{data['d']:.0f}、RSI{data['rsi']:.0f}、量{data['volume']/max(data['vol_ma5'],1):.2f}倍"
        _light3_detail = f"價格{'>' if data['price']>data['ma20'] else '≤'}MA20、多頭排列{'是' if (data['ma10']>data['ma20']>data['ma60']) else '否'}"
        st.caption(f"燈號依據：{'動能' if data['is_us'] else '籌碼'}［{_light1_detail}］｜量能［{_light2_detail}］｜趨勢［{_light3_detail}］")
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
        # 【調整】分頁順序改為：交易計畫優先（平常真正要照著做的依據，放第一分頁一打開就看到），
        # 決策時間軸放最後（比較像是有空才回顧的歷史資料）。只調整這裡的宣告順序，下面每個分頁
        # 實際內容的程式碼位置完全不動，用變數對應關係換順序，避免搬動大段內容時改錯。
        tab_c5, tab_c1, tab_c2, tab_c3, tab_c6, tab_c4 = st.tabs(["🗓️ 交易計畫", "⚙️ AI決策與SOP", "📉 技術數據", "🛡️ 風控點位", "📐 MACD動能背離", "📈 決策時間軸"])

        with tab_c1:
            st.markdown(f"<div class='ai-advice-box'><div style='font-size: 1.1em; font-weight: bold; margin-bottom: 8px;'>🤖 AI 執行建議：</div>{''.join([f'<div style=\"margin-bottom: 4px;\">{item}</div>' for item in data['ai_advice']])}</div>", unsafe_allow_html=True)
            st.markdown(f"**🧠 AI 戰力拆解 (總分 {data['ai_score']})**")
            # 【V2.11.9修正，P1-1】美股的 score_inst 實際上是用「price>MA60 且 MACD>0」算的
            # 動能/趨勢分數，跟台股用真正的法人籌碼資料算出來的 score_inst 不是同一件事，
            # 不該在美股卡片上也標成「籌碼/長線」，容易誤導成美股也有籌碼資料可看。
            _inst_label = "動能/趨勢" if data['is_us'] else "籌碼/長線"
            st.code(f"{_inst_label}: +{data['score_inst']:.0f} | 趨勢技術: +{data['score_tech']:.0f} | 量能指標: +{data['score_vol']:.0f} | 風控狀態: +{data['score_risk']:.0f}", language="text")
            # 【V2.9.5 修正】改用小方塊組成的迷你進度條（而非整條拉滿寬度的 st.progress），
            # 視覺上更接近「一排小方塊」的樣式，且寬度只跟着方塊數走、不會佔滿整個畫面寬度。
            _bar_rows = []
            for _label, _val, _max in [
                (_inst_label, data['score_inst'], 40),
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
            # 【V2.11.10新增，觀察用】動能加速度：獨立於上面總分之外的±10分參考數字，補上「現有分數
            # 看不出正在轉強還是轉弱」的缺口。明確標示為觀察用，目前不影響決策分數、不影響任何
            # 進場/加碼門檻——先觀察這個數字準不準，之後有需要再考慮要不要正式併入。
            _accel = data.get('momentum_accel_score', 0.0)
            _accel_icon = "🔺" if _accel > 2 else ("🔻" if _accel < -2 else "▪️")
            st.caption(f"{_accel_icon} 動能加速度（觀察用，不影響任何進場/加碼門檻）：{_accel:+.1f} 分 — {data.get('momentum_accel_detail', '')}")
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
            _branch_note_map = {
                "resistance": "（依前高壓力位）",
                "atr_fallback": "（前高不明顯，改用ATR外推）",
                "r_floor_t1": "（前高太近、值不到1.5R，改用風險倍數R floor）",
                "profit_gate": "（獲利尚未達門檻，暫用保底值）",
                "atr_unavailable": "（ATR資料無效，暫用保底值）",
                "no_position": "",
            }
            _branch_note = _branch_note_map.get(data.get('target_branch'), "")
            _stop_source_note = STOP_SOURCE_LABELS.get(data.get('atr_stop_source'), data.get('atr_stop_source', ''))
            st.write(f"**設定成本**: {data['cost']:.2f}\n**動態防守/停損**: {data['atr_stop_price']:.2f}（來源：{_stop_source_note}）\n**第一目標 T1**: {data['t1']:.2f} {_branch_note}\n**第二目標 T2**: {data['t2']:.2f}")
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
                "BREAKOUT_FAILED": "🟠 突破失敗",
            }
            st.markdown(f"**交易計畫狀態**：{_state_label_map.get(_plan_state, _plan_state)}")
            if data.get('plan_signal_reason'):
                st.caption(f"📝 {data['plan_signal_reason']}")

            # 【V2.11.41新增，Trade Plan Snapshot】只在「今天這根K棒可能還在變動」時才顯示，
            # 提醒你上面看到的這個版本可能是盤中還沒收盤的資料算出來的，附上「前一個確定版」讓你
            # 對照——平常（收盤後、或開盤前查看）不需要看這個，上面顯示的本來就是確定版，加這個
            # 提示反而多餘。
            if data.get('is_today_bar'):
                _snap = trade_plan_snapshots.get(str(data['code']))
                if _snap:
                    with st.expander(f"🔒 上一個確定版（{_snap.get('snapshot_date', '—')} 收盤後）— 目前這個版本可能還會變動"):
                        st.caption("上面顯示的計畫，是用今天盤中還沒收盤的資料算出來的，之後可能還會變。這裡是上一個交易日收盤後確定、且從未被覆寫過的版本：")
                        st.write(f"**狀態**：{_state_label_map.get(_snap.get('state', ''), _snap.get('state', ''))}")
                        if _snap.get('signal_reason'):
                            st.caption(f"📝 {_snap['signal_reason']}")
                        st.write(f"T1：{_snap.get('t1_price', '—')}　｜　T2：{_snap.get('t2_price', '—')}　｜　移動防守線：{_snap.get('current_trailing_stop', '—')}")
                        if str(_snap.get('addon_shares_approved', '') or '') not in ('', '0'):
                            st.write(f"建議加碼股數：{_snap['addon_shares_approved']}")
                        if str(_snap.get('partial_exit_shares', '') or '') not in ('', '0'):
                            st.write(f"建議分批出場股數：{_snap['partial_exit_shares']}")
                        if str(_snap.get('full_exit_shares', '') or '') not in ('', '0'):
                            st.write(f"建議全部出清股數：{_snap['full_exit_shares']}")
                        st.caption(f"（快照時間：{_snap.get('saved_at', '—')}）")

            # 【V2.11.12新增，第六點簡化版】人工覆核：只記錄「有沒有看過」，不記錄「同意/拒絕」
            # （那個判斷本來就會反映在你有沒有去改股數上，不需要再另外記錄一次主觀決定）。
            # 只在真的有作用中的計畫時才顯示（PREPARE代表還沒有任何訊號，不需要確認什麼）。
            if _plan_state != "PREPARE":
                if data.get('plan_review_state') == "PENDING":
                    _rc1, _rc2 = st.columns([3, 1])
                    _rc1.warning("🆕 這是尚未確認過的訊號")
                    if _rc2.button("✅ 已閱覽", key=f"ack_{data['code']}"):
                        if data['code'] in trade_plan_data:
                            trade_plan_data[data['code']]['review_state'] = 'ACKNOWLEDGED'
                            trade_plan_data[data['code']]['review_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            save_trade_plan(trade_plan_data)
                        st.rerun()
                else:
                    st.caption(f"✅ 已於 {data.get('plan_review_at') or '（時間未記錄）'} 確認閱覽過這個訊號")

            # 【V2.11.14新增，Next-Day Execution Gate】只在還沒真正持有、且交易計畫確實建議「下一交易日
            # 進場」或「已判定突破失敗」時才顯示，告訴你隔日開盤價落在哪個區間該怎麼做，
            # 不用自己心算現價相對突破價/追價上限/失效價的位置。
            if data.get('held_qty', 0) <= 0 and _plan_state in ("ENTER_NEXT_DAY", "BREAKOUT_FAILED"):
                _gate_code, _gate_note = data.get('plan_next_day_gate', ("NO_TRADE", ""))
                st.info(f"**隔日執行判斷**：{_gate_note}")

            # 【修正】原本用 st.columns(2) 左右分欄，在手機/平板較窄的螢幕上兩欄文字容易被硬擠在一起、
            # 看起來混在一起分不清楚，改成單欄、由上到下依序顯示，寬螢幕/窄螢幕都不會有排版混淆的問題。
            st.write(f"**訊號日期資料**\n台股資料日：{data.get('plan_taiwan_data_date') or '—'}\n美股資料日：{data.get('plan_us_data_date') or '—'}\n建議執行日：{data.get('plan_execution_date') or '—'}\n有效期限：{data.get('plan_valid_until') or '—'}")
            # 【V2.11.9修正，P1-4】原本這裡沒有分台股/美股，一律用台股假日表算下一交易日，
            # 對美股會不準。改成依這張卡片的股票代號判斷市場，跟其他地方判斷 is_us_stock 的方式一致。
            _card_is_us_stock = str(data.get('code', '')).isalpha() or str(data.get('code', '')).endswith('.US')
            st.write(f"**目前執行模式**：{_mode_display.get(execution_mode, execution_mode)}\n**下一交易日**：{_next_business_day(data.get('plan_taiwan_data_date') or today_str, _card_is_us_stock) or '—'}")

            # 【V2.11.43新增，懶人包摘要】把「建議股數」跟「該對照的價位」合併成單一句話，放在
            # 最上面，不用再往下翻、跨區塊對照。下面原本的詳細分區（空手訊號資訊／持倉計畫資訊）
            # 完整保留不變，這裡純粹是額外多加一個「一眼看懂」的摘要，不影響任何既有顯示內容。
            # 【V2.11.45修正】依你的要求：不管哪種情況，只要有建議股數異動，都要提供一個可對照的
            # 價位，不要出現「沒有價位」這種空白。對「條件觸發、本來就沒有目標價」的情況（加碼、
            # MACD轉弱提前停利），改用「現價」當參考——這裡要非常清楚標示這是「參考」不是「目標」：
            # 這幾種訊號本來就是「條件成立、隔天開盤就執行」，不是「等股價漲到/跌到某個價位才動作」，
            # 用現價當參考只是讓你有一個大概的數字可以對照，實際成交價會是隔天開盤價，不會剛好等於
            # 現在看到的這個數字，跟「突破價～追價上限」「T1/T2」這種真正的價位門檻，意義不一樣。
            _cur_price = _safe_float(data.get('price', 0))
            if _plan_state == "ENTER_NEXT_DAY":
                st.info(f"📋 **建議進場：{data.get('plan_suggested_shares', 0)} 股　｜　參考價位：{data.get('plan_breakout_price', 0):.2f} ～ {data.get('plan_chase_limit', 0):.2f}**（突破價～追價上限，開盤價超過上限則不追）")
            elif _plan_state == "ADD_NEXT_DAY":
                st.info(f"📋 **建議加碼：{data.get('plan_addon_shares_approved', 0)} 股　｜　參考現價：{_cur_price:.2f}**（加碼是條件達成即執行，沒有目標價門檻，隔日開盤附近成交，價格會跟現價略有出入）")
            elif _plan_state == "PARTIAL_EXIT_NEXT_DAY":
                # 分批停利有兩種完全不同的觸發原因，不能一律顯示T1/T2：T1_PARTIAL_EXIT／
                # T2_PARTIAL_EXIT 是「真的漲到目標價」觸發，T1/T2就是該對照的價位；但
                # MACD_REVERSAL_T1／MACD_REVERSAL_T2 是「還沒漲到目標價，MACD轉弱提前停利」，
                # 本質上跟加碼一樣是「條件成立就執行」，改用現價當參考，不能顯示T1/T2誤導成
                # 「要等漲到那個價位」。
                _partial_signal_type = data.get('plan_signal_type', '')
                if _partial_signal_type in ('T1_PARTIAL_EXIT', 'T2_PARTIAL_EXIT'):
                    _partial_target = data.get('plan_t1_price', 0) if _partial_signal_type == 'T1_PARTIAL_EXIT' else data.get('plan_t2_price', 0)
                    st.info(f"📋 **建議分批停利：{data.get('plan_partial_exit_shares', 0)} 股　｜　參考價位：{_partial_target:.2f}**（已觸及{'T1' if _partial_signal_type == 'T1_PARTIAL_EXIT' else 'T2'}目標價）")
                elif _partial_signal_type in ('MACD_REVERSAL_T1', 'MACD_REVERSAL_T2'):
                    st.info(f"📋 **建議分批停利：{data.get('plan_partial_exit_shares', 0)} 股　｜　參考現價：{_cur_price:.2f}**（尚未到達{'T1' if _partial_signal_type == 'MACD_REVERSAL_T1' else 'T2'}，是MACD轉弱提前停利，沒有目標價門檻，隔日開盤附近執行）")
                else:
                    st.info(f"📋 **建議分批停利：{data.get('plan_partial_exit_shares', 0)} 股　｜　參考價位：T1 {data.get('plan_t1_price', 0):.2f} ／ T2 {data.get('plan_t2_price', 0):.2f}**")
            elif _plan_state == "FULL_EXIT_NEXT_DAY":
                st.info(f"📋 **建議全部出清：{data.get('plan_full_exit_shares', 0)} 股　｜　參考價位：防守線 {data.get('plan_current_trailing_stop', 0):.2f}**（實際成交價可能因跳空而偏離，見下方警語）")

            if data.get('held_qty', 0) <= 0:
                st.markdown("**空手訊號資訊**")
                st.write(f"突破價：{data.get('plan_breakout_price', 0):.2f}　｜　追價上限：{data.get('plan_chase_limit', 0):.2f}")
                _bq_score = data.get('plan_breakout_quality_score', 0)
                _bq_grade = data.get('plan_breakout_quality_grade', '')
                if _bq_grade:
                    st.write(f"突破品質：{_bq_score:.0f}/100 {_bq_grade}級")
                st.write(f"回測區間：{data.get('plan_pullback_low', 0):.2f} ～ {data.get('plan_pullback_high', 0):.2f}")
                _retest_q = data.get('plan_retest_quality', '')
                if _retest_q == "RETESTED_HELD":
                    st.caption("✅ 回測品質：已實際拉回等待區間內確認支撐，屬真實回測後再突破")
                elif _retest_q == "NO_RETEST":
                    st.caption("⚠️ 回測品質：股價從未實際拉回等待區間，屬直接反彈站上（V型），未經真實回測確認")
                elif _retest_q == "INVALID_TOUCHED":
                    st.caption("⚠️ 回測品質：期間曾跌破失效價，訊號品質存疑")
                st.write(f"失效價：{data.get('plan_invalid_price', 0):.2f}　｜　建議進場股數：{data.get('plan_suggested_shares', 0)} 股")
                if _plan_state == "SUSPENDED_BY_REGIME":
                    st.warning("⏸️ 市場目前處於逆風狀態，新倉暫停，但交易計畫本身未被刪除，逆風解除後會自動恢復。")
            else:
                st.markdown("**持倉計畫資訊**")
                _pnl_cost = _safe_float(data.get('cost', 0))
                _pnl_qty = _safe_float(data.get('held_qty', 0))
                _pnl_price = _safe_float(data.get('price', 0))
                _pnl_amount = (_pnl_price - _pnl_cost) * _pnl_qty
                _pnl_pct = ((_pnl_price - _pnl_cost) / _pnl_cost * 100) if _pnl_cost > 0 else 0.0
                _pnl_color = "#f87171" if _pnl_amount > 0 else ("#34d399" if _pnl_amount < 0 else "inherit")  # 台股習慣：紅漲(賺)、綠跌(賠)
                st.write(f"持有股數：{data.get('held_qty', 0)} 股　｜　平均成本：{_pnl_cost:.2f}")
                st.markdown(f"損益：<span style='color:{_pnl_color}'>{'+' if _pnl_amount >= 0 else ''}{_pnl_amount:,.0f} 元（{'+' if _pnl_pct >= 0 else ''}{_pnl_pct:.2f}%）</span>", unsafe_allow_html=True)
                st.write(f"T1：{data.get('plan_t1_price', 0):.2f}（{'✅已執行' if data.get('plan_t1_taken') else '⬜未執行'}）　｜　T2：{data.get('plan_t2_price', 0):.2f}（{'✅已執行' if data.get('plan_t2_taken') else '⬜未執行'}）")
                st.write(f"初始防守線：{data.get('atr_stop_price', 0):.2f}　｜　今日移動防守線（計畫值）：{data.get('plan_current_trailing_stop', 0):.2f}（來源：{STOP_SOURCE_LABELS.get(data.get('plan_current_trailing_stop_source'), data.get('plan_current_trailing_stop_source', ''))}）")
                if _plan_state == "ADD_NEXT_DAY":
                    st.success(f"📈 建議加碼股數：{data.get('plan_addon_shares_approved', 0)} 股（下一交易日執行）")
                if _plan_state == "PARTIAL_EXIT_NEXT_DAY":
                    st.warning(f"🟠 建議分批停利股數：{data.get('plan_partial_exit_shares', 0)} 股（{data.get('plan_signal_type','')}，下一交易日執行）")
                if _plan_state == "FULL_EXIT_NEXT_DAY":
                    st.error(f"🔴 建議全部出清股數：{data.get('plan_full_exit_shares', 0)} 股（下一交易日執行）\n⚠️ 系統不保證一定能以防守觸發價成交，實際成交價可能因跳空而偏離，請留意跳空風險。")

            # 【V2.11.39新增，V2.11.40擴充選項，Decision Log】只在系統真的要求你做決定的狀態才顯示
            # 這個記錄小工具，平常的PREPARE/HOLD不需要記錄什麼。系統自己的建議/理由已經存在
            # trade_plan分頁裡，這裡只補系統不可能知道的那一半：你有沒有真的照做、實際成交價、
            # 偏離的原因。
            _actionable_states = {"ENTER_NEXT_DAY", "ADD_NEXT_DAY", "PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY", "BREAKOUT_FAILED"}
            if _plan_state in _actionable_states:
                with st.expander("📝 記錄我的決定（供之後回顧用，不影響系統判斷）"):
                    st.caption(f"系統建議：{_state_label_map.get(_plan_state, _plan_state)}｜{data.get('plan_signal_reason', '')}")
                    # 【V2.11.40】出場類訊號（分批/全部出清）跟進場類訊號（進場/加碼/突破失敗）的
                    # 「沒做」意義不一樣——出場訊號沒做＝續抱；進場訊號沒做＝放棄這次機會。用不同的
                    # 選項組合，記錄起來才準確，不用勉強套同一組選項。
                    if _plan_state in ("PARTIAL_EXIT_NEXT_DAY", "FULL_EXIT_NEXT_DAY"):
                        _log_options = ["完全照做", "延後出場", "部分出場/股數不同", "續抱未出場"]
                    else:
                        _log_options = ["完全照做", "延後執行", "部分執行/股數不同", "沒有進場/加碼"]
                    _log_action = st.radio("我實際上", _log_options, key=f"log_action_{data['code']}", horizontal=True)
                    _log_price_col, _log_reason_col = st.columns(2)
                    _log_price = _log_price_col.number_input("實際價位（選填）", min_value=0.0, value=0.0, step=0.01, key=f"log_price_{data['code']}")
                    _log_reason = _log_reason_col.text_input("原因（選填，例如：延後/沒做/續抱的理由）", key=f"log_reason_{data['code']}")
                    if st.button("儲存這筆紀錄", key=f"log_submit_{data['code']}"):
                        _ok = append_decision_log(
                            today_str, data['code'], f"{_plan_state}：{data.get('plan_signal_reason', '')}",
                            _log_action, _log_price if _log_price > 0 else None, _log_reason,
                        )
                        if _ok:
                            st.success("已記錄")

        with tab_c6:
            # 【新增】MACD 動能變化與背離分析：日線／週線分開顯示，格式對照四大模組
            # （技術指標現況診斷／訊號層級評估／交易決策建議／風險過濾提醒）。
            _action_style = {
                "核心進場": ("success", "🟢"), "分批試單": ("success", "🟢"), "觀望": ("info", "⚪"),
                "預警關注": ("warning", "🟡"), "減碼50%": ("warning", "🟠"), "出場": ("error", "🔴"),
                "資料不足": ("info", "⚪"),
            }
            for _tf_label, _macd_r in [("📅 日線", data.get('macd_daily')), ("🗓️ 週線", data.get('macd_weekly'))]:
                if _macd_r is None:
                    continue
                st.markdown(f"#### {_tf_label}")
                if _macd_r.error:
                    st.info(f"ℹ️ {_macd_r.error}")
                    st.divider()
                    continue

                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("DIF", f"{_macd_r.dif:.3f}" if _macd_r.dif is not None else "—")
                _mc2.metric("DEA", f"{_macd_r.dea:.3f}" if _macd_r.dea is not None else "—")
                _mc3.metric("柱狀體 OSC", f"{_macd_r.osc:.3f}" if _macd_r.osc is not None else "—", _macd_r.osc_status)

                st.write(f"**背離警示（近似演算法，非嚴謹判定）**：{_macd_r.divergence_type}")
                st.write(f"**訊號層級**：{_macd_r.osc_status} ｜ **背離警示**：{_macd_r.divergence_type}")

                _style, _icon = _action_style.get(_macd_r.signal_action, ("info", "⚪"))
                _action_msg = f"{_icon} **操作動作：{_macd_r.signal_action}**"
                if _style == "success": st.success(_action_msg)
                elif _style == "warning": st.warning(_action_msg)
                elif _style == "error": st.error(_action_msg)
                else: st.info(_action_msg)

                # 【新增】澄清這裡的「操作動作」只是純MACD規則算出來的理論建議，不是要你現在執行的指令；
                # 日線MACD已經實質影響🗓️交易計畫（頂背離/翻黑會暫停加碼，翻黑且有獲利會提前分批停利），
                # 但頂背離本身不會直接觸發賣出——避免對還在噴出階段、反覆出現頂背離的強勢股賣過頭。
                # 週線MACD目前完全沒有接入交易計畫判斷，純粹參考。這裡明確告知，避免誤以為兩邊互相矛盾。
                if _tf_label.startswith("📅"):
                    st.caption("ℹ️ 以上是純MACD規則算出的理論建議，非現在要執行的指令。日線MACD已實質影響「🗓️交易計畫」：頂背離/翻黑會暫停加碼，翻黑且有獲利會提前分批停利；但頂背離本身不會直接觸發賣出（避免對噴出階段反覆出現的頂背離反應過度）。實際要不要操作，請以「🗓️交易計畫」分頁的判斷為準。")
                else:
                    st.caption("ℹ️ 以上是純MACD規則算出的理論建議，目前週線MACD尚未接入「🗓️交易計畫」的判斷，僅供參考對照，不會影響加碼/出清/進場的正式決策。")

                st.caption(f"📝 {_macd_r.detail}")
                if _macd_r.risk_management is not None:
                    st.write(f"**失效停損點（關鍵支撐）**：{_macd_r.risk_management:.2f}")
                st.divider()

# --- 6. 主程式執行 ---
st.title(f"⚡ {APP_TITLE}")
st.warning("⚠️ 本系統僅為個人化技術指標整理與紀律提醒工具，所有分數、判定、建議均由你自訂的公式與參數計算而成，**不構成任何投資建議**，過去的訊號表現也不保證未來結果。所有操作決策與風險，仍需由你自己判斷並承擔。")

def calculate_data_freshness(data_date_str, reference_date_str, is_us_stock=False):
    """
    【V2.11.13新增，資料新鮮度分級】依照「資料日期」跟「今天」之間差幾個交易日（用既有的
    台股/美股假日行事曆計算，排除週末與已知假日），分成四級，比單純顯示一個日期字串更容易
    一眼判斷「這個資料到底新不新鮮」，不用自己心算日期差幾天。

    回傳 (等級文字, 說明文字, 顏色)：
      🟢 新鮮：資料截至上一個應有交易日（落後0個交易日）
      🟡 可用但延遲：落後1個交易日
      🟠 過期：落後2~3個交易日
      🔴 不可用：無法確認資料日期、日期格式錯誤、或落後超過3個交易日
    """
    if not data_date_str:
        return "🔴 不可用", "無法確認資料日期", "red"
    try:
        d = pd.Timestamp(data_date_str)
        t = pd.Timestamp(reference_date_str)
        if d > t:
            return "🔴 不可用", "資料日期在未來，資料來源可能有誤", "red"
        holidays = US_MARKET_HOLIDAYS if is_us_stock else TW_MARKET_HOLIDAYS
        lag, cursor = 0, d
        while cursor < t:
            cursor += pd.Timedelta(days=1)
            if cursor.weekday() < 5 and cursor.strftime("%Y-%m-%d") not in holidays:
                lag += 1
        if lag <= 0:
            return "🟢 新鮮", "資料截至上一個應有交易日", "green"
        elif lag == 1:
            return "🟡 可用但延遲", "資料落後1個交易日", "yellow"
        elif lag <= 3:
            return "🟠 過期", f"資料落後{lag}個交易日", "orange"
        else:
            return "🔴 不可用", f"資料落後{lag}個交易日，過久未更新", "red"
    except Exception:
        return "🔴 不可用", "資料日期格式錯誤", "red"

macro_data = fetch_macro_data()
st.markdown("### 🌍 雙軌市場環境總覽")
m_col1, m_col2, m_col3 = st.columns(3)

def _render_macro_asof(col, asof, is_us_stock=False):
    # 【V2.10.8 新增】顯示資料實際對應的交易日期，並在資料超過3天沒更新時跳出警示，
    # 讓使用者自己能判斷「這數字是不是卡住了」，不用只能憑感覺猜。
    # 【V2.11.13新增】疊加新鮮度分級徽章，不用自己心算日期差幾天。
    if asof is None:
        return
    _asof_ts = pd.Timestamp(asof)
    if _asof_ts.tzinfo is not None:
        _asof_ts = _asof_ts.tz_localize(None)
    _days_old = (pd.Timestamp(datetime.datetime.now()) - _asof_ts).days
    _date_str = _asof_ts.strftime("%Y-%m-%d")
    _freshness_label, _freshness_note, _ = calculate_data_freshness(_date_str, today_str, is_us_stock)
    if _days_old > 3:
        col.caption(f"⚠️ 資料日期：{_date_str}（{_days_old}天前，可能不是最新資料，建議留意）｜{_freshness_label}")
    else:
        col.caption(f"資料日期：{_date_str}｜{_freshness_label}（{_freshness_note}）")

tw_trend = macro_data.get('TW', {})
if tw_trend:
    m_col1.metric("🇹🇼 台股加權 (大盤方向)", f"{tw_trend['price']:,.0f}", tw_trend['trend'], delta_color="normal" if "多頭" in tw_trend['trend'] else "inverse")
    _render_macro_asof(m_col1, tw_trend.get('asof'), is_us_stock=False)
else: m_col1.metric("🇹🇼 台股加權", "連線中...")

us_trend = macro_data.get('US', {})
if us_trend:
    m_col2.metric("🇺🇸 那斯達克 (科技風向)", f"{us_trend['price']:,.0f}", us_trend['trend'], delta_color="normal" if "多頭" in us_trend['trend'] else "inverse")
    _render_macro_asof(m_col2, us_trend.get('asof'), is_us_stock=True)
else: m_col2.metric("🇺🇸 那斯達克", "連線中...")

vix_trend = macro_data.get('VIX', {})
if vix_trend:
    v_val = vix_trend['price']
    v_status, v_color = ("🚨 極度恐慌", "inverse") if v_val >= 25 else (("⚠️ 波動加劇", "off") if v_val >= 20 else ("🟢 環境穩定", "normal"))
    m_col3.metric("📉 VIX 恐慌指數", f"{v_val:.2f}", v_status, delta_color=v_color)
    _render_macro_asof(m_col3, vix_trend.get('asof'), is_us_stock=True)
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
if not PORTFOLIO_LOAD_OK:
    execution_mode = VIEW_ONLY  # 【V2.11.15新增】portfolio 讀取失敗，同樣強制唯讀，理由同上

market_regime_label = derive_market_regime(macro_data)
_mode_display = {"TAIWAN_CLOSE_UPDATE": "🇹🇼 台股收盤更新", "US_CLOSE_UPDATE": "🇺🇸 美股收盤更新", "VIEW_ONLY": "👁️ 唯讀檢視（無新資料）"}
st.caption(f"⚙️ 執行模式：**{_mode_display.get(execution_mode, execution_mode)}** ｜台股資料日期：{latest_tw_date or 'N/A'}｜美股資料日期：{latest_us_date or 'N/A'}｜市場燈號：{market_regime_label}｜上次已保存：台{saved_tw_date or 'N/A'} / 美{saved_us_date or 'N/A'}｜版本：{APP_VERSION}")
st.divider()

# --- 5-1. 策略回測（P2，V2.11.23新增）---
with st.expander("📊 策略回測（Backtest）— 用歷史資料驗證這套策略到底有沒有用", expanded=False):
    st.caption(
        "逐日重演過去2年的歷史資料，呼叫跟即時系統完全同一套決策函式（不是另外寫一份簡化邏輯），"
        "統計出勝率、Profit Factor、Expectancy等真正的績效數字。**這裡跑的是模擬帳本，不會動到你"
        "真實的 Google Sheet 持股/交易計畫資料。** 抓取歷史資料需要連網，一次跑完全部股票可能需要"
        "數分鐘，請耐心等待，不要中途重新整理頁面。"
    )
    _bt_default_codes = sorted(set(list(trade_plan_data.keys()) + list(portfolio.keys())))
    _bt_codes_input = st.text_area(
        "回測股票代碼（逗號分隔，預設帶入你目前追蹤過的所有代碼）",
        value=", ".join(_bt_default_codes), height=80, key="bt_codes_input"
    )
    _bt_run = st.button("🚀 開始回測", key="bt_run_button")

    if _bt_run:
        _bt_codes = [c.strip() for c in _bt_codes_input.split(",") if c.strip()]
        if not _bt_codes:
            st.warning("請至少輸入一檔股票代碼。")
        else:
            _bt_progress = st.progress(0, text="準備中...")
            _bt_status = st.empty()

            def _bt_progress_cb(idx, total, code):
                _bt_progress.progress((idx + 1) / total, text=f"正在回測第 {idx + 1}/{total} 檔：{code}")

            with st.spinner("回測執行中，請勿關閉頁面..."):
                _bt_trades_df, _bt_metrics, _bt_errors = run_backtest(_bt_codes, years=2, progress_callback=_bt_progress_cb)
            _bt_progress.progress(1.0, text="回測完成")

            if _bt_errors:
                with st.expander(f"⚠️ {len(_bt_errors)} 項資料抓取問題（不影響其他股票的回測結果）", expanded=False):
                    for _code, _msg in _bt_errors.items():
                        st.write(f"- **{_code}**：{_msg}")

            if _bt_metrics is None:
                st.info("這個範圍內沒有產生任何完整交易，可能是股票清單太少、資料不足，或這段期間確實沒有符合條件的訊號。")
            else:
                st.subheader("📈 績效總覽")
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric("總交易筆數", f"{_bt_metrics['total_trades']}")
                _m1.metric("勝率", f"{_bt_metrics['win_rate']:.1f}%")
                _m2.metric("平均獲利", f"{_bt_metrics['avg_win_pct']:.1f}%")
                _m2.metric("平均虧損", f"{_bt_metrics['avg_loss_pct']:.1f}%")
                _m3.metric("Profit Factor", f"{_bt_metrics['profit_factor']:.2f}" if _bt_metrics['profit_factor'] != float('inf') else "∞（無虧損交易）")
                _m3.metric("Expectancy（期望值）", f"{_bt_metrics['expectancy_pct']:.2f}%")
                _m4.metric("平均持有天數", f"{_bt_metrics['avg_holding_days']:.0f} 天")
                _m4.metric("最大回撤", f"{_bt_metrics['max_drawdown_pct']:.1f}%")
                st.caption(
                    f"T1命中率：{_bt_metrics['t1_hit_rate']:.1f}%　｜　T2命中率：{_bt_metrics['t2_hit_rate']:.1f}%　｜　"
                    f"停損/虧損出場率：{_bt_metrics['stop_rate']:.1f}%"
                )
                if _bt_metrics.get('avg_regime_score_win') is not None or _bt_metrics.get('avg_regime_score_loss') is not None:
                    _rw = _bt_metrics.get('avg_regime_score_win')
                    _rl = _bt_metrics.get('avg_regime_score_loss')
                    st.caption(
                        f"🌡️ 進場當下市場燈號分數：贏的交易平均 {f'{_rw:.1f}' if _rw is not None else 'N/A'} 分　"
                        f"｜　輸的交易平均 {f'{_rl:.1f}' if _rl is not None else 'N/A'} 分　"
                        "（分數越低代表大盤當時越弱，見「依股票代碼分組績效」下方交易明細的 `entry_regime_score` 欄位；"
                        "0~100分級距見說明書第14節Market Regime Score說明）"
                    )
                if _bt_metrics.get('avg_mfe_win') is not None:
                    _mfe_w = _bt_metrics.get('avg_mfe_win')
                    _mae_w = _bt_metrics.get('avg_mae_win')
                    _mae_l = _bt_metrics.get('avg_mae_loss')
                    _giveback = _mfe_w - _bt_metrics['avg_win_pct']
                    st.caption(
                        f"📐 MFE/MAE：贏的交易平均最高曾浮盈 {_mfe_w:.1f}%（實際出場拿到{_bt_metrics['avg_win_pct']:.1f}%，"
                        f"回吐約{_giveback:.1f}個百分點）｜贏的交易平均最深曾浮虧 {_mae_w:.1f}%　"
                        f"｜　輸的交易平均最深曾浮虧 {f'{_mae_l:.1f}%' if _mae_l is not None else 'N/A'}　"
                        "（回吐缺口大代表停利可能太鬆；贏的交易MAE很深代表停損可能設太緊，容易把後來會成功的交易洗掉；"
                        "輸的交易MAE可以拿來對照理論停損距離，看停損有沒有確實執行）"
                    )
                st.caption(
                    "⚠️ 最大回撤是用「逐筆交易累積報酬率」算出的簡化權益曲線，不是真實資金逐日回撤"
                    "（沒有考慮同時持有多檔部位的資金排擠、也沒有考慮交易成本/滑價，屬於P2-2/P2-3尚未實作範圍，"
                    "見說明書第14節）。"
                )

                st.subheader("📉 權益曲線（模擬，逐筆交易累積報酬率）")
                st.line_chart(_bt_metrics['equity_curve'])

                st.subheader("🧪 跨時段一致性檢查")
                st.caption(
                    "把交易依進場日期切成前後兩段，各自獨立算一次完整績效，檢查表現是不是集中在"
                    "某一段特定期間（過度擬合的警訊），還是不同時間段都維持一致的方向。**這不是"
                    "嚴謹學術定義的walk-forward optimization**（那需要每段自動重新尋優參數），"
                    "是更務實的一致性檢查：用你現在這組固定參數，看不同時段表現穩不穩定。"
                )
                _wf_segments = calculate_walk_forward_split(_bt_trades_df, n_periods=2)
                if len(_wf_segments) < 2:
                    st.info("交易數量或時間範圍不足，無法切成兩段比較。")
                else:
                    _wf_cols = st.columns(len(_wf_segments))
                    for _seg, _col in zip(_wf_segments, _wf_cols):
                        with _col:
                            st.markdown(f"**{_seg['period_label']}**　{_seg['start_date']} ～ {_seg['end_date']}")
                            if _seg['metrics'] is None:
                                st.write(f"這段沒有任何交易（共{_seg['trade_count']}筆）")
                            else:
                                _m = _seg['metrics']
                                st.metric("交易筆數", _seg['trade_count'])
                                st.metric("勝率", f"{_m['win_rate']:.1f}%")
                                st.metric("Profit Factor", f"{_m['profit_factor']:.2f}" if _m['profit_factor'] != float('inf') else "∞")
                                st.metric("Expectancy", f"{_m['expectancy_pct']:.2f}%")
                    _valid_segs = [s for s in _wf_segments if s['metrics'] is not None]
                    if len(_valid_segs) >= 2:
                        _exps = [s['metrics']['expectancy_pct'] for s in _valid_segs]
                        if all(e > 0 for e in _exps):
                            st.caption("✅ 各時間段的期望值都是正的，方向一致，比較不像是單一時段偶然撐起來的結果。")
                        elif all(e < 0 for e in _exps):
                            st.caption("⚠️ 各時間段的期望值都是負的——這組參數在這整段回測期間可能本來就沒有優勢。")
                        else:
                            st.caption("⚠️ 各時間段的期望值方向不一致（有正有負），目前看到的整體正期望值，可能是被表現特別好的那一段拉高的，建議謹慎看待，不要照單全收。")

                st.subheader("🧩 依股票代碼分組績效")
                st.caption(
                    "「總報酬貢獻%」是把該股票所有交易的報酬率直接加總，用來快速看出「哪幾檔是主要"
                    "賺錢引擎、哪幾檔是主要拖累」——這是粗略估計，不是依實際資金配置加權算出的貢獻度"
                    "（原因同上方權益曲線的警語：目前還沒實作真實資金分配）。"
                )
                _bt_by_code = calculate_backtest_by_code(_bt_trades_df)
                st.dataframe(_bt_by_code, use_container_width=True, hide_index=True)

                st.subheader("📋 交易明細")
                _bt_display_df = _bt_trades_df.sort_values('exit_date', ascending=False).copy()
                _bt_display_df['entry_date'] = pd.to_datetime(_bt_display_df['entry_date']).dt.strftime('%Y-%m-%d')
                _bt_display_df['exit_date'] = pd.to_datetime(_bt_display_df['exit_date']).dt.strftime('%Y-%m-%d')
                st.dataframe(_bt_display_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ 下載完整交易明細（CSV）",
                    data=_bt_trades_df.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f"backtest_trades_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

# --- 5-2. 決策日誌（V2.11.39新增）---
# 每一檔股票的「🗓️交易計畫」分頁，遇到系統要求你做決定的狀態時，會出現「📝記錄我的決定」小工具，
# 這裡是彙總所有已經記錄過的內容，供你事後回顧、下載帶去分析。
with st.expander("📝 決策日誌 — 系統建議 vs 你實際執行了什麼", expanded=False):
    st.caption(
        "在各股票的「🗓️交易計畫」分頁，遇到系統要求你做決定的狀態（進場/加碼/出場等）時，"
        "會出現「記錄我的決定」小工具，填完按儲存就會累積在這裡。系統自己的建議/理由已經存在"
        "trade_plan分頁裡，這裡只補系統不可能知道的那一半：你有沒有真的照做、實際成交價、"
        "偏離的原因。建議累積幾週到幾個月後，下載下來回顧或帶去分析。"
    )
    _decision_log_data = load_decision_log()
    if not _decision_log_data:
        st.info("目前還沒有任何紀錄。到某檔股票的「🗓️交易計畫」分頁，遇到需要決定的訊號時，展開「📝記錄我的決定」開始記錄。")
    else:
        _log_df = pd.DataFrame(_decision_log_data)
        st.dataframe(_log_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 下載決策日誌（CSV）",
            data=_log_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"decision_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

if not portfolio:
    if not PORTFOLIO_LOAD_OK:
        st.error("🛡️ 安全模式：持股資料本次讀取失敗，為避免用錯誤/空白資料跑分析或誤判「你目前沒有任何持股」，本次不顯示任何分析結果。請確認 Google Sheets 連線與服務帳號權限後，重新整理頁面重試。")
    else:
        st.info("👈 請先從左側邊欄新增股票代號！")
else:
    summary_data, card_data, paused_data = [], [], []
    macd_report_results: List[MACDSignalResult] = []
    _any_intraday_reevaluation = False  # 【新增】只要有任何一檔因「今天K棒尚未收斂」被強制重新評估過，
                                         # 就算全域執行模式是VIEW_ONLY，最後也要把這次的結果存回trade_plan。

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
            # 【V2.11.2 新增，V2.11.33修正真實bug】未完成K棒提醒：如果抓到的最後一筆資料日期是
            # 「今天」，代表這根K棒可能還在交易時段中持續變動（尤其影響量能、KD、RSI），收盤後
            # 數字才會定案。
            # 【V2.11.33修正】原本只比較日期（今天 vs K棒日期），完全沒有比較時間，導致收盤後很久
            # （例如下午5點，台股13:30早就收盤）系統還誤判「可能還在交易時段中」——因為
            # datetime.datetime.now() 用的是伺服器當地時間，跟股市實際開盤時段完全是兩回事，
            # 光看日期不看時鐘完全無法判斷「現在到底收盤了沒」。改成同時比較日期「和」市場實際
            # 交易時段（台股09:00~13:30／美股09:30~16:00，用zoneinfo正確處理時區與日光節約時間），
            # 兩者都符合才視為「今天這根K棒可能還在變動」，收盤後即使日期還是今天，也會正確判定
            # 為已經定案的資料。
            _is_us_stock_for_freshness = code.isalpha() or code.endswith('.US')
            _market_tz = ZoneInfo("America/New_York") if _is_us_stock_for_freshness else ZoneInfo("Asia/Taipei")
            _market_now = datetime.datetime.now(_market_tz)
            _last_bar_date = pd.Timestamp(df.index[-1])
            if _last_bar_date.tzinfo is not None:
                _last_bar_date = _last_bar_date.tz_localize(None)
            is_today_bar = _last_bar_date.date() == _market_now.date() and _is_market_session_open(_is_us_stock_for_freshness, _market_now)
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
            # 【V2.11.10新增】RSI完整序列（不只是最後一個數值），供動能加速度子分數算斜率用，
            # 跟上面的純量rsi分開算，不影響既有計算，只是多留一份序列版本。
            _up_series = delta.clip(lower=0).rolling(14).mean()
            _down_series = -1 * delta.clip(upper=0).rolling(14).mean()
            _rsi_series = 100 - (100 / (1 + (_up_series / (_down_series + 0.001))))
            # 【V2.11.2 修正】原本 range(-13,0) 只加總13天卻除以14，跟新增的 calc_atr_series()（14期）
            # 對不齊，微幅低估ATR。改成 range(-14,0) 真正取14天，兩處ATR計算基準一致。
            atr = float(sum([max(h.iloc[i]-l.iloc[i], abs(h.iloc[i]-c.iloc[i-1]), abs(l.iloc[i]-c.iloc[i-1])) for i in range(-14, 0)]) / 14)
            bias = float(((price - ma60) / ma60) * 100)
            # 【V2.11 修正②】布林上軌：用於跟 RSI 超買訊號交叉確認，不進計分公式，純粹是文字警示用。
            boll_upper = float((c.rolling(20).mean() + 2 * c.rolling(20).std()).iloc[-1])

            # 【V2.11.10新增，觀察用】動能加速度子分數：完全不影響 ai_score 總分或任何進場/加碼門檻，
            # 純粹補上「現有分數看不出正在轉強還是轉弱」的缺口，先觀察準不準，之後再決定要不要正式生效。
            try:
                _osc_full_series = calc_macd_full_series(c)[2]
                _momentum_accel_score, _momentum_accel_detail = calculate_momentum_acceleration_score(_rsi_series, _osc_full_series, v)
            except Exception:
                _momentum_accel_score, _momentum_accel_detail = 0.0, "計算失敗，暫不顯示"

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

            # ===== MACD 動能變化與背離分析模組（新增，與既有分數/final_status計算並行、互不影響）=====
            _macd_daily_result = macd_analyzer.analyze(code, name, df, timeframe="日線")
            # 週線分析需要較長歷史暖身，改抓獨立的2年期資料再做週線resample（不影響既有6個月日線抓取）。
            try:
                _df_long = fetch_stock_data_extended(code)
            except Exception:
                _df_long = pd.DataFrame()
            _weekly_df = resample_to_weekly(_df_long) if _df_long is not None and not _df_long.empty else pd.DataFrame()
            if len(_weekly_df) >= macd_analyzer.min_bars:
                _macd_weekly_result = macd_analyzer.analyze(code, name, _weekly_df, timeframe="週線")
            else:
                _macd_weekly_result = macd_analyzer._empty_result(code, name, "週線", f"週線資料不足（僅{len(_weekly_df)}週，需要至少{macd_analyzer.min_bars}週），可能是延伸資料抓取失敗或該標的上市時間過短")
            macd_report_results.append(_macd_daily_result)
            macd_report_results.append(_macd_weekly_result)

            inst = get_institutional_data(code)
            # 【V2.11.11修正：真實bug】is_us_stock 原本定義在T1/T2/防守線計算「之後」（第2263行附近），
            # 但V2.11.9把T1/T2/防守線計算往前移到這裡統一時，忘記把這個定義也一起搬過來，
            # 導致 calculate_target_plan() 在 is_us_stock 被賦值前就先用到它，整批股票分析全部
            # 因為 NameError（name 'is_us_stock' is not defined）而失敗。這裡把定義提前到最前面。
            is_us_stock = code.isalpha() or code.endswith('.US')
            # 【V2.11.9 修正】_old_plan 提前到這裡計算（原本在後面才算），因為現在防守線／T1/T2
            # 都要讀取「trade_plan裡已經持久化保存的上一次數值」當基準，UI跟狀態機才能真正算出
            # 同一個答案，不是只是公式一樣但各自從零開始算。
            _old_plan = _normalize_trade_plan_row(trade_plan_data.get(code, _trade_plan_defaults(code)))
            _prev_stop_for_calc = _old_plan.get("current_trailing_stop") or _old_plan.get("initial_stop")

            # 【V2.11.10新增，Trend Runner】波段低點：過去20天（不含今日）實際低點，當防守線的
            # 第三種候選方法（跟MA20−ATR、現價−1.5×ATR取最大值），用真實買盤支撐過的價位當防守線，
            # 不是只靠均線推算出來的理論值。
            _swing_low_window = l.iloc[-21:-1] if len(l) > 20 else l.iloc[:-1]
            _swing_low = float(_swing_low_window.min()) if len(_swing_low_window) > 0 else 0.0

            # 【V2.11.8 修正】T1/T2 目標價原本這裡跟正式交易計畫狀態機（calculate_exit_plan）各自維護
            # 一套公式：這裡有「獲利>10%才用結構分析、否則用cost×1.10」的門檻判斷，狀態機那邊卻永遠
            # 直接用前高、完全沒有這個門檻，導致「🛡️風控點位」跟「🗓️交易計畫」兩個分頁可能顯示不同
            # 的T1/T2數字。現在改成兩邊都呼叫同一個 calculate_target_plan()，公式統一、不再各自維護。
            _t_previous_high_window = h.iloc[-61:-1] if len(h) > 60 else h.iloc[:-1]
            _t_previous_high = float(_t_previous_high_window.max()) if len(_t_previous_high_window) > 0 else price

            # 【V2.11.9 修正】移動防守線原本這裡跟正式交易計畫狀態機各自維護一套公式：這裡用
            # calc_trailing_stop()（無狀態、每次重新掃描過去60天重建），狀態機用
            # calculate_trailing_stop_stateful()（有狀態增量），兩者連「有沒有10%獲利門檻」都不一樣，
            # 導致同一檔股票兩個分頁顯示不同防守線，甚至連「有沒有跌破」都可能不一致（P0-2）。
            # 現在改成兩邊都呼叫同一個 calculate_stop_plan()，且都讀取 trade_plan 持久化的
            # 上一次防守線（_prev_stop_for_calc）當基準，公式跟記憶體都統一。
            # atr_stop_price／take_profit_price 這兩個變數名稱保留不變，
            # 讓後面所有既有的分數/顯示邏輯不用跟著大改；take_profit_price = T1（較近的第一目標）。
            t1, t2, _target_branch = calculate_target_plan(price, cost, atr, _t_previous_high, is_us_stock,
                                                             entry_price=_old_plan.get("entry_price"), initial_stop=_old_plan.get("initial_stop"))
            # 【V2.11.18】這裡的「趨勢已確認」判斷跟後面 step3_pass（SOP三燈的趨勢燈）用同一套公式
            # （price>MA20 且 MA10>MA20>MA60），故意在這裡重算一次而不是等 step3_pass 算完再引用，
            # 是因為 step3_pass 定義在後面（第2773行附近）——V2.11.11 曾因為把計算搬到迴圈更前面、
            # 卻忘記把某個變數的定義一起搬過去，導致 NameError 整批分析失敗，這裡刻意不重蹈覆轍，
            # 用「原地重算同一條件」取代「往前搬變數定義」，公式必須跟下面 step3_pass 保持一致。
            _ui_trend_confirmed = price > ma20 and (ma10 > ma20 > ma60)
            atr_stop_price, _atr_stop_source = calculate_stop_plan(price, cost, atr, ma20, _prev_stop_for_calc, swing_low=_swing_low, trend_confirmed=_ui_trend_confirmed)

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

            # 【V2.11.21】改呼叫抽出來的 calculate_daily_score()，純函式重構，數字跟改版前逐行一致，
            # 目的是讓回測引擎能呼叫同一份公式，不會有UI跟回測各自維護一份、算出不同答案的風險。
            _score_result = calculate_daily_score(
                price, cost, ma10, ma20, ma60, macd, bias, k, d, rsi, volume, vol_ma5,
                atr_stop_price, take_profit_price, pivot_point, inst, is_us_stock,
                tw_bearish=bool(tw_trend and "空頭" in tw_trend.get('trend', '')),
                us_bearish=bool(us_trend and "空頭" in us_trend.get('trend', '')),
                vix_high=bool(vix_trend and vix_trend.get('price', 0) > 25),
            )
            ai_score, confidence = _score_result["ai_score"], _score_result["confidence"]
            step1_pass, step2_pass, step3_pass = _score_result["step1_pass"], _score_result["step2_pass"], _score_result["step3_pass"]
            macro_warnings, is_bull_aligned = _score_result["macro_warnings"], _score_result["is_bull_aligned"]
            score_inst, score_tech = _score_result["score_inst"], _score_result["score_tech"]
            score_vol, score_risk = _score_result["score_vol"], _score_result["score_risk"]
            score_forced_zero = bool(cost > 0 and price <= atr_stop_price)


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
            # 【修正②：UI加碼建議 與 狀態機加碼判斷 共用同一組品質關卡】
            # 這個旗標只會在下面 elif 鏈真正走到「可以加碼」的最終 else 分支時被設成 True，
            # 不是另外重寫一份條件——確保「UI會不會顯示可以加碼」跟「狀態機會不會核准加碼」
            # 永遠是同一份判斷邏輯算出來的同一個答案，不會再各說各話。
            _addon_quality_gate_pass = False
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
                elif _macd_daily_result.error is None and (_macd_daily_result.divergence_type == "頂背離" or _macd_daily_result.osc_status == "翻黑"):
                    # 【MACD深度整合】新增關卡：日線MACD出現頂背離或已翻黑，代表動能可能已經在轉弱，
                    # 這時候不該再加重部位，即使前面幾道關卡都通過也一樣暫停加碼。
                    ai_advice.append(f"⏸️ 暫不建議加碼：日線MACD出現「{_macd_daily_result.divergence_type if _macd_daily_result.divergence_type == '頂背離' else _macd_daily_result.osc_status}」，動能可能轉弱，暫緩加碼觀察後續。")
                else:
                    _addon_quality_gate_pass = True
                    _current_value = _held_qty * price
                    _remaining_room = max(0.0, cap - _current_value)
                    # 【V2.11.9 修正，P0-3統一】原本這裡自己重算一套「_risk_based_addon_cap」，跟狀態機的
                    # calculate_addon_shares() 是兩份獨立維護的公式（風險距離分母用的防守線也不一樣，見
                    # P0-2）。現在直接呼叫同一個函式，用統一後的 atr_stop_price（calculate_stop_plan算出）
                    # 當防守線，suggested_shares（原始、未經信心縮減）×0.5當上限，confidence_multiplier
                    # 統一在函式內部最後才乘一次，UI跟狀態機不會再算出兩個不同的加碼股數。
                    _addon_shares = calculate_addon_shares(
                        _held_qty, price, atr_stop_price, price, atr_stop_price,
                        cap, risk_pct, _remaining_room,
                        suggested_shares_cap=suggested_shares * 0.5 if suggested_shares > 0 else None,
                        confidence_multiplier=position_pct,
                    )
                    _remaining_risk_budget = max(risk_amount - _held_qty * _per_share_risk, 0.0)
                    if _remaining_risk_budget <= 0:
                        ai_advice.append("⏸️ 不建議加碼：目前持倉的風險已達（或超過）這檔股票原始設定的風險預算上限，加碼會讓總風險超出你原本能接受的範圍。")
                    elif _remaining_room <= 0:
                        ai_advice.append("⏸️ 不建議加碼：目前持有市值已達到你設定的分配資金上限，加碼會超出原本的資金規劃。")
                    elif _addon_shares > 0:
                        addon_shares_approved = _addon_shares
                        ai_advice.append(f"📈 可考慮加碼：SOP三燈全亮、決策信心{confidence}%、現價已與成本拉開足夠空間，資金額度內約可加碼 {_addon_shares} 股（同時受分配資金上限、原始建倉股數一半、加碼後總風險上限、決策信心縮減四重限制，避免單押過重）。")
                    else:
                        ai_advice.append("⏸️ 資金或風險額度所剩不多，加碼股數不足1股，暫不建議加碼。")

            # ===== V2.11.x 交易計畫狀態機：與上方既有 final_status 邏輯並行運作，不修改既有變數 =====
            # 只把「持久化的交易計畫」疊加上去，既有的 ai_score/final_status/atr_stop_price/t1/t2/
            # suggested_shares_adjusted/addon_shares_approved 全部原封不動，UI 既有分頁行為不受影響。
            _plan_data_date = _date_str(_last_bar_date)
            # 【V2.11.8】previous_high 沿用上面 T1/T2 統一計算時已經算好的 _t_previous_high，
            # 不再重複計算一次——確保「用來算 legacy T1/T2」跟「用來算進場R1／傳給狀態機」的前高，
            # 是同一個數字，不會因為兩處各自重算而有微小落差。
            _plan_previous_high = _t_previous_high

            # 【V2.11.22移除，真實P0 bug修復】原本這裡算的「進場專用R1」，用來餵給
            # calculate_entry_plan() 的 r1>=1.5 進場前預檢。這個算法在數學上永遠精確等於1.0
            # （因為突破價＝前高×1.005，而 calc_structural_target 用「同一個前高」當基準去找
            # 目標價，前高必然小於突破價，永遠掉進「突破價+2×ATR」備援公式，跟「突破價−2×ATR」
            # 的風險距離相除必然等於1.0），導致entry_gate_pass自V2.11.9這道關卡加入後就永遠是
            # False，系統完全無法偵測任何新的突破訊號。詳見說明書第16節V2.11.22項目。
            # 跟你討論後決定拿掉這道關卡本身（見 calculate_entry_plan 的docstring），而不是修補
            # 這個結構性有瑕疵的預檢公式，所以這裡也不再需要算這個數字，直接不傳（None），
            # calculate_breakout_quality_score() 會把 r1=None 當成「資料不足」給中性半分處理，
            # 不會因為少了這個數字就誤判成低分。
            _entry_r1 = None

            # 【V2.11.9】_old_plan 已經在前面（第2020行附近）算好並用於防守線/T1/T2計算，這裡不用重算。
            # 【MACD深度整合】把「日線」MACD結果讀出來，餵給交易計畫狀態機。用日線而不用週線，
            # 是因為週線需要至少35週歷史暖身、新股常常「資料不足」，若拿週線去擋新倉/加碼，
            # 會讓剛上市股票長期卡死無法進場；日線資料完整度高很多，適合當硬性關卡。
            # 資料不足（error不為None）時視為中性（None），不影響任何判斷，不會誤擋。
            _macd_osc_status = _macd_daily_result.osc_status if _macd_daily_result.error is None else None
            _macd_divergence_type = _macd_daily_result.divergence_type if _macd_daily_result.error is None else None
            # 【V2.11.30新增】柱狀體OSC的實際數值，供 calculate_breakout_quality_score() 的MACD動能
            # 強度子項使用（不再用entry_gate已經篩過一次的正值/翻紅分類，改用連續數值才有區分度）。
            _macd_osc_value = _macd_daily_result.osc if _macd_daily_result.error is None else None

            _plan_indicators = {
                "code": code, "price": price, "atr": atr, "ma20": ma20, "previous_high": _plan_previous_high,
                "decision_score": ai_score, "trend_gate": step3_pass, "chip_gate": step1_pass, "volume_gate": step2_pass,
                "r1": _entry_r1, "market_regime": "BEARISH" if _regime_is_bearish(macro_data, is_us_stock) else "NORMAL",
                "is_us_stock": is_us_stock, "data_date": _plan_data_date,
                "macd_osc_status": _macd_osc_status, "macd_divergence_type": _macd_divergence_type,
                "macd_osc_value": _macd_osc_value,
                "swing_low": _swing_low, "volume": volume, "vol_ma5": vol_ma5,
                "prev_close": float(c.iloc[-2]) if len(c) >= 2 else price,
            }
            _plan_portfolio_info = {"cost": cost, "cap": cap, "risk": risk_pct, "qty": _held_qty,
                                     "available_cash": max(0.0, cap - _held_qty * price),
                                     "addon_quality_gate": _addon_quality_gate_pass,
                                     "confidence_multiplier": position_pct}

            # 【修正：今天K棒尚未收斂時，即使全域執行模式判定為唯讀，這一檔也要強制重新評估】
            # 問題根源：系統用「資料日期字串有沒有變」判斷要不要重跑狀態機，但當最後一根K棒就是
            # 「今天」（is_today_bar=True）時，同一個日期底下的實際數值（量能/KD/RSI/收盤價）仍可能
            # 隨資料來源更新而改變——若這裡沿用純日期比對，會把「今天稍早、用還沒收斂的資料算出來的
            # 決策」凍結一整天，跟同一次畫面上「AI決策與SOP」分頁每次都重新即時計算的結果對不上。
            # 只要 is_today_bar 還是 True，就強制走跟 TAIWAN_CLOSE_UPDATE 一樣的完整重新評估，
            # 確保交易計畫會跟著當天最新資料調整，不會卡在早上的過時快照。
            #
            # 【V2.11.9 修正，P1-3】原本這裡沒有區分台股/美股：只要「最後一根K棒的日期＝系統今天」
            # 就會強制完整重算，但這個判斷完全繞過了 process_us_close_update() 原本設計的白名單保護
            # （只准動暫停/恢復欄位，不准碰價格/T1/T2/防守線）。美股在台灣時區下，資料日期跟台灣「今天」
            # 對齊純屬巧合，可能發生在美股盤中還沒收盤、資料還在形成的時候，若這時候強制跑完整
            # evaluate_trade_state()，可能拿美股盤中的暫時性下殺去觸發不該觸發的 FULL_EXIT_NEXT_DAY，
            # 等實際收盤資料出來才發現是誤判。修正：is_today_bar 的強制重算只適用台股，
            # 美股一律只透過 US_CLOSE_UPDATE（白名單保護）或每日一次的 TAIWAN_CLOSE_UPDATE 全域週期更新。
            _force_intraday_recheck = bool(is_today_bar) and not is_us_stock
            if execution_mode == TAIWAN_CLOSE_UPDATE or _force_intraday_recheck or (not _old_plan.get("taiwan_data_date") and _plan_data_date):
                # 台股有新日K、今天K棒尚未收斂需要持續追蹤、或這檔股票從未被 evaluate 過（首次遷移/新增持股的一次性 bootstrap）
                _new_plan = process_taiwan_close_update(_old_plan, _plan_indicators, macro_data, _plan_portfolio_info)
                # 【V2.11.41新增，Trade Plan Snapshot】在覆寫日期之前，先檢查這是不是「真的推進到
                # 新的一天」（不是同一天內的盤中重複評估）——如果是，把_old_plan（前一天最終確定的
                # 內容）凍結存一份快照，這樣不管接下來當天盤中重新評估幾次、覆寫幾次，使用者都還能
                # 回頭看到「前一天收盤後」那個版本，解決「盤中打開App，昨晚看到的計畫已經被當天
                # 盤中還沒收盤的資料覆寫掉」這個真實操作痛點。只在日期真的前進時觸發一次，同一天內
                # 重複的盤中重新評估（_old_plan的taiwan_data_date已經等於今天）不會重複快照。
                _old_data_date = _old_plan.get("taiwan_data_date", "")
                if _old_data_date and _plan_data_date and _old_data_date != _plan_data_date:
                    append_trade_plan_snapshot(code, _old_plan)
                _new_plan["taiwan_data_date"] = _plan_data_date
                if latest_us_date:
                    _new_plan["us_data_date"] = latest_us_date
                if _force_intraday_recheck and execution_mode == VIEW_ONLY:
                    _any_intraday_reevaluation = True
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
                "atr_stop_source": _atr_stop_source,
                "momentum_accel_score": _momentum_accel_score, "momentum_accel_detail": _momentum_accel_detail,
                # ===== V2.11.x 交易計畫（trade_plan）欄位，統一用 plan_ 前綴，跟既有欄位分開，互不覆蓋 =====
                "plan_state": trade_plan_data[code]["state"], "plan_origin_state": trade_plan_data[code]["origin_state"],
                "plan_signal_type": trade_plan_data[code]["signal_type"], "plan_signal_reason": trade_plan_data[code]["signal_reason"],
                "plan_entry_price": trade_plan_data[code]["entry_price"], "plan_breakout_price": trade_plan_data[code]["breakout_price"],
                "plan_pullback_low": trade_plan_data[code]["pullback_low"], "plan_pullback_high": trade_plan_data[code]["pullback_high"],
                "plan_retest_quality": trade_plan_data[code]["retest_quality"],
                "plan_breakout_quality_score": trade_plan_data[code]["breakout_quality_score"],
                "plan_breakout_quality_grade": trade_plan_data[code]["breakout_quality_grade"],
                "plan_chase_limit": trade_plan_data[code]["chase_limit"], "plan_invalid_price": trade_plan_data[code]["invalid_price"],
                "plan_t1_price": trade_plan_data[code]["t1_price"], "plan_t2_price": trade_plan_data[code]["t2_price"],
                "plan_t1_taken": trade_plan_data[code]["t1_taken"], "plan_t2_taken": trade_plan_data[code]["t2_taken"],
                "plan_current_trailing_stop": trade_plan_data[code]["current_trailing_stop"],
                "plan_current_trailing_stop_source": trade_plan_data[code]["current_trailing_stop_source"],
                "plan_review_state": trade_plan_data[code]["review_state"],
                "plan_review_at": trade_plan_data[code]["review_at"],
                "plan_next_day_gate": classify_next_day_execution(trade_plan_data[code], price),
                "plan_suggested_shares": trade_plan_data[code]["suggested_shares"],
                "plan_addon_shares_approved": trade_plan_data[code]["addon_shares_approved"],
                "plan_partial_exit_shares": trade_plan_data[code]["partial_exit_shares"],
                "plan_full_exit_shares": trade_plan_data[code]["full_exit_shares"],
                "plan_execution_date": trade_plan_data[code]["execution_date"], "plan_valid_until": trade_plan_data[code]["valid_until"],
                "plan_taiwan_data_date": trade_plan_data[code]["taiwan_data_date"], "plan_us_data_date": trade_plan_data[code]["us_data_date"],
                # ===== MACD 動能與背離分析（新增）=====
                "macd_daily": _macd_daily_result, "macd_weekly": _macd_weekly_result,
            })
        except Exception as e: st.error(f"分析 {code} 發生錯誤: {e}")

    save_history(system_history)
    # 【V2.11.x 新增】trade_plan 只在真的有新資料時才寫回（VIEW_ONLY 模式嚴格唯讀，不做任何寫入，
    # 避免同一天重複開啟頁面時，把「唯讀複本」誤當作正式結果覆蓋回 Google Sheet）。
    # 【修正】但若本次執行中有任何一檔因「今天K棒尚未收斂」被強制重新評估過，即使全域模式仍是
    # VIEW_ONLY，也要把這次追上最新資料的結果存回去，否則追平的計算白做了，畫面重整一次又會消失。
    if execution_mode != VIEW_ONLY or _any_intraday_reevaluation:
        save_trade_plan(trade_plan_data)
    if execution_mode == VIEW_ONLY and _any_intraday_reevaluation:
        st.caption("🔄 偵測到部分股票的今日K棒尚未收斂，已針對這幾檔強制重新評估交易計畫並存回，其餘股票維持唯讀。")

    # 【V2.10 新增②】AI 每日一句：從今天戰力最高的持股，自動拼一句話當作頭條，
    # 不用先看完整份排行榜跟卡片才知道「今天最值得注意的是哪一檔」。
    if card_data:
        _headline_top = max(card_data, key=lambda x: x['ai_score'])
        if _headline_top['ai_score'] > 0:
            # 【V2.11.9修正，P1-1】同一個「籌碼」誤標問題：今天戰力最高的若剛好是美股，
            # 這裡不該說優勢來自「籌碼/長線動能」，美股的這個子分數其實是動能/趨勢，跟籌碼無關。
            _inst_label_headline = "動能/趨勢" if _headline_top.get('is_us') else "籌碼/長線動能"
            _sub_scores = {_inst_label_headline: _headline_top['score_inst'], "趨勢技術": _headline_top['score_tech'], "量能表現": _headline_top['score_vol'], "風控狀態": _headline_top['score_risk']}
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

    # ===== 【新增】MACD 動能與背離分析報表：整合日線／週線結果，支援表格檢視與 CSV/JSON 匯出 =====
    if macd_report_results:
        st.markdown("### 📐 MACD 動能與背離分析報表")
        _df_macd = build_macd_report(macd_report_results)
        _macd_actionable = _df_macd[_df_macd["signal_action"].isin(["核心進場", "分批試單", "減碼50%", "出場"])]
        if not _macd_actionable.empty:
            st.markdown("**🔔 目前有明確訊號的標的**")
            st.dataframe(
                _macd_actionable[["stock_id", "stock_name", "timeframe", "osc_status", "divergence_type", "signal_action", "risk_management"]]
                .rename(columns={"stock_id": "代號", "stock_name": "名稱", "timeframe": "週期", "osc_status": "柱狀體狀態",
                                  "divergence_type": "背離型態", "signal_action": "操作動作", "risk_management": "失效停損參考價"})
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("目前所有追蹤股票的日線／週線都沒有明確的 MACD 訊號（觀望中）。")

        with st.expander("展開完整 MACD 分析報表（所有股票，日線＋週線）"):
            st.dataframe(
                _df_macd[["stock_id", "stock_name", "timeframe", "dif", "dea", "osc", "osc_status", "divergence_type", "signal_action", "risk_management", "detail"]]
                .rename(columns={"stock_id": "代號", "stock_name": "名稱", "timeframe": "週期", "dif": "DIF", "dea": "DEA", "osc": "OSC",
                                  "osc_status": "柱狀體狀態", "divergence_type": "背離型態", "signal_action": "操作動作",
                                  "risk_management": "失效停損參考價", "detail": "判斷依據"})
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
            _macd_csv = _df_macd.to_csv(index=False).encode("utf-8-sig")
            _macd_json = json.dumps([r.to_dict() for r in macd_report_results], ensure_ascii=False, default=str, indent=2).encode("utf-8")
            _dl_col1, _dl_col2 = st.columns(2)
            _dl_col1.download_button("📤 匯出 MACD 報表 (CSV)", _macd_csv, file_name="macd_signal_report.csv", mime="text/csv")
            _dl_col2.download_button("📤 匯出 MACD 報表 (JSON)", _macd_json, file_name="macd_signal_report.json", mime="application/json")
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
                # 【V2.11.12新增】還沒確認過的新訊號加上🆕標記，讓你在總覽清單就能分辨「這是今天才出現
                # 的新東西」還是「你已經看過、還在等執行的舊訊號」，不用點進每張卡片才知道。
                _ack_badge = "🆕" if data.get('plan_review_state') == "PENDING" else ""
                if _ps == "FULL_EXIT_NEXT_DAY":
                    action_sell.append(f"🔴{_ack_badge} **【交易計畫】全部出清**：{data['name']} {data.get('plan_signal_reason','')}，建議出清 {data.get('plan_full_exit_shares',0)} 股（{data.get('plan_execution_date','下一交易日')} 執行）。")
                elif _ps == "PARTIAL_EXIT_NEXT_DAY":
                    action_sell.append(f"🟠{_ack_badge} **【交易計畫】分批停利（{data.get('plan_signal_type','')}）**：{data['name']} 建議出脫 {data.get('plan_partial_exit_shares',0)} 股（{data.get('plan_execution_date','下一交易日')} 執行）。")
                elif _ps == "ADD_NEXT_DAY":
                    action_watch.append(f"📈{_ack_badge} **【交易計畫】下一交易日可加碼**：{data['name']} 核准加碼 {data.get('plan_addon_shares_approved',0)} 股。")
                elif _ps == "ENTER_NEXT_DAY" and data.get('held_qty', 0) <= 0:
                    action_buy.append(f"🎯{_ack_badge} **【交易計畫】下一交易日可進場**：{data['name']} 突破價 {data.get('plan_breakout_price',0):.2f}，建議股數 {data.get('plan_suggested_shares',0)} 股。")
                elif _ps == "SUSPENDED_BY_REGIME":
                    action_watch.append(f"⏸️{_ack_badge} **【交易計畫】市場逆風暫停**：{data['name']} 原訂計畫已保留，等待逆風解除後自動恢復。")

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
    st.markdown("### 📈 歷史訊號結果觀察（非正式回測）")
    _bt_stats = compute_signal_backtest(system_history)
    if not _bt_stats:
        st.info("目前累積的歷史記錄還太少（至少要有同一檔股票連續兩天以上的記錄才能比較），先讓系統多跑幾天，這裡的統計會隨時間慢慢累積。")
    else:
        _bt_rows = []
        for _status, _rets in _bt_stats.items():
            _win_rate = sum(1 for r in _rets if r > 0) / len(_rets) * 100
            _avg_ret = sum(_rets) / len(_rets)
            _bt_rows.append({"判定狀態": _status, "樣本數": len(_rets), "後續平均報酬%": round(_avg_ret, 2), "後續上漲比例%": round(_win_rate, 1)})
        _df_bt = pd.DataFrame(_bt_rows).sort_values("後續平均報酬%", ascending=False).reset_index(drop=True)
        st.dataframe(_df_bt, use_container_width=True, hide_index=True)
        st.caption("「後續平均報酬」＝拿每筆歷史記錄當天的價格，對照同一檔股票目前歷史中最新一筆的價格計算漲跌幅，再依「當時的判定狀態」分組平均。樣本數會隨使用天數增加而增加；目前每檔股票最多保留最近10筆記錄，天數越久統計越有參考價值。")
        st.caption("⚠️ 此統計未模擬固定持有期間、實際成交、交易成本、滑價、停損與重疊交易，**不是正式策略回測**，僅供檢視過去系統訊號與後續價格變化的粗略參考，不能理解成「這套系統的勝率」。")

if __name__ == "__main__":
    pass
