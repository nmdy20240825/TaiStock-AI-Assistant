import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai

# 1. 網頁基本配置
st.set_page_config(page_title="台股AI波段交易助手 V7.0", layout="wide", page_icon="📈")

# 2. 側邊欄配置
st.sidebar.title("🤖 台股AI波段助手 V7.0")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input("🔑 請輸入 Gemini API 金鑰", type="password")
if api_key:
    genai.configure(api_key=api_key)

st.sidebar.markdown("---")
page = st.sidebar.radio("功能選單", ["Dashboard（首頁）", "📌 AI 雷達", "我的持股", "自選股觀察", "💬 AI 教練模式", "歷史紀錄與績效", "系統設定"])

# 3. 初始化記憶體資料
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"代號": "2317", "名稱": "鴻海", "持有均價": 180.0, "股數": 1000},
        {"代號": "3017", "名稱": "奇鋐", "持有均價": 550.0, "股數": 1000},
        {"代號": "2382", "名稱": "廣達", "持有均價": 250.0, "股數": 2000}
    ])

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = pd.DataFrame([
        {"代號": "3037", "名稱": "欣興", "備註": "ABF載板龍頭"},
        {"代號": "2408", "名稱": "南亞科", "備註": "DRAM波段觀察"},
        {"代號": "3711", "名稱": "日月光投控", "備註": "封測龍頭"}
    ])

if 'sys_settings' not in st.session_state:
    st.session_state.sys_settings = {"stop_loss": 8.0, "take_profit": 15.0, "model": "gemini-1.5-pro"}

# 4. 核心功能模組 (完全移除 pandas_ta，改用純 Pandas 數學運算，百毒不侵)
def get_taiwan_stock_data(symbol):
    for suffix in ['.TW', '.TWO']:
        ticker = f"{symbol}{suffix}"
        df = yf.Ticker(ticker).history(period="6mo")
        if not df.empty:
            return df, ticker
    return pd.DataFrame(), None

def get_market_trend():
    try:
        twii = yf.Ticker("^TWII").history(period="1mo")
        if twii.empty: return "震盪 (無法取得指數)"
        latest_close = twii['Close'].iloc[-1]
        ma20 = twii['Close'].rolling(20).mean().iloc[-1]
        if latest_close > ma20 * 1.01: return "🔥 偏多 (站穩月線)"
        elif latest_close < ma20 * 0.99: return "🧊 偏空 (跌破月線)"
        else: return "⚖️ 震盪 (月線糾結)"
    except:
        return "⚖️ 震盪 (數據異常)"

def compute_signals(df):
    if df.empty or len(df) < 30:
        return None
    data = df.copy()
    try:
        # 內建 MACD 運算
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        m_l = exp1 - exp2
        m_s = m_l.ewm(span=9, adjust=False).mean()
        m_h = m_l - m_s
        
        # 內建 RSI 運算
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        r_v = 100 - (100 / (1 + rs))
        
        # 內建 KD 運算
        low_min = data['Low'].rolling(window=9, min_periods=1).min()
        high_max = data['High'].rolling(window=9, min_periods=1).max()
        rsv = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        rsv = rsv.fillna(50)
        k_v = rsv.ewm(com=2, adjust=False).mean()
        d_v = k_v.ewm(com=2, adjust=False).mean()
        
        # 均線與均量
        price_ma20 = data['Close'].rolling(20).mean()
        vol_ma20 = data['Volume'].rolling(20).mean()
        
        # 取得今日最新數值
        close_price = data['Close'].iloc[-1]
        current_vol = data['Volume'].iloc[-1]
        v_ma20_val = vol_ma20.iloc[-1]
        m_h_val = m_h.iloc[-1]
        k_val = k_v.iloc[-1]
        d_val = d_v.iloc[-1]
        r_val = r_v.iloc[-1]
        
        # 波段分數計分
        score = 0
        if close_price > price_ma20.iloc[-1]: score += 20
        if current_vol > v_ma20_val: score += 10
        if m_h_val > 0: score += 15
        if m_l.iloc[-1] > m_s.iloc[-1]: score += 10
        if k_val > d_val: score += 15
        if k_val > d_val and k_val < 35: score += 10
        if k_val > 85: score -= 10
        if 50 <= r_val <= 75: score += 20
        elif r_val > 75: score += 10
        score = max(0, min(100, score))
        
        if score >= 90: score_badge = "🟢 強烈留意"
        elif score >= 75: score_badge = "🟡 可續抱"
        elif score >= 60: score_badge = "🟠 觀察"
        else: score_badge = "🔴 檢視持股"
        
        return {
            "現價": round(close_price, 2),
            "漲跌幅": round(((close_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100, 2),
            "評級": score_badge,
            "純分數": score,
            "今日成交量": current_vol,
            "月均成交量": v_ma20_val,
            "K值": round(k_val, 2),
            "D值": round(d_val, 2),
            "MACD狀態": "多頭" if m_h_val > 0 else "空頭",
            "量能狀態": "量增" if current_vol > v_ma20_val else "量縮"
        }
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 頁面 UI 渲染
# -----------------------------------------------------------------------------
if page == "Dashboard（首頁）":
    st.title("🌅 AI 每日晨報 V7.0")
    st.markdown("🎯 **目標：30 秒內掌握今日戰略**")
    
    with st.spinner("正在掃描大盤與您的專屬股池..."):
        market_trend = get_market_trend()
        total_pnl = 0.0
        best_stock = {"名稱": "無", "分數": -1, "漲幅": 0}
        worst_stock = {"名稱": "無", "分數": 101, "跌幅": 0}
        
        all_stocks_to_scan = list(set(st.session_state.portfolio["代號"].tolist() + st.session_state.watchlist["代號"].tolist()))

        for sym in all_stocks_to_scan:
            df, _ = get_taiwan_stock_data(sym)
            sig = compute_signals(df)
            if sig:
                if sym in st.session_state.portfolio["代號"].values:
                    idx = st.session_state.portfolio[st.session_state.portfolio["代號"] == sym].index[0]
                    cost = st.session_state.portfolio.at[idx, "持有均價"]
                    shares = st.session_state.portfolio.at[idx, "股數"]
                    pnl = (sig["現價"] - cost) * shares
                    total_pnl += pnl
                
                if sig["純分數"] > best_stock["分數"]:
                    best_stock = {"名稱": sym, "分數": sig["純分數"], "漲幅": sig["漲跌幅"]}
                if sig["純分數"] < worst_stock["分數"]:
                    worst_stock = {"名稱": sym, "分數": sig["純分數"], "跌幅": sig["漲跌幅"]}

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("📊 今日大盤趨勢", market_trend)
        c2.metric("💰 預估庫存總損益", f"NT$ {total_pnl:,.0f}", f"{'▲ 獲利中' if total_pnl >=0 else '▼ 虧損中'}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.success(f"🎯 **今日最值得注意**\n\n**{best_stock['名稱']}** (分數: {best_stock['分數']})\n\n漲幅: {best_stock['漲幅']}%")
        with c4:
            st.error(f"⚠️ **今日最大風險**\n\n**{worst_stock['名稱']}** (分數: {worst_stock['分數']})\n\n跌幅: {worst_stock['跌幅']}%")
        
        st.markdown("---")
        st.subheader("🤖 AI 晨報 30 秒決策結論")
        if api_key:
            try:
                model = genai.GenerativeModel(st.session_state.sys_settings["model"])
                prompt = f"你是台股波段教練。大盤：{market_trend}，損益：{total_pnl}。強勢股：{best_stock['名稱']}。弱勢股：{worst_stock['名稱']}。請用一句話告訴我今天該怎麼做。"
                st.info(f"**{model.generate_content(prompt).text}**")
            except:
                st.warning("AI 產生失敗。")
        else:
            st.warning("⚠️ 請輸入 API 金鑰。")

elif page == "📌 AI 雷達":
    st.title("📡 AI 盤後主力與趨勢雷達")
    scan_pool = ["2330", "2317", "2382", "3017", "3037", "3711", "3443", "2454", "3231", "2376", "2603", "2303"]
    if st.button("🚀 啟動今日雷達掃描"):
        progress_bar = st.progress(0)
        results = []
        for i, sym in enumerate(scan_pool):
            df, _ = get_taiwan_stock_data(sym)
            sig = compute_signals(df)
            if sig:
                sig['代號'] = sym
                results.append(sig)
            progress_bar.progress((i + 1) / len(scan_pool))
        
        df_res = pd.DataFrame(results)
        if not df_res.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔥 強烈留意 (>90分)")
                st.dataframe(df_res[df_res['純分數'] >= 90][['代號', '現價', '漲跌幅', '評級']], hide_index=True)
            with c2:
                st.subheader("⚠️ 弱勢檢視 (<60分)")
                st.dataframe(df_res[df_res['純分數'] < 60][['代號', '現價', '漲跌幅', '評級']], hide_index=True)

elif page == "我的持股":
    st.title("💼 我的庫存持股管理")
    edited_portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)
    if st.button("🔄 儲存庫存變更"):
        st.session_state.portfolio = edited_portfolio
        st.rerun()

elif page == "自選股觀察":
    st.title("⭐ 自選股追蹤")
    edited_watchlist = st.data_editor(st.session_state.watchlist, num_rows="dynamic", use_container_width=True)
    if st.button("💾 儲存自選股"):
        st.session_state.watchlist = edited_watchlist
        st.rerun()

elif page == "💬 AI 教練模式":
    st.title("🏋️‍♂️ AI 專屬波段教練")
    all_stocks = list(set(st.session_state.portfolio["代號"].tolist() + st.session_state.watchlist["代號"].tolist()))
    selected_stock = st.selectbox("選擇要請教教練的股票代號", all_stocks)
    
    if st.button("🗣️ 呼叫教練給予指導"):
        if not api_key:
            st.warning("⚠️ 請先在左側欄輸入 API 金鑰。")
        else:
            with st.spinner(f"正在審視 {selected_stock}..."):
                df, _ = get_taiwan_stock_data(selected_stock)
                sig = compute_signals(df)
                if sig:
                    model = genai.GenerativeModel(st.session_state.sys_settings["model"])
                    prompt = f"你是嚴苛教練。標的：{selected_stock}。分數：{sig['純分數']}。現價：{sig['現價']}。KD：K={sig['K值']}, D={sig['D值']}。MACD：{sig['MACD狀態']}。量能：{sig['量能狀態']}。請照格式回覆：【教練指示】、【背後原因】(①②③)、【教練總結行動】。"
                    try:
                        st.success(model.generate_content(prompt).text)
                    except:
                        st.error("AI 錯誤。")

elif page == "歷史紀錄與績效":
    st.title("📚 波段績效分析")
    st.line_chart([100, 102, 101, 105, 109, 115])

elif page == "系統設定":
    st.title("⚙️ 風控參數設定")
    st.session_state.sys_settings["stop_loss"] = st.number_input("強制停損 (%)", value=st.session_state.sys_settings["stop_loss"], step=0.5)
    st.session_state.sys_settings["model"] = st.selectbox("核心 AI 模型", ["gemini-1.5-pro", "gemini-1.5-flash"])
    st.success("參數已儲存！")
