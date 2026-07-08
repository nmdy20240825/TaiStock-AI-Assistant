import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import requests

# 1. 網頁基本配置
st.set_page_config(page_title="台股AI波段交易助手 V13.0", layout="wide", page_icon="📈")

st.markdown("""
<style>
    div[data-testid="stSidebarNav"] span { font-size: 18px !important; font-weight: bold; }
    div[role="radiogroup"] label p { font-size: 18px !important; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# 2. 側邊欄配置
st.sidebar.title("🤖 波段助手 V13.0")
st.sidebar.markdown("---")

saved_key = ""
if "GEMINI_API_KEY" in st.secrets:
    saved_key = st.secrets["GEMINI_API_KEY"]

api_key = st.sidebar.text_input("🔑 Gemini API 金鑰", value=saved_key, type="password")
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
    # 關鍵修正 1：使用帶有 -latest 尾綴的安全模型名稱，避免 404 錯誤
    st.session_state.sys_settings = {"stop_loss": 8.0, "take_profit": 15.0, "model": "gemini-1.5-flash-latest"}

STOCK_MAPPING = {
    "2330": "台積電", "2317": "鴻海", "2382": "廣達", "3017": "奇鋐",
    "3037": "欣興", "3711": "日月光", "3443": "創意", "2454": "聯發科",
    "3231": "緯創", "2376": "技嘉", "2603": "長榮", "2303": "聯電",
    "3035": "智原", "2368": "金像電", "8046": "南電", "2313": "華通",
    "2308": "台達電", "2327": "國巨", "2408": "南亞科", "3324": "雙鴻"
}

http_session = requests.Session()
http_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
})

# 4. 核心功能模組
def get_taiwan_stock_data(symbol):
    for suffix in ['.TW', '.TWO']:
        ticker = f"{symbol}{suffix}"
        try:
            stock = yf.Ticker(ticker, session=http_session)
            df = stock.history(period="6mo")
            df = df.dropna(subset=['Close', 'Volume'])
            if not df.empty and len(df) > 30:
                return df, ticker
        except:
            continue
    return pd.DataFrame(), None

def get_market_trend():
    try:
        twii = yf.Ticker("^TWII", session=http_session).history(period="1mo")
        twii = twii.dropna(subset=['Close'])
        if twii.empty: return "⚖️ 無數據"
        latest_close = twii['Close'].iloc[-1]
        ma20 = twii['Close'].rolling(20).mean().iloc[-1]
        if latest_close > ma20 * 1.01: return f"🔥 偏多 (現價 {latest_close:.0f} > 月線 {ma20:.0f})"
        elif latest_close < ma20 * 0.99: return f"🧊 偏空 (現價 {latest_close:.0f} < 月線 {ma20:.0f})"
        else: return f"⚖️ 震盪 (現價 {latest_close:.0f} 月線糾結)"
    except:
        return "⚖️ 大盤連線失敗"

def compute_signals(df):
    if df.empty or len(df) < 30:
        return None
    data = df.copy()
    try:
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        m_l = exp1 - exp2
        m_s = m_l.ewm(span=9, adjust=False).mean()
        m_h = m_l - m_s
        
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        r_v = 100 - (100 / (1 + rs))
        
        low_min = data['Low'].rolling(window=9, min_periods=1).min()
        high_max = data['High'].rolling(window=9, min_periods=1).max()
        rsv = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        rsv = rsv.fillna(50)
        k_v = rsv.ewm(com=2, adjust=False).mean()
        d_v = k_v.ewm(com=2, adjust=False).mean()
        
        price_ma20 = data['Close'].rolling(20).mean()
        vol_ma20 = data['Volume'].rolling(20).mean()
        
        close_price = data['Close'].iloc[-1]
        current_vol = data['Volume'].iloc[-1]
        v_ma20_val = vol_ma20.iloc[-1]
        
        if pd.isna(close_price):
            return None

        m_h_val = m_h.iloc[-1]
        k_val = k_v.iloc[-1]
        d_val = d_v.iloc[-1]
        r_val = r_v.iloc[-1]
        
        ma20_status = "站上月線" if close_price > price_ma20.iloc[-1] else "跌破月線"
        vol_ratio = current_vol / v_ma20_val if v_ma20_val > 0 else 0
        
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
            "漲跌幅(%)": round(((close_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100, 2),
            "月線狀態": ma20_status,
            "主力動能(爆量倍數)": f"{vol_ratio:.1f} 倍",
            "純量化倍數": round(vol_ratio, 2), # 新增純數字欄位，防呆專用
            "K值": round(k_val, 2),
            "D值": round(d_val, 2),
            "MACD柱": round(m_h_val, 2),
            "RSI": round(r_val, 2),
            "評級": score_badge,
            "純分數": score
        }
    except:
        return None

# -----------------------------------------------------------------------------
# 頁面 UI 渲染
# -----------------------------------------------------------------------------
if page == "Dashboard（首頁）":
    st.title("🌅 AI 每日晨報")
    
    with st.spinner("正在向外抓取最新報價..."):
        market_trend = get_market_trend()
        total_pnl = 0.0
        best_stock = {"名稱": "無", "分數": -1, "漲幅": 0, "現價": 0}
        worst_stock = {"名稱": "無", "分數": 101, "跌幅": 0, "現價": 0}
        
        all_stocks_to_scan = list(set(st.session_state.portfolio["代號"].tolist() + st.session_state.watchlist["代號"].tolist()))

        for sym in all_stocks_to_scan:
            df, _ = get_taiwan_stock_data(sym)
            if df.empty: continue
                
            sig = compute_signals(df)
            if sig:
                stock_name = STOCK_MAPPING.get(sym, "自訂標的")
                if sym in st.session_state.portfolio["代號"].values:
                    idx = st.session_state.portfolio[st.session_state.portfolio["代號"] == sym].index[0]
                    cost = st.session_state.portfolio.at[idx, "持有均價"]
                    shares = st.session_state.portfolio.at[idx, "股數"]
                    total_pnl += (sig["現價"] - cost) * shares
                
                if sig["純分數"] > best_stock["分數"]:
                    best_stock = {"名稱": f"{sym} {stock_name}", "分數": sig["純分數"], "漲幅": sig["漲跌幅(%)"], "現價": sig["現價"]}
                if sig["純分數"] < worst_stock["分數"]:
                    worst_stock = {"名稱": f"{sym} {stock_name}", "分數": sig["純分數"], "跌幅": sig["漲跌幅(%)"], "現價": sig["現價"]}

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("📊 今日大盤與月線", market_trend)
        c2.metric("💰 預估庫存總損益", f"NT$ {total_pnl:,.0f}", f"{'▲ 獲利中' if total_pnl >=0 else '▼ 虧損中'}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.success(f"🎯 **今日最值得注意：\n{best_stock['名稱']}**\n\n現價：{best_stock['現價']}｜漲幅：{best_stock['漲幅']}%\n\n綜合分數：{best_stock['分數']} 分")
        with c4:
            st.error(f"⚠️ **今日最大風險：\n{worst_stock['名稱']}**\n\n現價：{worst_stock['現價']}｜跌幅：{worst_stock['跌幅']}%\n\n綜合分數：{worst_stock['分數']} 分")

elif page == "📌 AI 雷達":
    st.title("📡 AI 盤後主力與趨勢雷達")
    
    if st.button("🚀 啟動廣域雷達掃描"):
        progress_bar = st.progress(0)
        results = []
        tickers = list(STOCK_MAPPING.keys())
        
        for i, sym in enumerate(tickers):
            df, _ = get_taiwan_stock_data(sym)
            sig = compute_signals(df)
            if sig:
                sig['代號'] = sym
                sig['名稱'] = STOCK_MAPPING[sym]
                results.append(sig)
            progress_bar.progress((i + 1) / len(tickers))
        
        # 關鍵修正 2：捨棄 Pandas 過濾，改用純 Python 列表過濾，防呆 100%
        good_list = [r for r in results if r['純分數'] >= 90 or r['純量化倍數'] >= 1.5]
        bad_list = [r for r in results if r['純分數'] < 60]
        
        cols = ['代號', '名稱', '現價', '漲跌幅(%)', '主力動能(爆量倍數)', '月線狀態', 'K值', 'D值', 'MACD柱', 'RSI', '評級']
        
        st.subheader("🔥 強烈留意 (>90分) 或 主力爆量")
        if good_list:
            df_good = pd.DataFrame(good_list)[cols]
            st.dataframe(df_good, hide_index=True)
        else:
            st.write("今日無符合標的。")
            
        st.subheader("⚠️ 弱勢檢視 (<60分)")
        if bad_list:
            df_bad = pd.DataFrame(bad_list)[cols]
            st.dataframe(df_bad, hide_index=True)
        else:
            st.write("今日無符合標的。")

elif page == "我的持股":
    st.title("💼 我的庫存持股管理")
    st.info("👆 **上方看板**：自動即時抓取現價並計算實質損益 (僅供檢視)")
    live_portfolio = []
    for idx, row in st.session_state.portfolio.iterrows():
        df, _ = get_taiwan_stock_data(row['代號'])
        if not df.empty:
            current_price = df['Close'].iloc[-1]
            pnl = (current_price - row['持有均價']) * row['股數']
            roi = ((current_price - row['持有均價']) / row['持有均價']) * 100
            live_portfolio.append({
                "代號": row['代號'],
                "名稱": STOCK_MAPPING.get(row['代號'], row['名稱']),
                "持有均價": row['持有均價'],
                "現價": round(current_price, 2),
                "股數": row['股數'],
                "預估損益": round(pnl, 0),
                "報酬率(%)": round(roi, 2)
            })
        else:
            live_portfolio.append(row.to_dict())
            
    st.dataframe(pd.DataFrame(live_portfolio), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.warning("👇 **下方操作區**：請在表格內「點擊兩下」來修改您的真實持股，滑到最底下可新增。修改完務必按下儲存！")
    edited_portfolio = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)
    if st.button("🔄 儲存庫存變更"):
        st.session_state.portfolio = edited_portfolio
        st.rerun()

elif page == "自選股觀察":
    st.title("⭐ 自選股追蹤")
    st.write("將尚未買進的標的輸入於此，首頁與教練模式將會自動為您監控掃描。")
    edited_watchlist = st.data_editor(st.session_state.watchlist, num_rows="dynamic", use_container_width=True)
    if st.button("💾 儲存自選股"):
        st.session_state.watchlist = edited_watchlist
        st.rerun()

elif page == "💬 AI 教練模式":
    st.title("🏋️‍♂️ AI 專屬波段教練")
    all_stocks = list(set(st.session_state.portfolio["代號"].tolist() + st.session_state.watchlist["代號"].tolist()))
    
    stock_options = [f"{sym} {STOCK_MAPPING.get(sym, '')}" for sym in all_stocks]
    selected_option = st.selectbox("選擇要請教教練的股票", stock_options)
    selected_stock = selected_option.split(" ")[0]
    
    if st.button("🗣️ 呼叫教練給予指導"):
        if not api_key:
            st.warning("⚠️ 請先設定 API 金鑰。")
        else:
            with st.spinner(f"正在抓取 {selected_stock} 數據..."):
                df, _ = get_taiwan_stock_data(selected_stock)
                if df.empty:
                    st.error("抓不到報價。")
                else:
                    st.line_chart(df['Close'])
                    sig = compute_signals(df)
                    if sig:
                        st.write(f"📝 現價 {sig['現價']}｜月線：{sig['月線狀態']}｜主力動能(爆量)：{sig['主力動能(爆量倍數)']}｜K={sig['K值']}, D={sig['D值']}｜MACD柱={sig['MACD柱']}")
                        
                        model = genai.GenerativeModel(st.session_state.sys_settings["model"])
                        prompt = f"你是教練。標的：{selected_stock}。現價：{sig['現價']}。月線：{sig['月線狀態']}。爆量倍數：{sig['主力動能(爆量倍數)']}。KD：K={sig['K值']}, D={sig['D值']}。MACD柱：{sig['MACD柱']}。請回覆：\n【教練指示】\n【背後原因】(①均線與籌碼動能 ②KD ③MACD)\n【教練總結行動】"
                        try:
                            st.success(model.generate_content(prompt).text)
                        except Exception as e:
                            st.error(f"🚨 AI 錯誤：{str(e)}")

elif page == "歷史紀錄與績效":
    st.title("📚 波段績效分析")
    st.line_chart([100, 102, 101, 105, 109, 115])

elif page == "系統設定":
    st.title("⚙️ 風控參數設定")
    st.session_state.sys_settings["stop_loss"] = st.number_input("強制停損 (%)", value=st.session_state.sys_settings["stop_loss"], step=0.5)
    # 提供最安全與最新版的模型選項
    st.session_state.sys_settings["model"] = st.selectbox("核心 AI 模型", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro"])
    st.success("參數已儲存！")
