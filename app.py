import requests
import datetime
# 記得確保頂部有引入 numpy, pandas, streamlit 等既有套件

@st.cache_data(ttl=3600)  # 加入 1 小時快取，避免頻繁重整耗盡 API 免費額度
def get_institutional_data(code):
    try:
        # 1. 設定時間區間：抓取近 30 天的資料來計算連續買賣
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        # 2. 呼叫 FinMind 獲取法人數據
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": code,
            "start_date": start_date,
            "end_date": end_date,
        }
        resp = requests.get(url, params=parameter, timeout=5)
        data = resp.json()
        
        # 驗證資料是否成功回傳
        if data.get("msg") != "success" or not data.get("data"):
            return {"buy_sell": 0, "days": 0, "trend": "資料不足"}
            
        # 3. 資料清洗與轉換
        df_inst = pd.DataFrame(data["data"])
        
        # 計算淨買賣超 (FinMind 的單位為「股」)
        df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
        
        # 將外資、投信、自營商的數據「按日期」加總
        daily_net = df_inst.groupby('date')['net_buy'].sum().reset_index()
        # 日期由新到舊排序
        daily_net = daily_net.sort_values('date', ascending=False).reset_index(drop=True)
        
        if daily_net.empty:
            return {"buy_sell": 0, "days": 0, "trend": "近期無交易"}
            
        # 將最新一日的淨買超股數除以 1000，換算為「張數」
        latest_net = int(daily_net.iloc[0]['net_buy'] / 1000)
        
        # 4. 計算連續買賣天數邏輯
        days = 0
        is_buy = latest_net > 0
        
        for val in daily_net['net_buy']:
            if is_buy and val > 0:
                days += 1
            elif not is_buy and val < 0:
                days += 1
            else:
                break # 遇到反向操作即停止計算
                
        # 5. 輸出狀態字串
        if days == 0:
            trend_str = "盤整"
        else:
            trend_str = f"連{days}買" if is_buy else f"連{days}賣"
            
        return {"buy_sell": latest_net, "days": days, "trend": trend_str}
        
    except Exception as e:
        # 例外處理：若網路異常，回傳預設值以避免系統崩潰
        return {"buy_sell": 0, "days": 0, "trend": "API異常"}
