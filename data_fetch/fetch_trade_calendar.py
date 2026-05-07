import tushare as ts
import pandas as pd
import os

"""
本脚本用于获取全量交易日历并缓存至本地，
供回测框架(load_calendar.py)使用。
"""

def generate_trade_calendar_cache():
    TOKEN = os.getenv("TUSHARE_TOKEN")
    if not TOKEN:
        raise RuntimeError("Missing TUSHARE_TOKEN environment variable.")
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    cache_dir = r"E:\1_basement\quant_research\data\交易日历"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "rebalance_calendar_cache.parquet")

    print("正在从 Tushare 获取全量交易日历...")
    # 获取SSE交易日历
    df_cal = pro.trade_cal(exchange='SSE', is_open='1')
    
    # 重命名列以匹配 load_calendar.py 的期望 ('date' 列)
    df_cal = df_cal.rename(columns={'cal_date': 'date'})
    df_cal['date'] = pd.to_datetime(df_cal['date'])
    
    # 按照日期排序
    df_cal = df_cal.sort_values('date').reset_index(drop=True)

    # 存储为 Parquet
    df_cal.to_parquet(cache_path, index=False, engine='pyarrow')
    print(f"交易日历缓存已生成: {cache_path}")
    print(f"包含交易日数量: {len(df_cal)}")

if __name__ == "__main__":
    generate_trade_calendar_cache()
