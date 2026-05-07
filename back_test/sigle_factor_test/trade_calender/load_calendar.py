r"""
回测调仓日历生成器 (Backtest Rebalance Calendar Generator)
模块: factor_engineering / 阶段: p01-数据加载

功能:
  从交易日历缓存中提取指定区间，生成含买入/卖出日期的调仓日历。
  核心逻辑已调整为“月末因子观测 -> 月末收盘调仓”模式。

计算公式 (Formula Explanation):
  该脚本已强制对齐为严格月末模式：
  对于第 M 个调仓月 (For the M-th rebalancing month):
  1. 信号观测 (Signal): 
     Date_signal = LastTradeDay(Month_M)
  2. 买入执行 (Buy): 
     Date_buy = LastTradeDay(Month_M) (月末当日收盘买入)
  3. 卖出执行 (Sell): 
     Date_sell = LastTradeDay(Month_{M+1}) (次月月末收盘卖出)

  注意: delay_days 参数在此版本中已被忽略，以确保严格对齐月末。

CLI 用法:
  python load_calendar.py --start "2015-01-01" --end "2024-12-31"
"""

import os
import pandas as pd
import argparse
import sys
from datetime import datetime


def load_calendar_from_cache(
    start_date,
    end_date,
    buyday="month_end",
    sellday="month_end",
    delay_days=0,
    calendar_cache_path=None,
    logger=None
):
    """
    从交易日历缓存中选取回测区间，生成调仓日历。
    
    采用强制“月末对齐”逻辑：
    - 信号观测日 (Signal): 本月最后一个交易日
    - 买入执行日 (Buy): 本月最后一个交易日 (忽略 delay_days)
    - 卖出执行日 (Sell): 下月最后一个交易日
    
    注意: 为了兼顾兼容性，保留参数接口，但内部逻辑已固定为月末模式。
    
    公式:
        buy_date = LastTradeDay(M)
        sell_date = LastTradeDay(M+1)
    """
    if calendar_cache_path is None:
        calendar_cache_path = r'E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet'
    
    if not os.path.exists(calendar_cache_path):
        msg = f"[Calendar] 缓存文件不存在: {calendar_cache_path}"
        if logger: logger.error(msg)
        raise FileNotFoundError(msg)
    
    try:
        # 1. 加载全量交易日
        cache = pd.read_parquet(calendar_cache_path)
        cache["date"] = pd.to_datetime(cache["date"])
        cache = cache.sort_values("date").reset_index(drop=True)
        trade_dates_all = pd.DatetimeIndex(cache["date"].values)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 2. 确定涉及的月份区间 (增加前后缓冲区)
        cal_raw = cache[(cache["date"] >= (start_dt - pd.Timedelta(days=40))) & 
                        (cache["date"] <= (end_dt + pd.Timedelta(days=40)))].copy()
        cal_raw["ym"] = cal_raw["date"].dt.to_period("M")
        yms = sorted(cal_raw["ym"].unique())
        
        months_data = []
        for i in range(len(yms) - 1):
            cur_ym = yms[i]
            next_ym = yms[i + 1]
            
            # 月份过滤：确保在请求的区间内
            cur_month_start = pd.to_datetime(cur_ym.to_timestamp())
            if cur_month_start > end_dt:
                break
            
            cur_month_df = cal_raw[cal_raw["ym"] == cur_ym]
            next_month_df = cal_raw[cal_raw["ym"] == next_ym]
            
            if len(cur_month_df) == 0 or len(next_month_df) == 0:
                continue
                
            cur_month_dates = pd.DatetimeIndex(cur_month_df["date"].values)
            next_month_dates = pd.DatetimeIndex(next_month_df["date"].values)
            
            # --- 确定 Signal Date (本月末) ---
            signal_date = cur_month_dates[-1]
            
            # --- 确定 Buy/Sell (强制月末) ---
            # 无论参数如何，均设定为月末最后一个交易日
            buy_date = cur_month_dates[-1]
            sell_date = next_month_dates[-1]
            
            months_data.append({
                "year_month": str(cur_ym),
                "signal_date": signal_date,
                "sell_date": sell_date,
                "buy_date": buy_date,
                "is_rebalance_day": 1
            })
            
        df_calendar = pd.DataFrame(months_data)
        
        # 3. 最终按信号日筛选用户请求的区间
        df_calendar = df_calendar[(df_calendar["signal_date"] >= start_dt) & 
                                  (df_calendar["signal_date"] <= end_dt)].reset_index(drop=True)
        
        if logger:
            logger.info(f"[Calendar] 已生成对齐日历: {len(df_calendar)} 期")
            
        return df_calendar
        
    except Exception as e:
        if logger: logger.error(f"[Calendar] 运行异常: {e}")
        raise


def load_calendar(df_daily, delay_days=0, rebalance_at_month_end=True, logger=None):
    """
    生成严格的对齐调仓日历 (Unified Rebalance Calendar Engine)。
    
    默认模式 (rebalance_at_month_end=True):
        - 因子观测: 月末
        - 买入: 月末 (当天收盘)
        - 卖出: 下月末 (当天收盘)
        
    公式:
        T_signal(M) = LastTradeDay(Month_M)
        T_buy(M) = T_signal(M) + delay
        T_sell(M) = T_buy(M+1)
    """
    # 逻辑重用
    start_date = df_daily["date"].min()
    if isinstance(start_date, (pd.Timestamp, datetime)):
        start_date = start_date.strftime("%Y-%m-%d")
    
    end_date = df_daily["date"].max()
    if isinstance(end_date, (pd.Timestamp, datetime)):
        end_date = end_date.strftime("%Y-%m-%d")
        
    buyday = "month_end" if rebalance_at_month_end else "month_start"
    sellday = "month_end" if rebalance_at_month_end else "month_start"
    
    df_calendar = load_calendar_from_cache(
        start_date=start_date,
        end_date=end_date,
        buyday=buyday,
        sellday=sellday,
        delay_days=delay_days,
        logger=logger
    )
    
    # 获取全量交易日序列用于返回 (部分组件需要)
    calendar_cache_path = r'E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet'
    if not os.path.exists(calendar_cache_path):
        # Fallback if path differs
        calendar_cache_path = r'E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet'
    
    cache = pd.read_parquet(calendar_cache_path)
    trade_dates_all = pd.DatetimeIndex(pd.to_datetime(cache["date"]).values)
    
    return df_calendar, trade_dates_all


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(description="回测调仓日历生成器 - 月末对齐版本")
    parser.add_argument('--start', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--buyday', type=str, default='month_end', help='买入规则 (month_end/month_start)')
    parser.add_argument('--sellday', type=str, default='month_end', help='卖出规则 (month_end/month_start)')
    parser.add_argument('--delay', type=int, default=0, help='执行延迟天数')
    parser.add_argument('--output', type=str, default=None, help='输出路径 (.parquet/.csv)')
    
    args = parser.parse_args()
    
    df_calendar = load_calendar_from_cache(
        start_date=args.start,
        end_date=args.end,
        buyday=args.buyday,
        sellday=args.sellday,
        delay_days=args.delay
    )
    
    print(f"\n[Success] 生成调仓日历: {len(df_calendar)} 期")
    if not df_calendar.empty:
        print(df_calendar.head())
        print("...")
        print(df_calendar.tail())
    
    if args.output:
        if args.output.endswith('.parquet'):
            df_calendar.to_parquet(args.output, index=False)
        else:
            df_calendar.to_csv(args.output, index=False)
        print(f"\n[Saved] 已保存至: {args.output}")


if __name__ == "__main__":
    main()
