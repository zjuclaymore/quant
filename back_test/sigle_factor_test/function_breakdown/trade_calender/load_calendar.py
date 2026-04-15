r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  回测调仓日历生成器 (Backtest Rebalance Calendar Generator)                   ║
║  模块: factor_engineering / 阶段: p01-数据加载                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

功能:
  从交易日历缓存中提取指定区间，生成含买入/卖出日期的调仓日历
  支持灵活的调仓规则配置

CLI 用法:
  python p01_data_loading__load_calendar.py \\
    --start "YYYY-MM-DD" \\             # 回测开始日期（必填）
    --end "YYYY-MM-DD" \\               # 回测结束日期（必填）
    [--buyday <rule>] \\                # 买入日规则，默认 month_start
    [--sellday <rule>] \\               # 卖出日规则，默认 month_end
    [--delay <days>] \\                 # 交易延迟天数，默认 0
    [--cache <path>] \\                 # 日历缓存路径（自动检测）
    [--output <path>]                   # 输出文件路径（可选）

调仓规则:
  month_start: 月初第一个交易日（默认买入规则）
  month_end: 月末最后一个交易日（默认卖出规则）
  month_N: 月内第 N 个交易日（如 month_5）
  month_-N: 月内倒数第 N 个交易日（如 month_-1）

示例:
  # 基础：月底卖、月初买（标准月度调仓）
  python p01_data_loading__load_calendar.py \\
    --start "2015-01-01" --end "2024-12-31"

  # 自定义规则：月末倒数第2个交易日卖、月初第1个交易日买
  python p01_data_loading__load_calendar.py \\
    --start "2015-01-01" --end "2024-12-31" \\
    --sellday "month_-2" --buyday "month_1"

  # 加延迟：延迟2个交易日执行
  python p01_data_loading__load_calendar.py \\
    --start "2015-01-01" --end "2024-12-31" \\
    --delay 2 --output "calendar.parquet"

输出列: [year_month, signal_date, sell_date, buy_date, is_rebalance_day]
"""
import os
import pandas as pd
import argparse
import sys
from datetime import datetime


def load_calendar_from_cache(
    start_date,
    end_date,
    buyday="month_start",
    sellday="month_end",
    delay_days=0,
    calendar_cache_path=None,
    logger=None
):
    """
    从交易日历缓存中选取回测区间，生成调仓日历
    
    参数:
        start_date (str): 回测开始日期 (格式: 'YYYY-MM-DD' 如 '2015-01-01')
        end_date (str): 回测结束日期 (格式: 'YYYY-MM-DD' 如 '2024-12-31')
        buyday (str): 买入日标记方式
                     'month_start': 月初第一个交易日 (默认)
                     'month_Nth': 月内第N个交易日 (如 'month_5' = 月内第5个交易日)
        sellday (str): 卖出日标记方式
                      'month_end': 月末最后一个交易日 (默认)
                      'month_-Nth': 月内倒数第N个交易日 (如 'month_-1' = 月末最后一个)
        delay_days (int): 交易延迟天数 (在计算出buy/sell后再延迟N个交易日)
        calendar_cache_path (str): 交易日历缓存文件路径。若为None，使用默认路径。
        logger (logging.Logger): 日志对象
    
    返回:
        pd.DataFrame: 包含 ['year_month', 'signal_date', 'sell_date', 'buy_date', 'is_rebalance_day'] 列
    """
    # 默认缓存路径
    if calendar_cache_path is None:
        calendar_cache_path = r'E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet'
    
    if not os.path.exists(calendar_cache_path):
        msg = f"[Calendar] 缓存文件不存在: {calendar_cache_path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)
    
    try:
        # 读取交易日历缓存
        cache = pd.read_parquet(calendar_cache_path)
        cache["date"] = pd.to_datetime(cache["date"])
        cache = cache.sort_values("date").reset_index(drop=True)
        
        if logger:
            logger.info(f"[Calendar] 已加载缓存: {calendar_cache_path}")
            logger.info(f"[Calendar] 缓存范围: {cache['date'].min().date()} ~ {cache['date'].max().date()}")
        
        # 解析回测区间
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 验证缓存覆盖范围
        cmin, cmax = cache["date"].min(), cache["date"].max()
        if not (cmin <= start_dt and cmax >= end_dt):
            msg = (f"[Calendar] 缓存区间不足。"
                   f"缓存: {cmin.date()}~{cmax.date()}, "
                   f"需求: {start_dt.date()}~{end_dt.date()}")
            if logger:
                logger.error(msg)
            raise ValueError(msg)
        
        # 筛选回测区间
        cal_raw = cache[(cache["date"] >= start_dt) & (cache["date"] <= end_dt)].copy()
        trade_dates_list = pd.DatetimeIndex(cal_raw["date"].values)
        
        # 按月分组
        cal_raw["ym"] = cal_raw["date"].dt.to_period("M")
        months_data = []
        
        for ym in sorted(cal_raw["ym"].unique()):
            month_df = cal_raw[cal_raw["ym"] == ym].copy()
            month_dates = pd.DatetimeIndex(month_df["date"].values)
            
            # 确定卖出日
            if sellday == "month_end":
                sell_date = month_dates[-1]  # 月末最后一个交易日
            elif sellday.startswith("month_"):
                parts = sellday.replace("month_", "").split("_")
                if len(parts) == 1 and parts[0].lstrip('-').isdigit():
                    idx = int(parts[0])
                    if idx < 0:  # 倒数第N个
                        sell_date = month_dates[idx]
                    else:  # 正数第N个
                        sell_date = month_dates[idx - 1] if idx > 0 else month_dates[0]
                else:
                    sell_date = month_dates[-1]
            else:
                sell_date = month_dates[-1]
            
            # 确定买入日
            next_month_ym = ym + 1
            next_month_data = cal_raw[cal_raw["ym"] == next_month_ym]
            
            if len(next_month_data) == 0:
                # 已是最后一个月，跳过
                continue
            
            next_month_dates = pd.DatetimeIndex(next_month_data["date"].values)
            
            if buyday == "month_start":
                buy_date = next_month_dates[0]  # 月初第一个交易日
            elif buyday.startswith("month_"):
                parts = buyday.replace("month_", "").split("_")
                if len(parts) == 1 and parts[0].lstrip('-').isdigit():
                    idx = int(parts[0])
                    if idx < 0:  # 倒数第N个
                        buy_date = next_month_dates[idx]
                    else:  # 正数第N个
                        buy_date = next_month_dates[idx - 1] if idx > 0 else next_month_dates[0]
                else:
                    buy_date = next_month_dates[0]
            else:
                buy_date = next_month_dates[0]
            
            # 应用延迟
            if delay_days > 0:
                sell_idx = trade_dates_list.get_loc(sell_date)
                buy_idx = trade_dates_list.get_loc(buy_date)
                
                adj_sell_idx = min(sell_idx + delay_days, len(trade_dates_list) - 1)
                adj_buy_idx = min(buy_idx + delay_days, len(trade_dates_list) - 1)
                
                sell_date = trade_dates_list[adj_sell_idx]
                buy_date = trade_dates_list[adj_buy_idx]
            
            months_data.append({
                "year_month": str(ym),
                "signal_date": month_dates[-1],  # 信号日 = 月末
                "sell_date": sell_date,
                "buy_date": buy_date,
                "is_rebalance_day": 1  # 标记为调仓日
            })
        
        df_calendar = pd.DataFrame(months_data)
        
        if logger:
            logger.info(f"[Calendar] 生成调仓日历: {len(df_calendar)} 期")
            logger.info(f"[Calendar] 回测区间: {start_date} ~ {end_date}")
            logger.info(f"[Calendar] 买入规则: {buyday}, 卖出规则: {sellday}, 延迟天数: {delay_days}")
        
        return df_calendar
        
    except Exception as e:
        if logger:
            logger.error(f"[Calendar] 生成失败: {e}")
        raise


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(
        description="回测日历生成器 - 从交易日历缓存中生成调仓日历",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='回测开始日期 (格式: YYYY-MM-DD, 如 2015-01-01)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='回测结束日期 (格式: YYYY-MM-DD, 如 2024-12-31)'
    )
    
    parser.add_argument(
        '--buyday',
        type=str,
        default='month_start',
        help=(
            '买入日标记方式 (默认: month_start)\n'
            '  month_start: 月初第一个交易日\n'
            '  month_N: 月内第N个交易日 (如 month_5)\n'
            '  month_-N: 月内倒数第N个交易日 (如 month_-1)'
        )
    )
    
    parser.add_argument(
        '--sellday',
        type=str,
        default='month_end',
        help=(
            '卖出日标记方式 (默认: month_end)\n'
            '  month_end: 月末最后一个交易日\n'
            '  month_N: 月内第N个交易日 (如 month_5)\n'
            '  month_-N: 月内倒数第N个交易日 (如 month_-1)'
        )
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=0,
        help='交易延迟天数 (默认: 0, 在计算出buy/sell后再延迟N个交易日)'
    )
    
    parser.add_argument(
        '--cache',
        type=str,
        default=None,
        help='交易日历缓存文件路径 (默认: E:\\...\\data\\交易日历\\rebalance_calendar_cache.parquet)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 (可选，格式: parquet/csv)'
    )
    
    args = parser.parse_args()
    
    try:
        # 生成调仓日历
        df_calendar = load_calendar_from_cache(
            start_date=args.start,
            end_date=args.end,
            buyday=args.buyday,
            sellday=args.sellday,
            delay_days=args.delay,
            calendar_cache_path=args.cache
        )
        
        print(f"\n[Success] 调仓日历生成完成: {len(df_calendar)} 期")
        print(f"\n数据预览:")
        print(df_calendar.head(10))
        print(f"\n数据统计:")
        print(df_calendar.tail(5))
        
        # 保存输出
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            if args.output.endswith('.parquet'):
                df_calendar.to_parquet(args.output, index=False)
            else:
                df_calendar.to_csv(args.output, index=False)
            print(f"\n[Saved] 已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
自动拆分视图文件（仅用于阅读理解，不参与运行）。
来源文件: back_test/sigle_factor_test/src/data_loader.py
函数: load_calendar
类型: module_function
行号: 577-672
签名: def load_calendar(df_daily, delay_days=0, rebalance_month_start=False, logger=None)
作用概述: 生成严格的调仓日历 (Rebalance Calendar Engine)
"""
def load_calendar(df_daily, delay_days=0, rebalance_month_start=False, logger=None):
    """
    生成严格的调仓日历 (Rebalance Calendar Engine)

    根据官方交易日历和用户的信号延迟配置，计算每个月的信号发出日、买入日及卖出日。

    关键逻辑:
        1. 缓存依赖: 必须先运行 `generate_calendar_cache.py` 生成 Parquet 缓存，以保证回测的日期全局一致。
          2. 信号延迟 (`delay_days`): 支持实战仿真的“发车延迟”。若 `delay_days > 0`，
              所有的买入和卖出执行日将相对于标准计划日顺延对应个交易日。
          3. 月初调仓模式 (`rebalance_month_start`): 若开启，则当期信号对应“次月初买入、再下月初卖出”，
              让卖旧与买新在月初同一调仓点发生。
          4. 状态闭环: 确保回测区间完全被缓存覆盖，否则抛出异常。

    参数:
        df_daily (pd.DataFrame): 基础行情数据。
        delay_days (int): 交易延迟天数。
        rebalance_month_start (bool): 是否采用月初调仓口径。
        logger (logging.Logger): 日志对象。

    返回:
        tuple: (调仓日历 DataFrame, 全量交易日序列)。
    """
    data_paths = load_data_paths()
    calendar_dir = get_path_from_config(data_paths, "TradeCalendar", "data/交易日历")
    calendar_cache_path = os.path.join(calendar_dir, "rebalance_calendar_cache.parquet")

    data_min = pd.to_datetime(df_daily["date"].min()) - pd.Timedelta(days=30)
    data_max = pd.to_datetime(df_daily["date"].max()) + pd.Timedelta(days=30)

    if not os.path.exists(calendar_cache_path):
        msg = f"交易日历缓存文件不存在，请先运行 generate_calendar_cache.py: {calendar_cache_path}"
        if logger: logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        cache = pd.read_parquet(calendar_cache_path)
        cache["date"] = pd.to_datetime(cache["date"])
        cache = cache.sort_values("date").reset_index(drop=True)

        cmin, cmax = cache["date"].min(), cache["date"].max()
        if (cmin <= data_min) and (cmax >= data_max):
            cal_raw = cache[(cache["date"] >= data_min) & (cache["date"] <= data_max)].copy()
            trade_dates = pd.DatetimeIndex(cal_raw["date"].values)

            cal_raw["ym"] = cal_raw["date"].dt.to_period("M")
            first_days = cal_raw.groupby("ym")["date"].min()
            last_days = cal_raw.groupby("ym")["date"].max()

            cal_list = []
            yms = sorted(cal_raw["ym"].unique())
            max_i = len(yms) - (2 if rebalance_month_start else 1)
            if max_i <= 0:
                raise ValueError("交易日历月份数量不足，无法构建调仓序列")

            for i in range(max_i):
                cur_ym = yms[i]
                next_ym = yms[i + 1]
                
                base_buy = first_days[next_ym]
                if rebalance_month_start:
                    base_sell = first_days[yms[i + 2]]
                else:
                    base_sell = last_days[next_ym]
                
                # Apply delay_days
                buy_idx = trade_dates.get_loc(base_buy)
                sell_idx = trade_dates.get_loc(base_sell)
                
                adj_buy_idx = min(buy_idx + delay_days, len(trade_dates) - 1)
                adj_sell_idx = min(sell_idx + delay_days, len(trade_dates) - 1)
                
                cal_list.append(
                    {
                        "year_month": cur_ym,
                        "signal_date": last_days[cur_ym],
                        "buy_date": trade_dates[adj_buy_idx],
                        "sell_date": trade_dates[adj_sell_idx],
                    }
                )
            df_calendar = pd.DataFrame(cal_list)
            if logger:
                logger.info(
                    f"已从本地读取调仓日历: {len(df_calendar)} 期 "
                    f"(本地路径: {calendar_cache_path})"
                )
            return df_calendar, trade_dates
        else:
            msg = (f"本地区间不足。缓存区间: {cmin.date()}~{cmax.date()}，"
                   f"需求区间: {data_min.date()}~{data_max.date()}。"
                   f"请重新运行 generate_calendar_cache.py 扩充区间。")
            if logger: logger.error(msg)
            raise ValueError("Calendar cache date range is insufficient for the underlying data.")
    except Exception as e:
        if logger: logger.error(f"读取交易日历缓存失败: {e}")
        raise
