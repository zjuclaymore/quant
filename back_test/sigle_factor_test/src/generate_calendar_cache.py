"""
交易日历缓存生成器 (Trading Calendar Cache Generator)

该脚本负责从原始行情数据中提取官方交易日序列，并将其持久化为 Parquet 缓存。
这一步骤对于回测框架至关重要，因为它确保了在没有实时行情服务的环境下，回测引擎依然能够准确对齐
每一个交易日、信号日及调仓执行日。

输出目标:
    E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet
"""

import os
import pandas as pd
import logging

def generate_calendar_cache():
    """
    独立生成交易日历缓存的逻辑 (Cache Generation Engine)

    流程说明:
        1. 载入原始交易日历 CSV (包含全交易所数据)。
        2. 过滤规则: 仅保留 'exchange' == 'SSE' (上交所) 且 'is_open' == 1 (开市) 的交易日。
        3. 格式转换: 将 cal_date 映射为标准 pd.Timestamp 类型并排序。
        4. 持久化: 仅保存 `date` 列至 Parquet 格式，实现毫秒级加载。
    """
    logger = logging.getLogger("CalendarCache")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)

    cal_path = r"E:\1_basement\quant_research\data\交易日历\trade_calendar.csv"
    calendar_cache_path = r"E:\1_basement\quant_research\data\交易日历\rebalance_calendar_cache.parquet"

    if not os.path.exists(cal_path):
        logger.error(f"原始交易日历文件不存在: {cal_path}")
        return

    logger.info("开始处理原始交易日历...")
    try:
        cal_raw = pd.read_csv(cal_path)
        # 仅保留上交所开市日
        cal_raw = cal_raw[(cal_raw["exchange"] == "SSE") & (cal_raw["is_open"] == 1)].copy()
        cal_raw["date"] = pd.to_datetime(cal_raw["cal_date"].astype(str), format="%Y%m%d")
        cal_raw = cal_raw.sort_values("date").reset_index(drop=True)

        os.makedirs(os.path.dirname(calendar_cache_path), exist_ok=True)
        # 仅缓存 date 列以节省空间
        cal_raw[["date"]].to_parquet(calendar_cache_path, index=False)
        logger.info(f"交易日历缓存已成功写入: {calendar_cache_path}")
        logger.info(f"缓存范围: {cal_raw['date'].iloc[0].date()} ~ {cal_raw['date'].iloc[-1].date()}")
        logger.info(f"共包含 {len(cal_raw)} 个有效交易日")
        
    except Exception as e:
        logger.error(f"处理或写入缓存时发生错误: {e}")

if __name__ == "__main__":
    generate_calendar_cache()
