"""
download_daily_basic.py
-----------------------
从 Tushare Pro 下载 A 股每日指标数据（daily_basic），
按截面格式存储（每个交易日一个 pickle 文件）。

输出目录: E:\\1_basement\\quant_research\\data\\中国A股daily_basic
输出文件: YYYYMMDD.pickle  (如 20200102.pickle)

用法:
  python download_daily_basic.py                      # 全量下载
  python download_daily_basic.py --start 20140101 --end 20211231  # 指定日期范围
  python download_daily_basic.py --resume             # 跳过已存在文件
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import tushare as ts
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

# Token (从项目其他脚本中获取)
TOKEN = "2d4f555869182905bfd48ce1fd0f649015f2bf10b3ef4a7e558573bb"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "中国A股daily_basic"

# 交易日历路径
CALENDAR_PATH = PROJECT_ROOT / "data" / "交易日历" / "trade_calendar.csv"

# Tushare 调用频率限制 (每分钟最多200次，保守使用80次)
CALLS_PER_MINUTE = 80
SLEEP_TIME = 60.0 / CALLS_PER_MINUTE

# 需要下载的字段 (daily_basic 主要字段)
FIELDS = [
    "ts_code",  # TS代码
    "trade_date",  # 交易日期
    "close",  # 收盘价
    "turnover_rate",  # 换手率（%）
    "turnover_rate_f",  # 换手率（自由流通股）
    "volume_ratio",  # 量比
    "pe",  # 市盈率（日）
    "pe_ttm",  # 市盈率（TTM）
    "pb",  # 市净率
    "ps",  # 市销率
    "ps_ttm",  # 市销率（TTM）
    "dv_ratio",  # 股息率（%）
    "dv_ttm",  # 股息率（TTM）
    "total_mv",  # 总市值（万元）
    "circ_mv",  # 流通市值（万元）
    "free_share",  # 自由流通股本（万股）
    "total_share",  # 总股本（万股）
    "total_assets",  # 总资产（万元）
    "liquid_assets",  # 流动资产（万元）
    "fixed_assets",  # 固定资产（万元）
    "ebit",  # 息税前利润（万元）
    "ebitda",  # 息税折旧摊销前利润（万元）
    "cfps",  # 每股现金流
    "eps",  # 每股收益
    "bps",  # 每股净资产
    "bps",  # 每股净资产
    "dt_eps",  # 扣非每股收益
    "dt_eps_yoy",  # 扣非每股收益同比增长
    "bps_yoy",  # 每股净资产同比增长
]

# 去重后的字段列表
FIELDS = list(dict.fromkeys(FIELDS))

# ──────────────────────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────


def load_trade_calendar(start_date: str, end_date: str) -> list[str]:
    """
    从本地交易日历加载指定范围内的交易日列表

    参数:
        start_date: 起始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    返回:
        交易日列表 (YYYYMMDD格式)
    """
    if not os.path.exists(CALENDAR_PATH):
        logger.error(f"交易日历文件不存在: {CALENDAR_PATH}")
        sys.exit(1)

    df = pd.read_csv(CALENDAR_PATH)

    # 过滤上交所开市日
    df = df[(df["exchange"] == "SSE") & (df["is_open"] == 1)].copy()

    # 日期格式转换
    df["cal_date"] = df["cal_date"].astype(str)

    # 过滤日期范围
    df = df[(df["cal_date"] >= start_date) & (df["cal_date"] <= end_date)]

    trade_dates = df["cal_date"].sort_values().unique().tolist()
    logger.info(
        f"加载交易日历: {len(trade_dates)} 个交易日 ({start_date} ~ {end_date})"
    )

    return trade_dates


def get_existing_files() -> set[str]:
    """获取已下载的日期集合"""
    if not OUTPUT_DIR.exists():
        return set()

    files = list(OUTPUT_DIR.glob("*.pickle"))
    dates = {f.stem for f in files}
    return dates


def download_one_day(pro, trade_date: str) -> pd.DataFrame | None:
    """
    下载单个交易日的 daily_basic 数据

    参数:
        pro: Tushare Pro API 实例
        trade_date: 交易日期 (YYYYMMDD)

    返回:
        DataFrame 或 None (失败时)
    """
    max_retries = 5

    for attempt in range(max_retries):
        try:
            df = pro.daily_basic(trade_date=trade_date, fields=",".join(FIELDS))
            return df
        except Exception as e:
            error_msg = str(e)

            # 判断是否为限流错误
            if (
                "抱歉" in error_msg
                or "每分钟" in error_msg
                or "limit" in error_msg.lower()
            ):
                wait_time = 60 + attempt * 10
                logger.warning(
                    f"触发限流，等待 {wait_time} 秒后重试... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                logger.warning(
                    f"下载 {trade_date} 失败 (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2 + attempt)

    return None


def save_cross_section(df: pd.DataFrame, trade_date: str) -> None:
    """
    保存截面数据为 pickle 文件

    参数:
        df: 单日数据 DataFrame
        trade_date: 交易日期 (YYYYMMDD)
    """
    output_file = OUTPUT_DIR / f"{trade_date}.pickle"
    df.to_pickle(output_file, protocol=4)


# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="下载 Tushare daily_basic 数据（截面格式）"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="20000101",
        help="起始日期 (YYYYMMDD), 默认: 20000101",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="20251231",
        help="结束日期 (YYYYMMDD), 默认: 20251231",
    )
    parser.add_argument("--resume", action="store_true", help="跳过已下载的文件")
    parser.add_argument(
        "--workers", type=int, default=1, help="并发工作线程数 (默认: 1, 建议 ≤ 3)"
    )
    args = parser.parse_args()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化 Tushare
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    # 加载交易日历
    trade_dates = load_trade_calendar(args.start, args.end)

    if not trade_dates:
        logger.error("没有找到交易日，退出")
        return

    # 获取已下载文件
    existing_dates = get_existing_files() if args.resume else set()

    # 过滤需要下载的日期
    dates_to_download = [d for d in trade_dates if d not in existing_dates]

    logger.info(f"总交易日: {len(trade_dates)}")
    logger.info(f"已下载: {len(existing_dates)}")
    logger.info(f"待下载: {len(dates_to_download)}")

    if not dates_to_download:
        logger.info("所有交易日已下载，退出")
        return

    # ────────────────────────────────────────────────────────────
    # 下载循环
    # ────────────────────────────────────────────────────────────

    success_count = 0
    failed_count = 0
    failed_dates = []

    logger.info(f"开始下载 (字段数: {len(FIELDS)})")

    for i, trade_date in enumerate(tqdm(dates_to_download, desc="下载进度")):
        start_time = time.time()

        # 下载数据
        df = download_one_day(pro, trade_date)

        if df is not None and not df.empty:
            # 保存截面数据
            save_cross_section(df, trade_date)
            success_count += 1

            # 每100个文件打印一次进度
            if success_count % 100 == 0:
                logger.info(f"已成功下载 {success_count} 个交易日")
        else:
            failed_count += 1
            failed_dates.append(trade_date)
            logger.warning(f"交易日 {trade_date} 无数据或下载失败")

        # 频率限制
        elapsed = time.time() - start_time
        if elapsed < SLEEP_TIME:
            time.sleep(SLEEP_TIME - elapsed)

    # ────────────────────────────────────────────────────────────
    # 汇总报告
    # ────────────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("下载完成!")
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"失败: {failed_count} 个交易日")

    if failed_dates:
        logger.warning(f"失败的日期: {failed_dates[:20]}")  # 只显示前20个

    logger.info(f"输出目录: {OUTPUT_DIR}")

    # 统计文件总数
    total_files = len(list(OUTPUT_DIR.glob("*.pickle")))
    logger.info(f"目录中共有 {total_files} 个 pickle 文件")


if __name__ == "__main__":
    main()
