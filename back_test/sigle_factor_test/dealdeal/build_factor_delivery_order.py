#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于月度因子截面与交易日历生成拟交割单。

逻辑：
1. 在每个月的 sell day 读取因子截面并做横截面 rank；
2. 在对应 buy day 读取股票池 allow_flag，只保留 allow_flag=1 的标的；
3. 输出拟交割单 parquet / csv 到当前目录。

默认输入：
  - 因子：factor_engineering/factor_A021_mad3_zscore_neutral.parquet
  - 股票池：stock_pool/original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet
  - 日历：trade_calender/load_calendar.py
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import time
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown")
DEFAULT_FACTOR_PATH = ROOT / "factor_engineering" / "factor_A021_mad3_zscore_neutral.parquet"
DEFAULT_POOL_PATH = ROOT / "stock_pool" / "original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet"
DEFAULT_CALENDAR_PY = ROOT / "trade_calender" / "load_calendar.py"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("factor_A021_delivery_order.parquet")
DEFAULT_LOG_PATH = Path(__file__).with_name("build_factor_delivery_order.log")


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("factor_delivery_order")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _load_calendar_module(calendar_py_path: Path):
    spec = importlib.util.spec_from_file_location("load_calendar_module", calendar_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载交易日历模块: {calendar_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_calendar_from_cache"):
        raise RuntimeError("load_calendar.py 中未找到 load_calendar_from_cache")
    return module


def _normalize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    # 大表场景下，只有含交易所后缀时再触发正则替换，可显著减少耗时。
    has_suffix = s.str.contains(".", regex=False, na=False)
    if bool(has_suffix.any()):
        s = s.where(~has_suffix, s.str.replace(r"\.(SZ|SH|BJ)$", "", regex=True))
    return s.str.zfill(6)


def _to_int_date(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64")


def _progress(logger: logging.Logger, done: int, total: int, start_ts: float, label: str, extra: str = "") -> None:
    elapsed = time.perf_counter() - start_ts
    rate = done / elapsed if elapsed > 0 else 0.0
    remain = (total - done) / rate if rate > 0 else math.inf
    pct = done / total * 100 if total else 100.0
    eta = f"{remain:.1f}s" if math.isfinite(remain) else "N/A"
    suffix = f" | {extra}" if extra else ""
    logger.info("[%s] %.2f%% (%d/%d) | elapsed=%.1fs | eta=%s%s", label, pct, done, total, elapsed, eta, suffix)


def _load_calendar_df(calendar_py_path: Path, start_date: str, end_date: str, buyday: str, sellday: str) -> pd.DataFrame:
    module = _load_calendar_module(calendar_py_path)
    cal = module.load_calendar_from_cache(
        start_date=start_date,
        end_date=end_date,
        buyday=buyday,
        sellday=sellday,
        delay_days=0,
        calendar_cache_path=None,
        logger=None,
    )
    if cal is None or cal.empty:
        raise RuntimeError("交易日历为空，无法生成拟交割单")
    cal = cal.copy()
    cal["sell_int"] = pd.to_datetime(cal["sell_date"]).dt.strftime("%Y%m%d").astype(int)
    cal["buy_int"] = pd.to_datetime(cal["buy_date"]).dt.strftime("%Y%m%d").astype(int)
    cal["signal_int"] = pd.to_datetime(cal["signal_date"]).dt.strftime("%Y%m%d").astype(int)
    return cal


def build_delivery_order(
    factor_path: Path,
    pool_path: Path,
    calendar_py_path: Path,
    output_path: Path,
    log_path: Path,
    factor_col: str,
    buyday: str,
    sellday: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """
    构建拟交割单：在每个调仓月的 sell_date 读取因子截面并做横截面排名，
    在对应 buy_date 读取股票池 allow_flag，只保留 allow_flag=1 的标的。

    调仓日历范围优先使用 start_date/end_date（由上游 pipeline 显式传入），
    若均为 None 则 fallback 到因子文件本身的日期范围（可能包含额外历史数据）。

    参数:
        factor_path      (Path):      预处理后的因子 parquet。
        pool_path        (Path):      股票池 parquet（含 code/date/allow_flag）。
        calendar_py_path (Path):      load_calendar.py 的路径。
        output_path      (Path):      输出拟交割单路径。
        log_path         (Path):      日志文件路径。
        factor_col       (str):       因子值列名。
        buyday           (str):       买入日规则（如 'month_start'）。
        sellday          (str):       卖出日规则（如 'month_end'）。
        start_date       (str|None):  回测开始日期（YYYY-MM-DD）；None 时从因子推断。
        end_date         (str|None):  回测结束日期（YYYY-MM-DD）；None 时从因子推断。

    异常:
        FileNotFoundError: 输入文件不存在时抛出。
        RuntimeError: 日历为空或因子与日历无法匹配时抛出。
    """
    logger = _setup_logger(log_path)
    t0 = time.perf_counter()

    logger.info("[Order] 开始生成拟交割单")
    logger.info("[Order] 因子文件: %s", factor_path)
    logger.info("[Order] 股票池文件: %s", pool_path)
    logger.info("[Order] 日历脚本: %s", calendar_py_path)
    logger.info("[Order] 输出文件: %s", output_path)
    logger.info("[Order] 因子列: %s", factor_col)
    logger.info("[Order] 买入规则: %s, 卖出规则: %s", buyday, sellday)
    logger.info("[Order] 回测区间（显式传入）: %s ~ %s", start_date or "auto", end_date or "auto")

    if not factor_path.exists():
        raise FileNotFoundError(f"因子文件不存在: {factor_path}")
    if not pool_path.exists():
        raise FileNotFoundError(f"股票池文件不存在: {pool_path}")
    if not calendar_py_path.exists():
        raise FileNotFoundError(f"交易日历脚本不存在: {calendar_py_path}")

    t = time.perf_counter()
    factor = pd.read_parquet(factor_path)
    required_factor_cols = {"code", "date", factor_col}
    miss = [c for c in required_factor_cols if c not in factor.columns]
    if miss:
        raise ValueError(f"因子文件缺少必要列: {miss}")
    factor = factor[["code", "date", factor_col] + [c for c in ["lncap", "ind_code"] if c in factor.columns]].copy()
    factor["code"] = _normalize_code(factor["code"])
    factor["date"] = _to_int_date(factor["date"])
    factor = factor.dropna(subset=["code", "date", factor_col]).copy()
    factor["date"] = factor["date"].astype("int64")
    factor["date"] = pd.to_datetime(factor["date"].astype(str), format="%Y%m%d")
    factor["year_month"] = factor["date"].dt.to_period("M").astype(str)
    logger.info("[Order] 读取因子完成: 行数=%d, 代码数=%d, 耗时=%.2fs", len(factor), factor["code"].nunique(), time.perf_counter() - t)

    t = time.perf_counter()
    pool = pd.read_parquet(pool_path, columns=["code", "date", "allow_flag"])
    pool["code"] = _normalize_code(pool["code"])
    pool["date"] = _to_int_date(pool["date"])
    pool = pool.dropna(subset=["code", "date"]).copy()
    pool["date"] = pool["date"].astype("int64")
    pool["allow_flag"] = pd.to_numeric(pool["allow_flag"], errors="coerce").fillna(0).astype("int8")
    logger.info("[Order] 读取股票池完成: 行数=%d, 代码数=%d, allow_flag=1占比=%.2f%%, 耗时=%.2fs", len(pool), pool["code"].nunique(), pool["allow_flag"].mean() * 100, time.perf_counter() - t)

    # ── 确定调仓日历范围 ──────────────────────────────────────────────────────
    # 优先使用上游显式传入的 start_date/end_date，避免加载不必要的历史调仓期。
    # 若未传入（None），则 fallback 到因子文件本身的日期范围（向前兼容旧用法）。
    if start_date is None:
        start_date = pd.to_datetime(factor["date"].min()).strftime("%Y-%m-%d")
        logger.warning("[Order] --start 未指定，从因子文件自动推断起始日期: %s", start_date)
    if end_date is None:
        end_date = pd.to_datetime(factor["date"].max()).strftime("%Y-%m-%d")
        logger.warning("[Order] --end 未指定，从因子文件自动推断截止日期: %s", end_date)

    t = time.perf_counter()
    cal = _load_calendar_df(calendar_py_path, start_date, end_date, buyday, sellday)
    logger.info("[Order] 调仓日历完成: 期数=%d, 范围=%s~%s, 耗时=%.2fs",
                len(cal), start_date, end_date, time.perf_counter() - t)

    cal["year_month"] = cal["year_month"].astype(str)

    # 每月使用“最后一个因子有值日”进行排序，避免因子自然月末与交易月末不一致导致断层。
    month_last = factor.groupby("year_month")["date"].max().reset_index(name="factor_date")
    factor = factor.merge(month_last, on="year_month", how="inner")
    factor = factor[factor["date"] == factor["factor_date"]].copy()

    factor = factor.merge(
        cal[["year_month", "signal_date", "sell_date", "buy_date", "sell_int", "buy_int", "signal_int"]],
        on="year_month",
        how="inner",
    )
    if factor.empty:
        raise RuntimeError("因子按月份无法匹配到任何调仓期，请检查日期口径")

    factor = factor.sort_values(["year_month", factor_col, "code"], ascending=[True, False, True]).reset_index(drop=True)
    factor["factor_rank"] = factor.groupby("year_month")[factor_col].rank(method="first", ascending=False).astype("int64")
    factor["factor_rank_pct"] = factor.groupby("year_month")[factor_col].rank(method="average", ascending=False, pct=True)

    factor["buy_int"] = factor["buy_int"].astype("int64")
    factor["sell_int"] = factor["sell_int"].astype("int64")

    t = time.perf_counter()
    eligible = pool[pool["allow_flag"] == 1].copy()
    eligible = eligible.rename(columns={"date": "buy_int"})
    order = factor.merge(eligible, on=["code", "buy_int"], how="inner", suffixes=("", "_pool"))
    logger.info("[Order] buyday allow_flag 过滤完成: 命中行数=%d, 耗时=%.2fs", len(order), time.perf_counter() - t)

    order = order.rename(columns={factor_col: "factor_value"})
    order["buy_date"] = pd.to_datetime(order["buy_int"], format="%Y%m%d")
    order["sell_date"] = pd.to_datetime(order["sell_int"], format="%Y%m%d")
    order["signal_date"] = pd.to_datetime(order["signal_int"], format="%Y%m%d")
    order["factor_date"] = pd.to_datetime(order["factor_date"])
    order["year_month"] = order["year_month"].astype(str)
    order["signal_month"] = order["signal_date"].dt.to_period("M").astype(str)
    order["buy_month"] = order["buy_date"].dt.to_period("M").astype(str)
    order = order[
        [
            "year_month",
            "signal_month",
            "buy_month",
            "signal_date",
            "factor_date",
            "sell_date",
            "buy_date",
            "code",
            "factor_value",
            "factor_rank",
            "factor_rank_pct",
            "allow_flag",
        ]
        + [c for c in ["lncap", "ind_code"] if c in order.columns]
    ].copy()
    order = order.sort_values(["buy_date", "factor_rank", "code"], ascending=[True, True, True]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    if output_path.suffix.lower() == ".csv":
        order.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        order.to_parquet(output_path, index=False)
    logger.info("[Order] 输出写盘完成: %s, 耗时=%.2fs", output_path, time.perf_counter() - t)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary = (
        order.groupby(["year_month", "signal_month", "buy_month", "buy_date"], as_index=False)
        .agg(
            total_orders=("code", "size"),
            unique_codes=("code", "nunique"),
            avg_rank=("factor_rank", "mean"),
            min_rank=("factor_rank", "min"),
            max_rank=("factor_rank", "max"),
        )
        .sort_values(["buy_date", "year_month"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logger.info("[Order] 最终行数: %d", len(order))
    logger.info("[Order] 期数: %d", order["year_month"].nunique())
    logger.info("[Order] buy_date 数: %d", order["buy_date"].nunique())
    logger.info("[Order] summary: %s", summary_path)
    logger.info("[Order] 因子 rank 范围: %s ~ %s", int(order["factor_rank"].min()), int(order["factor_rank"].max()))
    logger.info("[Order] allow_flag=1 行数: %d", int((order["allow_flag"] == 1).sum()))
    logger.info("[Order] 总耗时: %.2fs", time.perf_counter() - t0)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 包含全部配置项的参数对象。
    """
    parser = argparse.ArgumentParser(description="基于 sell day 排名、buy day allow_flag 过滤生成拟交割单")
    parser.add_argument("--factor",      type=str, default=str(DEFAULT_FACTOR_PATH), help="输入因子 parquet")
    parser.add_argument("--pool",        type=str, default=str(DEFAULT_POOL_PATH),   help="股票池 parquet")
    parser.add_argument("--calendar-py", type=str, default=str(DEFAULT_CALENDAR_PY), help="load_calendar.py 路径")
    parser.add_argument("--output",      type=str, default=str(DEFAULT_OUTPUT_PATH), help="输出拟交割单路径")
    parser.add_argument("--log",         type=str, default=str(DEFAULT_LOG_PATH),    help="日志文件路径")
    parser.add_argument("--factor-col",  type=str, default="Factor_A021",            help="因子列名")
    parser.add_argument("--buyday",      type=str, default="month_end",            help="买入日规则")
    parser.add_argument("--sellday",     type=str, default="month_end",              help="卖出日规则")
    # ── 回测区间（限定调仓日历范围，避免加载全量历史） ──────────────────────────
    parser.add_argument(
        "--start", type=str, default=None,
        help="回测开始日期 YYYY-MM-DD；\n"
             "指定后调仓日历仅生成该日期之后的调仓期，\n"
             "默认 None（从因子文件日期范围自动推断，会打 WARNING）",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="回测结束日期 YYYY-MM-DD；\n"
             "默认 None（从因子文件日期范围自动推断，会打 WARNING）",
    )
    return parser.parse_args()


def main() -> int:
    """
    命令行入口。解析参数后调用 build_delivery_order()。

    返回:
        int: 0 表示成功。
    """
    args = parse_args()
    build_delivery_order(
        factor_path=Path(args.factor),
        pool_path=Path(args.pool),
        calendar_py_path=Path(args.calendar_py),
        output_path=Path(args.output),
        log_path=Path(args.log),
        factor_col=args.factor_col,
        buyday=args.buyday,
        sellday=args.sellday,
        start_date=args.start,   # 显式传入回测起始日，None 时函数内 fallback 并打 WARNING
        end_date=args.end,       # 显式传入回测截止日，None 时函数内 fallback 并打 WARNING
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
