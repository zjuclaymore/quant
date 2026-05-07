#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _to_int_date(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64")


def _is_beijing_code(code_series: pd.Series) -> pd.Series:
    c = code_series.astype(str).str.zfill(6)
    # 北交所常见代码段：8xxxxx / 4xxxxx / 92xxxx
    return c.str.startswith(("8", "4", "92"))


def _load_calendar_df(calendar_py_path: str, start_date: str, end_date: str, buyday: str, sellday: str) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("load_calendar_module", calendar_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载日历模块: {calendar_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_calendar_from_cache"):
        raise RuntimeError("load_calendar.py 中未找到 load_calendar_from_cache")

    cal = module.load_calendar_from_cache(
        start_date=start_date,
        end_date=end_date,
        buyday=buyday,
        sellday=sellday,
        delay_days=0,
        calendar_cache_path=None,
        logger=None,
    )
    if cal is None or len(cal) == 0:
        raise RuntimeError("交易日历为空，无法生成 allow_flag")
    return cal


def build_allow_flag(input_path: str, output_path: str, calendar_py_path: str, buyday: str, sellday: str) -> None:
    print(f"[Info] 读取输入文件: {input_path}")
    df = pd.read_parquet(input_path)

    required = ["date", "code", "is_st"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"输入缺少必要列: {miss}")

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = _to_int_date(df["date"])
    df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype("int8")

    if "list_cal90_date" in df.columns:
        df["list90"] = _to_int_date(df["list_cal90_date"])
    elif "list_td90_date" in df.columns:
        df["list90"] = _to_int_date(df["list_td90_date"])
    elif "first_date" in df.columns:
        first = pd.to_datetime(_to_int_date(df["first_date"]).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        df["list90"] = pd.to_datetime(first + pd.to_timedelta(90, unit="D")).dt.strftime("%Y%m%d")
        df["list90"] = _to_int_date(df["list90"])
    else:
        raise ValueError("输入缺少 list_cal90_date/list_td90_date/first_date，无法判断90天新股")

    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].astype(np.int64)
    df["is_bj"] = _is_beijing_code(df["code"]).astype("int8")

    min_dt = pd.to_datetime(str(int(df["date"].min())), format="%Y%m%d")
    max_dt = pd.to_datetime(str(int(df["date"].max())), format="%Y%m%d")
    start_date = min_dt.strftime("%Y-%m-%d")
    end_date = max_dt.strftime("%Y-%m-%d")

    print(f"[Info] 载入交易日历: {calendar_py_path}")
    print(f"[Info] 日历区间: {start_date} ~ {end_date}")
    cal = _load_calendar_df(
        calendar_py_path=calendar_py_path,
        start_date=start_date,
        end_date=end_date,
        buyday=buyday,
        sellday=sellday,
    )

    cal = cal[["year_month", "sell_date", "buy_date"]].copy()
    cal["sell_int"] = pd.to_datetime(cal["sell_date"]).dt.strftime("%Y%m%d").astype(int)
    cal["buy_int"] = pd.to_datetime(cal["buy_date"]).dt.strftime("%Y%m%d").astype(int)

    df["allow_flag"] = 0

    print("[Info] 按 sell_day 条件筛选，并映射到 buy_day 赋值 allow_flag=1")
    updates = 0
    for r in cal.itertuples(index=False):
        sell_day = int(r.sell_int)
        buy_day = int(r.buy_int)

        snap = df[df["date"] == sell_day][["code", "is_st", "list90", "is_bj"]].copy()
        if snap.empty:
            continue

        # sell_day 同时满足：非ST、非90天新股、非北交所
        cond = (
            (snap["is_st"] == 0)
            & (snap["list90"].notna())
            & (sell_day >= snap["list90"].astype(np.int64))
            & (snap["is_bj"] == 0)
        )
        eligible_codes = set(snap.loc[cond, "code"].tolist())
        if not eligible_codes:
            continue

        buy_mask = (df["date"] == buy_day) & (df["code"].isin(eligible_codes))
        cnt = int(buy_mask.sum())
        if cnt > 0:
            df.loc[buy_mask, "allow_flag"] = 1
            updates += cnt

    keep_cols = [c for c in ["list90", "is_bj"] if c in df.columns]
    if keep_cols:
        df = df.drop(columns=keep_cols)

    print(f"[Info] 写出结果: {output_path}")
    df.to_parquet(output_path, index=False, compression="snappy")

    print("[Done] 完成")
    print(f"[Done] 总行数: {len(df):,}")
    print(f"[Done] allow_flag=1 行数: {int((df['allow_flag'] == 1).sum()):,}")
    print(f"[Done] 本次更新 buy_day 命中行数: {updates:,}")


def parse_args():
    parser = argparse.ArgumentParser(description="基于 sell_day 条件在 buy_day 赋值 allow_flag")
    parser.add_argument(
        "--input",
        type=str,
        default=r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet",
        help="输入 parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet",
        help="输出 parquet",
    )
    parser.add_argument(
        "--calendar-py",
        type=str,
        default=r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\trade_calender\load_calendar.py",
        help="load_calendar.py 路径",
    )
    parser.add_argument(
        "--buyday",
        type=str,
        default="month_start",
        help="买入日规则，默认 month_start",
    )
    parser.add_argument(
        "--sellday",
        type=str,
        default="month_end",
        help="卖出日规则，默认 month_end",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_allow_flag(
        input_path=args.input,
        output_path=args.output,
        calendar_py_path=args.calendar_py,
        buyday=args.buyday,
        sellday=args.sellday,
    )
