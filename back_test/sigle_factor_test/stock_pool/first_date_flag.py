#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _to_int_date_series(series):
    vals = pd.to_numeric(series, errors="coerce")
    return vals.astype("Int64")


def _nth_trading_date(date_array, n):
    """返回第 n 个交易日（n从1开始），不足则返回 <NA>。"""
    idx = n - 1
    if len(date_array) <= idx:
        return pd.NA
    return int(date_array[idx])


def _calendar_plus(first_date_int, days):
    """first_date(YYYYMMDD int) + 自然日，返回YYYYMMDD int。"""
    if pd.isna(first_date_int):
        return pd.NA
    dt = pd.to_datetime(str(int(first_date_int)), format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return pd.NA
    return int((dt + timedelta(days=days)).strftime("%Y%m%d"))


def _normalize_code_to_6(code_series):
    s = code_series.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
    return s.str.zfill(6)


def load_listing_info(basic_info_path):
    """从 A股基本资料.pickle 读取上市/退市日期。"""
    basic_df = pd.read_pickle(basic_info_path)
    cols = basic_df.columns.tolist()

    code_col = "Wind代码" if "Wind代码" in cols else None
    list_col = "上市日期" if "上市日期" in cols else None
    delist_col = "退市日期" if "退市日期" in cols else None

    if code_col is None:
        candidates = [c for c in cols if "Wind" in str(c) or "代码" in str(c)]
        if candidates:
            code_col = candidates[0]
    if list_col is None:
        candidates = [c for c in cols if "上市" in str(c) and "日期" in str(c)]
        if candidates:
            list_col = candidates[0]
    if delist_col is None:
        candidates = [c for c in cols if "退市" in str(c) and "日期" in str(c)]
        if candidates:
            delist_col = candidates[0]

    if code_col is None or list_col is None:
        raise ValueError(
            f"A股基本资料缺少必要字段，当前列: {cols}，至少需要代码列和上市日期列"
        )

    use_cols = [code_col, list_col] + ([delist_col] if delist_col else [])
    listing = basic_df[use_cols].copy()
    rename_map = {code_col: "code", list_col: "first_date"}
    if delist_col:
        rename_map[delist_col] = "delist_date"
    else:
        listing["delist_date"] = pd.NA

    listing = listing.rename(columns=rename_map)
    listing["code"] = _normalize_code_to_6(listing["code"])
    listing["first_date"] = pd.to_numeric(listing["first_date"], errors="coerce").astype("Int64")
    listing["delist_date"] = pd.to_numeric(listing["delist_date"], errors="coerce").astype("Int64")

    listing = listing.dropna(subset=["code", "first_date"]).drop_duplicates(subset=["code"], keep="first")
    return listing[["code", "first_date", "delist_date"]]


def build_first_date_features(pool_df, listing_df):
    """
    为每只股票计算（上市/退市来自 A股基本资料）：
    1) first_date / delist_date
    2) 上市后第30/60/90/365个市场交易日
    3) first_date + 60/90/365 自然日
    """
    required = ["date", "code"]
    missing = [c for c in required if c not in pool_df.columns]
    if missing:
        raise ValueError(f"输入股票池缺少必要字段: {missing}")

    work = pool_df[["code", "date"]].copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["date"] = _to_int_date_series(work["date"])
    work = work.dropna(subset=["code", "date"]).drop_duplicates(subset=["code", "date"])
    work["date"] = work["date"].astype(np.int64)
    trade_dates = np.sort(work["date"].unique())

    rows = []
    for _, row in listing_df.iterrows():
        code = str(row["code"]).zfill(6)
        first_date = row["first_date"]
        delist_date = row["delist_date"]

        if pd.isna(first_date):
            arr = np.array([], dtype=np.int64)
            first_date_int = pd.NA
        else:
            first_date_int = int(first_date)
            arr = trade_dates[trade_dates >= first_date_int]

        rows.append(
            {
                "code": code,
                "first_date": first_date_int,
                "delist_date": (int(delist_date) if not pd.isna(delist_date) else pd.NA),
                "list_td30_date": _nth_trading_date(arr, 30),
                "list_td60_date": _nth_trading_date(arr, 60),
                "list_td90_date": _nth_trading_date(arr, 90),
                "list_td365_date": _nth_trading_date(arr, 365),
                "list_cal60_date": _calendar_plus(first_date_int, 60),
                "list_cal90_date": _calendar_plus(first_date_int, 90),
                "list_cal365_date": _calendar_plus(first_date_int, 365),
            }
        )

    feat = pd.DataFrame(rows)
    int_cols = [
        "first_date",
        "delist_date",
        "list_td30_date",
        "list_td60_date",
        "list_td90_date",
        "list_td365_date",
        "list_cal60_date",
        "list_cal90_date",
        "list_cal365_date",
    ]
    for c in int_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").astype("Int64")
    return feat


def merge_first_dates(
    input_path,
    basic_info_path,
    output_path,
):
    print(f"[Info] 读取输入股票池: {input_path}")
    pool = pd.read_parquet(input_path)
    print(f"[Info] 输入行数: {len(pool):,}")

    print(f"[Info] 读取上市退市信息: {basic_info_path}")
    listing = load_listing_info(basic_info_path)
    print(f"[Info] 上市退市记录数: {len(listing):,}")

    print("[Info] 计算每只股票上市里程碑日期")
    feat = build_first_date_features(pool, listing)
    print(f"[Info] 计算完成股票数: {len(feat):,}")

    print("[Info] 合并回股票池")
    pool["code"] = pool["code"].astype(str).str.zfill(6)

    # 若输入中已存在旧版本字段，先删除，避免 merge 后出现 _x/_y 列
    replace_cols = [
        "first_date",
        "delist_date",
        "list_td30_date",
        "list_td60_date",
        "list_td90_date",
        "list_td365_date",
        "list_cal60_date",
        "list_cal90_date",
        "list_cal365_date",
        "is_listed",
    ]
    existing_replace_cols = [c for c in replace_cols if c in pool.columns]
    if existing_replace_cols:
        print(f"[Info] 删除旧字段后覆盖更新: {existing_replace_cols}")
        pool = pool.drop(columns=existing_replace_cols)

    merged = pool.merge(feat, on="code", how="left")

    # 上市退市状态标记：上市后且未退市(或尚未到退市日)
    merged["date"] = pd.to_numeric(merged["date"], errors="coerce").astype("Int64")
    merged["is_listed"] = (
        merged["first_date"].notna()
        & (merged["date"] >= merged["first_date"])
        & (merged["delist_date"].isna() | (merged["date"] <= merged["delist_date"]))
    ).astype("int8")

    print(f"[Info] 写出结果: {output_path}")
    merged.to_parquet(output_path, index=False, compression="snappy")
    print("[Done] 完成")
    print(f"[Done] 输出行数: {len(merged):,}")
    print(f"[Done] 输出列: {merged.columns.tolist()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="给 original_stock_pool_with_st.parquet 合并首日和上市里程碑日期特征"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st.parquet",
        help="输入 parquet 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--basic-info",
        type=str,
        default=r"e:\1_basement\quant_research\data\中国A股基本资料_AShareDescription\A股基本资料.pickle",
        help="A股基本资料 pickle 路径（用于上市/退市标记）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_first_dates(
        input_path=args.input,
        basic_info_path=args.basic_info,
        output_path=args.output,
    )
