#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import pandas as pd


def _standardize_code_date(df):
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_numeric(out["date"], errors="coerce")
    if out["date"].isna().any():
        out = out[out["date"].notna()].copy()
    out["date"] = out["date"].astype(np.int64)
    return out


def add_volume_liquidity_features(input_path, output_path, window):
    print(f"[Info] 读取输入: {input_path}")
    df = pd.read_parquet(input_path)

    required = ["date", "code", "volume"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"输入缺少必要列: {miss}")

    df = _standardize_code_date(df)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["code", "date"], kind="mergesort")

    print(f"[Info] 计算每只股票 {window} 日滚动成交量均值")
    df["volume_ma20"] = (
        df.groupby("code", sort=False)["volume"]
        .transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    )

    print("[Info] 计算每日截面滚动均量百分位")
    df["volume_ma20_pct"] = df.groupby("date", sort=False)["volume_ma20"].transform(
        lambda s: s.rank(method="average", pct=True)
    )

    print(f"[Info] 写出结果: {output_path}")
    df.to_parquet(output_path, index=False, compression="snappy")

    ma_valid = float(df["volume_ma20"].notna().mean()) if len(df) else 0.0
    pct_valid = float(df["volume_ma20_pct"].notna().mean()) if len(df) else 0.0
    print("[Done] 计算完成")
    print(f"[Done] 行数: {len(df):,}")
    print(f"[Done] volume_ma20 有值率: {ma_valid:.2%}")
    print(f"[Done] volume_ma20_pct 有值率: {pct_valid:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description="给股票池计算20日滚动均量及日截面百分位")
    parser.add_argument(
        "--input",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet",
        help="输入股票池 parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet",
        help="输出 parquet",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="滚动窗口（日）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_volume_liquidity_features(
        input_path=args.input,
        output_path=args.output,
        window=args.window,
    )
