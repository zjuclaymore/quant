#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd


def build_normal_listed_pool(input_path: str, output_path: str, stats_path: str, require_not_st: bool) -> None:
    print(f"[Info] 读取: {input_path}")
    df = pd.read_parquet(input_path)

    needed = ["date", "code", "is_listed"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"输入缺少必要字段: {miss}，请先运行 first_date_flag.py 生成 is_listed")

    df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")

    mask = df["is_listed"].fillna(0).astype("int8") == 1
    if require_not_st and "is_st" in df.columns:
        mask = mask & (pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype("int8") == 0)

    out = df.loc[mask].copy()
    out["year"] = (out["date"] // 10000).astype("Int64")

    # 年度截面统计：每年正常上市股票数、交易日数、行数
    yearly = (
        out.groupby("year", dropna=True)
        .agg(
            stocks=("code", lambda s: s.astype(str).str.zfill(6).nunique()),
            trade_days=("date", "nunique"),
            rows=("date", "size"),
        )
        .reset_index()
        .sort_values("year")
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_parquet(output_path, index=False, compression="snappy")
    yearly.to_csv(stats_path, index=False, encoding="utf-8-sig")

    print("[Done] 输出完成")
    print(f"[Done] 输出文件: {output_path}")
    print(f"[Done] 年度统计: {stats_path}")
    print(f"[Done] 输出行数: {len(out):,}")
    print("[Done] 年度统计预览:")
    print(yearly.tail(10).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="构建每年仅保留正常上市股票的截面池")
    parser.add_argument(
        "--input",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv.parquet",
        help="输入 parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet",
        help="输出 parquet（仅正常上市）",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\yearly_normal_listed_stats.csv",
        help="年度统计 CSV 输出",
    )
    parser.add_argument(
        "--require-not-st",
        choices=["yes", "no"],
        default="yes",
        help="是否要求 is_st=0（默认 yes）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_normal_listed_pool(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats_output,
        require_not_st=(args.require_not_st == "yes"),
    )
