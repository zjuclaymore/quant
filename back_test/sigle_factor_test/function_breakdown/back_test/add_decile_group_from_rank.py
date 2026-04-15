#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\dealdeal\factor_A021_real_delivery_order.parquet"
)
DEFAULT_OUTPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\dealdeal\factor_A021_real_delivery_order_with_group.parquet"
)


def assign_groups_by_rank(df: pd.DataFrame, date_col: str, rank_col: str, group_col: str, n_groups: int) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"缺少日期列: {date_col}")
    if rank_col not in df.columns:
        raise ValueError(f"缺少排序列: {rank_col}")

    out = df.copy()
    out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce")

    valid = out[rank_col].notna()
    out[group_col] = pd.NA

    # rank 数值越小表示越靠前；组号 1 表示最高组，n_groups 表示最低组。
    # 先在每个交易日截面内重新按 rank 位置排序，再等分为 n_groups 组。
    sub = out.loc[valid].copy()
    tie_cols = [c for c in ["code", "year_month"] if c in sub.columns]
    sub = sub.sort_values([date_col, rank_col] + tie_cols, ascending=[True, True] + [True] * len(tie_cols))

    sub["_pos"] = sub.groupby(date_col).cumcount() + 1
    sub["_n"] = sub.groupby(date_col)[rank_col].transform("size")
    sub[group_col] = (((sub["_pos"] - 1) * n_groups / sub["_n"]).astype("int64") + 1).clip(1, n_groups).astype("Int64")

    out.loc[sub.index, group_col] = sub[group_col]

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 rank 在交易日截面做 10 等分分组")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入 parquet")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出 parquet")
    parser.add_argument("--date-col", type=str, default="buy_date", help="交易日列")
    parser.add_argument("--rank-col", type=str, default="factor_rank", help="rank 列")
    parser.add_argument("--group-col", type=str, default="group_id", help="输出组号列名")
    parser.add_argument("--n-groups", type=int, default=10, help="分组数，默认 10")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_parquet(input_path)
    df2 = assign_groups_by_rank(
        df=df,
        date_col=args.date_col,
        rank_col=args.rank_col,
        group_col=args.group_col,
        n_groups=args.n_groups,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(output_path, index=False)

    print(f"[Done] input rows: {len(df):,}")
    print(f"[Done] output rows: {len(df2):,}")
    print(f"[Done] output file: {output_path}")
    print(f"[Done] group col: {args.group_col}")
    print("[Done] group distribution sample:")
    print(df2[args.group_col].value_counts(dropna=False).sort_index().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
