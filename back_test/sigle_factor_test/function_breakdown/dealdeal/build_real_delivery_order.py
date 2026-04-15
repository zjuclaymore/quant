#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from trade_mask_flag import get_trade_mask_from_pool


ROOT = Path(r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown")
DEFAULT_PAPER_ORDER = ROOT / "dealdeal" / "factor_A021_delivery_order.parquet"
DEFAULT_POOL = ROOT / "stock_pool" / "original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet"
DEFAULT_OUT = ROOT / "dealdeal" / "factor_A021_real_delivery_order.parquet"
DEFAULT_SUMMARY = ROOT / "dealdeal" / "factor_A021_real_delivery_order_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 trade_mask_flag 在 buyday 过滤拟交割单，生成真实交割单")
    parser.add_argument("--paper-order", type=str, default=str(DEFAULT_PAPER_ORDER), help="拟交割单 parquet")
    parser.add_argument("--pool", type=str, default=str(DEFAULT_POOL), help="股票池 parquet")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT), help="真实交割单 parquet")
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY), help="真实交割单汇总 csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    paper_path = Path(args.paper_order)
    pool_path = Path(args.pool)
    out_path = Path(args.output)
    summary_path = Path(args.summary)

    paper = pd.read_parquet(paper_path)
    need_cols = ["year_month", "buy_date", "code"]
    miss = [c for c in need_cols if c not in paper.columns]
    if miss:
        raise ValueError(f"拟交割单缺少必要列: {miss}")

    pool_cols = set(pq.read_schema(pool_path).names)
    use_cols = [c for c in ["date", "code", "allow_flag", "volume"] if c in pool_cols]
    pool = pd.read_parquet(pool_path, columns=use_cols)

    paper = paper.copy()
    paper["code"] = paper["code"].astype(str).str.zfill(6)
    paper["buy_int"] = pd.to_datetime(paper["buy_date"]).dt.strftime("%Y%m%d").astype(int)

    buy_dates = sorted(paper["buy_int"].unique().tolist())
    mask = get_trade_mask_from_pool(pool, buy_dates)
    mask = mask.rename(columns={"date": "buy_int"})

    # 仅保留可买键，避免大规模 left merge 带来的性能开销。
    elig = mask[mask["can_buy"]][["buy_int", "code", "allow_flag", "tradable_base", "can_buy"]].copy()
    real_order = paper.merge(elig, on=["buy_int", "code"], how="inner")
    real_order = real_order.drop(columns=["buy_int"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    real_order.to_parquet(out_path, index=False)

    summary = (
        real_order.groupby(["year_month", "buy_date"], as_index=False)
        .agg(total_orders=("code", "size"), unique_codes=("code", "nunique"))
        .sort_values(["buy_date", "year_month"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[Done] paper rows: {len(paper):,}")
    print(f"[Done] real rows : {len(real_order):,}")
    print(f"[Done] kept ratio: {len(real_order) / len(paper):.2%}")
    print(f"[Done] out: {out_path}")
    print(f"[Done] summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
