#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ORDER = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\back_test\factor_A021_real_delivery_order_with_group.parquet"
)
DEFAULT_PRICE_DIR = Path(
    r"E:\1_basement\quant_research\data\中国A股日行情_AShareEODPrices"
)
DEFAULT_OUTPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\back_test\factor_A021_real_delivery_order_with_group_adjclose.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为真实交割单回填 buy_date 复权收盘价与 buy_month 月末交易日复权收盘价"
    )
    parser.add_argument("--order", type=str, default=str(DEFAULT_ORDER), help="真实交割单 parquet")
    parser.add_argument("--price-dir", type=str, default=str(DEFAULT_PRICE_DIR), help="中国A股日行情目录")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出 parquet")
    return parser.parse_args()


def code_to_wind(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("8", "4")) or c.startswith("92"):
        return f"{c}.BJ"
    if c.startswith(("5", "6", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"


def load_adj_close_for_date(price_dir: Path, ymd: int) -> pd.DataFrame:
    p = price_dir / f"{ymd}.pickle"
    if not p.exists():
        return pd.DataFrame(columns=["Wind代码", "adj_close"])

    df = pd.read_pickle(p)
    need = ["Wind代码", "复权收盘价(元)"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"价格文件缺少必要列 {miss}: {p}")

    out = df[["Wind代码", "复权收盘价(元)"]].copy()
    out = out.rename(columns={"复权收盘价(元)": "adj_close"})
    out["Wind代码"] = out["Wind代码"].astype(str).str.upper()
    out = out.dropna(subset=["Wind代码", "adj_close"]).drop_duplicates("Wind代码", keep="last")
    return out


def build_month_end_map(trade_dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(trade_dates.astype(str), format="%Y%m%d", errors="coerce")
    m = d.dt.to_period("M").astype(str)
    x = pd.DataFrame({"buy_month": m, "trade_date": d})
    x = x.dropna(subset=["buy_month", "trade_date"]).sort_values("trade_date")
    z = x.groupby("buy_month", as_index=False)["trade_date"].max()
    z["buy_month_end_date"] = z["trade_date"].dt.strftime("%Y%m%d").astype(int)
    return z[["buy_month", "buy_month_end_date"]]


def main() -> int:
    args = parse_args()
    order_path = Path(args.order)
    price_dir = Path(args.price_dir)
    output_path = Path(args.output)

    if not order_path.exists():
        raise FileNotFoundError(f"交割单不存在: {order_path}")
    if not price_dir.exists():
        raise FileNotFoundError(f"价格目录不存在: {price_dir}")

    order = pd.read_parquet(order_path)
    req = ["code", "buy_date"]
    miss = [c for c in req if c not in order.columns]
    if miss:
        raise ValueError(f"交割单缺少必要列: {miss}")

    order = order.copy()
    order["code"] = order["code"].astype(str).str.zfill(6)
    order["Wind代码"] = order["code"].map(code_to_wind)
    order["buy_date"] = pd.to_datetime(order["buy_date"], errors="coerce")
    order["buy_date_int"] = pd.to_numeric(order["buy_date"].dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")
    order["buy_month"] = order["buy_date"].dt.to_period("M").astype(str)

    # 用目录中文件名作为交易日全集，计算每个月最后一个交易日。
    trade_dates = pd.Series([p.stem for p in price_dir.glob("*.pickle") if p.stem.isdigit()])
    month_end_map = build_month_end_map(pd.to_numeric(trade_dates, errors="coerce").dropna().astype(int))
    order = order.merge(month_end_map, on="buy_month", how="left")

    buy_dates_needed = sorted(order["buy_date_int"].dropna().astype(int).unique().tolist())
    month_end_dates_needed = sorted(order["buy_month_end_date"].dropna().astype(int).unique().tolist())

    all_dates = sorted(set(buy_dates_needed + month_end_dates_needed))
    price_map = {}
    for d in all_dates:
        price_map[d] = load_adj_close_for_date(price_dir, d)

    buy_parts = []
    for d in buy_dates_needed:
        px = price_map[d].copy()
        if px.empty:
            continue
        px["buy_date_int"] = d
        px = px.rename(columns={"adj_close": "buy_date_adj_close"})
        buy_parts.append(px[["Wind代码", "buy_date_int", "buy_date_adj_close"]])
    buy_df = pd.concat(buy_parts, ignore_index=True) if buy_parts else pd.DataFrame(columns=["Wind代码", "buy_date_int", "buy_date_adj_close"])

    end_parts = []
    for d in month_end_dates_needed:
        px = price_map[d].copy()
        if px.empty:
            continue
        px["buy_month_end_date"] = d
        px = px.rename(columns={"adj_close": "buy_month_end_adj_close"})
        end_parts.append(px[["Wind代码", "buy_month_end_date", "buy_month_end_adj_close"]])
    end_df = pd.concat(end_parts, ignore_index=True) if end_parts else pd.DataFrame(columns=["Wind代码", "buy_month_end_date", "buy_month_end_adj_close"])

    out = order.merge(buy_df, on=["Wind代码", "buy_date_int"], how="left")
    out = out.merge(end_df, on=["Wind代码", "buy_month_end_date"], how="left")

    # 月收益率: 从 buy_date 到 buy_month 最后一个交易日的复权收盘收益。
    out["monthly_return"] = pd.NA
    valid = (
        out["buy_date_adj_close"].notna()
        & out["buy_month_end_adj_close"].notna()
        & (out["buy_date_adj_close"] != 0)
    )
    out.loc[valid, "monthly_return"] = (
        out.loc[valid, "buy_month_end_adj_close"] / out.loc[valid, "buy_date_adj_close"] - 1.0
    )

    out["buy_month_end_date"] = pd.to_datetime(
        out["buy_month_end_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )

    drop_cols = [
        "Wind代码",
        "buy_date_int",
        "allow_flag",
        "allow_flag_x",
        "allow_flag_y",
        "tradable_base",
        "can_buy",
        "can_sell",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"[Done] rows: {len(out):,}")
    print(f"[Done] output: {output_path}")
    print(f"[Done] buy_date_adj_close 缺失率: {out['buy_date_adj_close'].isna().mean():.2%}")
    print(f"[Done] buy_month_end_adj_close 缺失率: {out['buy_month_end_adj_close'].isna().mean():.2%}")
    print(f"[Done] monthly_return 缺失率: {out['monthly_return'].isna().mean():.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
