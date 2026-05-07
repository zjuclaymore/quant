#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


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
        description="为真实交割单回填两套收益率（Scheme 1: 月首交割，Scheme 2: 月末交割）"
    )
    parser.add_argument(
        "--order", type=str, default=str(DEFAULT_ORDER), help="真实交割单 parquet"
    )
    parser.add_argument(
        "--price-dir",
        type=str,
        default=str(DEFAULT_PRICE_DIR),
        help="中国A股日行情目录",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="输出 parquet"
    )
    parser.add_argument(
        "--pool", type=str, default=None, help="股票池 parquet (用于二次校验可交易性)"
    )
    return parser.parse_args()


def code_to_wind(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("8", "4")) or c.startswith("92"):
        return f"{c}.BJ"
    if c.startswith(("5", "6", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"


def load_price_data_for_date(price_dir: Path, ymd: int) -> pd.DataFrame:
    p = price_dir / f"{ymd}.pickle"
    if not p.exists():
        return pd.DataFrame(columns=["Wind代码", "adj_close", "close", "limit_down"])

    df = pd.read_pickle(p)
    # 收盘价(元) -> close, 复权收盘价(元) -> adj_close, 跌停价(元) -> limit_down
    mapping = {
        "Wind代码": "Wind代码",
        "复权收盘价(元)": "adj_close",
        "收盘价(元)": "close",
        "跌停价(元)": "limit_down"
    }
    
    miss = [c for c in mapping.keys() if c not in df.columns]
    if miss:
        # 兼容性处理：如果没跌停价，默认为0，由下游 can_sell 处理
        if "跌停价(元)" in miss:
            df["跌停价(元)"] = 0.0
            miss.remove("跌停价(元)")
        if miss:
            raise ValueError(f"价格文件缺少必要列 {miss}: {p}")

    out = df[list(mapping.keys())].copy()
    out = out.rename(columns=mapping)
    out["Wind代码"] = out["Wind代码"].astype(str).str.upper()
    out = out.dropna(subset=["Wind代码", "adj_close"]).drop_duplicates(
        "Wind代码", keep="last"
    )
    return out


def build_ym_calendar_map(trade_dates: pd.Series) -> pd.DataFrame:
    d = (
        pd.to_datetime(trade_dates.astype(str), format="%Y%m%d", errors="coerce")
        .dropna()
        .sort_values()
    )
    df = pd.DataFrame({"trade_date": d, "ym": d.dt.to_period("M").astype(str)})

    # 获取每个月的开头和结尾
    m_start = (
        df.groupby("ym", as_index=False)["trade_date"]
        .min()
        .rename(columns={"trade_date": "m_start"})
    )
    m_end = (
        df.groupby("ym", as_index=False)["trade_date"]
        .max()
        .rename(columns={"trade_date": "m_end"})
    )

    cal = pd.merge(m_start, m_end, on="ym").sort_values("ym").reset_index(drop=True)
    cal["m_start_int"] = cal["m_start"].dt.strftime("%Y%m%d").astype(int)
    cal["m_end_int"] = cal["m_end"].dt.strftime("%Y%m%d").astype(int)

    # 偏移计算未来的交易日
    cal["next_ym_m_start_int"] = cal["m_start_int"].shift(-1).astype("Int64")
    cal["next_ym_m_end_int"] = cal["m_end_int"].shift(-1).astype("Int64")
    cal["next_next_ym_m_start_int"] = cal["m_start_int"].shift(-2).astype("Int64")

    return cal[
        [
            "ym",
            "m_end_int",
            "next_ym_m_start_int",
            "next_ym_m_end_int",
            "next_next_ym_m_start_int",
        ]
    ]


def main() -> int:
    args = parse_args()
    order_path = Path(args.order)
    price_dir = Path(args.price_dir)
    output_path = Path(args.output)
    pool_path = Path(args.pool) if args.pool else None

    # 加入 dealdeal 路径
    dealdeal_path = str(Path(__file__).resolve().parent.parent / "dealdeal")
    if dealdeal_path not in sys.path:
        sys.path.append(dealdeal_path)
    from trade_mask_flag import get_trade_mask_from_pool

    if not order_path.exists():
        raise FileNotFoundError(f"交割单不存在: {order_path}")
    if not price_dir.exists():
        raise FileNotFoundError(f"价格目录不存在: {price_dir}")

    order = pd.read_parquet(order_path)
    order = order.copy()
    order["code"] = order["code"].astype(str).str.zfill(6)
    order["Wind代码"] = order["code"].map(code_to_wind)

    trade_dates = pd.Series([p.stem for p in price_dir.glob("*.pickle") if p.stem.isdigit()])
    month_cal = build_ym_calendar_map(pd.to_numeric(trade_dates, errors="coerce").dropna().astype(int))

    order = order.merge(month_cal, left_on="year_month", right_on="ym", how="left")

    # 统一为月底模式 (Unified Monthly Pattern)
    # 逻辑：对于 Signal Month T，买入点为 row[T].buy_date (通常为 T 月末)，卖出点为 row[T+1].buy_date (通常为 T+1 月末)
    # 
    # 1. 提取所有涉及的交易日期并加载价格
    all_buy_dates = sorted(order["buy_date"].unique().tolist())
    all_dates = pd.to_datetime(all_buy_dates)
    price_map = {d.strftime("%Y%m%d"): load_price_data_for_date(price_dir, int(d.strftime("%Y%m%d"))) for d in all_dates}

    mask_map = {}
    if pool_path and pool_path.exists():
        print(f"[Info] 加载交易掩码: {pool_path}")
        pool = pd.read_parquet(pool_path)
        # 获取所有日期的 mask
        all_date_ints = [int(p) for p in price_map.keys()]
        full_mask = get_trade_mask_from_pool(pool, all_date_ints)
        for d, group in full_mask.groupby("date"):
            mask_map[str(d)] = group.set_index("code")

    # 2. 为每一行（Signal Month T）挂载买入价格与买入状态
    def get_info_for_date(ymd_str, suffix):
        px = price_map.get(ymd_str, pd.DataFrame()).copy()
        if px.empty: return pd.DataFrame()
        px["code"] = px["Wind代码"].str[:6]
        px = px.rename(columns={
            "adj_close": f"adj_close_{suffix}",
            "close": f"close_{suffix}",
            "limit_down": f"limit_down_{suffix}"
        })
        if ymd_str in mask_map:
            m = mask_map[ymd_str].copy().rename(columns={"can_buy": f"can_buy_{suffix}", "can_sell": f"can_sell_{suffix}", "is_limit_down": f"is_limit_down_{suffix}"})
            px = px.merge(m[[f"can_buy_{suffix}", f"can_sell_{suffix}", f"is_limit_down_{suffix}"]], on="code", how="left")
        else:
            px[f"can_buy_{suffix}"] = True
            px[f"can_sell_{suffix}"] = True
            px[f"is_limit_down_{suffix}"] = False
        return px[["code", f"adj_close_{suffix}", f"close_{suffix}", f"limit_down_{suffix}", f"can_buy_{suffix}", f"can_sell_{suffix}", f"is_limit_down_{suffix}"]]

    # 3. 计算下一期的买入点作为本期的卖出点
    order = order.sort_values(["code", "buy_date"])
    order["next_buy_date"] = order.groupby("code")["buy_date"].shift(-1)
    
    # 4. 合并数据
    out = order.copy()
    out["buy_ymd"] = out["buy_date"].dt.strftime("%Y%m%d")
    out["sell_ymd"] = out["next_buy_date"].dt.strftime("%Y%m%d")

    # 预初始化所有列为 NaN，避免赋值时的 KeyError
    all_attr_cols = [
        "adj_close_buy", "close_buy", "limit_down_buy", "can_buy_buy", "can_sell_buy", "is_limit_down_buy",
        "adj_close_sell", "close_sell", "limit_down_sell", "can_buy_sell", "can_sell_sell", "is_limit_down_sell"
    ]
    for c in all_attr_cols:
        out[c] = np.nan

    # 批量加载以提高性能
    print("[Info] 合并买入/卖出价格数据...")
    for d_str in out["buy_ymd"].dropna().unique():
        info = get_info_for_date(d_str, "buy")
        if info.empty: continue
        mask = (out["buy_ymd"] == d_str)
        # 注意：使用 values 赋值要求顺序一致，merge 后需保持原索引顺序或重新映射
        merged = out.loc[mask, ["code"]].merge(info, on="code", how="left")
        out.loc[mask, info.columns] = merged[info.columns].values

    for d_str in out["sell_ymd"].dropna().unique():
        info = get_info_for_date(d_str, "sell")
        if info.empty: continue
        mask = (out["sell_ymd"] == d_str)
        merged = out.loc[mask, ["code"]].merge(info, on="code", how="left")
        out.loc[mask, info.columns] = merged[info.columns].values

    # 5. 收益率计算逻辑
    # 买入判定
    valid_entry = (out["adj_close_buy"].notna() & out["can_buy_buy"].fillna(False))
    
    # 卖出判定与跌停补偿
    out["final_adj_sell"] = out["adj_close_sell"]
    # 跌停补偿逻辑：如果卖出日 (T+1 买入点) 无法卖出 (跌停/停牌)，强制使用跌停价
    mask_blocked = (out["can_sell_sell"].fillna(False) == False) & (out["adj_close_sell"].notna())
    out.loc[mask_blocked, "final_adj_sell"] = (
        out.loc[mask_blocked, "limit_down_sell"] * 
        (out.loc[mask_blocked, "adj_close_sell"] / out.loc[mask_blocked, "close_sell"].replace(0, 1))
    )

    valid_trade = valid_entry & out["final_adj_sell"].notna()
    
    out["monthly_return"] = pd.NA
    out.loc[valid_trade, "monthly_return"] = (out.loc[valid_trade, "final_adj_sell"] / out.loc[valid_trade, "adj_close_buy"]) - 1.0

    # 保持列名兼容
    out["monthly_return_scheme2"] = out["monthly_return"]
    out["monthly_return_scheme1"] = pd.NA

    # 清理中间列
    drop_cols = ["buy_ymd", "sell_ymd", "next_buy_date", "adj_close_buy", "close_buy", "limit_down_buy", "can_buy_buy", "can_sell_buy", "is_limit_down_buy",
                 "adj_close_sell", "close_sell", "limit_down_sell", "can_buy_sell", "can_sell_sell", "is_limit_down_sell", "final_adj_sell"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[Done] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
