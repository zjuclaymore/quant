#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_st_data(st_path=None):
    """加载 ST 原始数据。"""
    if st_path is None:
        st_path = r"e:\1_basement\quant_research\data\中国A股特别处理_AShareST\ST.pickle"
    elif os.path.isdir(st_path):
        st_path = os.path.join(st_path, "ST.pickle")

    if not os.path.exists(st_path):
        raise FileNotFoundError(f"未找到ST数据文件: {st_path}")

    try:
        return pd.read_pickle(st_path)
    except Exception as e:
        raise RuntimeError(f"加载 ST 数据失败: {e}") from e


def _normalize_st_intervals(st_df):
    """将 ST 数据标准化为区间表: code, start_date, end_date, st_type。"""
    required_cols = ["Wind代码", "实施日期", "撤销日期", "特别处理类型"]
    missing = [c for c in required_cols if c not in st_df.columns]
    if missing:
        raise ValueError(f"ST数据缺少字段: {missing}")

    out = st_df[required_cols].copy()
    out["code"] = out["Wind代码"].astype(str).str.split(".").str[0]

    out["start_date"] = pd.to_numeric(out["实施日期"], errors="coerce")
    out["end_date"] = pd.to_numeric(out["撤销日期"], errors="coerce")
    out["st_type"] = out["特别处理类型"].astype(str)

    out = out.dropna(subset=["code", "start_date"])
    out["start_date"] = out["start_date"].astype(np.int64)
    out["end_date"] = out["end_date"].fillna(20991231).astype(np.int64)

    # 防御性修正: 若结束日早于起始日，则将结束日修正为起始日
    bad = out["end_date"] < out["start_date"]
    if bad.any():
        out.loc[bad, "end_date"] = out.loc[bad, "start_date"]

    return out[["code", "start_date", "end_date", "st_type"]]


def build_st_daily_pairs(st_intervals, trading_dates):
    """基于 ST 区间和交易日，生成逐日 ST 标记对(code,date)。"""
    tdates = np.array(sorted(set(int(x) for x in trading_dates)), dtype=np.int64)
    pairs = []

    for row in st_intervals.itertuples(index=False):
        left = np.searchsorted(tdates, int(row.start_date), side="left")
        right = np.searchsorted(tdates, int(row.end_date), side="right")
        if right <= left:
            continue
        d = tdates[left:right]
        if d.size == 0:
            continue
        pairs.append(
            pd.DataFrame(
                {
                    "date": d,
                    "code": row.code,
                    "is_st": 1,
                    "st_type": row.st_type,
                }
            )
        )

    if not pairs:
        return pd.DataFrame(columns=["date", "code", "is_st", "st_type"])

    st_daily = pd.concat(pairs, ignore_index=True)
    # 同日同票可能有多条原因，保留第一条即可
    st_daily = st_daily.drop_duplicates(subset=["date", "code"], keep="first")
    return st_daily


def merge_st_to_stock_pool(stock_pool_path, st_path=None, output_path=None):
    """将 ST 标记合并到 original_stock_pool。"""
    if not os.path.exists(stock_pool_path):
        raise FileNotFoundError(f"未找到股票池文件: {stock_pool_path}")

    print(f"[Info] 读取股票池: {stock_pool_path}")
    pool = pd.read_parquet(stock_pool_path)
    required_pool_cols = ["date", "code"]
    missing_pool = [c for c in required_pool_cols if c not in pool.columns]
    if missing_pool:
        raise ValueError(f"股票池缺少字段: {missing_pool}")

    pool["date"] = pd.to_numeric(pool["date"], errors="coerce").astype(np.int64)
    pool["code"] = pool["code"].astype(str).str.zfill(6)

    print("[Info] 读取 ST 原始数据")
    st_raw = load_st_data(st_path)
    st_intervals = _normalize_st_intervals(st_raw)

    print("[Info] 生成逐交易日 ST 标记")
    st_daily = build_st_daily_pairs(st_intervals, pool["date"].unique())

    print("[Info] 合并 ST 到股票池")
    merged = pool.merge(st_daily, on=["date", "code"], how="left")
    merged["is_st"] = merged["is_st"].fillna(0).astype(np.int8)
    merged["st_type"] = merged["st_type"].fillna("")

    if output_path is None:
        output_path = str(Path(stock_pool_path).with_name("original_stock_pool_with_st.parquet"))

    print(f"[Info] 写出结果: {output_path}")
    merged.to_parquet(output_path, index=False, compression="snappy")

    st_ratio = float(merged["is_st"].mean()) if len(merged) else 0.0
    print("[Done] 合并完成")
    print(f"[Done] 行数: {len(merged):,}")
    print(f"[Done] ST占比: {st_ratio:.4%}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="读取ST数据并合并到 original_stock_pool.parquet")
    parser.add_argument(
        "--stock-pool",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool.parquet",
        help="原始股票池 parquet 路径",
    )
    parser.add_argument(
        "--st-path",
        type=str,
        default=r"e:\1_basement\quant_research\data\中国A股特别处理_AShareST\ST.pickle",
        help="ST 源文件路径（pickle）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"e:\1_basement\quant_research\back_test\sigle_factor_test\function_breakdown\stock_pool\original_stock_pool_with_st.parquet",
        help="输出 parquet 路径",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_st_to_stock_pool(
        stock_pool_path=args.stock_pool,
        st_path=args.st_path,
        output_path=args.output,
    )
