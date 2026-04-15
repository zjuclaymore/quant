#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trade mask 工具。

说明:
- 原始拆分文件来自类方法片段，无法直接运行；此处提供可执行版本。
- 在当前股票池字段下（无 is_limit_up/is_limit_down/tradable_base），
  用如下口径构造 can_buy:
    tradable_base = (allow_flag == 1) & (volume > 0)
    can_buy = tradable_base
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _to_int_date(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def get_trade_mask_from_pool(pool_df: pd.DataFrame, target_dates: Iterable[int]) -> pd.DataFrame:
    required = ["date", "code", "allow_flag"]
    miss = [c for c in required if c not in pool_df.columns]
    if miss:
        raise ValueError(f"股票池缺少必要列: {miss}")

    df = pool_df.copy()
    df["date"] = _to_int_date(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["allow_flag"] = pd.to_numeric(df["allow_flag"], errors="coerce").fillna(0).astype("int8")

    target_dates = [int(x) for x in target_dates]
    df = df[df["date"].isin(target_dates)].copy()

    if "volume" in df.columns:
        vol_ok = pd.to_numeric(df["volume"], errors="coerce").fillna(0) > 0
    else:
        vol_ok = pd.Series(True, index=df.index)

    df["tradable_base"] = (df["allow_flag"] == 1) & vol_ok
    df["can_buy"] = df["tradable_base"]

    return df[["date", "code", "allow_flag", "tradable_base", "can_buy"]].copy()
