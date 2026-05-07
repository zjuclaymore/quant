#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade mask 工具。

构造每个买入日（buy_date）截面的可买性标志，用于从拟交割单生成真实交割单。

可买性判断口径（按优先级叠加）:
    tradable_base = (allow_flag == 1) & (volume > 0)
        → allow_flag=1: 未停牌、非ST/退市预警、处于正常上市状态
        → volume > 0:   当日有成交量，排除无流动性标的

    涨停板过滤（A 股涨停无法主动买入）:
        优先级 1 — 直接读 is_limit_up 列（布尔型，True=涨停）
        优先级 2 — 从 pct_chg 列推断：pct_chg >= 9.8 视为涨停
                   （宽松阈值：ST 股 5% 涨停亦触发；科创板/创业板为 20%，
                    请确认日行情数据的 pct_chg 单位为百分比形式，如 9.85）
        无法推断  — 打印 WARNING，跳过涨停过滤（不静默忽略）

    can_buy = tradable_base & (~is_limit_up)
"""

from __future__ import annotations

import warnings
from typing import Iterable

import pandas as pd

# 涨停推断阈值（pct_chg 列，单位：百分比，如 9.85）
_LIMIT_UP_PCT_THRESHOLD = 9.8


def _to_int_date(series: pd.Series) -> pd.Series:
    """
    将日期列转换为 YYYYMMDD 整数格式（Int64）。

    参数:
        series (pd.Series): 日期序列，支持 int / float / 字符串。

    返回:
        pd.Series: YYYYMMDD 格式的 Int64 序列；无法解析的值置为 NA。
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _infer_limit_up(df: pd.DataFrame) -> pd.Series:
    """
    从股票池 DataFrame 推断当日是否涨停。

    推断优先级:
        1. is_limit_up 列：布尔型，True 表示涨停，直接使用。
        2. 价格对比：若含 '收盘价(元)' 和 '涨停价(元)'，
           则 (收盘价 >= 涨停价 - 0.01) 视为涨停。
        3. pct_chg 列：涨跌幅（百分比，如 9.85），
           pct_chg >= _LIMIT_UP_PCT_THRESHOLD 视为涨停。
        4. 以上均无：返回全 False 并打印 WARNING。

    参数:
        df (pd.DataFrame): 已过滤至目标日期的股票池切片。

    返回:
        pd.Series[bool]: 与 df 索引对齐，True 表示该标的当日涨停。
    """
    if "is_limit_up" in df.columns:
        return df["is_limit_up"].fillna(False).astype(bool)

    if "收盘价(元)" in df.columns and "涨停价(元)" in df.columns:
        close = pd.to_numeric(df["收盘价(元)"], errors="coerce").fillna(0.0)
        l_up = pd.to_numeric(df["涨停价(元)"], errors="coerce").fillna(1e9)
        return close >= (l_up - 0.01)

    if "pct_chg" in df.columns:
        pct = pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0.0)
        return pct >= _LIMIT_UP_PCT_THRESHOLD

    warnings.warn(
        "[trade_mask] 股票池中未找到可用字段 (is_limit_up / 价格 / pct_chg)，"
        "无法精确判断涨停，can_buy 将不排除涨停标的。",
        stacklevel=3,
    )
    return pd.Series(False, index=df.index)


def get_trade_mask_from_pool(
    pool_df: pd.DataFrame,
    target_dates: Iterable[int],
) -> pd.DataFrame:
    """
    为指定买入日集合构造每只股票的可买性掩码。

    可买性规则（逐层叠加，全部满足才可买）:
        1. allow_flag == 1  : 未停牌、非ST、上市状态正常
        2. volume > 0       : 当日有成交量（非零流动性）
        3. 非涨停板         : 涨停时无法主动成交
                              → 优先读 is_limit_up；次选 pct_chg >= 9.8 推断；
                                两者均缺时打 WARNING 并跳过此条件。

    参数:
        pool_df      (pd.DataFrame):  股票池 DataFrame，必须包含
                                      ['date', 'code', 'allow_flag'] 三列，
                                      可选 'volume'、'is_limit_up'、'pct_chg'。
        target_dates (Iterable[int]): 需要计算掩码的日期集合（YYYYMMDD 整数）。

    返回:
        pd.DataFrame: 仅包含 target_dates 中存在的行，含以下列：
            - date         : YYYYMMDD 整数
            - code         : 6 位股票代码字符串
            - allow_flag   : 原始 allow_flag
            - tradable_base: allow_flag==1 且 volume>0
            - is_limit_up  : 是否涨停（来源见 _infer_limit_up 说明）
            - can_buy      : tradable_base & ~is_limit_up，最终可买标志

    异常:
        ValueError: pool_df 缺少必要列时抛出。
    """
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

    # ── 流动性过滤：volume > 0 ────────────────────────────────────────────────
    if "volume" in df.columns:
        vol_ok = pd.to_numeric(df["volume"], errors="coerce").fillna(0) > 0
    else:
        vol_ok = pd.Series(True, index=df.index)

    # tradable_base：allow_flag=1 且有成交量
    df["tradable_base"] = (df["allow_flag"] == 1) & vol_ok

    # ── 涨跌停板过滤 ──────────────────────────────────────────────────────────
    # 买入限制：涨停无法买入（以免回测由于假设以涨停价成交而系统性高估收益）
    # 卖出限制：跌停无法卖出（以免回测由于假设以跌停价成交而系统性高估收益）
    df["is_limit_up"] = _infer_limit_up(df)
    
    # 推断跌停
    if "is_limit_down" in df.columns:
        df["is_limit_down"] = df["is_limit_down"].fillna(False).astype(bool)
    elif "收盘价(元)" in df.columns and "跌停价(元)" in df.columns:
        close = pd.to_numeric(df["收盘价(元)"], errors="coerce").fillna(0.0)
        l_down = pd.to_numeric(df["跌停价(元)"], errors="coerce").fillna(-1.0)
        df["is_limit_down"] = close <= (l_down + 0.01)
    elif "pct_chg" in df.columns:
        # pct_chg <= -9.8 视为跌停
        pct = pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0.0)
        df["is_limit_down"] = pct <= -9.8
    else:
        df["is_limit_down"] = False

    # can_buy：能买（基础准入且非涨停）
    df["can_buy"] = df["tradable_base"] & (~df["is_limit_up"])
    
    # can_sell：能卖（基础准入且非跌停）
    df["can_sell"] = df["tradable_base"] & (~df["is_limit_down"])

    return df[["date", "code", "allow_flag", "tradable_base", "is_limit_up", "is_limit_down", "can_buy", "can_sell"]].copy()
