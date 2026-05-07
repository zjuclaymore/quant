#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截面交割单诊断检查器 (Cross-Section Delivery Inspection)
=========================================================

读取 enriched 真实交割单（含分组、复权价格、月收益），
输出两个可直接用 Excel / 数据库查看的 CSV 检查文件：

  1. cross_section_stock_detail.csv  — 股票明细层
     每行 = 一期 × 一只股票，完整呈现该股票在该期的：
       持仓信息（因子值、排名、分组）、
       价格信息（买入价、月末价）、
       收益信息（月收益率）

  2. cross_section_group_summary.csv — 组汇总层
     每行 = 一期 × 一个分组，呈现该期该组的：
       持仓数量（count）、
       因子统计（mean/min/max）、
       收益统计（mean/std/median/p25/p75/min/max）、
       胜率（个股正收益比例）、
       等权月收益（等于 mean）

列说明（stock_detail）:
    year_month          信号月份（因子生成月份，格式 YYYY-MM）
    buy_date            实际买入日（该月第一个交易日）
    sell_date / buy_month_end_date  卖出日（月末最后交易日）
    code                股票代码
    factor_value        经预处理（3MAD + Z-Score）的因子值
    factor_rank         截面内因子排名（1=最大值）
    factor_rank_pct     截面内百分位（0~1，1 = 最高）
    group_id            分组编号（1=最高因子分位）
    buy_date_adj_close  买入日复权收盘价
    buy_month_end_adj_close  持仓月末复权收盘价
    monthly_return      月持仓收益率 = (月末价/买入价) - 1（无未来函数）

列说明（group_summary）:
    year_month          信号月份
    buy_date            实际买入日
    group_id            分组（1~N）
    n_stocks            该期该组实际持仓数（已过 trade_mask 过滤）
    n_return_valid      有效收益数（monthly_return 非 NaN）
    factor_mean/min/max 因子值分布
    ret_mean            等权月收益均值（= 该组 NAV 月涨幅）
    ret_std             收益标准差（组内离散度）
    ret_median          收益中位数
    ret_p25 / ret_p75   四分位数
    ret_min / ret_max   极端收益
    win_rate            正收益股票占比（= 个股胜率）

注意:
    ⚠ monthly_return 的计算口径为当月 buy_date → buy_month_end_date，
      不含跨月持仓，符合月频调仓假设，无未来函数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 默认路径（直接运行时使用）
# ─────────────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent / "output"

DEFAULT_INPUT = (
    _BASE / "test_turnover_rate"
    / "real_delivery_order_with_group_adjclose.parquet"
)
DEFAULT_OUT_DIR = _BASE / "test_turnover_rate" / "inspection"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="截面交割单诊断检查器 — 输出股票明细与组汇总",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--input",   type=str, default=str(DEFAULT_INPUT),
                   help="enriched 交割单 parquet（含 group_id / monthly_return）")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="输出目录（默认在 input 同目录下创建 inspection/）")
    p.add_argument("--start",   type=str, default=None,
                   help="筛选起始月 YYYY-MM（过滤 year_month）")
    p.add_argument("--end",     type=str, default=None,
                   help="筛选截止月 YYYY-MM（过滤 year_month）")
    p.add_argument("--group",   type=int, default=0,
                   help="只输出指定 group_id（0 = 全部）")
    p.add_argument("--period",  type=str, default=None,
                   help="只输出指定 year_month（如 2015-06），用于单期深度查看")
    p.add_argument("--encoding", type=str, default="utf-8-sig",
                   help="输出 CSV 编码（默认 utf-8-sig，Excel 中文友好）")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _pct(val: float, fmt: str = ".2%") -> str:
    """将浮点数格式化为百分比字符串，NaN 返回 'N/A'。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}"


def build_stock_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建股票明细层：每行 = 一期 × 一只股票。

    输出列顺序经过整理，便于从左到右"读故事"：
      信号期 → 交割日期 → 代码 → 因子相关 → 价格 → 收益

    参数:
        df (pd.DataFrame): enriched 交割单，含全部原始列。

    返回:
        pd.DataFrame: 按 (buy_date, group_id, factor_rank) 升序排列的明细表。
    """
    COLS_ORDER = [
        "year_month",          # 信号月
        "buy_date",            # 买入日
        "sell_date",           # 卖出日（月末）
        "buy_month_end_date",  # 持仓月末日期（与 sell_date 一致）
        "code",                # 股票代码
        "group_id",            # 分组
        "factor_value",        # 因子值（预处理后）
        "factor_rank",         # 截面排名（1=最高）
        "factor_rank_pct",     # 截面百分位
        "buy_date_adj_close",         # 买入价（复权）
        "buy_month_end_adj_close",    # 月末价（复权）
        "monthly_return",             # 月收益率
    ]
    available = [c for c in COLS_ORDER if c in df.columns]
    detail = df[available].copy()

    # 排序：期 → 组 → 排名
    sort_keys = [c for c in ["buy_date", "group_id", "factor_rank"] if c in detail.columns]
    detail = detail.sort_values(sort_keys).reset_index(drop=True)
    return detail


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建组汇总层：每行 = 一期 × 一个分组。

    每组统计指标（全部在已实现收益上计算，无未来函数）:
        n_stocks       : 持仓数量（经 trade_mask 过滤后）
        n_return_valid : 有效月收益数（monthly_return 非 NaN）
        factor_mean/min/max : 截面内因子值分布
        ret_mean       : 等权月收益均值                    → 对应 NAV 涨幅
        ret_std        : 收益标准差                        → 组内离散度
        ret_median     : 中位数收益
        ret_p25/ret_p75: 四分位范围
        ret_min/ret_max: 极端收益（含正常大涨/大跌）
        win_rate       : 个股正收益比例 = P(monthly_return > 0)

    参数:
        df (pd.DataFrame): enriched 交割单。

    返回:
        pd.DataFrame: 按 (buy_date, group_id) 升序排列的汇总表。
    """
    ret_col = "monthly_return"
    fac_col = "factor_value"

    rows = []
    group_keys = ["year_month", "buy_date", "group_id"]
    group_keys = [c for c in group_keys if c in df.columns]

    for keys, g in df.groupby(group_keys, sort=True):
        key_dict = dict(zip(group_keys, keys if isinstance(keys, tuple) else [keys]))

        ret_s   = pd.to_numeric(g[ret_col],  errors="coerce") if ret_col in g.columns else pd.Series(dtype=float)
        fac_s   = pd.to_numeric(g[fac_col],  errors="coerce") if fac_col in g.columns else pd.Series(dtype=float)
        valid_r = ret_s.dropna()

        row = {
            **key_dict,
            # ── 持仓数量 ──────────────────────────────────────────────────
            "n_stocks":        len(g),
            "n_return_valid":  len(valid_r),
            # ── 因子分布 ──────────────────────────────────────────────────
            "factor_mean":     round(float(fac_s.mean()), 6)   if not fac_s.empty else np.nan,
            "factor_min":      round(float(fac_s.min()),  6)   if not fac_s.empty else np.nan,
            "factor_max":      round(float(fac_s.max()),  6)   if not fac_s.empty else np.nan,
            # ── 收益统计 ──────────────────────────────────────────────────
            # ret_mean = 该组等权月收益，与 analyze_group_nav_and_ic 口径一致
            "ret_mean":        round(float(valid_r.mean()),   6) if len(valid_r) >= 1 else np.nan,
            "ret_std":         round(float(valid_r.std(ddof=1)), 6) if len(valid_r) >= 2 else np.nan,
            "ret_median":      round(float(valid_r.median()), 6) if len(valid_r) >= 1 else np.nan,
            "ret_p25":         round(float(valid_r.quantile(0.25)), 6) if len(valid_r) >= 2 else np.nan,
            "ret_p75":         round(float(valid_r.quantile(0.75)), 6) if len(valid_r) >= 2 else np.nan,
            "ret_min":         round(float(valid_r.min()), 6) if len(valid_r) >= 1 else np.nan,
            "ret_max":         round(float(valid_r.max()), 6) if len(valid_r) >= 1 else np.nan,
            # ── 胜率 ──────────────────────────────────────────────────────
            # 个股正收益比例：反映组内大多数股票是否跑赢 0
            "win_rate":        round(float((valid_r > 0).mean()), 4) if len(valid_r) >= 1 else np.nan,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        sort_keys = [c for c in ["buy_date", "group_id"] if c in summary.columns]
        summary = summary.sort_values(sort_keys).reset_index(drop=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    主执行函数。

    步骤:
        1. 读取 enriched 交割单
        2. 按 --start / --end / --group / --period 参数过滤
        3. 构建股票明细层（cross_section_stock_detail.csv）
        4. 构建组汇总层（cross_section_group_summary.csv）
        5. 输出 CSV 并打印统计摘要

    返回:
        int: 0 表示成功。
    """
    args = parse_args()
    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)
    enc        = args.encoding

    # ── 读取数据 ─────────────────────────────────────────────────────────────
    print(f"[Info] 读取: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[Info] 原始行数: {len(df):,}  列: {list(df.columns)}")

    # ── 过滤：时间范围 ────────────────────────────────────────────────────────
    if "year_month" in df.columns:
        if args.start:
            df = df[df["year_month"] >= args.start[:7]]
        if args.end:
            df = df[df["year_month"] <= args.end[:7]]
    elif "buy_date" in df.columns:
        df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
        if args.start:
            df = df[df["buy_date"] >= pd.to_datetime(args.start)]
        if args.end:
            df = df[df["buy_date"] <= pd.to_datetime(args.end)]

    # ── 过滤：单期 ────────────────────────────────────────────────────────────
    if args.period and "year_month" in df.columns:
        df = df[df["year_month"] == args.period[:7]]
        print(f"[Info] 单期过滤: year_month = {args.period[:7]}")

    # ── 过滤：单组 ────────────────────────────────────────────────────────────
    if args.group > 0 and "group_id" in df.columns:
        df = df[df["group_id"] == args.group]
        print(f"[Info] 单组过滤: group_id = {args.group}")

    if df.empty:
        print("[WARN] 过滤后无数据，请检查参数。")
        return 1

    print(f"[Info] 过滤后行数: {len(df):,}  "
          f"期数: {df['year_month'].nunique() if 'year_month' in df.columns else 'N/A'}  "
          f"股票数: {df['code'].nunique() if 'code' in df.columns else 'N/A'}")

    # ── Step A  股票明细层 ───────────────────────────────────────────────────
    print("\n[Step A] 构建股票明细层...")
    detail = build_stock_detail(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "cross_section_stock_detail.csv"
    detail.to_csv(detail_path, index=False, encoding=enc)
    print(f"[Done]  stock_detail  → {detail_path}  ({len(detail):,} 行)")

    # ── Step B  组汇总层 ─────────────────────────────────────────────────────
    print("\n[Step B] 构建组汇总层...")
    summary = build_group_summary(df)
    summary_path = out_dir / "cross_section_group_summary.csv"
    summary.to_csv(summary_path, index=False, encoding=enc)
    print(f"[Done]  group_summary → {summary_path}  ({len(summary):,} 行)")

    # ── 控制台预览：展示每期组 1 vs 组 10 的均值收益对比 ──────────────────────
    if "group_id" in summary.columns and "ret_mean" in summary.columns:
        g1  = summary[summary["group_id"] == 1]["ret_mean"]
        g10 = summary[summary["group_id"] == summary["group_id"].max()]["ret_mean"]
        n_periods = summary["year_month"].nunique() if "year_month" in summary.columns else "N/A"
        spread = float(g1.mean() - g10.mean()) if not g1.empty and not g10.empty else float("nan")
        print(f"\n{'─'*50}")
        print(f"  截面汇总（{n_periods} 期）")
        print(f"{'─'*50}")
        print(f"  组 1  平均月收益 : {_pct(float(g1.mean()))}")
        print(f"  组 {summary['group_id'].max():2d} 平均月收益 : {_pct(float(g10.mean()))}")
        print(f"  多空利差 (G1-G10) : {_pct(spread)}")
        print(f"{'─'*50}")
        print(f"\n  组汇总预览（前 5 行）：")
        print(summary.head(5).to_string(index=False))
        print()

    print(f"\n[Done] 输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
