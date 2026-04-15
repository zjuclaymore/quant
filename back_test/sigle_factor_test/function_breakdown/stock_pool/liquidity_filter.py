#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流动性过滤器 (Liquidity Filter)
================================

基于交易日截面的 20 日滚动均量百分位（volume_ma20_pct）对股票池执行
流动性过滤，剔除流动性过差的标的，输出精简后的新股票池。

过滤逻辑:
    liquidity_flag = 1  当且仅当  volume_ma20_pct >= threshold（默认 0.10）
                    0  否则（流动性不足，按交易日截面后 10% 分位以下）

    仅保留 liquidity_flag == 1 的行，构成新股票池。

说明:
    volume_ma20_pct 含义（来自 liquidity.py）:
        每只股票在每个自然交易日的 20 日滚动成交量均值（volume_ma20），
        在同日截面内按 average 方法进行百分位排名（0~1 范围）。
        此字段必须在输入股票池中已经预先计算好；
        若字段不存在，脚本会报错提示先运行 liquidity.py。

    "按截面后 10% 剔除"的含义:
        每个交易日独立排名，若某股票当日的 volume_ma20_pct < 0.10，
        则认为其当日流动性处于全市场末尾 10% 分位，流动性过差，予以剔除。
        该过滤为"截面独立判断"，不同日期对同一股票的判断可以不同。

    ⚠  拒绝未来函数：volume_ma20 的滚动窗口仅使用历史 [t-19, t] 共 20 日数据，
       不向前看；百分位排名同理，仅在当日截面内计算，无信息泄露。

CLI 用法:
    # 基础：从默认输入路径读取，输出到同目录新文件
    python liquidity_filter.py

    # 指定输入/输出
    python liquidity_filter.py \\
        --input  "E:/path/to/stock_pool.parquet" \\
        --output "E:/path/to/stock_pool_liq_filtered.parquet"

    # 调整阈值（如改为 15% 分位）
    python liquidity_filter.py --threshold 0.15

    # 只打印统计，不写文件（dry-run）
    python liquidity_filter.py --dry-run

输出文件:
    新股票池 parquet，包含原始全部列（去掉 liquidity_flag 列，因直接 drop 低流动性行）

输出统计示例:
    输入行数        : 16,615,496
    输出行数        : 12,801,234
    过滤行数        : 3,814,262 (22.96%)
    volume_ma20_pct NaN 行（已保留，不参与过滤）: 1,023,411
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 默认路径
# ─────────────────────────────────────────────────────────────────────────────

_POOL_DIR = Path(__file__).parent

DEFAULT_INPUT = (
    _POOL_DIR / "original_stock_pool_with_st_and_first_dates_liq_mv__listed_only.parquet"
)
DEFAULT_OUTPUT = (
    _POOL_DIR / "original_stock_pool_liq_filtered.parquet"
)

REQUIRED_COL = "volume_ma20_pct"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 包含 input / output / threshold / dry_run 的参数对象。
    """
    p = argparse.ArgumentParser(
        description="按截面 volume_ma20_pct 阈值过滤股票池，剔除低流动性标的",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"输入股票池 parquet（必须已含 {REQUIRED_COL} 列）\n默认: {DEFAULT_INPUT}",
    )
    p.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"输出新股票池 parquet\n默认: {DEFAULT_OUTPUT}",
    )
    p.add_argument(
        "--threshold", type=float, default=0.10,
        help="流动性阈值（截面分位，0~1）：\n"
             "  volume_ma20_pct >= threshold → liquidity_flag=1（保留）\n"
             "  volume_ma20_pct <  threshold → 剔除\n"
             "  默认 0.10（剔除截面后 10%% 分位以下的标的）",
    )
    p.add_argument(
        "--keep-nan", action="store_true", default=False,
        help="是否保留 volume_ma20_pct 为 NaN 的行（默认保留，不参与过滤）\n"
             "  NaN 通常来自新股上市初期（不足 20 个交易日数据），默认不剔除。\n"
             "  加此参数无效果（等同于默认行为）；若要剔除 NaN，请使用 --drop-nan。",
    )
    p.add_argument(
        "--drop-nan", action="store_true", default=False,
        help="剔除 volume_ma20_pct 为 NaN 的行（新股暖机期数据）\n"
             "  默认 False：NaN 行当作「不过滤」保留。",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="只打印统计，不写文件",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────────────────────────────────────

def apply_liquidity_filter(
    df: pd.DataFrame,
    threshold: float = 0.10,
    drop_nan: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对股票池截面数据执行流动性过滤。

    过滤规则（按交易日截面独立判断，不引入未来函数）:
        - 若 volume_ma20_pct >= threshold → liquidity_flag = 1 → 保留
        - 若 volume_ma20_pct <  threshold → liquidity_flag = 0 → 剔除
        - 若 volume_ma20_pct 为 NaN（新股暖机期）:
              drop_nan=False → 保留（liquidity_flag = 1，不惩罚新股）
              drop_nan=True  → 剔除

    参数:
        df        (pd.DataFrame): 完整股票池，必须包含 volume_ma20_pct 列。
        threshold (float):        截面流动性阈值，默认 0.10。
        drop_nan  (bool):         是否剔除 volume_ma20_pct 为 NaN 的行。

    返回:
        tuple[pd.DataFrame, pd.DataFrame]:
            (过滤后的新股票池, 被剔除的行)
    """
    if REQUIRED_COL not in df.columns:
        raise ValueError(
            f"输入数据缺少必要列 '{REQUIRED_COL}'。\n"
            f"请先运行 liquidity.py 计算 volume_ma20 和 volume_ma20_pct，"
            f"再执行过滤。"
        )

    pct = pd.to_numeric(df[REQUIRED_COL], errors="coerce")

    # 构造 liquidity_flag：
    #   volume_ma20_pct >= threshold → 1（保留）
    #   volume_ma20_pct <  threshold → 0（剔除）
    #   NaN：按 drop_nan 参数决定
    if drop_nan:
        # NaN 视为流动性不足 → 剔除
        keep_mask = pct >= threshold
    else:
        # NaN 视为「不确定但保留」→ is_nan | (pct >= threshold)
        keep_mask = pct.isna() | (pct >= threshold)

    df_keep = df[keep_mask].copy()
    df_drop = df[~keep_mask].copy()

    return df_keep, df_drop


def _print_stats(
    df_in:    pd.DataFrame,
    df_keep:  pd.DataFrame,
    df_drop:  pd.DataFrame,
    threshold: float,
    drop_nan:  bool,
) -> None:
    """
    打印过滤统计摘要。

    参数:
        df_in     (pd.DataFrame): 过滤前原始数据。
        df_keep   (pd.DataFrame): 过滤后保留数据。
        df_drop   (pd.DataFrame): 被剔除数据。
        threshold (float):        过滤阈值。
        drop_nan  (bool):         是否剔除 NaN 标记。
    """
    n_in   = len(df_in)
    n_keep = len(df_keep)
    n_drop = len(df_drop)
    n_nan  = int(pd.to_numeric(df_in[REQUIRED_COL], errors="coerce").isna().sum())

    pct_col = pd.to_numeric(df_in[REQUIRED_COL], errors="coerce")
    n_below = int((pct_col < threshold).sum())   # 含 NaN=False

    print(f"\n{'─'*56}")
    print(f"  流动性过滤统计摘要")
    print(f"{'─'*56}")
    print(f"  过滤阈值                : volume_ma20_pct >= {threshold:.2%}")
    print(f"  NaN 处理方式            : {'剔除' if drop_nan else '保留（不参与过滤）'}")
    print(f"{'─'*56}")
    print(f"  输入总行数              : {n_in:>12,}")
    print(f"  volume_ma20_pct NaN 行  : {n_nan:>12,}  ({n_nan/n_in:.2%})")
    print(f"  低于阈值行（不含NaN）   : {n_below:>12,}  ({n_below/n_in:.2%})")
    print(f"  剔除行数                : {n_drop:>12,}  ({n_drop/n_in:.2%})")
    print(f"  保留行数 (输出)         : {n_keep:>12,}  ({n_keep/n_in:.2%})")
    print(f"{'─'*56}")

    # 按年份或日期范围展示覆盖情况（若有 date 列）
    if "date" in df_in.columns:
        d_min = int(df_in["date"].min())
        d_max = int(df_in["date"].max())
        print(f"  日期范围                : {d_min} ~ {d_max}")

    if "code" in df_in.columns:
        n_codes_in   = df_in["code"].nunique()
        n_codes_keep = df_keep["code"].nunique()
        print(f"  过滤前股票数            : {n_codes_in:>12,}")
        print(f"  过滤后出现过的股票数    : {n_codes_keep:>12,}")
    print(f"{'─'*56}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    主执行函数。

    流程:
        1. 读取含 volume_ma20_pct 的股票池 parquet
        2. 按截面阈值过滤（liquidity_flag = 1 保留）
        3. 打印统计摘要
        4. 写出新股票池 parquet（除非 --dry-run）

    返回:
        int: 0 表示成功。
    """
    args = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)
    threshold   = args.threshold
    drop_nan    = args.drop_nan
    dry_run     = args.dry_run

    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        return 1

    if not (0.0 < threshold < 1.0):
        print(f"[ERROR] --threshold 必须在 (0, 1) 范围内，当前值: {threshold}", file=sys.stderr)
        return 1

    print(f"[Info] 读取股票池: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[Info] 原始行数: {len(df):,}，列数: {len(df.columns)}")

    # ── 过滤 ────────────────────────────────────────────────────────────────
    df_keep, df_drop = apply_liquidity_filter(
        df=df,
        threshold=threshold,
        drop_nan=drop_nan,
    )

    # ── 统计摘要 ─────────────────────────────────────────────────────────────
    _print_stats(df, df_keep, df_drop, threshold, drop_nan)

    if dry_run:
        print("[dry-run] 未写文件。")
        return 0

    # ── 写出 ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Info] 写出过滤后股票池: {output_path}")
    df_keep.to_parquet(output_path, index=False, compression="snappy")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[Done] 完成。输出文件大小: {size_mb:.1f} MB")
    print(f"[Done] 输出路径: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
