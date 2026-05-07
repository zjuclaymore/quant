#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分组标签生成器（Step 4）及批量分析调度器（--batch 模式）
==========================================================

**基本用法（Step 4：按 factor_rank 给真实交割单打分组标签）**

    python add_decile_group_from_rank.py \\
        --input  real_delivery_order.parquet \\
        --output real_delivery_order_with_group.parquet \\
        --n-groups 10

分组算法:
    在每个买入日（date_col）截面内，按 factor_rank 升序排列后，
    用等宽分箱公式分配组号（group_id）：

        _pos  = 该股在截面内的排名位置（从1开始）
        _n    = 截面总股票数
        group = floor((_pos - 1) * n_groups / _n) + 1

    其中 group_id=1 表示因子值最高组，group_id=n_groups 最低组。
    [注] factor_rank 越小 = 因子值越高（在上游 build_factor_delivery_order
         中按因子值降序排名，rank=1 代表最大值）。

**批量分析模式（--batch）**

    在完成 Step 4 分组后，自动对以下 max_stocks_per_group 取值批量
    执行 Step 6（IC/NAV 分析）+ Step 7（HTML 净值图）：

        BATCH_CAPS = [0, 10, 25, 50, 100, 200, 500]
        （0 = 不限制，即全部股票等权；其余为每组持仓上限）

    每种设置的结果独立落地到：
        {batch-out-dir}/cap_{n}/analysis/   ← IC 汇总、净值 CSV
        {batch-out-dir}/cap_{n}/group_nav_curve.html

    批量结束后打印各 cap 的 RankIC_mean / RankIC_IR_ann 对比表，
    方便快速判断哪种持仓上限的因子表现最稳定。

用法示例::

    # 仅做 Step 4
    python add_decile_group_from_rank.py \\
        --input real_delivery_order.parquet \\
        --output real_delivery_order_with_group.parquet

    # Step 4 + 批量 Step 6/7（需先完成 Step 5 生成 enriched_order）
    python add_decile_group_from_rank.py \\
        --input  real_delivery_order.parquet \\
        --output real_delivery_order_with_group.parquet \\
        --batch \\
        --enriched-input real_delivery_order_with_group_adjclose.parquet \\
        --batch-out-dir  ./analysis_batch \\
        --start 2015-01-01 \\
        --end   2024-12-31 \\
        --factor-title "Factor_A021"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test"
    r"\function_breakdown\dealdeal\factor_A021_real_delivery_order.parquet"
)
DEFAULT_OUTPUT = Path(
    r"E:\1_basement\quant_research\back_test\sigle_factor_test"
    r"\function_breakdown\dealdeal\factor_A021_real_delivery_order_with_group.parquet"
)

# 批量分析时固定使用的 max_stocks_per_group 取值列表
# 0 = 不限制（全部等权），其余为每组持仓上限
BATCH_CAPS: list[int] = [0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 325, 350]

# 分析脚本 / 绘图脚本路径（相对于本文件所在目录）
_HERE = Path(__file__).resolve().parent
_ANALYZE_SCRIPT = _HERE / "analyze_group_nav_and_ic.py"
_PLOT_SCRIPT = _HERE.parent / "plot" / "plot_group_nav_curve.py"


# ─────────────────────────────────────────────────────────────────────────────
# Step 4  核心：按 rank 打分组标签
# ─────────────────────────────────────────────────────────────────────────────


def assign_groups_by_rank(
    df: pd.DataFrame,
    date_col: str,
    rank_col: str,
    group_col: str,
    n_groups: int,
) -> pd.DataFrame:
    """
    在每个买入日截面内，按因子排名等分 n_groups 组，写入 group_col 列。

    分组公式（等宽分箱，组号从 1 开始，1 = 最高因子分位）::

        _pos  = 股票在截面内的排名位置（rank 升序排列后的位次，从 1 开始）
        _n    = 截面内有效股票总数
        group = floor((_pos - 1) × n_groups / _n) + 1   ∈ [1, n_groups]

    原理：将 [0, _n) 等分为 n_groups 段，每只股票按 _pos 落入对应段。
    因子 rank 越小（因子值越高），_pos 越靠前，group_id 越接近 1。

    参数:
        df         (pd.DataFrame): 真实交割单，至少包含 date_col 和 rank_col。
        date_col   (str):          买入日列名（截面分组依据）。
        rank_col   (str):          因子排名列名；数值越小表示因子值越高。
        group_col  (str):          输出的分组标签列名。
        n_groups   (int):          分组数，如 10 = 十分位。

    返回:
        pd.DataFrame: 添加了 group_col 列的新 DataFrame（原始行数不变）。

    异常:
        ValueError: date_col 或 rank_col 不在 df.columns 时抛出。
    """
    if date_col not in df.columns:
        raise ValueError(f"缺少日期列: {date_col}")
    if rank_col not in df.columns:
        raise ValueError(f"缺少排序列: {rank_col}")

    out = df.copy()
    out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce")

    valid = out[rank_col].notna()
    out[group_col] = pd.NA

    # 在每个买入日截面内按 rank 升序排列，再按等宽分箱公式分配组号
    sub = out.loc[valid].copy()
    tie_cols = [c for c in ["code", "year_month"] if c in sub.columns]
    sub = sub.sort_values(
        [date_col, rank_col] + tie_cols,
        ascending=[True, True] + [True] * len(tie_cols),
    )

    # _pos：截面内位置（从 1 开始），_n：截面内有效股数
    sub["_pos"] = sub.groupby(date_col).cumcount() + 1
    sub["_n"] = sub.groupby(date_col)[rank_col].transform("size")
    # 等宽分箱：group = floor((_pos - 1) * n / _n) + 1，clip 防止浮点越界
    sub[group_col] = (
        (((sub["_pos"] - 1) * n_groups / sub["_n"]).astype("int64") + 1)
        .clip(1, n_groups)
        .astype("Int64")
    )

    out.loc[sub.index, group_col] = sub[group_col]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 批量分析（--batch 模式）
# ─────────────────────────────────────────────────────────────────────────────


def _safe_run(cmd: list, label: str) -> bool:
    """
    执行子进程命令，返回是否成功（True = 退出码 0）。

    参数:
        cmd   (list): 命令及参数列表（传给 subprocess.run）。
        label (str):  步骤名称，仅用于日志打印。

    返回:
        bool: True 表示成功，False 表示失败（不抛异常，由调用方决定后续行为）。
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"    [>] {cmd_str}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"    [FAIL] {label} 退出码={result.returncode}")
        return False
    print(f"    [OK]  {label}")
    return True


def run_batch_analysis(
    enriched_input: Path,
    batch_out_dir: Path,
    start: str | None,
    end: str | None,
    factor_title: str,
) -> None:
    """
    对 BATCH_CAPS 中每个 max_stocks_per_group 值，依次执行：
        Step 6: analyze_group_nav_and_ic.py  → IC 汇总 + NAV CSV
        Step 7: plot_group_nav_curve.py      → 交互式 HTML 净值图

    结果落地目录结构::

        {batch_out_dir}/
            cap_0/
                analysis/
                    group_nav_curve.csv
                    ic_rankic_timeseries.csv
                    ic_summary.csv
                group_nav_curve.html
            cap_10/
                ...

    批量结束后打印 RankIC_mean / RankIC_IR_ann 对比汇总表。

    参数:
        enriched_input (Path):      Step 5 输出的 enriched 交割单 parquet。
        batch_out_dir  (Path):      批量结果根目录。
        start          (str|None):  回测开始日期（YYYY-MM-DD），可为 None。
        end            (str|None):  回测结束日期（YYYY-MM-DD），可为 None。
        factor_title   (str):       图表标题前缀（如因子名称）。

    异常:
        FileNotFoundError: enriched_input 或分析脚本不存在时抛出。
    """
    if not enriched_input.exists():
        raise FileNotFoundError(f"enriched 交割单不存在: {enriched_input}")
    if not _ANALYZE_SCRIPT.exists():
        raise FileNotFoundError(f"分析脚本不存在: {_ANALYZE_SCRIPT}")
    if not _PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"绘图脚本不存在: {_PLOT_SCRIPT}")

    bar = "─" * 60
    print(f"\n┌{bar}┐")
    print(f"│  批量分析模式  BATCH_CAPS = {BATCH_CAPS}")
    print(f"│  enriched 输入: {enriched_input}")
    print(f"│  结果根目录   : {batch_out_dir}")
    print(f"└{bar}┘")

    summary_rows: list[dict] = []

    for scheme, ret_col in [("Monthly", "monthly_return")]:
        for cap in BATCH_CAPS:
            cap_label = f"cap_{cap}" if cap > 0 else f"cap_0_no_limit"
            cap_dir = batch_out_dir / cap_label
            ana_dir = cap_dir / "analysis"
            nav_csv = ana_dir / "group_nav_curve.csv"
            ic_csv = ana_dir / "ic_summary.csv"
            plot_html = cap_dir / "group_nav_curve.html"

            ana_dir.mkdir(parents=True, exist_ok=True)

            cap_display = f"{scheme} cap={cap} {'(不限制)' if cap == 0 else f'(每组≤{cap}只)'}"
            print(f"\n  ── {cap_display} ──────────────────────────────")

            # Step 6: 分析
            cmd6 = [
                sys.executable,
                str(_ANALYZE_SCRIPT),
                "--input",
                str(enriched_input),
                "--out-dir",
                str(ana_dir),
                "--max-stocks-per-group",
                str(cap),
                "--ret-col",
                ret_col
            ]
            if start:
                cmd6 += ["--start", start]
            if end:
                cmd6 += ["--end", end]

            ok6 = _safe_run(cmd6, f"Step 6 analyze {scheme} cap={cap}")
            if not ok6:
                summary_rows.append({"scheme": scheme, "cap": cap, "status": "FAILED (Step6)"})
                continue

            # Step 7: 绘图
            chart_title = (
                f"{factor_title} [{scheme}] | cap={cap if cap > 0 else 'ALL'}  "
                f"{'  ' + start if start else ''}{'→' + end if end else ''}"
            )
            cmd7 = [
                sys.executable,
                str(_PLOT_SCRIPT),
                "--input",
                str(nav_csv),
                "--output",
                str(plot_html),
                "--title",
                chart_title,
            ]
            ok7 = _safe_run(cmd7, f"Step 7 plot {scheme} cap={cap}")

            # 读取 IC 汇总，用于最终对比表
            row: dict = {"scheme": scheme, "cap": cap, "status": "OK" if ok7 else "plot_FAIL"}
            if ic_csv.exists():
                try:
                    ic_df = pd.read_csv(ic_csv)
                    if not ic_df.empty:
                        r = ic_df.iloc[0]
                        row["RankIC_mean"] = round(
                            float(r.get("rank_ic", float("nan"))), 4
                        )
                        row["RankIC_IR_ann"] = round(
                            float(r.get("annualized_rank_ir", float("nan"))), 4
                        )
                        row["IC_mean"] = round(float(r.get("ic", float("nan"))), 4)
                        row["IC_IR_ann"] = round(
                            float(r.get("annualized_ir", float("nan"))), 4
                        )
                        row["periods"] = int(r.get("periods", -1))
                        
                        rows_cnt = float(r.get("rows", 0))
                        periods_cnt = float(r.get("periods", 1))
                        avg_stocks = rows_cnt / periods_cnt if periods_cnt > 0 else 0
                        row["avg_stocks"] = round(avg_stocks, 0)
                except Exception as e:  # noqa: BLE001
                    row["note"] = f"read_ic_failed: {e}"
            summary_rows.append(row)

    # ── 对比汇总表 ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 95}")
    print(f"  批量分析完成  ─ max_stocks_per_group & Scheme 对比汇总")
    print(f"{'═' * 95}")
    hdr = f"  {'scheme':>8} │ {'cap':>6} │ {'status':^10} │ {'RankIC_mean':>11} │ {'RankIC_IR_ann':>13} │ {'IC_mean':>8} │ {'IC_IR_ann':>9} │ {'avg_stocks':>10}"
    print(hdr)
    print(
        f"  {'─' * 8}─┼─{'─' * 6}─┼─{'─' * 10}─┼─{'─' * 11}─┼─{'─' * 13}─┼─{'─' * 8}─┼─{'─' * 9}─┼─{'─' * 10}"
    )
    for row in summary_rows:
        scheme = row.get("scheme", "?")
        cap = row.get("cap", "?")
        status = row.get("status", "?")
        ric = f"{row['RankIC_mean']:+.4f}" if "RankIC_mean" in row else "  N/A"
        ric_ir = f"{row['RankIC_IR_ann']:+.4f}" if "RankIC_IR_ann" in row else "  N/A"
        ic = f"{row['IC_mean']:+.4f}" if "IC_mean" in row else "  N/A"
        ic_ir = f"{row['IC_IR_ann']:+.4f}" if "IC_IR_ann" in row else "  N/A"
        stocks = (
            f"{int(row['avg_stocks']):>10}"
            if "avg_stocks" in row and pd.notna(row.get("avg_stocks"))
            else "  N/A"
        )
        print(
            f"  {scheme:>8} │ {cap:>6} │ {status:^10} │ {ric:>11} │ {ric_ir:>13} │ {ic:>8} │ {ic_ir:>9} │ {stocks:>10}"
        )
    print(f"{'═' * 82}\n")

    try:
        csv_path = batch_out_dir / f"batch_summary_combinations.csv"
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  [+] 批量组合回测结果已汇总至: {csv_path}\n")
    except Exception as e:
        print(f"  [-] 写入 CSV 汇总失败: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 包含全部配置项的参数对象。
    """
    p = argparse.ArgumentParser(
        description="Step 4：按 factor_rank 打分组标签；可选 --batch 批量跑 Step 6/7",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Step 4 核心参数 ──
    g4 = p.add_argument_group("Step 4 分组参数")
    g4.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="真实交割单 parquet（Step 3 输出）",
    )
    g4.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="含分组标签的输出 parquet",
    )
    g4.add_argument(
        "--date-col",
        type=str,
        default="buy_date",
        help="买入日列名（截面分组依据，默认 buy_date）",
    )
    g4.add_argument(
        "--rank-col",
        type=str,
        default="factor_rank",
        help="因子排名列名（越小越靠前，默认 factor_rank）",
    )
    g4.add_argument(
        "--group-col",
        type=str,
        default="group_id",
        help="输出分组号列名（默认 group_id）",
    )
    g4.add_argument(
        "--n-groups", type=int, default=10, help="分组数（默认 10，即十分位）"
    )

    # ── 批量分析参数（--batch 时有效） ──
    gb = p.add_argument_group("批量分析参数（--batch 模式）")
    gb.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="启用批量分析模式：完成 Step 4 后，对\n"
        f"  BATCH_CAPS = {BATCH_CAPS}\n"
        "自动执行 Step 6（IC/NAV 分析）+ Step 7（绘图）",
    )
    gb.add_argument(
        "--enriched-input",
        type=str,
        default=None,
        help="Step 5 输出的 enriched 交割单 parquet（--batch 时必填）",
    )
    gb.add_argument(
        "--batch-out-dir",
        type=str,
        default=None,
        help="批量分析结果根目录\n（默认：与 --output 同级的 analysis_batch/ 子目录）",
    )
    gb.add_argument(
        "--start",
        type=str,
        default=None,
        help="回测开始日期 YYYY-MM-DD（转发至 Step 6）",
    )
    gb.add_argument(
        "--end", type=str, default=None, help="回测结束日期 YYYY-MM-DD（转发至 Step 6）"
    )
    gb.add_argument(
        "--factor-title",
        type=str,
        default="Factor",
        help="图表标题前缀（因子名称，默认 'Factor'）",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """
    主执行入口。

    流程:
        1. 解析参数。
        2. 执行 Step 4：assign_groups_by_rank，写出含 group_id 的 parquet。
        3. 若 --batch：调用 run_batch_analysis，批量执行 Step 6/7，
           对每个 cap 值输出独立目录，最终打印 RankIC 对比汇总表。

    返回:
        int: 0 表示成功，1 表示异常。
    """
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # ── Step 4 ────────────────────────────────────────────────────────────
    print(f"[Step4] 读取: {input_path}")
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

    print(f"[Step4] input rows : {len(df):,}")
    print(f"[Step4] output rows: {len(df2):,}")
    print(f"[Step4] output file: {output_path}")
    print(f"[Step4] group col  : {args.group_col}")
    print("[Step4] group distribution:")
    print(df2[args.group_col].value_counts(dropna=False).sort_index().to_string())

    # ── 批量分析 ──────────────────────────────────────────────────────────
    if args.batch:
        if not args.enriched_input:
            print(
                "[ERROR] --batch 模式需要指定 --enriched-input"
                "（Step 5 输出的 enriched 交割单 parquet）",
                file=sys.stderr,
            )
            return 1

        enriched_input = Path(args.enriched_input)
        batch_out_dir = (
            Path(args.batch_out_dir)
            if args.batch_out_dir
            else output_path.parent / "analysis_batch"
        )

        run_batch_analysis(
            enriched_input=enriched_input,
            batch_out_dir=batch_out_dir,
            start=args.start,
            end=args.end,
            factor_title=args.factor_title,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
