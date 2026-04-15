#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化回测与绘图工作流
输入：
    1. 因子原始序列（parquet/csv）
    2. 回测时间区间（start, end）
    3. 股票池（可选，csv/parquet/代码列表）
输出：
    统一输出到 output 目录，按步骤和参数自动命名
"""
import argparse
from pathlib import Path
import shutil
import pandas as pd
import subprocess
import sys
import datetime

# 默认路径
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
BACKTEST_DIR = BASE_DIR / "back_test"
PLOT_DIR = BASE_DIR / "plot"

# 工具脚本路径
ENRICH_SCRIPT = BACKTEST_DIR / "enrich_real_order_adj_close.py"
ANALYZE_SCRIPT = BACKTEST_DIR / "analyze_group_nav_and_ic.py"
PLOT_SCRIPT = PLOT_DIR / "plot_group_nav_curve.py"


def parse_args():
    parser = argparse.ArgumentParser(description="自动化回测与绘图工作流")
    parser.add_argument("--factor", type=str, required=True, help="因子原始序列文件（parquet/csv）")
    parser.add_argument("--start", type=str, required=True, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--stock-pool", type=str, default=None, help="股票池文件或代码列表（可选）")
    parser.add_argument("--name", type=str, default=None, help="任务名（可选，自动生成）")
    return parser.parse_args()


def safe_run(cmd, cwd=None):
    print(f"[RUN] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}")
    return result


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 生成任务名
    if args.name:
        task_name = args.name
    else:
        factor_base = Path(args.factor).stem
        date_tag = f"{args.start.replace('-', '')}_{args.end.replace('-', '')}"
        pool_tag = Path(args.stock_pool).stem if args.stock_pool else "all"
        task_name = f"{factor_base}_{pool_tag}_{date_tag}"
    task_dir = OUTPUT_DIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # 步骤1：准备交割单（假设已有脚本/函数，或直接复制因子文件）
    # 这里可扩展为调用因子处理脚本
    factor_file = Path(args.factor)
    factor_dst = task_dir / f"factor_input{factor_file.suffix}"
    shutil.copy(factor_file, factor_dst)

    # 步骤2：生成 enriched 交割单
    enriched_path = task_dir / "real_delivery_order_with_group_adjclose.parquet"
    safe_run([
        sys.executable, str(ENRICH_SCRIPT),
        "--order", str(factor_dst),
        "--output", str(enriched_path)
    ])

    # 步骤3：回测分析
    analysis_dir = task_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    safe_run([
        sys.executable, str(ANALYZE_SCRIPT),
        "--input", str(enriched_path),
        "--start", args.start,
        "--end", args.end,
        "--out-dir", str(analysis_dir)
    ])

    # 步骤4：绘图
    nav_curve = analysis_dir / "group_nav_curve.csv"
    plot_out = task_dir / "group_nav_curve.html"
    safe_run([
        sys.executable, str(PLOT_SCRIPT),
        "--input", str(nav_curve),
        "--output", str(plot_out)
    ])

    print(f"[Done] 所有结果已输出到: {task_dir}")

if __name__ == "__main__":
    main()
