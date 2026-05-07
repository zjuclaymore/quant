#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子回测自动化 Pipeline (Factor Backtest Pipeline)
==============================================

功能:
    一键完成从因子文件到多维度可视化报告的全流程。
    主要步骤：
    1. 建立结构化目录 (output/{factor_name}/data, /reports, /analysis_batch)
    2. 生成拟交割单 (Step 3: build_factor_delivery_order.py)
    3. 回填收益率与价格 (Step 5: enrich_real_order_adj_close.py)
    4. 执行分层测试与 IC 分析 (Step 4/6/7: add_decile_group_from_rank.py --batch)
    5. 生成最终回归分析报告 (plot/plot_regression_report.py)

用法:
    python factor_pipeline.py --factor-name A021 --factor-file ./output/factor_A021_...parquet --pool ./stock_pool/stock_pool_base.parquet
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_cmd(cmd, label):
    logger.info(f"==> 执行 [{label}]: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        logger.error(f"[FAIL] {label} 失败，退出码: {res.returncode}")
        sys.exit(res.returncode)
    logger.info(f"[OK] {label} 完成")

def main():
    parser = argparse.ArgumentParser(description="因子回测一键流水线")
    parser.add_argument("--factor-name", type=str, required=True, help="因子名称 (用于文件夹命名)")
    parser.add_argument("--factor-file", type=str, required=True, help="因子 parquet 文件路径")
    parser.add_argument("--pool", type=str, default="stock_pool/stock_pool_base.parquet", help="股票池路径")
    parser.add_argument("--start", type=str, default="2008-04-01", help="回测起始日期")
    parser.add_argument("--out-root", type=str, default="output", help="输出根目录")
    
    args = parser.parse_args()
    
    # 1. 建立目录结构
    root_dir = Path(args.out_root) / args.factor_name
    data_dir = root_dir / "data"
    reports_dir = root_dir / "reports"
    # Note: add_decile_group_from_rank.py --batch 默认会在 --output 同级创建 analysis_batch
    # 为了保持逻辑，我们把 grouped 文件直接放 data，分析放 root_dir
    
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始执行因子回测流水线: {args.factor_name}")
    
    # 定义中间文件路径
    order_file = data_dir / f"{args.factor_name}_order.parquet"
    enriched_file = data_dir / f"{args.factor_name}_enriched.parquet"
    grouped_file = data_dir / f"{args.factor_name}_grouped.parquet"
    
    # 2. 生成交割单 (Step 3)
    cmd3 = [
        sys.executable, "dealdeal/build_factor_delivery_order.py",
        "--factor", args.factor_file,
        "--pool", args.pool,
        "--start", args.start,
        "--output", str(order_file)
    ]
    run_cmd(cmd3, "Step 3: 生成交割单")
    
    # 3. 回填收益率 (Step 5)
    cmd5 = [
        sys.executable, "back_test/enrich_real_order_adj_close.py",
        "--order", str(order_file),
        "--output", str(enriched_file),
        "--pool", args.pool
    ]
    run_cmd(cmd5, "Step 5: 回填收益率")
    
    # 4. 执行分组测试分析 (Step 4/6/7 - Batch Mode)
    # 我们把 input 指向回填后的文件，以便同时添加 group_id 并分析
    cmd467 = [
        sys.executable, "back_test/add_decile_group_from_rank.py",
        "--input", str(enriched_file),
        "--output", str(grouped_file),
        "--batch",
        "--enriched-input", str(grouped_file),
        "--start", args.start,
        "--batch-out-dir", str(root_dir / "analysis_batch"),
        "--factor-title", args.factor_name
    ]
    run_cmd(cmd467, "Step 4/6/7: 分组测试与 IC 批量分析")
    
    # 5. 生成最终回归分析报告 (Dashboard)
    # 默认使用 cap_0 (不设限) 的结果来生成汇总报告
    analysis_dir_cap0 = root_dir / "analysis_batch" / "cap_0_no_limit" / "analysis"
    final_report = reports_dir / f"regression_report_{args.factor_name}_cap0.html"
    batch_summary_csv = root_dir / "analysis_batch" / "batch_summary_combinations.csv"

    cmd_plot = [
        sys.executable, "plot/plot_regression_report.py",
        "--analysis-dir", str(analysis_dir_cap0),
        "--output", str(final_report),
        "--title", f"{args.factor_name} Factor Regression Report"
    ]
    if batch_summary_csv.exists():
        cmd_plot.extend(["--batch-summary", str(batch_summary_csv)])
        
    run_cmd(cmd_plot, "Plot: 生成最终回归看板")
    
    logger.info("=" * 50)
    logger.info(f"Pipeline 执行完毕！")
    logger.info(f"因子目录: {root_dir}")
    logger.info(f"最终报告: {final_report}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
