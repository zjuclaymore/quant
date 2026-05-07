#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
因子回归报告生成脚本 (Factor Regression & Performance Report)

功能:
  整合分层回测与 IC 分析结果，生成 Premium 交互式 HTML 报告。
  1. 核心指标看板 (Sharpe, MDD, Annualized Ret, IC IR)
  2. 分组净值曲线 (Group NAV) 与 多空对冲 (Spread)
  3. 因子单调性校验 (Annualized Return by Group)
  4. 因子稳定性分析 (RankIC TS & Rolling Mean)

数据源 (默认位于 analysis_batch/{cap}/analysis/):
  - group_nav_curve.csv
  - ic_summary.csv
  - ic_rankic_timeseries.csv
  - ic_decay.csv
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(nav_series, periods_per_year=12):
    """计算年化收益、夏普、回撤等指标"""
    ret = nav_series.pct_change().dropna()
    if len(ret) == 0:
        return 0, 0, 0
    ann_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (periods_per_year / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    # MDD
    cum_max = nav_series.cummax()
    dd = (nav_series - cum_max) / cum_max
    mdd = dd.min()
    return ann_ret, sharpe, mdd

def main():
    parser = argparse.ArgumentParser(description="生成因子回归分析报告")
    parser.add_argument('--analysis-dir', type=str, required=True, help='分析结果 CSV 所在目录')
    parser.add_argument('--batch-summary', type=str, default=None, help='批量汇总 CSV 路径 (可选)')
    parser.add_argument('--output', type=str, default='regression_report.html', help='输出 HTML 路径')
    parser.add_argument('--title', type=str, default='Factor Regression Report', help='报告标题')
    args = parser.parse_args()

    ana_dir = Path(args.analysis_dir)
    nav_file = ana_dir / "group_nav_curve.csv"
    ic_sum_file = ana_dir / "ic_summary.csv"
    ic_ts_file = ana_dir / "ic_rankic_timeseries.csv"
    decay_file = ana_dir / "ic_decay.csv"

    if not all([nav_file.exists(), ic_sum_file.exists(), ic_ts_file.exists()]):
        logger.error(f"分析目录缺失必要 CSV 文件: {ana_dir}")
        return

    # 1. 载入数据
    nav_df = pd.read_csv(nav_file, index_col=0, parse_dates=True)
    ic_sum_df = pd.read_csv(ic_sum_file)
    ic_ts_df = pd.read_csv(ic_ts_file, index_col=0, parse_dates=True)
    decay_df = pd.read_csv(decay_file) if decay_file.exists() else None
    
    batch_df = None
    if args.batch_summary:
        b_path = Path(args.batch_summary)
        if b_path.exists():
            batch_df = pd.read_csv(b_path).sort_values('cap')
        else:
            logger.warning(f"未能找到批量汇总文件: {args.batch_summary}")

    group_cols = [c for c in nav_df.columns if c.startswith('group_')]
    if not group_cols:
        logger.error("nav_file 中未发现 group_ 列")
        return
    
    # 2. 计算分组指标
    group_stats = []
    for col in group_cols:
        ann_ret, sharpe, mdd = calculate_metrics(nav_df[col])
        group_stats.append({
            'Group': col,
            'AnnRet': ann_ret,
            'Sharpe': sharpe,
            'MaxDD': mdd
        })
    stats_df = pd.DataFrame(group_stats)

    # (Spread calculation removed per user request)

    # 4. 构建 Plotly Dashboard
    n_rows = 4 if batch_df is not None else 3
    row_heights = [0.35, 0.25, 0.2, 0.2] if batch_df is not None else [0.4, 0.3, 0.3]
    h = 1600 if batch_df is not None else 1200
    
    titles = [
        '分组累计净值 (Group NAV)',
        '分组年化收益 (Monotonicity)', 'RankIC 稳定性 (Rolling 12M Mean)',
        'IC 衰减分析 (IC Decay)', 'IC 统计分布 (RankIC Histogram)'
    ]
    if batch_df is not None:
        titles.append('因子容量分析 (Capacity Analysis: Cap vs Performance)')

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {}],
               [{}, {}],
               [{"colspan": 2, "secondary_y": True}, None]] if batch_df is not None else 
              [[{"colspan": 2}, None],
               [{}, {}],
               [{}, {}]],
        subplot_titles=tuple(titles),
        vertical_spacing=0.08,
        row_heights=row_heights
    )

    # Plot 1: NAV Curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, col in enumerate(group_cols):
        fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[col], name=col, line=dict(color=colors[i%len(colors)], width=1.5)), row=1, col=1)
    # (Spread trace removed)

    # Plot 2: Monotonicity (AnnRet Bar)
    fig.add_trace(go.Bar(x=stats_df['Group'], y=stats_df['AnnRet'], marker_color='#1f77b4', name='Ann. Return'), row=2, col=1)

    # Plot 3: RankIC Rolling Mean
    rolling_ic = ic_ts_df['rank_ic'].rolling(12).mean()
    fig.add_trace(go.Scatter(x=ic_ts_df.index, y=ic_ts_df['rank_ic'], name='RankIC', line=dict(color='lightgrey', width=1), opacity=0.5), row=2, col=2)
    fig.add_trace(go.Scatter(x=ic_ts_df.index, y=rolling_ic, name='RankIC Rolling 12M', line=dict(color='red', width=2)), row=2, col=2)

    # Plot 4: IC Decay
    if decay_df is not None:
        fig.add_trace(go.Bar(x=decay_df['lag'], y=decay_df['rank_ic'], marker_color='#ff7f0e', name='Lag RankIC'), row=3, col=1)
        fig.update_xaxes(title_text="Lag (Months)", row=3, col=1)
    else:
        logger.warning("未发现 ic_decay.csv，Plot 4 将留空")

    # Plot 5: IC Histogram
    fig.add_trace(go.Histogram(x=ic_ts_df['rank_ic'], name='RankIC Dist', nbinsx=30, marker_color='#2ca02c'), row=3, col=2)

    # Plot 6: Capacity Analysis (Optional)
    if batch_df is not None:
        # 使用辅助轴显示 RankIC 和 IR
        # RankIC Mean (Left Y)
        fig.add_trace(go.Scatter(x=batch_df['cap'], y=batch_df['RankIC_mean'], name='RankIC Mean', 
                                 line=dict(color='blue', width=2), marker=dict(size=8)), row=4, col=1, secondary_y=False)
        
        # RankIC IR Ann (Right Y)
        fig.add_trace(go.Scatter(x=batch_df['cap'], y=batch_df['RankIC_IR_ann'], name='RankIC IR Ann', 
                                 line=dict(color='green', width=2, dash='dot'), marker=dict(size=8)), row=4, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="Cap (Max stocks per group)", row=4, col=1)
        fig.update_yaxes(title_text="RankIC Mean", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="RankIC IR Ann", row=4, col=1, secondary_y=True)

    # Summary Table as Annotation
    ic_row = ic_sum_df.iloc[0]
    summary_text = (
        f"<b>Factor Summary: {args.title}</b><br>"
        f"RankIC Mean: {ic_row['rank_ic']:.4f} | RankIC IR: {ic_row['annualized_rank_ir']:.4f} | "
        f"Start: {ic_row['start']} | End: {ic_row['end']}"
    )
    fig.add_annotation(
        text=summary_text, xref="paper", yref="paper", x=0, y=1.12,
        showarrow=False, font=dict(size=14), align="left"
    )

    fig.update_layout(height=h, template='plotly_white', hovermode='x unified', title=dict(text=args.title, font=dict(size=24)))
    
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"回测报告已生成: {output_path}")

if __name__ == '__main__':
    main()
