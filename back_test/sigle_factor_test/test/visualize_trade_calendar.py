#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
回测调仓日历可视化与质量检测脚本 (Rebalance Calendar Visualization & Quality Check)

本脚本用于对由 load_calendar.py 生成的调仓日历进行深度分析，涵盖：
1. 时序规则验证：展示每月 Signal, Sell, Buy 日期的相对位置。
2. 执行延迟 (Execution Lag) 分析：统计信号发出到执行买卖之间的自然日间隔。
3. 月内分布概率：分析调仓动作通常发生在月份的哪些天。
4. 周期稳定性：检测是否存在月份遗漏。

计算公式说明：
- 卖出延迟 (Sell Lag): $Lag_{sell} = SellDate - SignalDate$ (单位: 自然日)
- 买入延迟 (Buy Lag): $Lag_{buy} = BuyDate - SignalDate$ (单位: 自然日)
- 调仓窗口宽度 (Window): $Window = BuyDate - SellDate$
  用于评估调仓过程中的资金空置时间或重叠时间。
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    主执行逻辑：解析命令行、加载调仓日历、生成可视化看版。
    """
    parser = argparse.ArgumentParser(description="回测调仓日历可视化分析工具")
    parser.add_argument('--input', type=str, required=True, help='输入的调仓日历文件路径 (parquet/csv)')
    parser.add_argument('--output-dir', type=str, default='analysis', help='输出分析结果的根目录')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    # 定义输出目录
    input_stem = input_path.stem
    output_dir = Path(args.output_dir) / input_stem
    output_file = output_dir / 'trade_calendar_analysis.html'
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger.info(f"正在载入日历数据: {input_path}...")
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # 1. 转换日期格式
    for col in ['signal_date', 'sell_date', 'buy_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # 2. 计算延迟与特征
    logger.info("正在执行数据诊断与特征计算...")
    df['sell_lag'] = (df['sell_date'] - df['signal_date']).dt.days
    df['buy_lag'] = (df['buy_date'] - df['signal_date']).dt.days
    df['exec_window'] = (df['buy_date'] - df['sell_date']).dt.days
    
    # 月内天数
    df['signal_dom'] = df['signal_date'].dt.day
    df['sell_dom'] = df['sell_date'].dt.day
    df['buy_dom'] = df['buy_date'].dt.day
    
    # 周几分布 (0=Mon, 6=Sun)
    df['buy_dow'] = df['buy_date'].dt.day_name()
    
    # 统计信息概要
    stats = {
        "总期数": len(df),
        "开始时间": df['signal_date'].min().strftime('%Y-%m-%d'),
        "结束时间": df['signal_date'].max().strftime('%Y-%m-%d'),
        "平均卖出延迟": f"{df['sell_lag'].mean():.2f} 天",
        "平均买入延迟": f"{df['buy_lag'].mean():.2f} 天"
    }

    # ---------------------------------------------------------
    # 可视化布局
    # ---------------------------------------------------------
    logger.info("正在构建可视化看板...")
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '调仓日历概览 (Signal / Sell / Buy)',
            '执行延迟分布 (自然日)',
            '月内日期分布 (Day of Month)',
            '买入执行周几分布 (Day of Week)',
            '调仓周期稳定性 (每月记录数)',
            '买卖执行窗口 (Buy - Sell Gap)'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Colors
    colors = {'signal': '#1F77B4', 'sell': '#FF7F0E', 'buy': '#2CA02C', 'gap': '#D62728'}

    # Plot 1: 时序概览 (最后 36 期)
    df_recent = df.tail(36)
    fig.add_trace(go.Scatter(x=df_recent['signal_date'], y=[1]*len(df_recent), name='信号日', mode='markers', marker=dict(size=10, color=colors['signal'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_recent['sell_date'], y=[1.1]*len(df_recent), name='卖出日', mode='markers', marker=dict(size=10, color=colors['sell'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_recent['buy_date'], y=[1.2]*len(df_recent), name='买入日', mode='markers', marker=dict(size=10, color=colors['buy'])), row=1, col=1)

    # Plot 2: 延迟分布
    fig.add_trace(go.Histogram(x=df['sell_lag'], name='卖出延迟', marker_color=colors['sell'], opacity=0.6), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['buy_lag'], name='买入延迟', marker_color=colors['buy'], opacity=0.6), row=1, col=2)
    fig.update_layout(barmode='overlay')

    # Plot 3: 月内日期分布
    fig.add_trace(go.Box(y=df['signal_dom'], name='信号日(月内)', marker_color=colors['signal']), row=2, col=1)
    fig.add_trace(go.Box(y=df['sell_dom'], name='卖出日(月内)', marker_color=colors['sell']), row=2, col=1)
    fig.add_trace(go.Box(y=df['buy_dom'], name='买入日(月内)', marker_color=colors['buy']), row=2, col=1)

    # Plot 4: 周几分布
    dow_counts = df['buy_dow'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    fig.add_trace(go.Bar(x=dow_counts.index, y=dow_counts.values, marker_color=colors['buy'], name='买入执行周布'), row=2, col=2)

    # Plot 5: 稳定性 (每年调仓次数)
    df['year'] = df['signal_date'].dt.year
    year_counts = df.groupby('year').size()
    fig.add_trace(go.Bar(x=year_counts.index, y=year_counts.values, marker_color=colors['signal'], name='年调仓频次'), row=3, col=1)

    # Plot 6: 窗口宽度
    fig.add_trace(go.Histogram(x=df['exec_window'], name='执行窗口(天)', marker_color=colors['gap']), row=3, col=2)

    # 布局定制
    diag_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in stats.items()])
    
    fig.update_layout(
        height=1000,
        template='plotly_white',
        title=dict(
            text=f'调仓日历质量诊断看板: {input_stem}',
            x=0.05, y=0.98,
            font=dict(size=20)
        ),
        margin=dict(t=150, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 添加诊断信息
    fig.add_annotation(
        text=diag_text,
        xref="paper", yref="paper",
        x=0, y=1.12,
        showarrow=False,
        align="left",
        font=dict(size=12, color="#555")
    )

    # 保存
    logger.info(f"正在保存可视化结果至: {output_file}")
    fig.write_html(output_file)
    logger.info("分析完成。")

if __name__ == '__main__':
    main()
