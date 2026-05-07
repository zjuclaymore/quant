#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
交割单收益与 IC 质量诊断脚本 (Enriched Order Quality & IC Analysis)

功能:
  本脚本针对 Step 5 生成的 enriched 交割单进行深度可视化，涵盖：
  1. IC/RankIC 时间序列及其稳定性 (Predictive Power Stability)
  2. 交割单全量信号的等权累计收益 (Cumulative Return of Signals)
  3. IC 统计属性 (Mean, IR, Win Rate)
  4. 截面相关性快照 (Cross-sectional Snapshots)

计算公式:
  - IC (Information Coefficient): $IC_t = Corr(Factor_t, Return_{t+1})$
  - RankIC: $RankIC_t = Corr(Rank(Factor_t), Rank(Return_{t+1}))$
  - IC IR: $IR = \frac{E[IC]}{\sigma(IC)}$
  - IC 胜率 (Win Rate): $P(IC > 0)$ (若因子预期为正相关) 或 $P(IC < 0)$ (若因子预期为负相关)
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
    parser = argparse.ArgumentParser(description="交割单收益与 IC 质量诊断工具")
    parser.add_argument('--input', type=str, required=True, help='Enriched 交割单 Parquet 文件路径')
    parser.add_argument('--ret-col', type=str, default='monthly_return', help='收益率列名')
    parser.add_argument('--factor-col', type=str, default='factor_value', help='因子值列名')
    parser.add_argument('--output-dir', type=str, default='analysis', help='输出目录')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    # 1. 加载数据
    logger.info(f"载入交割单数据: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 转换日期
    if 'buy_date' in df.columns:
        df['date_dt'] = pd.to_datetime(df['buy_date'])
    else:
        logger.error("交割单必须包含 buy_date 列")
        return

    # 2. 计算 IC 时间序列
    logger.info("正在执行 IC/RankIC 时序计算...")
    ic_ts = df.groupby('date_dt').apply(
        lambda x: pd.Series({
            'ic': x[args.factor_col].corr(x[args.ret_col]),
            'rank_ic': x[args.factor_col].rank().corr(x[args.ret_col].rank()),
            'avg_ret': x[args.ret_col].mean(),
            'count': len(x)
        })
    ).reset_index()

    # 填充缺失值 (部分月份可能无法计算相关性)
    ic_ts = ic_ts.fillna(0)

    # 3. 计算累计收益 (全量等权)
    ic_ts = ic_ts.sort_values('date_dt')
    ic_ts['cum_ret'] = (1 + ic_ts['avg_ret']).cumprod()

    # 4. IC 统计汇总
    ic_mean = ic_ts['ic'].mean()
    rank_ic_mean = ic_ts['rank_ic'].mean()
    ic_ir = ic_mean / ic_ts['ic'].std() if ic_ts['ic'].std() != 0 else 0
    rank_ic_ir = rank_ic_mean / ic_ts['rank_ic'].std() if ic_ts['rank_ic'].std() != 0 else 0
    
    # 根据 IC 均值符号自动判断胜率方向
    if rank_ic_mean >= 0:
        win_rate = (ic_ts['rank_ic'] > 0).mean()
    else:
        win_rate = (ic_ts['rank_ic'] < 0).mean()

    stats_summary = {
        "Mean IC": f"{ic_mean:.4f}",
        "Mean RankIC": f"{rank_ic_mean:.4f}",
        "IC IR": f"{ic_ir:.4f}",
        "RankIC IR": f"{rank_ic_ir:.4f}",
        "Win Rate (RankIC)": f"{win_rate*100:.2f}%",
        "Total Periods": len(ic_ts)
    }

    # 5. 输出准备
    input_stem = input_path.stem
    output_dir = Path(args.output_dir) / input_stem
    output_file = output_dir / 'enriched_order_analysis.html'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # ---------------------------------------------------------
    # 可视化构建
    # ---------------------------------------------------------
    logger.info("构建可视化看板...")
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'RankIC 时序变化', '全量信号等权累计收益 (Cumulative Return)',
            'IC 统计字典 (Summary Stats)', '每月平均收益分布 (Monthly Avg Return)',
            'IC 之柱 (IC Histogram)', '信号覆盖数时序 (Sample Count)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Plot 1: RankIC TS
    fig.add_trace(go.Bar(x=ic_ts['date_dt'], y=ic_ts['rank_ic'], name='RankIC', marker_color='#636EFA', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=ic_ts['date_dt'], y=ic_ts['rank_ic'].rolling(12).mean(), name='RankIC (12M Moving Avg)', line=dict(color='#EF553B', width=2)), row=1, col=1)

    # Plot 2: Cumulative Return
    fig.add_trace(go.Scatter(x=ic_ts['date_dt'], y=ic_ts['cum_ret'], name='累计净值', line=dict(color='#2CA02C', width=2), fill='tozeroy'), row=1, col=2)

    # Plot 3: Stats Table (As Annotation or Table)
    table_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in stats_summary.items()])
    fig.add_annotation(
        text=table_text,
        xref="paper", yref="paper",
        x=0.02, y=0.58,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#888",
        borderwidth=1,
        row=2, col=1
    )

    # Plot 4: Monthly Avg Return Distribution
    fig.add_trace(go.Histogram(x=ic_ts['avg_ret'], name='月均收益分布', marker_color='#FECB52', nbinsx=50), row=2, col=2)

    # Plot 5: IC Histogram
    fig.add_trace(go.Histogram(x=ic_ts['rank_ic'], name='RankIC 分布', marker_color='#636EFA', nbinsx=30), row=3, col=1)

    # Plot 6: Sample Count
    fig.add_trace(go.Scatter(x=ic_ts['date_dt'], y=ic_ts['count'], name='样本数', line=dict(color='#19D3F3')), row=3, col=2)

    # 布局外观
    fig.update_layout(
        height=1200,
        template='plotly_white',
        title=dict(
            text=f'交割单收益与 IC 质量诊断图: {input_stem}',
            x=0.05, y=0.98,
            font=dict(size=22)
        ),
        margin=dict(t=120, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 轴命名
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="NAV", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=2)

    # 保存
    logger.info(f"保存分析至: {output_file}")
    fig.write_html(output_file)
    logger.info("分析看板生成完毕。")

if __name__ == '__main__':
    main()
