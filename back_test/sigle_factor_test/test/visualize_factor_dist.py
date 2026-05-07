#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
因子分布分析与质量检测可视化脚本 (Factor Distribution Analysis & Visualization)

功能:
  本脚本用于对因子文件进行多维分布分析，帮助研究人员识别：
  1. 数据覆盖稳定性 (Coverage Stability)
  2. 截面分布形态 (Distribution Shape: Skewness/Kurtosis)
  3. 异常值影响与预处理效果 (Outliers & Preprocessing Impact)
  4. 分位数演变 (Quantile Evolution)

计算公式说明 (Formulas):
  - 覆盖率 (Coverage): 执行期间每日非空因子值的代码总数。
  - 偏度 (Skewness): $\gamma = E[(\frac{X-\mu}{\sigma})^3]$, 衡量采样分布的对称性。
  - 峰度 (Kurtosis): $K = E[(\frac{X-\mu}{\sigma})^4] - 3$, 衡量分布的肥尾程度（超额峰度）。
  - 分位数演变: 追踪 5%, 25%, 50%, 75%, 95% 分位随时间的变化情况。
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import logging
import argparse
from scipy.stats import skew, kurtosis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="因子分布多维看板生成工具")
    parser.add_argument('--input', type=str, required=True, help='因子 Parquet 文件路径')
    parser.add_argument('--factor-col', type=str, default=None, help='因子列名 (不指定则自动推断)')
    parser.add_argument('--output-dir', type=str, default='analysis', help='结果输出根目录')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return

    # 1. 加载数据
    logger.info(f"正在读取因子文件: {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 自动推断因子列 (排除 code, date 和 year_month 等常规列)
    if args.factor_col:
        factor_col = args.factor_col
    else:
        candidates = [c for c in df.columns if c not in ['code', 'date', 'year_month', 'factor_date']]
        if not candidates:
            logger.error("无法识别因子列，请显式指定 --factor-col")
            return
        factor_col = candidates[0]
        logger.info(f"自动识别因子列为: {factor_col}")

    # 日期转换
    if df['date'].dtype == 'int64':
        df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    else:
        df['date_dt'] = pd.to_datetime(df['date'])

    # 2. 统计特征提取
    logger.info("执行截面特征聚合计算...")
    grouped = df.groupby('date_dt')[factor_col]
    
    ts_stats = grouped.agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()

    # 计算偏度和峰度 (scipy)
    ts_stats['skew'] = grouped.apply(lambda x: skew(x.dropna())).values
    ts_stats['kurt'] = grouped.apply(lambda x: kurtosis(x.dropna())).values
    
    # 计算分位数
    logger.info("计算分位数轨迹...")
    quantiles = grouped.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack().reset_index()
    quantiles.columns = ['date_dt', 'q05', 'q25', 'q50', 'q75', 'q95']

    # 3. 输出路径准备
    input_stem = input_path.stem
    output_dir = Path(args.output_dir) / input_stem
    output_file = output_dir / 'factor_dist_analysis.html'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # ---------------------------------------------------------
    # 可视化看板构建
    # ---------------------------------------------------------
    logger.info("构建交互式可视化看板...")
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '因子覆盖稳定性 (个股总数)', '分位数演变区间 (5% - 95%)',
            '标准差 (波动性) 趋势', '均值与中位数对比',
            '分布偏度 (Symmetry)', '分布峰度 (Tails)',
            '最新截面分布 (Histogram)', '最新截面箱型图 (Boxplot Analysis)'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    theme_color = '#636EFA'
    alt_color = '#EF553B'

    # Plot 1: Coverage
    fig.add_trace(go.Scatter(x=ts_stats['date_dt'], y=ts_stats['count'], name='覆盖股数', fill='tozeroy', line=dict(color='#19D3F3')), row=1, col=1)

    # Plot 2: Quantiles
    fig.add_trace(go.Scatter(x=quantiles['date_dt'], y=quantiles['q95'], line=dict(width=0), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=quantiles['date_dt'], y=quantiles['q05'], line=dict(width=0), fill='tonexty', fillcolor='rgba(99, 110, 250, 0.2)', name='95%-5% 区间'), row=1, col=2)
    fig.add_trace(go.Scatter(x=quantiles['date_dt'], y=quantiles['q50'], line=dict(color=theme_color, width=2), name='中位数 (50%)'), row=1, col=2)

    # Plot 3: Std Dev
    fig.add_trace(go.Scatter(x=ts_stats['date_dt'], y=ts_stats['std'], name='标准差', line=dict(color=alt_color)), row=2, col=1)

    # Plot 4: Mean vs Median
    fig.add_trace(go.Scatter(x=ts_stats['date_dt'], y=ts_stats['mean'], name='均值', line=dict(dash='dash', color='#AB63FA')), row=2, col=2)
    fig.add_trace(go.Scatter(x=ts_stats['date_dt'], y=ts_stats['median'], name='中位数', line=dict(color=theme_color)), row=2, col=2)

    # Plot 5: Skewness
    fig.add_trace(go.Scatter(x=ts_stats['date_dt'], y=ts_stats['skew'], name='偏度', line=dict(color='#00CC96')), row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5, row=3, col=1)

    # Plot 6: Kurtosis
    fig.add_trace(go.Scatter(x=ts_stats['date_dt'], y=ts_stats['kurt'], name='峰度', line=dict(color='#FECB52')), row=3, col=2)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5, row=3, col=2)

    # Plot 7 & 8: Latest Snapshot
    latest_date = df['date_dt'].max()
    latest_data = df[df['date_dt'] == latest_date][factor_col].dropna()
    
    fig.add_trace(go.Histogram(x=latest_data, nbinsx=50, name='分布直方图', marker_color=theme_color, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Box(x=latest_data, name='分布箱型图', boxpoints='outliers', marker_color=alt_color), row=4, col=2)

    # 布局外观设置
    fig.update_layout(
        height=1400,
        template='plotly_white',
        title=dict(
            text=f'因子多维分布诊断看板: {factor_col} ({input_stem})',
            x=0.05, y=0.98,
            font=dict(size=22)
        ),
        margin=dict(t=120, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 轴命名
    fig.update_yaxes(title_text="个股数", row=1, col=1)
    fig.update_yaxes(title_text="因子值", row=1, col=2)
    fig.update_yaxes(title_text="标准差", row=2, col=1)
    fig.update_yaxes(title_text="偏度 (Skew)", row=3, col=1)
    fig.update_yaxes(title_text="峰度 (Kurt)", row=3, col=2)

    # 保存
    logger.info(f"保存分析报表至: {output_file}")
    fig.write_html(output_file)
    logger.info("分析看板生成完毕。")

if __name__ == '__main__':
    main()
