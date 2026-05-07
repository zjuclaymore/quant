#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
因子截面全量矩阵可视化脚本 (Cross-sectional Full Matrix Visualization)

功能:
  将历史上所有的调仓截面以“矩阵网格”的形式平铺在一个页面上。
  1. 快速全局视察：一眼看清因子的历史表现稳定性。
  2. 颜色编码：根据 IC 正负自动着色标题，识别回撤期。
  3. 紧凑布局：高密度展示，适合打印或长截图。

技术实现:
  使用 Plotly Subplots 预设 190+ 个坐标轴。
  使用 Scattergl (WebGL) 确保在大规模点数下的渲染性能。
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
    parser = argparse.ArgumentParser(description="因子截面全量矩阵平铺看板生成工具")
    parser.add_argument('--input', type=str, required=True, help='Enriched 交割单 Parquet 路径')
    parser.add_argument('--cols', type=int, default=6, help='每行显示的图表列数')
    parser.add_argument('--ret-col', type=str, default='monthly_return', help='收益率列名')
    parser.add_argument('--factor-col', type=str, default='factor_value', help='因子值列名')
    parser.add_argument('--output-dir', type=str, default='analysis', help='输出目录')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    # 1. 加载数据
    logger.info(f"加载数据: {input_path}")
    df = pd.read_parquet(input_path, columns=['buy_date', 'code', args.factor_col, args.ret_col])
    df['date_dt'] = pd.to_datetime(df['buy_date'])
    df = df.dropna(subset=[args.factor_col, args.ret_col])
    
    dates = sorted(df['date_dt'].unique())
    n_dates = len(dates)
    n_cols = args.cols
    n_rows = int(np.ceil(n_dates / n_cols))
    
    logger.info(f"共检测到 {n_dates} 个截面日，将生成 {n_rows}x{n_cols} 矩阵")

    # 2. 构建子图布局
    # subplot_titles 将显示日期和 IC
    subplot_titles = []
    logger.info("计算各期 IC 以准备标题...")
    date_groups = list(df.groupby('date_dt'))
    
    for d, sub in date_groups:
        ic = sub[args.factor_col].corr(sub[args.ret_col])
        d_str = d.strftime('%Y-%m')
        subplot_titles.append(f"{d_str} (IC:{ic:.2f})")
        
    # 补齐空位标题
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.03 # 进一步调小间距以配合高密度显示
    )

    # 3. 填充图表
    logger.info("正在渲染子图矩阵 (使用 WebGL 优化性能)...")
    for i, (d, sub) in enumerate(date_groups):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        ic = sub[args.factor_col].corr(sub[args.ret_col])
        color = '#2CA02C' if ic >= 0 else '#D62728' # 正向绿，负向红 (假设正相关为佳)
        
        # 使用 Scattergl 进行性能优化
        fig.add_trace(
            go.Scattergl(
                x=sub[args.factor_col], 
                y=sub[args.ret_col], 
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.4),
                name=d.strftime('%Y-%m'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # 隐藏坐标轴刻度以保持整洁，仅保留子图标题
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showticklabels=False, row=row, col=col)

    # 4. 整体布局定制
    # 每行高度 250px
    total_height = max(800, n_rows * 250)
    
    fig.update_layout(
        height=total_height,
        width=1800, # 增加宽度以容纳多列
        template='plotly_white',
        title=dict(
            text=f"因子截面矩阵平铺看板: {args.factor_col} | 每行 {n_cols} 列 | 红色=负IC, 绿色=正IC",
            x=0.05, y=0.99,
            font=dict(size=24)
        ),
        margin=dict(t=120, b=50, l=50, r=50),
    )

    # 修改子图标题样式 (Font Size)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)

    # 5. 保存
    input_stem = input_path.stem
    output_dir = Path(args.output_dir) / input_stem
    output_outfile = output_dir / 'cross_section_matrix.html'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    logger.info(f"保存矩阵看板至: {output_outfile}")
    fig.write_html(output_outfile)
    logger.info("矩阵分析看板生成完毕。")

if __name__ == '__main__':
    main()
