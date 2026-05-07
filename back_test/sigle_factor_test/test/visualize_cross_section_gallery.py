#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
因子截面逐月画廊可视化脚本 (Cross-sectional Monthly Gallery Visualization)

功能:
  本脚本通过交互式下拉菜单，允许研究员逐月查看因子的截面表现：
  1. 截面散点图 (Factor vs Return): 识别相关性强度及极端异常个股。
  2. 截面直方图 (Factor Distribution): 观察因子随时间的分布漂移。
  3. 实时指标统计 (Current Stats): 自动计算当月 IC, RankIC, 样本数。

技术实现:
  利用 Plotly 的 Visibility 属性实现多图层切换，无需后端 Flask，单个 HTML 即可运行。
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
    parser = argparse.ArgumentParser(description="因子截面逐月画廊可视化工具")
    parser.add_argument('--input', type=str, required=True, help='Enriched 交割单 Parquet 路径')
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
    logger.info(f"共检测到 {len(dates)} 个截面交易日")

    # 2. 构建画廊画布
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('因子值 vs 次月收益率 (Scatter)', '因子值截面分布 (Histogram)'),
        horizontal_spacing=0.1
    )

    # 预先生成所有月份的 Traces
    # 每个月生成 2 个 Trace: 1. Scatter, 2. Histogram
    buttons = []
    
    logger.info("正在为每个截面生成可视化图层...")
    for i, d in enumerate(dates):
        sub = df[df['date_dt'] == d]
        d_str = d.strftime('%Y-%m-%d')
        
        # 计算当月指标
        ic = sub[args.factor_col].corr(sub[args.ret_col])
        rank_ic = sub[args.factor_col].rank().corr(sub[args.ret_col].rank())
        count = len(sub)
        
        is_visible = (i == 0) # 仅第一个月默认可见
        
        # Trace 0: Scatter
        fig.add_trace(
            go.Scatter(
                x=sub[args.factor_col], 
                y=sub[args.ret_col], 
                mode='markers', 
                name='Signals',
                text=sub['code'],
                marker=dict(size=6, color='#636EFA', opacity=0.6, line=dict(width=1, color='white')),
                visible=is_visible
            ),
            row=1, col=1
        )
        
        # Trace 1: Histogram
        fig.add_trace(
            go.Histogram(
                x=sub[args.factor_col], 
                nbinsx=40, 
                name='Distribution',
                marker_color='#FECB52',
                opacity=0.7,
                visible=is_visible
            ),
            row=1, col=2
        )
        
        # 构建下拉菜单按钮
        # 按钮控制可见性矩阵: 每个按钮控制 i*2 和 i*2+1 两个 trace
        vis_mask = [False] * (len(dates) * 2)
        vis_mask[i*2] = True
        vis_mask[i*2+1] = True
        
        button = dict(
            label=d_str,
            method="update",
            args=[
                {"visible": vis_mask},
                {"title": f"因子截面看板: {d_str} | IC: {ic:.4f} | RankIC: {rank_ic:.4f} | Counts: {count}"}
            ]
        )
        buttons.append(button)

    # 3. 布局与菜单设置
    logger.info("正在优化交互菜单布局...")
    
    # 初始标题
    first_d = dates[0].strftime('%Y-%m-%d')
    first_sub = df[df['date_dt'] == dates[0]]
    first_ic = first_sub[args.factor_col].corr(first_sub[args.ret_col])
    first_rank_ic = first_sub[args.factor_col].rank().corr(first_sub[args.ret_col].rank())
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                showactive=True,
                x=0, y=1.1,
                xanchor="left", yanchor="top"
            ),
        ],
        title=dict(
            text=f"因子截面看板: {first_d} | IC: {first_ic:.4f} | RankIC: {first_rank_ic:.4f} | Counts: {len(first_sub)}",
            x=0.05, y=0.98,
            font=dict(size=20)
        ),
        height=800,
        template='plotly_white',
        margin=dict(t=150, b=50),
        showlegend=False
    )

    fig.update_xaxes(title_text="Factor Value", row=1, col=1)
    fig.update_yaxes(title_text="Next Month Return", row=1, col=1)
    fig.update_xaxes(title_text="Factor Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    # 4. 保存
    input_stem = input_path.stem
    output_dir = Path(args.output_dir) / input_stem
    output_outfile = output_dir / 'cross_section_gallery.html'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    logger.info(f"正在保存截面画廊至: {output_outfile}")
    fig.write_html(output_outfile)
    logger.info("交割单截面分析看板生成完毕。")

if __name__ == '__main__':
    main()
