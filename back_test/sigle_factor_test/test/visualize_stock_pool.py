#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
股票池多准则可视化与诊断脚本 (Stock Pool Visualization & Diagnostics)

本脚本用于对股票池文件（parquet格式）进行深度分析，涵盖：
1. 规模演变：分析全样本与可交易样本的随时间变化。
2. 通过率诊断：识别市场异常波动导致的流动性枯竭或大规模剔除。
3. 板块分布：依据代码前缀识别上市板块（主板、创业板、科创板、北交所）的构成比例。
4. 稳定性分析：计算每日流入流出股票池的个股数量（Churn）。
5. 处理逻辑识别：通过特征列自动推断该股票池经历的处理阶段。

计算公式说明：
- 通过率 (Pass Rate): $R_{pass}(t) = \frac{N_{allowed}(t)}{N_{total}(t)}$
  用于衡量当前过滤准则对全市场的覆盖广度。
- 每日进入数 (Entry): $E(t) = \sum_{i} [allow\_flag_{i}(t)=1 \land allow\_flag_{i}(t-1)=0]$
  反映新通过准则的股票数量。
- 每日退出数 (Exit): $X(t) = \sum_{i} [allow\_flag_{i}(t)=0 \land allow\_flag_{i}(t-1)=1]$
  反映因停牌、剔除或次新判定而离开可交易池的股票数量。
- 股票池波动项 (Stability/Churn): $Stability(t) = E(t) + X(t)$
  用于衡量股票池的换手稳定性，波动越大意味着交易机会越不稳定。
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

def get_board_name(code_str: str) -> str:
    """
    依据股票代码前缀推断上市板块。
    
    算法规则：
    - 688: 沪市科创板
    - 60: 沪市主板
    - 300: 深市创业板
    - 00: 深市主板/中小板
    - 43/83/87/88/920: 北京证券交易所
    - 其他: 其他/无法识别
    
    Args:
        code_str (str): 6位或8位股票代码。
        
    Returns:
        str: 中文板块名称。
    """
    # 填充为6位字符串
    c = str(code_str).zfill(6)
    prefix3 = c[:3]
    prefix2 = c[:2]
    
    if prefix3 == '688':
        return '沪市科创板'
    if prefix2 == '60':
        return '沪市主板'
    if prefix3 == '300':
        return '深市创业板'
    if prefix2 == '00':
        return '深市主板/中小板'
    if prefix2 in ['43', '83', '87', '88'] or prefix3 == '920':
        return '北交所'
    return '其他板块'

def detect_processing_steps(columns: list) -> list:
    """
    通过列名检测股票池经历的处理步骤。
    
    检测逻辑：
    - 存在 'is_listed': 已关联上市/退市状态。
    - 存在 'first_date'/'list_date': 已关联上市日期。
    - 存在 'st_type': 已标记 ST 状态。
    - 存在 'volume': 已涵盖成交量/流动性信息。
    - 存在 'lncap': 已涵盖市值信息。
    - 存在 'is_subnew': 已执行次新股判定。
    - 存在 'allow_flag': 已执行终极可交易准则过滤。
    
    Args:
        columns (list): DataFrame 的列名列表。
        
    Returns:
        list: 已识别的处理步骤描述列表。
    """
    steps = []
    if 'is_listed' in columns: steps.append("上市状态过滤 (Listed/Delisted)")
    if 'first_date' in columns or 'list_date' in columns: steps.append("上市时间关联 (Listing Date)")
    if 'st_type' in columns: steps.append("ST/退市风险警示识别")
    if 'volume' in columns: steps.append("流动性信息检测 (Liquidity/Volume)")
    if 'lncap' in columns: steps.append("市值规模识别 (Market Cap)")
    if 'is_subnew' in columns: steps.append("次新股识别 (Sub-new Filter)")
    if 'allow_flag' in columns: steps.append("综合准则放行 (Combined Gate)")
    
    if not steps:
        steps.append("基础原始池 (纯日期/代码架构)")
    return steps

def main():
    """
    主执行逻辑：解析命令行、加载数据、识别处理步骤、生成可视化。
    """
    parser = argparse.ArgumentParser(description="股票池多准则分析与可视化工具")
    parser.add_argument('--input', type=str, required=True, help='输入的股票池 Parquet 文件路径')
    parser.add_argument('--output-dir', type=str, default='analysis', help='输出分析结果的根目录 (默认为 analysis)')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    
    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        return

    # 定义输出子路径
    input_stem = input_path.stem
    output_dir = Path(args.output_dir) / input_stem
    output_file = output_dir / 'stock_pool_analysis.html'
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logger.info(f"正在载入数据: {input_path}...")
    # 尽可能只读入需要的列以节省内存
    all_cols = pd.read_parquet(input_path).columns.tolist()
    steps_detected = detect_processing_steps(all_cols)
    logger.info(f"检测到处理流程: {', '.join(steps_detected)}")

    # 核心分析流程所需列
    needed = ['date', 'code']
    if 'allow_flag' in all_cols:
        needed.append('allow_flag')
    
    df = pd.read_parquet(input_path, columns=needed)
    
    # 如果没有 allow_flag，默认全通过
    if 'allow_flag' not in df.columns:
        df['allow_flag'] = 1
    
    # 1. 基础处理：板块推断
    logger.info("正在执行板块推断...")
    df['board'] = df['code'].astype(str).str.zfill(6).map(get_board_name)
    
    # 2. 每日统计
    logger.info("正在聚合每日统计数据 (通过率分析)...")
    daily_counts = df.groupby('date').agg(
        total_count=('allow_flag', 'count'),
        allowed_count=('allow_flag', 'sum')
    ).reset_index()
    daily_counts['date_dt'] = pd.to_datetime(daily_counts['date'].astype(str), format='%Y%m%d')
    daily_counts['pass_rate'] = daily_counts['allowed_count'] / daily_counts['total_count']
    daily_counts['excluded_count'] = daily_counts['total_count'] - daily_counts['allowed_count']
    
    # 3. 板块分布
    logger.info("正在计算板块构成...")
    board_dist = df[df['allow_flag'] == 1].groupby(['date', 'board']).size().unstack(fill_value=0).reset_index()
    board_dist['date_dt'] = pd.to_datetime(board_dist['date'].astype(str), format='%Y%m%d')
    
    # 4. 稳定性 (Churn)
    logger.info("正在计算股票池变动稳定性...")
    df = df.sort_values(['code', 'date'])
    df['prev_allow'] = df.groupby('code')['allow_flag'].shift(1).fillna(0).astype(int)
    df['is_entry'] = ((df['allow_flag'] == 1) & (df['prev_allow'] == 0)).astype(int)
    df['is_exit'] = ((df['allow_flag'] == 0) & (df['prev_allow'] == 1)).astype(int)
    
    stability = df.groupby('date').agg(
        entry_count=('is_entry', 'sum'),
        exit_count=('is_exit', 'sum')
    ).reset_index()
    stability['date_dt'] = pd.to_datetime(stability['date'].astype(str), format='%Y%m%d')
    
    # ---------------------------------------------------------
    # 可视化布局
    # ---------------------------------------------------------
    logger.info("正在构建分析看板...")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{}], [{}], [{}],],
        subplot_titles=(
            '股票池规模演变 & 剔除样本(倒置)',
            '股票池通过率 (%)',
            '放行样本的板块构成',
            '股票池变动稳定性 (新增进入 vs 剔除退出)'
        )
    )
    
    # 配色方案
    colors = {
        'total': '#1F77B4',     # 样本总数
        'allowed': '#2CA02C',   # 放行总数
        'excluded': '#D3D3D3',  # 剔除总数
        'pass_rate': '#D62728', # 通过率
        'entry': '#AB63FA',     # 进入
        'exit': '#FFA15A',      # 退出
    }

    # Plot 1: 规模演变
    fig.add_trace(go.Scatter(x=daily_counts['date_dt'], y=daily_counts['total_count'], name='样本总数', line=dict(color=colors['total'], width=1.5), fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily_counts['date_dt'], y=daily_counts['allowed_count'], name='放行总数', line=dict(color=colors['allowed'], width=2), fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Bar(x=daily_counts['date_dt'], y=daily_counts['excluded_count'], name='剔除总数(倒置)', marker_color=colors['excluded'], opacity=0.4), row=1, col=1, secondary_y=True)

    # Plot 2: 通过率
    fig.add_trace(go.Scatter(x=daily_counts['date_dt'], y=daily_counts['pass_rate'] * 100, name='通过率 (%)', line=dict(color=colors['pass_rate'], width=1.5)), row=2, col=1)

    # Plot 3: 板块分布
    board_cols = [c for c in board_dist.columns if c not in ['date', 'date_dt']]
    for b_col in board_cols:
        fig.add_trace(go.Scatter(x=board_dist['date_dt'], y=board_dist[b_col], name=b_col, stackgroup='one', mode='lines', line=dict(width=0.5)), row=3, col=1)

    # Plot 4: 稳定性 (折线面积图以提供更好的视觉连续性)
    fig.add_trace(go.Scatter(x=stability['date_dt'], y=stability['entry_count'], name='每日进入', fill='tozeroy', line=dict(color=colors['entry'], width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=stability['date_dt'], y=-stability['exit_count'], name='每日退出', fill='tozeroy', line=dict(color=colors['exit'], width=1)), row=4, col=1)

    # 布局定制
    # 处理链自适应换行 (每 4 个步骤换一行)
    formatted_steps = []
    for i in range(0, len(steps_detected), 4):
        formatted_steps.append(" -> ".join(steps_detected[i:i+4]))
    steps_text = "<br>".join(formatted_steps)

    summary_text = f"<b>数据文件:</b> {input_stem}.parquet<br><b>识别到的处理链:</b><br>{steps_text}"
    
    fig.update_layout(
        height=1400, # 稍微增加总高度
        template='plotly_white',
        title=dict(
            text=f'股票池全局分析看板: {input_stem}',
            x=0.05,
            y=0.98,
            xanchor='left',
            yanchor='top',
            font=dict(size=20)
        ),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.01, 
            xanchor="right", 
            x=1,
            font=dict(size=10)
        ),
        margin=dict(t=220, b=100, l=60, r=60) # 显著增加顶部边距以容纳诊断信息
    )

    # 添加诊断信息作为 Annotation，位置固定在左上角
    fig.add_annotation(
        text=summary_text,
        xref="paper", yref="paper",
        x=0, y=1.15, # 放在标题下方，图表上方
        showarrow=False,
        align="left",
        font=dict(size=12, color="#555")
    )
    
    # 轴配置
    fig.update_yaxes(title_text="个股数量", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="剔除样本", row=1, col=1, secondary_y=True, autorange='reversed', showgrid=False)
    fig.update_yaxes(title_text="通过率 (%)", row=2, col=1, fixedrange=False)
    fig.update_yaxes(title_text="板块分类数", row=3, col=1, fixedrange=False)
    fig.update_yaxes(title_text="变动数量 (+/-)", row=4, col=1, fixedrange=False)
    
    # 时间控制
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.04,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(count=5, label="5年", step="year", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        ),
        row=4, col=1
    )

    # 保存
    logger.info(f"保存分析结果至: {output_file}...")
    fig.write_html(output_file)
    logger.info("可视化与诊断工作圆满完成。")

if __name__ == '__main__':
    main()
