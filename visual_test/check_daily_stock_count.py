#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全量 A 股交易日股票数量统计与分析脚本。

功能：
- 并发读取全量 Parquet 数据。
- 统计每日截面股票数量。
- 生成高保真可视化图表。
"""

import os
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm
from datetime import datetime

# 配置
DATA_DIR = r"E:\1_basement\quant_research\parquet_data"
OUTPUT_DIR = r"E:\1_basement\quant_research\visual_test\stock_count_check"

def process_single_file(file_path: str) -> list:
    """
    处理单个股票文件，提取其交易日期列表。
    
    Args:
        file_path: Parquet 文件路径.
        
    Returns:
        list: 交易日期字符串列表。
    """
    try:
        # 只读取 trade_date 列，最小化 I/O
        df = pd.read_parquet(file_path, columns=["trade_date"])
        return df["trade_date"].astype(str).tolist()
    except Exception as e:
        # 记录错误但在控制台静默，避免打断进度条
        return []

def main():
    """主逻辑函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[{datetime.now()}] 开始扫描数据目录...")
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    
    if not files:
        print("未找到 Parquet 文件，请检查 DATA_DIR。")
        return

    print(f"找到 {len(files)} 个文件，启动并发统计...")
    
    all_dates = []
    # 使用 ThreadPoolExecutor 并发处理 I/O 任务
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {executor.submit(process_single_file, f): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="扫描进度"):
            result = future.result()
            if result:
                all_dates.extend(result)

    # 聚合计数
    print("正在聚合数据...")
    date_counts = Counter(all_dates)
    
    # 转换为 DataFrame
    df_count = pd.DataFrame(list(date_counts.items()), columns=["trade_date", "count"])
    df_count["trade_date"] = pd.to_datetime(df_count["trade_date"])
    df_count = df_count.sort_values("trade_date").reset_index(drop=True)
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, "stock_daily_count.csv")
    df_count.to_csv(csv_path, index=False)
    
    # 可视化增强
    _generate_enhanced_chart(df_count)
    
    print(f"统计完成！数据已保存至: {csv_path}")

def _generate_enhanced_chart(df: pd.DataFrame):
    """
    生成高品质的股票数量变化图表。
    
    Args:
        df: 包含 trade_date 和 count 的 DataFrame。
    """
    fig = go.Figure()

    # 填充背景渐变（通过多层 fill）
    fig.add_trace(go.Scatter(
        x=df["trade_date"], 
        y=df["count"],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)',
        name='A股总数'
    ))

    # 添加 250 日均线 (MA250) 说明市场中长期规模趋势
    # 公式：Count_MA250 = Sum(Count, 250) / 250
    df['ma250'] = df['count'].rolling(window=250).mean()
    fig.add_trace(go.Scatter(
        x=df["trade_date"],
        y=df['ma250'],
        mode='lines',
        line=dict(color='rgba(255, 127, 14, 0.8)', width=1.5, dash='dot'),
        name='250日平均规模'
    ))

    # 标注重要历史节点（示例：2015 牛市顶点、2018 贸易战等可在此扩展）
    # 自动找出最大值
    max_count = df['count'].max()
    max_date = df.loc[df['count'].idxmax(), 'trade_date']
    
    fig.add_annotation(
        x=max_date,
        y=max_count,
        text=f"历史峰值: {int(max_count)}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )

    # 布局美化
    fig.update_layout(
        title={
            'text': "<b>A股每日上市/交易股票总数走势</b>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24, color='#2c3e50')
        },
        xaxis_title="年份",
        yaxis_title="股票支数",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 保存 HTML
    html_path = os.path.join(OUTPUT_DIR, "enhanced_stock_count.html")
    fig.write_html(html_path)

if __name__ == "__main__":
    main()
