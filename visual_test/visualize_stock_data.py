#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A 股股票量价高保真可视化检查脚本。

功能：
- 随机采样多只股票。
- 绘制包含均线系统（MA5/20/60）的专业 K 线图。
- 展示成交量柱状图。
- 采用收益率离群值检测算法进行数据质量评估。
"""

import os
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 配置
DATA_DIR = r"E:\1_basement\quant_research\parquet_data"
OUTPUT_DIR = r"E:\1_basement\quant_research\visual_test\stock_visualizations"
STOCKS_SAMPLES = 5  # 随机抽检数量

class StockVisualizer:
    """股票数据可视化与分析类。"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, ts_code: str) -> pd.DataFrame:
        """加载并预处理数据。"""
        path = os.path.join(self.data_dir, f"{ts_code}.parquet")
        df = pd.read_parquet(path)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df.sort_values("trade_date").reset_index(drop=True)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算常用技术指标。
        
        MA_N = Sum(Close, N) / N
        Pct_Change = (Close - Close_Prev) / Close_Prev
        """
        df = df.copy()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['pct_chg'] = df['close'].pct_change()
        return df

    def analyze_quality(self, df: pd.DataFrame, ts_code: str):
        """执行专业级数据质量分析。"""
        print(f"\n[Quality Report: {ts_code}]")
        print(f"数据区间: {df['trade_date'].min().date()} 至 {df['trade_date'].max().date()} ({len(df)} 行)")
        
        # 异常涨跌幅检查 (A股 10%/20% 限制，设置 21% 为硬阔值)
        outliers = df[df['pct_chg'].abs() > 0.21]
        if not outliers.empty:
            print(f"!!! 检测到潜在异常涨跌幅 ({len(outliers)} 处):")
            print(outliers[['trade_date', 'pct_chg']])
        
        # 缺失值检查
        nulls = df.isnull().sum()
        if nulls.any():
            print(f"[Error] 存在缺失字段: {nulls[nulls > 0].to_dict()}")
        else:
            print("[Success] 数据完整性优良")

    def create_professional_plot(self, df: pd.DataFrame, ts_code: str):
        """创建专业级的交互式 K 线图。"""
        # 创建带有共享 X 轴的子图
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=(f'{ts_code} K线与均线', '成交量'),
            row_heights=[0.7, 0.3]
        )

        # 1. K线图
        fig.add_trace(go.Candlestick(
            x=df['trade_date'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name='K线',
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
        ), row=1, col=1)

        # 2. 均线
        colors = {'ma5': '#f1c40f', 'ma20': '#3498db', 'ma60': '#9b59b6'}
        for ma in ['ma5', 'ma20', 'ma60']:
            fig.add_trace(go.Scatter(
                x=df['trade_date'], y=df[ma],
                mode='lines', name=ma.upper(),
                line=dict(width=1, color=colors[ma])
            ), row=1, col=1)

        # 3. 成交量
        # 标注上涨和下跌成交量颜色
        colors_vol = ['#ef5350' if x >= 0 else '#26a69a' for x in df['pct_chg'].fillna(0)]
        fig.add_trace(go.Bar(
            x=df['trade_date'], y=df['vol'],
            name='成交量',
            marker_color=colors_vol,
            opacity=0.8
        ), row=2, col=1)

        # 样式优化
        fig.update_layout(
            height=900,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            title={
                'text': f"<b>{ts_code} 深度量价动态可视化</b>",
                'y':0.95, 'x':0.5, 'xanchor': 'center'
            },
            showlegend=True,
            hovermode='x unified'
        )

        output_path = os.path.join(self.output_dir, f"{ts_code}_analysis.html")
        fig.write_html(output_path)
        return output_path

def main():
    visualizer = StockVisualizer(DATA_DIR, OUTPUT_DIR)
    
    # 获取候选股票
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    if not all_files:
        print("无数据，请先抓取。")
        return
        
    samples = random.sample(all_files, min(len(all_files), STOCKS_SAMPLES))
    
    for filename in samples:
        ts_code = filename.replace(".parquet", "")
        try:
            df = visualizer.load_data(ts_code)
            df = visualizer.calculate_indicators(df)
            
            # 报告与可视化
            visualizer.analyze_quality(df, ts_code)
            path = visualizer.create_professional_plot(df, ts_code)
            print(f"Done: 可视化完成: {path}")
            
        except Exception as e:
            print(f"处理 {ts_code} 失败: {e}")

if __name__ == "__main__":
    main()
