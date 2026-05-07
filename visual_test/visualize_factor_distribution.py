import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

"""
本脚本用于深度分析因子数据的截面属性。
优化点：采用高性能的向量化计算构建因子分布热力图，并集成至统一的深色模式报告中。
"""

# 配置
DATA_DIR = r"E:\1_basement\quant_research\factor_base\log_mv_1"
OUTPUT_DIR = r"E:\1_basement\quant_research\visual_test\factor_analysis"
FACTOR_COL = "log_mv"

def load_data(file_path: str) -> pd.DataFrame:
    """并发加载函数"""
    try:
        return pd.read_parquet(file_path)
    except:
        return pd.DataFrame()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    print(f"Loading {len(files)} files...")
    
    all_data = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(load_data, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="读取中"):
            df_slice = future.result()
            if not df_slice.empty:
                all_data.append(df_slice)
    
    print("正在合并数据并进行空间分析...")
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined['trade_date'] = pd.to_datetime(df_combined['trade_date'].astype(str))
    
    # 按照日期和因子的 Bin 进行向量化聚合
    print("构建分布式热力图矩阵 (Vectorized)...")
    
    # 自动计算 Bin 范围，剔除极值以增强可视化对比度
    f_min, f_max = df_combined[FACTOR_COL].quantile(0.001), df_combined[FACTOR_COL].quantile(0.999)
    bins = np.linspace(f_min, f_max, 80)
    
    # 使用 pd.cut 进行分箱
    df_combined['bin'] = pd.cut(df_combined[FACTOR_COL], bins=bins)
    
    # 统计每个日期每个箱体的数量
    # 公式：Density(Date, Bin) = Count(Date, Bin) / TotalCount(Date)
    dist_matrix = df_combined.groupby(['trade_date', 'bin'], observed=False).size().unstack(fill_value=0)
    
    # 归一化（每行占比）
    dist_matrix_norm = dist_matrix.div(dist_matrix.sum(axis=1), axis=0)
    
    # 计算统计线
    daily_stats = df_combined.groupby('trade_date')[FACTOR_COL].quantile([0.05, 0.5, 0.95]).unstack()
    daily_stats.columns = ['q5', 'median', 'q95']
    daily_count = df_combined.groupby('trade_date').size().reset_index(name='sample_count')

    # 开始绘图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("样本规模动态", f"因子 [{FACTOR_COL}] 分布演变热力图 (密度占比)"),
        row_heights=[0.2, 0.8]
    )

    # 1. 样本数
    fig.add_trace(go.Scatter(
        x=daily_count['trade_date'], y=daily_count['sample_count'],
        name="有效样本", fill='tozeroy', line=dict(color='#16a085', width=1.5),
        fillcolor='rgba(22, 160, 133, 0.1)'
    ), row=1, col=1)

    # 2. 热力图
    # Y 轴显示 Bin 的中心值
    bin_centers = [(b.left + b.right)/2 for b in dist_matrix.columns]
    
    fig.add_trace(go.Heatmap(
        x=dist_matrix_norm.index,
        y=bin_centers,
        z=dist_matrix_norm.values.T,
        colorscale='Magma', # 使用更高级的 Magma 色阶，适合深色模式
        colorbar=dict(title="分布密度", x=1.02, len=0.8),
        name='分布密度'
    ), row=2, col=1)

    # 3. 叠加分位线
    q_lines = {'q5': 'rgba(255,255,255,0.4)', 'median': 'white', 'q95': 'rgba(255,255,255,0.4)'}
    for q, color in q_lines.items():
        fig.add_trace(go.Scatter(
            x=daily_stats.index, y=daily_stats[q],
            mode='lines', name=q.upper(),
            line=dict(color=color, width=1.5 if q=='median' else 1, dash='dash' if q!='median' else 'solid'),
            hoverinfo='skip'
        ), row=2, col=1)

    fig.update_layout(
        height=1000,
        template="plotly_dark",
        title={
            'text': f"<b>因子 [{FACTOR_COL}] 横截面分布深度报告</b>",
            'y':0.98, 'x':0.5, 'xanchor': 'center'
        },
        hovermode='x unified'
    )
    
    report_path = os.path.join(OUTPUT_DIR, "factor_distribution_heatmap.html")
    fig.write_html(report_path)
    # 删除旧的、效果不佳的直方图文件
    old_hist = os.path.join(OUTPUT_DIR, "latest_factor_histogram.html")
    if os.path.exists(old_hist): os.remove(old_hist)
    
    print(f"🎉 优化版报告已生成: {report_path}")

if __name__ == "__main__":
    main()
