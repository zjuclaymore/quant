from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
截面分钟线高保真可视化模块 (Cross-Section Minute K-Line Visualizer)
阶段: 高级可视化呈现
功能: 在同一截面日内，对比多个核心指数的分钟级 K 线走势，直观展现日内的“失真”与背离现象。
"""

def visualize_cross_section_mins(data_dir: str, output_dir: str, target_date: str = None):
    """
    绘制指定截面日的多个指数分钟 K 线对比图。
    如果 target_date 为 None，则自动选择数据中最新的有效交易日。
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [path for path in data_dir.iterdir() if path.name.endswith("_1min.parquet")]
    if not files:
        print("尚未发现 1min 数据文件，请等待后台抓取脚本运行完毕。")
        return

    print(f"找到 {len(files)} 个指数数据文件，正在加载...")

    all_data = []
    for path in files:
        try:
            df = pd.read_parquet(path)
            all_data.append(df)
        except Exception as e:
            print(f"加载 {path.name} 失败: {e}")

    if not all_data:
        return

    df_combined = pd.concat(all_data, ignore_index=True)

    df_combined['trade_date_str'] = df_combined['trade_time'].astype(str).str[:10]

    if target_date is None:
        target_date = df_combined['trade_date_str'].max()
        print(f"未指定截面日，自动选择最新日期: {target_date}")

    df_section = df_combined[df_combined['trade_date_str'] == target_date].copy()
    if df_section.empty:
        print(f"错误: 截面日 {target_date} 无数据。")
        return

    df_section['trade_time'] = pd.to_datetime(df_section['trade_time'])
    df_section = df_section.sort_values('trade_time')

    DISPLAY_ORDER = ['上证指数', '深证成指', '科创综指', '创业板指', '北证50']
    indices = [i for i in DISPLAY_ORDER if i in df_section['index_name'].unique()]

    fig = make_subplots(
        rows=1, cols=1, 
        subplot_titles=[f"指数日内涨跌幅对比 ({target_date})"]
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    ref_df = pd.DataFrame()

    for idx, index_name in enumerate(indices):
        df_idx_full = df_combined[df_combined['index_name'] == index_name].copy()
        df_past = df_idx_full[df_idx_full['trade_date_str'] < target_date].sort_values('trade_time')

        df_idx = df_section[df_section['index_name'] == index_name].copy()
        df_idx = df_idx.sort_values('trade_time').reset_index(drop=True)

        if ref_df.empty and not df_idx.empty:
            ref_df = df_idx.copy()

        if not df_idx.empty:
            df_idx['close'] = df_idx['close'].astype(float)

            if not df_past.empty:
                pre_close = float(df_past['close'].iloc[-1])
            else:
                pre_close = float(df_idx['open'].iloc[0])

            df_idx['pct_change'] = (df_idx['close'] / pre_close - 1) * 100

            color = colors[idx % len(colors)]

            fig.add_trace(go.Scatter(
                x=df_idx.index,
                y=df_idx['pct_change'],
                mode='lines',
                name=index_name,
                line=dict(width=2, color=color),
                customdata=df_idx['close'],
                hovertemplate='%{y:.2f}% (点位: %{customdata:.2f})'
            ), row=1, col=1)
        
    tick_vals = []
    tick_texts = []
    if not ref_df.empty:
        tick_vals = list(range(0, len(ref_df), 30))
        if len(ref_df) - 1 not in tick_vals:
            tick_vals.append(len(ref_df) - 1)
        tick_texts = ref_df['trade_time'].dt.strftime('%H:%M').iloc[tick_vals].tolist()

    fig.update_layout(
        height=600,
        autosize=True,
        title={
            'text': f"<b>主要指数日内走势对比 (截面日: {target_date})</b>",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=22, color='#333333')
        },
        margin=dict(l=10, r=10, t=80, b=10),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    fig.update_yaxes(
        title_text="日内涨跌幅 (%)",
        ticksuffix="%",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0,0,0,0.2)',
        row=1, col=1
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_texts,
        tickangle=0,
        showgrid=True,
        row=1, col=1
    )

    output_base = output_dir / "index"
    output_base.mkdir(parents=True, exist_ok=True)
    output_path = output_base / "index_distortion_dashboard.html"
    fig.write_html(output_path)
    print(f"\n[Done] 指数失真看板已保存至: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "index_distortion_analysis" / "data_1min"
    OUTPUT_DIR = BASE_DIR / "index_distortion_analysis" / "visual_output"

    raw_date = sys.argv[1] if len(sys.argv) > 1 else None
    if raw_date and len(raw_date) == 8:
        raw_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

    visualize_cross_section_mins(DATA_DIR, OUTPUT_DIR, target_date=raw_date)
