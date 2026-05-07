from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 各指数近似总市值（万亿元），用作质心权重
INDEX_MARKET_CAP = {
    '上证指数': 45.0,
    '深证成指': 25.0,
    '科创综指': 8.0,
    '创业板指': 12.0,
    '北证50':   1.5,
}

DISPLAY_ORDER = list(INDEX_MARKET_CAP.keys())

# 背离阈值（相对质心的涨跌幅差，单位：百分点）
DIVERGENCE_THRESHOLD = 0.3


def _load_section(data_dir: Path, target_date: str) -> pd.DataFrame:
    files = list(data_dir.glob("*_1min.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df['trade_date_str'] = df['trade_time'].astype(str).str[:10]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined['trade_time'] = pd.to_datetime(combined['trade_time'])
    combined = combined.sort_values('trade_time')

    section = combined[combined['trade_date_str'] == target_date].copy()
    prev = combined[combined['trade_date_str'] < target_date]
    return section, prev


def _calc_pct(section: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """计算每个指数每分钟相对昨收的涨跌幅"""
    records = []
    for name in DISPLAY_ORDER:
        df = section[section['index_name'] == name].copy().reset_index(drop=True)
        if df.empty:
            continue
        df['close'] = df['close'].astype(float)

        prev_idx = prev[prev['index_name'] == name]
        pre_close = float(prev_idx['close'].iloc[-1]) if not prev_idx.empty else float(df['open'].iloc[0])

        df['pct'] = (df['close'] / pre_close - 1) * 100
        df['weight'] = INDEX_MARKET_CAP[name]
        records.append(df[['trade_time', 'index_name', 'pct', 'weight']])

    return pd.concat(records, ignore_index=True)


def _calc_centroid(df_pct: pd.DataFrame, anchor_end: str) -> pd.Series:
    """
    以开盘后前5分钟（9:30-9:34）各指数的平均涨幅为基准，
    构建质心系：质心涨幅 = Σ(weight_i * pct_i) / Σ(weight_i)，按分钟计算。
    返回 Series，index 为 trade_time。
    """
    pivot = df_pct.pivot(index='trade_time', columns='index_name', values='pct')
    weights = pd.Series(INDEX_MARKET_CAP)

    def weighted_mean(row):
        valid = row.dropna()
        w = weights[valid.index]
        return (valid * w).sum() / w.sum()

    centroid = pivot.apply(weighted_mean, axis=1)
    return centroid


def detect_divergence(data_dir: str, output_dir: str, target_date: str = None,
                      threshold: float = DIVERGENCE_THRESHOLD):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    section, prev = _load_section(data_dir, target_date)
    if section.empty:
        print(f"无数据: {target_date}")
        return

    df_pct = _calc_pct(section, prev)

    # 质心时间序列
    centroid = _calc_centroid(df_pct, target_date)

    # 每个指数相对质心的偏离
    pivot = df_pct.pivot(index='trade_time', columns='index_name', values='pct')
    deviation = pivot.subtract(centroid, axis=0)  # 正=跑赢质心，负=跑输

    # 背离标记：首次超过阈值的时间点
    divergence_events = {}
    for name in DISPLAY_ORDER:
        if name not in deviation.columns:
            continue
        col = deviation[name].dropna()
        # 只看9:35之后
        col = col[col.index.time >= pd.Timestamp('09:35').time()]
        above = col[col.abs() > threshold]
        if not above.empty:
            first_t = above.index[0]
            divergence_events[name] = {
                'time': first_t,
                'deviation': above.iloc[0],
                'direction': '跑赢' if above.iloc[0] > 0 else '跑输'
            }

    # ---- 绘图 ----
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.6, 0.4],
                        shared_xaxes=True,
                        subplot_titles=[f"指数日内涨跌幅 vs 质心 ({target_date})",
                                        "相对质心偏离 (%)"])

    times = centroid.index
    x = list(range(len(times)))
    time_labels = [t.strftime('%H:%M') for t in times]

    # 上图：各指数涨跌幅 + 质心
    for i, name in enumerate(DISPLAY_ORDER):
        if name not in pivot.columns:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=pivot[name].values,
            mode='lines', name=name,
            line=dict(width=2, color=colors[i]),
            hovertemplate='%{y:.2f}%'
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=centroid.values,
        mode='lines', name='质心',
        line=dict(width=2, color='black', dash='dash'),
        hovertemplate='质心: %{y:.2f}%'
    ), row=1, col=1)

    # 下图：偏离量 + 背离标记
    for i, name in enumerate(DISPLAY_ORDER):
        if name not in deviation.columns:
            continue
        dev_aligned = deviation[name].reindex(times)
        fig.add_trace(go.Scatter(
            x=x, y=dev_aligned.values,
            mode='lines', name=f'{name}偏离',
            line=dict(width=1.5, color=colors[i]),
            showlegend=False,
            hovertemplate='%{y:.2f}%'
        ), row=2, col=1)

        # 背离标记点
        if name in divergence_events:
            ev = divergence_events[name]
            t_idx = times.get_loc(ev['time']) if ev['time'] in times else None
            if t_idx is not None:
                fig.add_trace(go.Scatter(
                    x=[t_idx], y=[ev['deviation']],
                    mode='markers+text',
                    marker=dict(size=10, color=colors[i],
                                symbol='triangle-up' if ev['deviation'] > 0 else 'triangle-down'),
                    text=[f"{name}\n{ev['direction']}"],
                    textposition='top center' if ev['deviation'] > 0 else 'bottom center',
                    showlegend=False,
                    hovertemplate=f"{name} 首次背离: {ev['time'].strftime('%H:%M')}<br>偏离: {ev['deviation']:.2f}%"
                ), row=2, col=1)

    # 阈值参考线
    fig.add_hline(y=threshold, line_dash='dot', line_color='red', opacity=0.4, row=2, col=1)
    fig.add_hline(y=-threshold, line_dash='dot', line_color='green', opacity=0.4, row=2, col=1)
    fig.add_hline(y=0, line_color='black', opacity=0.2, row=2, col=1)

    # x轴刻度（每30分钟）
    tick_step = 30
    tick_vals = list(range(0, len(x), tick_step))
    tick_texts = [time_labels[i] for i in tick_vals]

    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_texts, row=1, col=1)
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_texts, row=2, col=1)
    fig.update_yaxes(ticksuffix='%', zeroline=True, zerolinewidth=1.5,
                     zerolinecolor='rgba(0,0,0,0.2)', row=1, col=1)
    fig.update_yaxes(ticksuffix='%', title_text='偏离质心(%)', row=2, col=1)

    fig.update_layout(
        height=750, autosize=True,
        title=dict(text=f'<b>指数质心背离检测 ({target_date})  阈值={threshold}%</b>',
                   x=0.5, font=dict(size=20)),
        margin=dict(l=10, r=10, t=80, b=10),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    out_dir = output_dir / 'index'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'divergence_dashboard.html'
    fig.write_html(out_path)
    print(f'[Done] 背离检测看板: {out_path}')

    if divergence_events:
        print(f'\n背离事件 (阈值={threshold}%):')
        for name, ev in divergence_events.items():
            print(f'  {name}: {ev["time"].strftime("%H:%M")}  偏离={ev["deviation"]:+.2f}%  ({ev["direction"]}质心)')
    else:
        print('无背离事件')

    return out_path


if __name__ == '__main__':
    import sys
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / 'index_distortion_analysis' / 'data_1min'
    OUTPUT_DIR = BASE_DIR / 'index_distortion_analysis' / 'visual_output'

    raw_date = sys.argv[1] if len(sys.argv) > 1 else None
    if raw_date and len(raw_date) == 8:
        raw_date = f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}'

    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DIVERGENCE_THRESHOLD
    detect_divergence(DATA_DIR, OUTPUT_DIR, target_date=raw_date, threshold=threshold)
