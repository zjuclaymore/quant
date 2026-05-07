import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

"""
连板梯度演变看板 (30-Day Refined Version)
功能: 
1. 30 日趋势展示。
2. 仅为 1, 2, 3 板绘制趋势线，高连板仅展示柱状分布。
3. 优化配色，增强对比度。
"""

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "special_data_analysis" / "data"
OUTPUT_DIR = BASE_DIR / "special_data_analysis" / "visual_output" / "ladder"
TARGET_HISTORY_DAYS = 30
MAX_SCAN_BUSINESS_DAYS = 120
EMPTY_STREAK_LIMIT = 5
REMOTE_LIMIT_HINT = "最近 30 个交易日"


def _iter_business_dates(target_date, limit=MAX_SCAN_BUSINESS_DAYS):
    current = datetime.strptime(target_date, "%Y%m%d")
    yielded = 0
    while yielded < limit:
        if current.weekday() < 5:
            yield current.strftime("%Y%m%d")
            yielded += 1
        current -= timedelta(days=1)


def _normalize_cache_frame(df):
    required_columns = ["日期", "1板", "2板", "3板", "4板及以上", "total"]
    if df is None or df.empty:
        return pd.DataFrame(columns=required_columns)

    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    if "日期" not in normalized.columns:
        return pd.DataFrame(columns=required_columns)

    normalized["日期"] = normalized["日期"].astype(str).str.extract(r"(\d{8})", expand=False)
    normalized = normalized.dropna(subset=["日期"])
    for column in ("1板", "2板", "3板"):
        if column not in normalized.columns:
            normalized[column] = 0
    if "4板及以上" not in normalized.columns:
        four_board = pd.to_numeric(normalized.get("4板", 0), errors="coerce").fillna(0)
        high_board = pd.to_numeric(normalized.get("5板及以上", 0), errors="coerce").fillna(0)
        normalized["4板及以上"] = (four_board + high_board).astype(int)

    for column in ("1板", "2板", "3板", "4板及以上"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype(int)

    if "total" not in normalized.columns:
        normalized["total"] = normalized[["1板", "2板", "3板", "4板及以上"]].sum(axis=1)
    normalized["total"] = pd.to_numeric(normalized["total"], errors="coerce").fillna(
        normalized[["1板", "2板", "3板", "4板及以上"]].sum(axis=1)
    ).astype(int)
    normalized = normalized[required_columns].drop_duplicates(subset=["日期"], keep="last")
    return normalized.sort_values("日期").reset_index(drop=True)


def _load_ladder_cache():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_frames = []
    for cache_path in sorted(DATA_DIR.glob("limit_ladder*.csv")):
        try:
            cache_frames.append(_normalize_cache_frame(pd.read_csv(cache_path)))
        except Exception as exc:
            print(f"[WARN] 读取连板缓存失败 {cache_path.name}: {exc}")
    if not cache_frames:
        return pd.DataFrame(columns=["日期", "1板", "2板", "3板", "4板及以上", "total"])
    combined = pd.concat(cache_frames, ignore_index=True)
    return _normalize_cache_frame(combined)


def _save_ladder_cache(df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "limit_ladder_cache.csv"
    _normalize_cache_frame(df).to_csv(cache_path, index=False, encoding="utf_8_sig")
    return cache_path


def get_ladder_stats_plotly(target_date=None, output_dir=None):
    target_date = target_date or datetime.now().strftime("%Y%m%d")
    print(f"正在分析截至 {target_date} 的 30 日连板梯度精修版...")
    cache_df = _load_ladder_cache()
    cache_map = {
        row["日期"]: {
            "日期": row["日期"],
            "1板": int(row["1板"]),
            "2板": int(row["2板"]),
            "3板": int(row["3板"]),
            "4板及以上": int(row["4板及以上"]),
            "total": int(row["total"]),
        }
        for _, row in cache_df.iterrows()
    }

    all_ladders = {}
    new_rows = []
    failures = []
    remote_disabled = False
    remote_boundary_date = None
    empty_streak = 0

    for trade_date in _iter_business_dates(target_date):
        if trade_date in cache_map:
            all_ladders[trade_date] = cache_map[trade_date]
            empty_streak = 0
        elif not remote_disabled:
            try:
                df_zt = ak.stock_zt_pool_em(date=trade_date)
                if df_zt is None or df_zt.empty:
                    empty_streak += 1
                    failures.append((trade_date, "empty result"))
                    if empty_streak >= EMPTY_STREAK_LIMIT:
                        remote_disabled = True
                        remote_boundary_date = trade_date
                        print(f"[INFO] 连板梯度连续 {EMPTY_STREAK_LIMIT} 个日期无远端结果，后续仅使用本地缓存。")
                    continue

                empty_streak = 0
                board_series = pd.to_numeric(df_zt.iloc[:, 14], errors='coerce').fillna(1)
                ladder = board_series.value_counts().sort_index().to_dict()
                row = {
                    "日期": trade_date,
                    "1板": int(ladder.get(1, 0)),
                    "2板": int(ladder.get(2, 0)),
                    "3板": int(ladder.get(3, 0)),
                    "4板及以上": int(sum(value for board, value in ladder.items() if board >= 4)),
                    "total": int(sum(ladder.values())),
                }
                all_ladders[trade_date] = row
                cache_map[trade_date] = row
                new_rows.append(row)
                time.sleep(0.05)
            except Exception as exc:
                message = str(exc)
                if REMOTE_LIMIT_HINT in message:
                    remote_disabled = True
                    remote_boundary_date = trade_date
                    print(f"[INFO] 连板梯度命中历史边界 {trade_date}，后续仅使用本地缓存。")
                else:
                    failures.append((trade_date, message))
                    print(f"[WARN] 连板梯度抓取失败 {trade_date}: {message}")

        if len(all_ladders) >= TARGET_HISTORY_DAYS:
            break

    if not all_ladders:
        raise RuntimeError("Limit ladder analysis produced no rows. No dashboard was generated.")

    if new_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        cache_path = _save_ladder_cache(cache_df)
        print(f"[INFO] 连板梯度缓存已更新: {cache_path}")

    df = pd.DataFrame(all_ladders.values()).sort_values("日期").tail(TARGET_HISTORY_DAYS)
    df['display_date'] = df['日期'].apply(lambda x: f"{x[4:6]}/{x[6:8]}")

    fig = go.Figure()

    bar_colors = ['#FFEBEE', '#FF8A80', '#D32F2F', '#880E4F']
    line_colors = ['#EF5350', '#E53935', '#B71C1C', None]
    stages = ['1板', '2板', '3板', '4板及以上']

    for i, stage in enumerate(stages):
        fig.add_trace(go.Bar(
            x=df['display_date'],
            y=df[stage],
            name=f"{stage}",
            marker_color=bar_colors[i],
            opacity=0.4 if i == 0 else 0.7, # 1板给高透明度作为背景
            legendgroup=stage
        ))

        if i < 3:
            fig.add_trace(go.Scatter(
                x=df['display_date'],
                y=df[stage],
                name=f"{stage}趋势",
                line=dict(color=line_colors[i], width=3),
                mode='lines+markers',
                marker=dict(size=5),
                legendgroup=stage,
                showlegend=False
            ))

    fig.add_trace(go.Scatter(
        x=df['display_date'],
        y=df['total'],
        name='总涨停家数',
        line=dict(color='#455A64', width=2, dash='dot'),
        mode='lines',
        opacity=0.4
    ))

    fig.update_layout(
        title={
            'text': f"<b>市场连板梯队月度演变趋势 (精修版)</b>",
            'x': 0.5, 'y': 0.95, 'xanchor': 'center',
            'font': dict(size=24, color='#263238')
        },
        xaxis=dict(title="交易日期", type='category', tickangle=-45),
        yaxis=dict(title="家数", gridcolor='#f5f5f5'),
        barmode='stack',
        template='plotly_white',
        height=700,
        margin=dict(l=50, r=50, t=100, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        hovermode="x unified"
    )

    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "limit_ladder_dashboard.html"
    fig.write_html(output_path)
    if len(df) < TARGET_HISTORY_DAYS:
        print(f"[WARN] 连板梯度仅拿到 {len(df)} 个交易日，未满 {TARGET_HISTORY_DAYS} 个。")
    if remote_boundary_date:
        print(f"[INFO] 连板梯度远端抓取在 {remote_boundary_date} 后停止，历史补齐依赖缓存。")
    if failures:
        print(f"[WARN] 连板梯度统计存在 {len(failures)} 个失败或空结果日期。")
    print(f"30日连板精修看板已生成: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    get_ladder_stats_plotly(sys.argv[1] if len(sys.argv) > 1 else None)
